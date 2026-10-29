import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from processor import atlas as A
from processor import inputs as I
from processor import main as M
from processor import normative as N
from processor import spectral as S
from tests import make_fixture as F

ATLAS_REC = "sub-HUPatlas002_ses-atlas_task-interictal_run-01"
NEW_REC = "sub-P01_ses-implant1_task-interictal01_run-01"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    F.main(str(d))
    return d / "linked"


def run(monkeypatch, input_dir, output_dir, **env):
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


def table(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --- units -----------------------------------------------------------------------------

def test_auc():
    assert N.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert N.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert N.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert N.auc([1, np.nan, 3], [0, 1, 1]) == 1.0
    assert np.isnan(N.auc([1, 2], [1, 1]))


def test_zscore_neighbourhood_and_leave_one_out():
    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal(0, 3, (40, 3)), rng.normal(100, 3, (5, 3))])
    patient = np.r_[np.repeat([1, 2], 20), np.full(5, 3)]
    vals = rng.normal(0, 1, (45, 2))
    ref = N.Reference(xyz, patient, vals)
    test_xyz = np.array([[0, 0, 0], [100, 100, 100], [np.nan] * 3, [500, 0, 0]], float)
    r = N.zscore(ref, test_xyz, np.array([[3, -3], [0, 0], [0, 0], [0, 0]], float), None, 20, 10)
    assert list(r.status) == ["ok", "insufficient", "unknown", "insufficient"]
    assert r.n_reference[0] == 40 and r.n_reference[1] == 5
    m, s = vals[:40].mean(0), vals[:40].std(0, ddof=1)
    assert np.allclose(r.z[0], (np.array([3, -3]) - m) / s)
    assert np.isnan(r.z[1:]).all()
    r1 = N.zscore(ref, test_xyz[:1], np.array([[3, -3]], float), 1, 20, 10)
    assert r1.n_reference[0] == 20          # patient 1 left out


def test_feature_transforms():
    a, r, n = np.array([10.0]), np.array([0.1]), np.array([0.01])
    assert N.transform("norm_median", a, r, n)[0] == 0.01
    assert N.transform("log_rel_power", a, r, n)[0] == pytest.approx(-1)
    assert N.transform("log_abs_power", a, r, n)[0] == pytest.approx(1)
    with pytest.raises(ValueError):
        N.transform("nope", a, r, n)


def test_atlas_patient_detection():
    ie = I.Ieeg(atlas_patient={"P77": 12})
    assert I.atlas_patient_of("HUPatlas007", ie) == 7
    assert I.atlas_patient_of("HUPatlas123", ie) == 123
    assert I.atlas_patient_of("P77", ie) == 12
    assert I.atlas_patient_of("HUP224", ie) is None
    assert I.is_mni("MNI152NLin2009aSym") and I.is_mni("MNI152") and not I.is_mni("ACPC") and not I.is_mni("Other")


def test_atlas_load_v5(fixture_dir):
    at = A.load(fixture_dir / "reference" / "HUP_atlas_final.mat")
    assert at.n_channels == 144 and at.wake_clip.shape == (12000, 144)
    assert at.soz.sum() == 18 and at.resected.sum() == 24 and set(np.unique(at.patient)) == set(range(1, 7))


def test_atlas_v73_without_h5py_is_explained(tmp_path):
    p = tmp_path / "x.mat"
    p.write_bytes(b"\x00" * 512 + b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert A.is_hdf5(p)


def test_reference_uses_the_shared_spectral_code(fixture_dir):
    """The reference band table equals what spectral.stream gives for the same clip (bandpower's code path)."""
    at = A.load(fixture_dir / "reference" / "HUP_atlas_final.mat")
    p = S.SpectralParams()
    tab = A.reference_features(at, p, batch=50)
    x = at.wake_clip[:, :3].T
    acc, _ = S.stream(lambda a, b: x[:, a:b], [(0, x.shape[1])], 200, 3, p, prefiltered=True)
    t2 = S.band_table(*acc.psd(), p.bands)
    assert np.allclose(tab.norm_median[:3], t2.norm_median) and np.allclose(tab.abs_power[:3], t2.abs_power)


# --- end to end ------------------------------------------------------------------------

def test_run(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir, out) == 0
    ab = table(out / "abnormality.csv")
    assert list(ab[0]) == ["recording_id", "channel", "band", "z", "abs_z", "region", "n_reference"]
    ch = {(r["recording_id"], r["channel"]): r for r in table(out / "channel_abnormality.csv")}
    assert list(next(iter(ch.values()))) == ["recording_id", "channel", "max_abs_z", "max_band", "region",
                                             "n_reference"]
    assert ch[(NEW_REC, "LX01-LX02")]["region"] == "unknown" and ch[(NEW_REC, "LX01-LX02")]["max_abs_z"] == ""
    assert ch[(NEW_REC, "RZ01-RZ02")]["region"] == "insufficient"
    assert ch[(NEW_REC, "LA01-LA02")]["region"] == "within-20mm"
    # the delta-excess channels are the abnormal ones
    z_soz = [float(ch[(NEW_REC, c)]["max_abs_z"]) for c in ("LB01-LB02", "LB02-LB03")]
    z_ok = [float(ch[(NEW_REC, c)]["max_abs_z"]) for c in ("LA01-LA02", "LA02-LA03")]
    assert min(z_soz) > 3 * max(z_ok)
    for r in ab:
        if r["z"]:
            assert float(r["abs_z"]) == pytest.approx(abs(float(r["z"])), rel=1e-3)
    # every band present per channel
    assert len([r for r in ab if (r["recording_id"], r["channel"]) == (NEW_REC, "LA01-LA02")]) == 5
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["recordings"][ATLAS_REC]["atlas_patient_left_out"] == 2
    assert prov["reference"]["normative_channels"] == 120
    val = {(r["scope"], r["label"]): r for r in table(out / "validation.csv")}
    assert float(val[("all", "soz")]["auc_max_abs_z"]) > 0.9


def test_leave_one_patient_out_matters(fixture_dir, tmp_path, monkeypatch):
    """n_reference of the atlas patient's channels drops when its own channels are left out."""
    assert run(monkeypatch, fixture_dir, tmp_path / "lopo") == 0
    assert run(monkeypatch, fixture_dir, tmp_path / "all", LEAVE_PATIENT_OUT="false") == 0
    a = {r["channel"]: int(r["n_reference"]) for r in table(tmp_path / "lopo" / "channel_abnormality.csv")
         if r["recording_id"] == ATLAS_REC}
    b = {r["channel"]: int(r["n_reference"]) for r in table(tmp_path / "all" / "channel_abnormality.csv")
         if r["recording_id"] == ATLAS_REC}
    assert all(a[c] < b[c] for c in a)


def test_parameters(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir, tmp_path / "o", RADIUS_MM=5, MIN_REFERENCE=50) == 0
    ch = table(tmp_path / "o" / "channel_abnormality.csv")
    assert all(r["region"] in ("insufficient", "unknown") for r in ch)
    assert run(monkeypatch, fixture_dir, tmp_path / "o2", FEATURE="log_rel_power", EXCLUDE_IRRITATIVE="true",
               LABEL="x") == 0
    prov = json.loads((tmp_path / "o2" / "x" / "provenance.json").read_text())
    assert "spike_24h" in prov["reference"]["normative_rule"]
    assert run(monkeypatch, fixture_dir, tmp_path / "o3", FEATURE="bogus") == 1


def test_flat_input_layout(fixture_dir, tmp_path, monkeypatch):
    """No links: features, ieeg and reference merged side by side at the root of INPUT_DIR."""
    inp = tmp_path / "in"
    inp.mkdir()
    for sub in ("features", "reference"):
        for p in (fixture_dir / sub).iterdir():
            os.symlink(p, inp / p.name)
    for p in (fixture_dir / "ieeg").iterdir():
        os.symlink(p, inp / p.name)
    assert run(monkeypatch, inp, tmp_path / "out") == 0
    assert len(table(tmp_path / "out" / "channel_abnormality.csv")) == 30


def test_missing_reference_or_features_fails(fixture_dir, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    shutil.copytree(fixture_dir, inp)
    (inp / "reference" / "HUP_atlas_final.mat").unlink()
    assert run(monkeypatch, inp, tmp_path / "o") == 1
    inp2 = tmp_path / "in2"
    shutil.copytree(fixture_dir, inp2)
    shutil.rmtree(inp2 / "features")
    assert run(monkeypatch, inp2, tmp_path / "o2") == 1


def test_no_coordinates_gives_unknown(fixture_dir, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    shutil.copytree(fixture_dir, inp)
    shutil.rmtree(inp / "ieeg")
    assert run(monkeypatch, inp, tmp_path / "o") == 0
    assert {r["region"] for r in table(tmp_path / "o" / "channel_abnormality.csv")} == {"unknown"}
    assert not (tmp_path / "o" / "validation.csv").exists()


def test_lambda_handler(fixture_dir, tmp_path, monkeypatch):
    from processor import handler
    res = handler.handler({"inputDir": str(fixture_dir), "outputDir": str(tmp_path / "out"), "RADIUS_MM": 25}, None)
    assert res["status"] == "ok"
    assert {r["region"] for r in table(tmp_path / "out" / "channel_abnormality.csv")} >= {"within-25mm"}
    with pytest.raises(RuntimeError):
        handler.handler({"inputDir": str(tmp_path / "none"), "outputDir": str(tmp_path / "o2")}, None)


def test_spectral_module_is_vendored_identically():
    other = Path(__file__).resolve().parents[2] / "processor-ieeg-bandpower" / "processor" / "spectral.py"
    if not other.exists():
        pytest.skip("processor-ieeg-bandpower not checked out next to this repo")
    mine = Path(__file__).resolve().parents[1] / "processor" / "spectral.py"
    assert mine.read_bytes() == other.read_bytes(), "spectral.py differs from processor-ieeg-bandpower's copy"
