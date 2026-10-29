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
from processor import reference as R
from processor import score as Sc
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


# --- scoring units ---------------------------------------------------------------------

def tiny_ref(rng, n_pat=4, per=12, roi=False):
    """A RefSet of n_pat patients, channels near the origin, one band of coherence edges."""
    n = n_pat * per
    patient = np.repeat([f"P{i}" for i in range(n_pat)], per).astype(object)
    xyz = rng.normal(0, 4, (n, 3))
    rois = np.array(["A" if i % 2 else "B" for i in range(n)], dtype=object) if roi else np.full(n, "", dtype=object)
    ref = R.RefSet("table", Path("."), patient.copy(), np.array([f"c{i}" for i in range(n)], dtype=object), xyz, rois,
                   patient, np.ones(n, bool), np.zeros(n, bool), features={"normpow_delta": rng.normal(0, 1, n)})
    a, b = [], []
    for p in range(n_pat):
        iu, ju = np.triu_indices(per, 1)
        a.append(p * per + iu)
        b.append(p * per + ju)
    ref.edge_a, ref.edge_b = np.concatenate(a), np.concatenate(b)
    ref.edge_bands = ["delta"]
    ref.edge_v = rng.uniform(0, 1, (len(ref.edge_a), 1))
    return ref


def test_edge_zscores_match_brute_force():
    rng = np.random.default_rng(5)
    ref = tiny_ref(rng)
    ref.xyz[ref.patient == "P1"] += 30            # patient 1 sits away from the test channels
    test = Sc.TestRec("t", ["x", "y", "w"], np.array([[0, 0, 0], [3, 0, 0], [30, 0, 0]], float),
                      np.full(3, "", dtype=object), {"normpow_delta": np.array([0.0, 2.0, 1.0])})
    test.pair_i, test.pair_j, test.pair_bands = np.array([0, 0, 1]), np.array([1, 2, 2]), ["delta"]
    test.pair_v = np.array([[0.9], [0.1], [0.5]])
    p = Sc.Params(radius=8, min_reference=5)
    s = Sc.score(test, ref, p, exclude={"P0"})
    M, _ = Sc.regions(test, ref, p, {"P0"})
    for e, (i, j) in enumerate(zip(test.pair_i, test.pair_j)):
        vals = [ref.edge_v[k, 0] for k, (a, b) in enumerate(zip(ref.edge_a, ref.edge_b))
                if (M[i, a] and M[j, b]) or (M[j, a] and M[i, b])]
        assert s.edge_n[e, 0] == len(vals)
        if len(vals) >= 5:
            assert s.edge_z[e, 0] == pytest.approx((test.pair_v[e, 0] - np.mean(vals)) / np.std(vals, ddof=1))
        else:
            assert np.isnan(s.edge_z[e, 0])
    assert not M[:, ref.patient == "P0"].any()      # left out
    # node z of channel 0 against its neighbourhood
    sel = M[0]
    v = ref.features["normpow_delta"][sel]
    assert s.z[0, 0] == pytest.approx((0.0 - v.mean()) / v.std(ddof=1))
    # connectivity of channel 0 = 75th percentile (MATLAB prctile) of its two edges' |z|
    zz = np.abs(s.edge_z[[0, 1], 0])
    if np.isfinite(zz).sum() >= Sc.MIN_EDGES:
        assert s.conn[0, 0] == pytest.approx(np.percentile(zz[np.isfinite(zz)], 75, method="hazen"))


def test_connectivity_percentile_and_summary():
    n = 4
    s = Sc.Scores(["normpow_delta", "normpow_theta", "coh_delta", "coh_theta", "spike_rate"],
                  np.array([[1, -3, 0.5, 1, 2]] * n, float), np.full((n, 5), 20), np.full(n, "within-20mm", object),
                  np.full(n, 20), ["delta", "theta"], np.array([[2, 4]] * (n - 1) + [[np.nan, np.nan]], float),
                  np.full((n, 2), 5), np.zeros((0, 2)), np.zeros((0, 2), int))
    out = Sc.summarise(s, Sc.Params())
    assert out["max_abs_z"][0] == 3 and out["max_band"][0] == "theta" and out["spectral_mean_abs_z"][0] == 2
    assert out["coherence_mean_abs_z"][0] == 0.75 and out["connectivity_mean"][0] == 3
    assert out["connectivity_source"][0] == "edges" and out["connectivity_source"][3] == "node-strength"
    assert out["connectivity_mean"][3] == 0.75                      # fallback to node strength
    assert out["combined"][0] == pytest.approx(np.mean([1, 3, 2, 4]))
    assert out["combined"][3] == pytest.approx(np.mean([1, 3, 0.5, 1]))
    assert out["other_max_abs_z"][0] == 2
    assert Sc.family("conn_delta") == "connectivity" and Sc.family("logpow_beta") == "spectral"


def test_roi_regions():
    rng = np.random.default_rng(6)
    ref = tiny_ref(rng, roi=True)
    test = Sc.TestRec("t", ["a", "b", "c"], np.array([[0, 0, 0], [0, 0, 0], [np.nan] * 3]),
                      np.array(["A", "", "B"], dtype=object), {"normpow_delta": np.zeros(3)})
    M, label = Sc.regions(test, ref, Sc.Params(region="auto"), set())
    assert list(label) == ["roi:A", "within-20mm", "roi:B"]
    assert (ref.roi[M[0]] == "A").all() and M[0].sum() == (ref.roi == "A").sum()
    M, label = Sc.regions(test, ref, Sc.Params(region="roi"), set())
    assert list(label) == ["roi:A", "unknown", "roi:B"]
    M, label = Sc.regions(test, ref, Sc.Params(region="radius"), set())
    assert list(label) == ["within-20mm", "within-20mm", "unknown"]


# --- end to end ------------------------------------------------------------------------

def test_run(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir, out) == 0
    ab = table(out / "abnormality.csv")
    assert list(ab[0]) == ["recording_id", "channel", "feature", "z", "abs_z", "region", "n_reference", "family"]
    ch = {(r["recording_id"], r["channel"]): r for r in table(out / "channel_abnormality.csv")}
    assert list(next(iter(ch.values()))) == ["recording_id", "channel", "max_abs_z", "max_band", "region",
                                             "n_reference"] + Sc.SUMMARY_COLUMNS[2:]
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
    mine = [r for r in ab if (r["recording_id"], r["channel"]) == (NEW_REC, "LA01-LA02")]
    bands = ("delta", "theta", "alpha", "beta", "gamma")
    assert {r["feature"] for r in mine} == {f"{p}_{b}" for p in ("relpow", "normpow", "logpow", "coh", "conn")
                                            for b in bands}
    conn = [r for r in mine if r["family"] == "connectivity"]
    assert all(r["z"] == "" and float(r["abs_z"]) >= 0 for r in conn)
    assert ch[(NEW_REC, "LA01-LA02")]["connectivity_source"] == "edges"
    ed = table(out / "edge_abnormality.csv")
    assert list(ed[0]) == ["recording_id", "channel_a", "channel_b", "band", "coherence", "z", "abs_z", "n_reference"]
    assert len([r for r in ed if r["recording_id"] == NEW_REC]) == 15 * 5
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["recordings"][ATLAS_REC]["left_out"] == ["HUPatlas002"]
    assert prov["reference"]["kind"] == "atlas" and "spike_24h" in prov["reference"]["normative_rule"]
    assert prov["reference"]["normative_channels"] < 120          # 120 minus the irritative zone (default)
    val = {(r["scope"], r["label"], r["score"]): r for r in table(out / "validation.csv")}
    assert float(val[("all", "soz", "spectral_max_abs_z")]["auc"]) > 0.9
    assert {k[2] for k in val} >= {"spectral_max_abs_z", "spectral_mean_abs_z", "coherence_mean_abs_z",
                                   "connectivity_mean", "combined"}


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
    assert run(monkeypatch, fixture_dir, tmp_path / "o2", SPECTRAL_FEATURES="logpow", EXCLUDE_IRRITATIVE="false",
               EDGES="false", LABEL="x") == 0
    prov = json.loads((tmp_path / "o2" / "x" / "provenance.json").read_text())
    assert "spike_24h" not in prov["reference"]["normative_rule"] and prov["reference"]["normative_channels"] == 120
    ch = table(tmp_path / "o2" / "x" / "channel_abnormality.csv")
    assert {r["spectral_max_feature"].split("_")[0] for r in ch if r["spectral_max_feature"]} == {"logpow"}
    assert {r["connectivity_source"] for r in ch if r["region"].startswith("within")} == {"node-strength"}
    assert run(monkeypatch, fixture_dir, tmp_path / "o3", REGION_MODE="aal") == 1
    assert run(monkeypatch, fixture_dir, tmp_path / "o4", SPECTRAL_FEATURES="coh") == 1


def test_format_a_table_equals_atlas(fixture_dir, tmp_path, monkeypatch):
    """The atlas exported to Format A by scripts/build_reference.py gives the same z-scores as the .mat."""
    from scripts import build_reference as B
    inp = tmp_path / "in"
    inp.mkdir()
    for sub in ("features", "ieeg"):
        os.symlink(fixture_dir / sub, inp / sub)
    B.main(["--from-atlas", str(fixture_dir / "reference" / "HUP_atlas_final.mat"), "--out", str(inp / "reference")])
    assert run(monkeypatch, inp, tmp_path / "a") == 0
    assert run(monkeypatch, fixture_dir, tmp_path / "b") == 0
    za = {(r["recording_id"], r["channel"], r["feature"]): r for r in table(tmp_path / "a" / "abnormality.csv")}
    zb = {(r["recording_id"], r["channel"], r["feature"]): r for r in table(tmp_path / "b" / "abnormality.csv")}
    assert za.keys() == zb.keys()
    for k, r in zb.items():
        assert za[k]["region"] == r["region"] and za[k]["n_reference"] == r["n_reference"]
        if r["abs_z"]:
            assert float(za[k]["abs_z"]) == pytest.approx(float(r["abs_z"]), rel=2e-3, abs=2e-3)
    prov = json.loads((tmp_path / "a" / "provenance.json").read_text())
    assert prov["reference"]["kind"] == "table" and prov["recordings"][ATLAS_REC]["left_out"] == ["HUPatlas002"]


def test_build_reference_from_cohort(fixture_dir, tmp_path, monkeypatch):
    from scripts import build_reference as B
    out = tmp_path / "ref"
    B.main(["--ieeg", str(fixture_dir / "ieeg"), "--features", str(fixture_dir / "features"), "--out", str(out)])
    rows = table(out / "reference_channels.csv")
    assert list(rows[0]) == B.COLUMNS and len(rows) == 30
    norm = {(r["recording_id"], r["channel"]): r["normative"] for r in rows}
    assert norm[(NEW_REC, "LB01-LB02")] == "false" and norm[(NEW_REC, "LA01-LA02")] == "true"
    assert {r["patient"] for r in rows} == {"HUPatlas002", "P01"}
    assert (out / "coherence.csv").is_file() and json.loads((out / "provenance.json").read_text())["spectral"]
    # P01 scored against it: P01 is left out, so only HUPatlas002's channels remain
    inp = tmp_path / "in"
    inp.mkdir()
    for sub in ("features", "ieeg"):
        os.symlink(fixture_dir / sub, inp / sub)
    os.symlink(out, inp / "reference")
    assert run(monkeypatch, inp, tmp_path / "o", MIN_REFERENCE=3) == 0
    prov = json.loads((tmp_path / "o" / "provenance.json").read_text())
    assert prov["recordings"][NEW_REC]["left_out"] == ["P01"]


def test_legacy_bandpower_only(fixture_dir, tmp_path, monkeypatch):
    """An older ieeg-bandpower output (bandpower.csv only) still works: spectral features, no connectivity."""
    inp = tmp_path / "in"
    shutil.copytree(fixture_dir, inp)
    (inp / "features" / "features.csv").unlink()
    (inp / "features" / "coherence.csv").unlink()
    assert run(monkeypatch, inp, tmp_path / "o") == 0
    ab = table(tmp_path / "o" / "abnormality.csv")
    assert {r["family"] for r in ab} == {"spectral"}
    ch = {(r["recording_id"], r["channel"]): r for r in table(tmp_path / "o" / "channel_abnormality.csv")}
    assert float(ch[(NEW_REC, "LB01-LB02")]["max_abs_z"]) > 3


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
    assert len(table(tmp_path / "out" / "edge_abnormality.csv")) > 0


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
