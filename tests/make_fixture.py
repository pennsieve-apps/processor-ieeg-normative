"""
A tiny synthetic input for the tests and `make run`, laid out as the three linked ports:

  <out>/linked/reference/HUP_atlas_final.mat   a synthetic atlas in the Discover 179 layout
        (MATLAB 5 file): 6 patients x 24 bipolar channels, 60 s at 200 Hz, MNI centroids in two
        clusters (left and right temporal), soz_ch/resected_ch on 3 channels per patient,
        spike_24h. Channels: 1/f background; SOZ channels carry extra delta.
  <out>/linked/ieeg/            an ieeg-prep derivative without signals: recordings.csv and the
        _channels.tsv (x, y, z, space, soz, resected) of two recordings, plus _electrodes.tsv
        - sub-HUPatlas002: atlas patient 2's own channels (left out of the reference)
        - sub-P01: 6 channels: two normal, two with a strong delta excess (soz), one without
          coordinates (region unknown), one far from any atlas channel (insufficient)
  <out>/linked/features/        features.csv, coherence.csv, bandpower.csv + provenance.json for both
        recordings, computed with processor/spectral.py (as ieeg-bandpower would)

Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import io as sio

from processor import spectral as S

FS = 200
N_S = 12000
CLUSTERS = {"L": (-30.0, -15.0, -15.0), "R": (30.0, -15.0, -15.0)}


def pink(rng, n_ch: int, n: int = N_S) -> np.ndarray:
    white = rng.standard_normal((n_ch, n))
    spec = np.fft.rfft(white, axis=1)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec /= np.maximum(f, 0.5) ** 0.75
    x = np.fft.irfft(spec, n=n, axis=1)
    return 20 * x / x.std(axis=1, keepdims=True)


def delta_burst(rng, n: int = N_S) -> np.ndarray:
    t = np.arange(n) / FS
    return 60 * np.sin(2 * np.pi * (2 + 0.3 * rng.standard_normal()) * t)


def make_atlas(path: Path, rng) -> dict:
    n_pat, per = 6, 24
    clip, xyz, pat, soz, res, spk = [], [], [], [], [], []
    for p in range(1, n_pat + 1):
        x = pink(rng, per)
        for c in range(per):
            cl = CLUSTERS["L" if c < per // 2 else "R"]
            xyz.append(np.array(cl) + rng.normal(0, 6, 3))
            is_soz = c < 3
            if is_soz:
                x[c] += delta_burst(rng)
            soz.append(1.0 if is_soz else 0.0)
            res.append(1.0 if c < 4 else 0.0)
            spk.append(float(rng.integers(0, 30)) if not is_soz else 500.0)   # a few irritative (> 24)
            pat.append(float(p))
        clip.append(S.preprocess(x, FS, S.SpectralParams()))  # "filtered 0.5-80 Hz", like the real atlas
    data = {"wake_clip": np.concatenate(clip).T, "mni_coords": np.array(xyz),
            "patient_no": np.array(pat)[:, None], "soz_ch": np.array(soz)[:, None],
            "resected_ch": np.array(res)[:, None], "spike_24h": np.array(spk)[:, None],
            "depth_elecs": np.ones((len(pat), 1))}
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), data, do_compression=False)
    return data


def write_tsv(path: Path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows([["n/a" if v is None or v == "" or (isinstance(v, float) and np.isnan(v)) else v for v in r]
                     for r in rows])


def bandpower_rows(rid: str, names: list[str], x: np.ndarray, params: S.SpectralParams, prefiltered=False):
    """(bandpower.csv, features.csv, coherence.csv rows), as ieeg-bandpower writes them."""
    acc, _ = S.stream(lambda a, b: x[:, a:b], [(0, x.shape[1])], FS, x.shape[0], params, prefiltered=prefiltered,
                      cross=True)
    f, p = acc.psd()
    t = S.band_table(f, p, params.bands)
    bp = [[rid, n, band, f"{t.abs_power[i, b]:.6g}", f"{t.rel_power[i, b]:.6g}", f"{t.norm_median[i, b]:.6g}"]
          for i, n in enumerate(names) for b, band in enumerate(t.bands)]
    cn, cb = S.coherence_bands(acc.coh.freqs[: acc.coh.nf], acc.coh.coherence(), params.bands)
    ft = [[rid, n, name, f"{v[i]:.6g}"] for name, v in S.channel_features(t, (cn, cb)) for i, n in enumerate(names)]
    coh = [[rid, names[i], names[j], band, f"{cb[i, j, b]:.6g}"] for i in range(len(names))
           for j in range(i + 1, len(names)) for b, band in enumerate(cn)]
    return bp, ft, coh


def main(out: str) -> None:
    rng = np.random.default_rng(42)
    root = Path(out) / "linked"
    atlas = make_atlas(root / "reference" / "HUP_atlas_final.mat", rng)
    params = S.SpectralParams()
    ieeg = root / "ieeg"
    feats = root / "features"
    feats.mkdir(parents=True, exist_ok=True)
    rows, frows, crows, recs = [], [], [], []
    ch_cols = ["name", "type", "units", "status", "contacts", "soz", "resected", "x", "y", "z", "space", "roi"]

    # atlas patient 2, as ieeg-prep would deliver it (sub-HUPatlas002)
    m = atlas["patient_no"].ravel() == 2
    rid = "sub-HUPatlas002_ses-atlas_task-interictal_run-01"
    names = [f"A{i + 1:02d}-A{i + 2:02d}" for i in range(int(m.sum()))]
    d = ieeg / "sub-HUPatlas002" / "ses-atlas" / "ieeg"
    d.mkdir(parents=True, exist_ok=True)
    write_tsv(d / f"{rid}_desc-preproc_channels.tsv", ch_cols, [
        [n, "SEEG", "µV", "good", n.replace("-", "|"), "true" if s else "false", "true" if r else "false", *xyz,
         "MNI152", ""] for n, s, r, xyz in zip(names, atlas["soz_ch"].ravel()[m], atlas["resected_ch"].ravel()[m],
                                                atlas["mni_coords"][m])])
    write_tsv(d / "sub-HUPatlas002_ses-atlas_desc-preproc_electrodes.tsv", ["name", "x", "y", "z", "atlas_patient"],
              [["A01", -30, -15, -15, 2]])
    for acc_, new in zip((rows, frows, crows), bandpower_rows(rid, names, atlas["wake_clip"][:, m].T, params,
                                                             prefiltered=True)):
        acc_ += new
    recs.append([rid, "HUPatlas002", "atlas", "interictal", "01"])

    # a new patient
    rid = "sub-P01_ses-implant1_task-interictal01_run-01"
    names = ["LA01-LA02", "LA02-LA03", "LB01-LB02", "LB02-LB03", "LX01-LX02", "RZ01-RZ02"]
    x = pink(rng, 6)
    x[2] += 3 * delta_burst(rng)
    x[3] += 3 * delta_burst(rng)
    L = CLUSTERS["L"]
    coords = [L, (L[0] - 3, L[1], L[2]), (L[0] - 6, L[1] + 3, L[2]), (L[0] - 9, L[1] + 3, L[2]),
              (np.nan, np.nan, np.nan), (60.0, 60.0, 60.0)]
    soz = ["false", "false", "true", "true", "false", "false"]
    d = ieeg / "sub-P01" / "ses-implant1" / "ieeg"
    d.mkdir(parents=True, exist_ok=True)
    write_tsv(d / f"{rid}_desc-preproc_channels.tsv", ch_cols, [
        [n, "SEEG", "µV", "good", n.replace("-", "|"), s, s, *c, "" if np.isnan(c[0]) else "MNI152NLin2009aSym", ""]
        for n, s, c in zip(names, soz, coords)])
    for acc_, new in zip((rows, frows, crows), bandpower_rows(rid, names, x, params)):
        acc_ += new
    recs.append([rid, "P01", "implant1", "interictal01", "01"])

    with open(ieeg / "recordings.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "subject", "session", "task", "run"])
        w.writerows(recs)
    with open(feats / "bandpower.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel", "band", "abs_power", "rel_power", "norm_median"])
        w.writerows(rows)
    with open(feats / "features.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel", "feature", "value"])
        w.writerows(frows)
    with open(feats / "coherence.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel_a", "channel_b", "band", "coherence"])
        w.writerows(crows)
    (feats / "provenance.json").write_text(json.dumps({"app": "processor-ieeg-bandpower", "spectral": params.as_dict()},
                                                      indent=2))
    print(f"fixture written to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")
