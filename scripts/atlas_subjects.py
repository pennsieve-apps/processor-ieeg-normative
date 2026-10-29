"""
Writes atlas patients as a small ieeg-prep-style BIDS-iEEG derivative (conventions.md), so the
whole chain ieeg-bandpower -> ieeg-normative can be validated on them while ieeg-prep does not
yet emit atlas subjects:

  <out>/recordings.csv
  <out>/sub-HUPatlas<NNN>/ses-atlas/ieeg/
      sub-HUPatlas<NNN>_ses-atlas_task-interictal_run-01_desc-preproc_ieeg.edf      200 Hz, 60 s, EDF 16-bit
      ..._desc-preproc_channels.tsv      x/y/z (MNI152 centroid), soz, resected, low/high cutoff 0.5/80
      ..._desc-preproc_events.tsv        (empty)
      sub-HUPatlas<NNN>_ses-atlas_desc-preproc_electrodes.tsv   atlas_patient

  python -m scripts.atlas_subjects <HUP_atlas_final.mat> <out-dir> [--patients 1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from processor import atlas as A


def write_edf(path: Path, data_uv: np.ndarray, sfreq: int, labels: list[str]) -> None:
    """EDF, 16-bit, 1 s records, per-channel physical range = robust max |x| * 1.1 (conventions.md "Precision")."""
    n_ch, n = data_uv.shape
    n_rec = n // sfreq
    data_uv = data_uv[:, : n_rec * sfreq]
    pmax = np.maximum(np.abs(data_uv).max(axis=1) * 1.1, 1.0)
    pmin = -pmax
    dmin, dmax = -32768, 32767
    dig = np.round((data_uv - pmin[:, None]) / (pmax - pmin)[:, None] * (dmax - dmin) + dmin)
    dig = np.clip(dig, dmin, dmax).astype("<i2")

    def f(s, w):
        return str(s)[:w].ljust(w).encode("ascii")

    hdr = b"".join([f("0", 8), f("X X X X", 80), f("Startdate 01-JAN-2020 X X X", 80), f("01.01.20", 8),
                    f("00.00.00", 8), f(256 * (n_ch + 1), 8), f("", 44), f(n_rec, 8), f(1, 8), f(n_ch, 4)])
    cols = [[f(lab, 16) for lab in labels], [f("", 80)] * n_ch, [f("uV", 8)] * n_ch,
            [f(f"{v:.2f}", 8) for v in pmin], [f(f"{v:.2f}", 8) for v in pmax], [f(dmin, 8)] * n_ch,
            [f(dmax, 8)] * n_ch, [f("HP:0.5Hz LP:80Hz", 80)] * n_ch, [f(sfreq, 8)] * n_ch, [f("", 32)] * n_ch]
    hdr += b"".join(b"".join(c) for c in cols)
    path.write_bytes(hdr + dig.reshape(n_ch, n_rec, sfreq).transpose(1, 0, 2).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat")
    ap.add_argument("out")
    ap.add_argument("--patients", default="1,2,3,4,5")
    args = ap.parse_args()
    atlas = A.load(Path(args.mat))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_description.json").write_text(json.dumps({
        "Name": "HUP Interictal iEEG Atlas patients (Discover 179) as a BIDS-iEEG derivative", "BIDSVersion": "1.9.0",
        "DatasetType": "derivative", "GeneratedBy": [{"Name": "processor-ieeg-normative scripts/atlas_subjects.py"}]},
        indent=2))
    rows = []
    for p in [int(x) for x in args.patients.split(",")]:
        m = np.where(atlas.patient == p)[0]
        sub = f"HUPatlas{p:03d}"
        d = out / f"sub-{sub}" / "ses-atlas" / "ieeg"
        d.mkdir(parents=True, exist_ok=True)
        stem = f"sub-{sub}_ses-atlas_task-interictal_run-01"
        names = [f"CH{i + 1:03d}-CH{i + 2:03d}" for i in range(len(m))]  # the atlas has no channel names
        write_edf(d / f"{stem}_desc-preproc_ieeg.edf", atlas.wake_clip[:, m].T, 200, names)
        with open(d / f"{stem}_desc-preproc_channels.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "status",
                        "status_description", "contacts", "soz", "resected", "onset", "spread", "x", "y", "z", "space",
                        "roi"])
            for n, i in zip(names, m):
                w.writerow([n, "SEEG", "µV", 0.5, 80, 200, "good", "n/a", n.replace("-", "|"),
                            str(bool(atlas.soz[i])).lower(), str(bool(atlas.resected[i])).lower(), "n/a", "n/a",
                            *[f"{v:.3f}" for v in atlas.xyz[i]], "MNI152", "n/a"])
        (d / f"{stem}_desc-preproc_events.tsv").write_text("onset\tduration\ttrial_type\tsource\n")
        (d / f"sub-{sub}_ses-atlas_desc-preproc_electrodes.tsv").write_text(
            "name\tx\ty\tz\tatlas_patient\n" + "".join(
                f"{n.split('-')[0]}\t{atlas.xyz[i][0]:.3f}\t{atlas.xyz[i][1]:.3f}\t{atlas.xyz[i][2]:.3f}\t{p}\n"
                for n, i in zip(names, m)))
        rows.append([stem, sub, "atlas", "interictal", "01", f"sub-{sub}/ses-atlas/ieeg/{stem}_desc-preproc_ieeg.edf",
                     200, len(m), len(m), 60, "bipolar", "interictal", "", "", "", "mat"])
        print(f"sub-{sub}: {len(m)} channels, {int(atlas.soz[m].sum())} soz, {int(atlas.resected[m].sum())} resected")
    with open(out / "recordings.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good",
                    "duration_s", "montage", "kind", "onset_s", "baseline_start_s", "baseline_end_s", "source_format"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
