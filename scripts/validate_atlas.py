"""
Leave-one-patient-out validation directly on the HUP atlas: every atlas patient is scored
against the normative channels of the other 59 (the same code the app runs), and the AUC of
max_abs_z for SOZ and for resected channels is reported, pooled over channels and per patient.

  python -m scripts.validate_atlas <HUP_atlas_final.mat> [--patients 1,2,3] [--out validation.csv]

Grid: FEATURE x BANDS preset x RADIUS_MM x irritative-zone exclusion (MIN_REFERENCE fixed).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from processor import atlas as A
from processor import normative as N
from processor import spectral as S


def run(atlas: A.Atlas, table: S.BandTable, feature: str, radius: float, min_ref: int, exclude_iz: bool,
        patients: list[int]) -> dict:
    nb = len(table.bands) - 1  # without broadband
    vals = N.transform(feature, table.abs_power[:, :nb], table.rel_power[:, :nb], table.norm_median[:, :nb])
    normative = ~atlas.soz & ~atlas.resected
    if exclude_iz and atlas.spike_24h is not None:
        normative &= ~(atlas.spike_24h > A.IRRITATIVE_SPIKES_24H)
    ref = N.Reference(atlas.xyz[normative], atlas.patient[normative], vals[normative])
    maxz, soz, res, pat, scored = [], [], [], [], 0
    per_soz, per_res = [], []
    total = 0
    for p in patients:
        m = atlas.patient == p
        r = N.zscore(ref, atlas.xyz[m], vals[m], exclude_patient=p, radius=radius, min_reference=min_ref)
        mz = np.nanmax(np.abs(r.z), axis=1, initial=-np.inf)
        mz[r.status != "ok"] = np.nan
        total += m.sum()
        scored += int((r.status == "ok").sum())
        maxz.append(mz), soz.append(atlas.soz[m]), res.append(atlas.resected[m]), pat.append(np.full(m.sum(), p))
        per_soz.append(N.auc(mz, atlas.soz[m]))
        per_res.append(N.auc(mz, atlas.resected[m]))
    mz, sz, rs = np.concatenate(maxz), np.concatenate(soz), np.concatenate(res)
    return {"auc_soz": N.auc(mz, sz), "auc_resected": N.auc(mz, rs),
            "auc_soz_patient_median": float(np.nanmedian(per_soz)),
            "auc_resected_patient_median": float(np.nanmedian(per_res)),
            "scored": scored, "channels": int(total), "n_normative": int(normative.sum()),
            "per_patient_soz": per_soz, "per_patient_res": per_res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat")
    ap.add_argument("--patients", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--radii", default="10,15,20,30")
    ap.add_argument("--min-reference", type=int, default=10)
    args = ap.parse_args()
    t = time.monotonic()
    atlas = A.load(Path(args.mat))
    print(f"atlas: {atlas.n_channels} channels, {len(np.unique(atlas.patient))} patients, loaded {time.monotonic() - t:.1f}s")
    patients = [int(x) for x in args.patients.split(",")] if args.patients else sorted(np.unique(atlas.patient).tolist())
    rows = []
    for preset in ("conventions", "bernabei2022"):
        t = time.monotonic()
        table = A.reference_features(atlas, S.SpectralParams(bands=S.BAND_PRESETS[preset]))
        print(f"reference features ({preset}): {time.monotonic() - t:.1f}s")
        for feature in ("norm_median", "taylor2022", "log_rel_power"):
            for exclude_iz in (False, True):
                for radius in [float(r) for r in args.radii.split(",")]:
                    r = run(atlas, table, feature, radius, args.min_reference, exclude_iz, patients)
                    row = {"bands": preset, "feature": feature, "exclude_irritative": exclude_iz, "radius_mm": radius,
                           "min_reference": args.min_reference, "n_normative": r["n_normative"],
                           "scored": r["scored"], "channels": r["channels"],
                           "auc_soz": round(r["auc_soz"], 3), "auc_resected": round(r["auc_resected"], 3),
                           "auc_soz_patient_median": round(r["auc_soz_patient_median"], 3),
                           "auc_resected_patient_median": round(r["auc_resected_patient_median"], 3)}
                    rows.append(row)
                    print(", ".join(f"{k}={v}" for k, v in row.items()))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
