"""
The AUC used to validate the scores, and the original single-feature neighbourhood z-score that
scripts/validate_atlas.py and scripts/atlas_checks.py still use for the band-power grid. The app
itself scores with processor/score.py (every feature, edges, families).

Every test channel with MNI coordinates is compared with the normative atlas channels whose
MNI centroid lies within RADIUS_MM of it (Euclidean), excluding the tested patient's own atlas
channels: z = (x - mean) / sd of the reference values, per band (sd with ddof=1). Fewer than
MIN_REFERENCE reference channels -> region "insufficient", no z.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("norm_median", "taylor2022", "log_rel_power", "rel_power", "log_abs_power")


def transform(feature: str, abs_power: np.ndarray, rel_power: np.ndarray, norm_median: np.ndarray) -> np.ndarray:
    """
    The value that is z-scored, from the three columns of bandpower.csv ((n_channels, n_bands) each):

    - norm_median: Bernabei et al. 2022, the median of the sum-1-normalised PSD in the band;
    - taylor2022: Taylor et al. 2022, log10 band power, normalised to sum to one over the bands
      of each channel (L1 norm);
    - log_rel_power, rel_power, log_abs_power: log10 or plain share of 1-80 Hz, log10 µV².
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if feature == "norm_median":
            return np.asarray(norm_median, dtype=float)
        if feature == "rel_power":
            return np.asarray(rel_power, dtype=float)
        if feature == "log_rel_power":
            return np.log10(np.asarray(rel_power, dtype=float))
        if feature == "taylor2022":
            lg = np.log10(np.atleast_2d(np.asarray(abs_power, dtype=float)))
            return lg / lg.sum(axis=1, keepdims=True)
        if feature == "log_abs_power":
            return np.log10(np.asarray(abs_power, dtype=float))
    raise ValueError(f"FEATURE must be one of {', '.join(FEATURES)}")


@dataclass
class Reference:
    xyz: np.ndarray          # (n_ref, 3) normative channels only
    patient: np.ndarray      # (n_ref,)
    values: np.ndarray       # (n_ref, n_bands) transformed feature


@dataclass
class ZResult:
    z: np.ndarray            # (n_test, n_bands), NaN when not scored
    n_reference: np.ndarray  # (n_test,) int
    status: np.ndarray       # (n_test,) "ok" | "insufficient" | "unknown"


def zscore(ref: Reference, xyz: np.ndarray, values: np.ndarray, exclude_patient: int | None, radius: float,
           min_reference: int) -> ZResult:
    n = xyz.shape[0]
    nb = ref.values.shape[1]
    z = np.full((n, nb), np.nan)
    nref = np.zeros(n, dtype=int)
    status = np.full(n, "unknown", dtype=object)
    keep = np.ones(ref.xyz.shape[0], dtype=bool) if exclude_patient is None else ref.patient != exclude_patient
    rx, rv = ref.xyz[keep], ref.values[keep]
    for i in range(n):
        if not np.all(np.isfinite(xyz[i])):
            continue
        d = np.sqrt(((rx - xyz[i]) ** 2).sum(axis=1))
        sel = d <= radius
        vals = rv[sel]
        nref[i] = int(sel.sum())
        if nref[i] < max(2, min_reference):
            status[i] = "insufficient"
            continue
        mu = np.nanmean(vals, axis=0)
        sd = np.nanstd(vals, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[i] = np.where(sd > 0, (values[i] - mu) / sd, np.nan)
        status[i] = "ok"
    return ZResult(z, nref, status)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U / (n_pos * n_neg)), ties counted half. NaN scores dropped."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks of ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
