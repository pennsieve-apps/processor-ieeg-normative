"""
Normative z-scores of every feature a recording has, against a RefSet (processor/reference.py).

Node features (relpow_*, normpow_*, logpow_*, coh_*, or any other feature in features.csv):
each test channel gets a reference set of normative channels, its *region*, and
z = (x - mean) / sd over that set (sd with ddof 1), per feature.

  region  REGION_MODE=radius: normative channels whose MNI centroid lies within RADIUS_MM.
          REGION_MODE=roi: normative channels with the same `roi` label.
          REGION_MODE=auto (default): roi when the test channel and the reference both have roi
          labels, otherwise radius.
  The tested patient's own reference channels are left out (LEAVE_PATIENT_OUT).

Edge-level connectivity (Bernabei et al. 2022, Methods "Bivariate features"): the coherence of
each test channel pair (i, j) in a band is z-scored against the normative edges between the
same pair of regions: every reference edge (a, b) between two normative channels of one
reference recording with a in region(i) and b in region(j), or the other way round. With the
radius method a "region pair" is the pair of neighbourhoods, which keeps the reference edges at
about the distance of the test edge (coherence falls with distance). A channel's connectivity
abnormality in a band is the 75th percentile of |z| over its edges (the paper's choice: "a high
level of abnormality but not as sensitive to outliers"), from MIN_EDGES scored edges up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from processor import spectral as S
from processor.reference import RefSet

CONN_PERCENTILE = 75
MIN_EDGES = 3
EDGE_CHUNK = 128
SPECTRAL_PREFIXES = ("relpow", "normpow", "logpow")


def family(feature: str) -> str:
    """spectral, coherence (node strength), connectivity (edge-based, conn_<band>), or other."""
    if feature.startswith("conn_"):
        return "connectivity"
    return S.feature_family(feature)


@dataclass
class TestRec:
    recording_id: str
    channels: list[str]
    xyz: np.ndarray                               # (n, 3) MNI, NaN when unknown
    roi: np.ndarray                               # (n,) str, "" when unknown
    features: dict[str, np.ndarray]               # name -> (n,)
    pair_i: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    pair_j: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    pair_bands: list[str] = field(default_factory=list)
    pair_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass
class Params:
    radius: float = 20.0
    min_reference: int = 10
    region: str = "auto"                          # auto | radius | roi
    edges: bool = True
    spectral: tuple[str, ...] = ("normpow",)      # the prefixes summarised as the spectral family


@dataclass
class Scores:
    features: list[str]
    z: np.ndarray                 # (n, n_features)
    n_ref: np.ndarray             # (n, n_features) reference values used
    region: np.ndarray            # (n,) region label, or "insufficient" / "unknown"
    n_region: np.ndarray          # (n,) normative reference channels in the region
    conn_bands: list[str]
    conn: np.ndarray              # (n, n_conn_bands) 75th percentile of edge |z|
    conn_edges: np.ndarray        # (n, n_conn_bands) edges scored
    edge_z: np.ndarray            # (n_pairs, n_conn_bands)
    edge_n: np.ndarray            # (n_pairs, n_conn_bands) reference edges


def regions(test: TestRec, ref: RefSet, p: Params, exclude: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """(M (n_test, n_ref) bool: the reference channels of each test channel's region, labels (n_test,))."""
    pool = ref.normative & ~np.isin(ref.patient, list(exclude)) if exclude else ref.normative.copy()
    has_ref_roi = bool((ref.roi[pool] != "").any())
    ref_xyz_ok = np.isfinite(ref.xyz).all(axis=1)
    n = len(test.channels)
    M = np.zeros((n, ref.n), dtype=bool)
    label = np.full(n, "unknown", dtype=object)
    for i in range(n):
        use_roi = test.roi[i] != "" and (p.region == "roi" or (p.region == "auto" and has_ref_roi))
        if use_roi:
            M[i] = pool & (ref.roi == test.roi[i])
            label[i] = f"roi:{test.roi[i]}"
        elif p.region != "roi" and np.isfinite(test.xyz[i]).all():
            d = np.sqrt(((ref.xyz - test.xyz[i]) ** 2).sum(axis=1))
            M[i] = pool & ref_xyz_ok & (np.nan_to_num(d, nan=np.inf) <= p.radius)
            label[i] = f"within-{p.radius:g}mm"
    return M, label


def score(test: TestRec, ref: RefSet, p: Params, exclude: set[str] | None = None) -> Scores:
    exclude = exclude or set()
    M, label = regions(test, ref, p, exclude)
    n = len(test.channels)
    n_region = M.sum(axis=1)
    feats = [f for f in test.features if f in ref.features]
    z = np.full((n, len(feats)), np.nan)
    n_ref = np.zeros((n, len(feats)), dtype=int)
    if feats:
        V = np.column_stack([ref.features[f] for f in feats])
        T = np.column_stack([test.features[f] for f in feats])
        for i in range(n):
            if not M[i].any():
                continue
            vals = V[M[i]]
            fin = np.isfinite(vals)
            cnt = fin.sum(axis=0)
            n_ref[i] = cnt
            with np.errstate(invalid="ignore", divide="ignore"):
                v0 = np.where(fin, vals, 0.0)
                mu = v0.sum(axis=0) / cnt
                sd = np.sqrt(np.where(fin, (vals - mu) ** 2, 0.0).sum(axis=0) / (cnt - 1))
                ok = (cnt >= max(2, p.min_reference)) & (sd > 0)
                z[i] = np.where(ok, (T[i] - mu) / sd, np.nan)
    region = label.copy()
    region[(label != "unknown") & (n_region < max(2, p.min_reference))] = "insufficient"

    bands = [b for b in test.pair_bands if b in ref.edge_bands] if p.edges else []
    n_pairs = len(test.pair_i)
    conn = np.full((n, len(bands)), np.nan)
    conn_edges = np.zeros((n, len(bands)), dtype=int)
    edge_z = np.full((n_pairs, len(bands)), np.nan)
    edge_n = np.zeros((n_pairs, len(bands)), dtype=int)
    if bands and n_pairs and len(ref.edge_a):
        pool = M.any(axis=0)
        keep = pool[ref.edge_a] & pool[ref.edge_b]       # only edges that can be in some region pair
        ea, eb = ref.edge_a[keep], ref.edge_b[keep]
        ev = ref.edge_v[keep][:, [ref.edge_bands.index(b) for b in bands]]
        tv = test.pair_v[:, [test.pair_bands.index(b) for b in bands]]
        fin = np.isfinite(ev)
        v0 = np.where(fin, ev, 0.0)
        Ma, Mb = M[:, ea], M[:, eb]
        for c0 in range(0, n_pairs, EDGE_CHUNK):
            pi, pj = test.pair_i[c0: c0 + EDGE_CHUNK], test.pair_j[c0: c0 + EDGE_CHUNK]
            mask = ((Ma[pi] & Mb[pj]) | (Ma[pj] & Mb[pi])).astype(np.float64)
            cnt = mask @ fin.astype(np.float64)
            s1 = mask @ v0
            s2 = mask @ (v0 ** 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = s1 / cnt
                sd = np.sqrt(np.maximum(s2 - s1 * mu, 0.0) / (cnt - 1))
                ok = (cnt >= max(2, p.min_reference)) & (sd > 0)
                edge_z[c0: c0 + EDGE_CHUNK] = np.where(ok, (tv[c0: c0 + EDGE_CHUNK] - mu) / sd, np.nan)
            edge_n[c0: c0 + EDGE_CHUNK] = cnt.astype(int)
        az = np.abs(edge_z)
        for c in range(n):
            touch = (test.pair_i == c) | (test.pair_j == c)
            for b in range(len(bands)):
                v = az[touch, b]
                v = v[np.isfinite(v)]
                conn_edges[c, b] = len(v)
                if len(v) >= MIN_EDGES:
                    # MATLAB prctile, which the authors used, interpolates as numpy's "hazen"
                    conn[c, b] = np.percentile(v, CONN_PERCENTILE, method="hazen")
    return Scores(feats, z, n_ref, region, n_region, bands, conn, conn_edges, edge_z, edge_n)


SUMMARY_COLUMNS = ["max_abs_z", "max_band", "spectral_max_abs_z", "spectral_mean_abs_z", "spectral_max_feature",
                   "coherence_max_abs_z", "coherence_mean_abs_z", "connectivity_max", "connectivity_mean",
                   "connectivity_source", "combined", "other_max_abs_z"]
VALIDATION_SCORES = ["spectral_max_abs_z", "spectral_mean_abs_z", "coherence_mean_abs_z", "connectivity_mean",
                     "combined", "other_max_abs_z"]


def _nan_stat(a: np.ndarray, fn) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    ok = np.isfinite(a).any(axis=1)
    if ok.any():
        out[ok] = fn(np.where(np.isfinite(a[ok]), a[ok], np.nan), axis=1)
    return out


def summarise(s: Scores, p: Params) -> dict[str, np.ndarray]:
    """Per channel, per feature family (SUMMARY_COLUMNS)."""
    n = s.z.shape[0]
    absz = np.abs(s.z)
    fam = [family(f) for f in s.features]
    spec = [k for k, f in enumerate(s.features) if fam[k] == "spectral" and f.split("_", 1)[0] in p.spectral]
    coh = [k for k, f in enumerate(s.features) if fam[k] == "coherence"]
    oth = [k for k, f in enumerate(s.features) if fam[k] == "other"]
    out: dict[str, np.ndarray] = {}
    sa = absz[:, spec] if spec else np.full((n, 0), np.nan)
    out["spectral_max_abs_z"] = _nan_stat(sa, np.nanmax) if spec else np.full(n, np.nan)
    out["spectral_mean_abs_z"] = _nan_stat(sa, np.nanmean) if spec else np.full(n, np.nan)
    feat = np.full(n, "", dtype=object)
    for i in range(n):
        if spec and np.isfinite(sa[i]).any():
            feat[i] = s.features[spec[int(np.nanargmax(sa[i]))]]
    out["spectral_max_feature"] = feat
    out["max_abs_z"] = out["spectral_max_abs_z"]
    out["max_band"] = np.array([f.split("_", 1)[1] if f else "" for f in feat], dtype=object)
    ca = absz[:, coh] if coh else np.full((n, 0), np.nan)
    out["coherence_max_abs_z"] = _nan_stat(ca, np.nanmax) if coh else np.full(n, np.nan)
    out["coherence_mean_abs_z"] = _nan_stat(ca, np.nanmean) if coh else np.full(n, np.nan)
    cmax = _nan_stat(s.conn, np.nanmax) if s.conn.shape[1] else np.full(n, np.nan)
    cmean = _nan_stat(s.conn, np.nanmean) if s.conn.shape[1] else np.full(n, np.nan)
    source = np.where(np.isfinite(cmean), "edges", "").astype(object)
    fb = ~np.isfinite(cmean) & np.isfinite(out["coherence_mean_abs_z"])
    cmax[fb], cmean[fb], source[fb] = out["coherence_max_abs_z"][fb], out["coherence_mean_abs_z"][fb], "node-strength"
    out["connectivity_max"], out["connectivity_mean"], out["connectivity_source"] = cmax, cmean, source
    # combined: the mean of the paper's ten features, |z| of the spectral family's bands and the
    # connectivity per band (edge-based, or node strength when a channel has no edge score)
    conn_part = np.where(fb[:, None], ca, s.conn) if (coh and s.conn.shape[1] == len(coh)) else \
        (s.conn if s.conn.shape[1] else ca)
    parts = np.hstack([sa, conn_part])
    out["combined"] = _nan_stat(parts, np.nanmean) if parts.shape[1] else np.full(n, np.nan)
    out["other_max_abs_z"] = _nan_stat(absz[:, oth], np.nanmax) if oth else np.full(n, np.nan)
    return out
