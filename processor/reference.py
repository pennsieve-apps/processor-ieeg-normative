"""
The normative reference that ieeg-normative z-scores against, in one shape (RefSet) whatever it
came from.

Format A: a reference feature table. Any normative cohort run through ieeg-prep and
ieeg-bandpower can serve as one; scripts/build_reference.py writes it. The files sit in one folder:

  reference_channels.csv  recording_id, channel, x, y, z, space, roi (optional), patient, normative,
                          and optionally spike_24h (> 24 = irritative zone) or irritative (true/false)
  features.csv            recording_id, channel, feature, value (the shared long shape, conventions.md)
  coherence.csv           recording_id, channel_a, channel_b, band, coherence (optional; needed for
                          edge-level connectivity)
  provenance.json         ieeg-bandpower's, for its spectral settings (optional)

Format B: the HUP Interictal iEEG Atlas .mat (Discover 179). It is converted here by running
the same feature code (processor/spectral.py) on each patient's 60-s wake clip, with the
spectral settings of the bandpower run being scored.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from processor import atlas as A
from processor import inputs as I
from processor import spectral as S

log = logging.getLogger("processor-ieeg-normative")

CHANNELS_FILE = "reference_channels.csv"
IRRITATIVE_SPIKES_24H = A.IRRITATIVE_SPIKES_24H


def atlas_patient_key(n: int) -> str:
    """Atlas patient_no -> the subject label ieeg-prep gives it (sub-HUPatlas001)."""
    return f"HUPatlas{int(n):03d}"


@dataclass
class RefSet:
    kind: str                         # "table" (Format A) or "atlas" (Format B)
    path: Path
    recording: np.ndarray             # (n,) str, edges only join channels of the same recording
    channel: np.ndarray               # (n,) str
    xyz: np.ndarray                   # (n, 3) MNI, NaN when unknown or not MNI
    roi: np.ndarray                   # (n,) str, "" when unknown
    patient: np.ndarray               # (n,) str
    normative: np.ndarray             # (n,) bool, the reference pool (irritative zone already applied)
    irritative: np.ndarray            # (n,) bool
    features: dict[str, np.ndarray] = field(default_factory=dict)     # name -> (n,)
    edge_a: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    edge_b: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    edge_bands: list[str] = field(default_factory=list)
    edge_v: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (n_edges, n_bands) coherence
    spectral: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.channel)

    def summary(self) -> dict:
        both = self.normative[self.edge_a] & self.normative[self.edge_b] if len(self.edge_a) else np.zeros(0, bool)
        return {"kind": self.kind, "channels": self.n, "patients": int(len(np.unique(self.patient))),
                "normative_channels": int(self.normative.sum()), "irritative_channels": int(self.irritative.sum()),
                "with_mni": int(np.isfinite(self.xyz).all(axis=1).sum()), "with_roi": int((self.roi != "").sum()),
                "features": sorted(self.features), "edge_bands": self.edge_bands,
                "normative_edges": int(both.sum()), "notes": self.notes}


def apply_irritative(ref: RefSet, exclude: bool) -> RefSet:
    if exclude:
        ref.normative = ref.normative & ~ref.irritative
    return ref


def pair_table(index: dict[tuple[str, str], int], rows, bands: list[str] | None = None):
    """(recording_id, channel_a, channel_b, band, coherence) tuples -> (a, b, bands, (n_pairs, n_bands));
    pairs whose channels are not in index are skipped."""
    pairs: dict[tuple[int, int], dict[str, float]] = {}
    seen_bands: list[str] = list(bands or [])
    for rid, ca, cb, band, value in rows:
        a = index.get((rid, ca))
        b = index.get((rid, cb))
        if a is None or b is None or a == b or not band:
            continue
        if band not in seen_bands:
            seen_bands.append(band)
        pairs.setdefault((min(a, b), max(a, b)), {})[band] = value
    keys = sorted(pairs)
    v = np.array([[pairs[k].get(bd, np.nan) for bd in seen_bands] for k in keys]).reshape(len(keys), len(seen_bands))
    a = np.array([k[0] for k in keys], dtype=int)
    b = np.array([k[1] for k in keys], dtype=int)
    return a, b, seen_bands, v


def load_table(folder: Path, exclude_iz: bool) -> RefSet:
    """Format A: reference_channels.csv + features.csv (+ coherence.csv, provenance.json) in one folder."""
    rows = I.read_table(folder / CHANNELS_FILE)
    if not rows:
        raise ValueError(f"{CHANNELS_FILE} has no rows")
    cols = set(rows[0])
    for need in ("recording_id", "channel", "normative"):
        if need not in cols:
            raise ValueError(f"{CHANNELS_FILE}: missing column {need}")
    notes = []
    rec, ch, xyz, roi, pat, norm, irr = [], [], [], [], [], [], []
    non_mni = 0
    for r in rows:
        rec.append(r["recording_id"])
        ch.append(r["channel"])
        c = (I.fnum(r.get("x")), I.fnum(r.get("y")), I.fnum(r.get("z")))
        if r.get("space") and not I.is_mni(r["space"]):
            non_mni += 1
            c = (np.nan,) * 3
        xyz.append(c)
        roi.append(r.get("roi", ""))
        pat.append(r.get("patient") or I.subject_of(r["recording_id"]) or r["recording_id"])
        n = I.fbool(r.get("normative"))
        norm.append(bool(n))
        if "irritative" in cols and I.fbool(r.get("irritative")) is not None:
            irr.append(bool(I.fbool(r.get("irritative"))))
        else:
            s = I.fnum(r.get("spike_24h"))
            irr.append(bool(np.isfinite(s) and s > IRRITATIVE_SPIKES_24H))
    if non_mni:
        notes.append(f"{non_mni} reference channel(s) with coordinates in a non-MNI space: not used with REGION=radius")
    if "patient" not in cols:
        notes.append("no patient column: the BIDS subject of recording_id is the patient")
    ref = RefSet("table", folder, np.array(rec, dtype=object), np.array(ch, dtype=object), np.array(xyz, float),
                 np.array(roi, dtype=object), np.array(pat, dtype=object), np.array(norm), np.array(irr), notes=notes)
    index = {(a, b): i for i, (a, b) in enumerate(zip(rec, ch))}
    fpath = folder / "features.csv"
    if not fpath.is_file():
        raise ValueError(f"{folder.name}/: {CHANNELS_FILE} without features.csv beside it")
    for r in I.read_table(fpath):
        i = index.get((r.get("recording_id", ""), r.get("channel", "")))
        if i is None or not r.get("feature"):
            continue
        ref.features.setdefault(r["feature"], np.full(ref.n, np.nan))[i] = I.fnum(r.get("value"))
    if (folder / "coherence.csv").is_file():
        ref.edge_a, ref.edge_b, ref.edge_bands, ref.edge_v = pair_table(index, I.read_coherence(folder / "coherence.csv"))
    prov = folder / "provenance.json"
    if prov.is_file():
        try:
            ref.spectral = json.loads(prov.read_text()).get("spectral")
        except (OSError, json.JSONDecodeError):
            notes.append("provenance.json unreadable")
    return apply_irritative(ref, exclude_iz)


def from_atlas(atlas: A.Atlas, params: S.SpectralParams, exclude_iz: bool, coherence: bool = True) -> RefSet:
    """
    Format B: features of every atlas channel, one patient at a time (coherence and edges only
    join channels of the same patient), with the same code ieeg-bandpower runs. The clips are
    already 0.5-80 Hz at 200 Hz, so they are not filtered or resampled again.
    """
    n = atlas.n_channels
    keys = np.array([atlas_patient_key(p) for p in atlas.patient], dtype=object)
    irr = (atlas.spike_24h > IRRITATIVE_SPIKES_24H) if atlas.spike_24h is not None else np.zeros(n, bool)
    ref = RefSet("atlas", atlas.path, keys.copy(), np.array([f"ch{i + 1:04d}" for i in range(n)], dtype=object),
                 atlas.xyz.copy(), np.full(n, "", dtype=object), keys, ~atlas.soz & ~atlas.resected, irr)
    if atlas.spike_24h is None:
        ref.notes.append("no spike_24h in the atlas: the irritative zone cannot be excluded")
    ea, eb, ev = [], [], []
    bands: list[str] = []
    n_s = atlas.wake_clip.shape[0]
    for p in np.unique(atlas.patient):
        idx = np.where(atlas.patient == p)[0]
        clip = atlas.wake_clip[:, idx]

        def read(s0, s1, clip=clip):
            return clip[s0:s1].T

        acc, _ = S.stream(read, [(0, n_s)], A.ATLAS_SFREQ, len(idx), params, prefiltered=True, cross=coherence)
        bt = S.band_table(*acc.psd(), params.bands)
        coh = None
        if coherence and len(idx) > 1:
            coh = S.coherence_bands(acc.coh.freqs[: acc.coh.nf], acc.coh.coherence(), params.bands)
            bands = coh[0]
            iu, ju = np.triu_indices(len(idx), 1)
            ea.append(idx[iu])
            eb.append(idx[ju])
            ev.append(coh[1][iu, ju, :])
        for name, vals in S.channel_features(bt, coh):
            ref.features.setdefault(name, np.full(n, np.nan))[idx] = vals
    if ea:
        ref.edge_a, ref.edge_b, ref.edge_bands, ref.edge_v = (np.concatenate(ea), np.concatenate(eb), bands,
                                                              np.concatenate(ev))
    ref.spectral = params.as_dict()
    return apply_irritative(ref, exclude_iz)


def find(files: list[Path]) -> tuple[str, Path] | None:
    """('table', folder) when a reference_channels.csv exists, else ('atlas', .mat path), else None."""
    tables = [f for f in files if f.name == CHANNELS_FILE]
    if tables:
        if len(tables) > 1:
            log.warning("%d %s files; using %s", len(tables), CHANNELS_FILE, tables[0])
        return "table", tables[0].parent
    mat = A.find(Path("."), files)
    return ("atlas", mat) if mat is not None else None
