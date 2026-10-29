"""
Builds a Format A reference for ieeg-normative (processor/reference.py) from a normative cohort:

  python -m scripts.build_reference --ieeg <ieeg-prep derivative> --features <ieeg-bandpower output> --out <dir>
         [--all-normative] [--patient subject|recording]
  python -m scripts.build_reference --from-atlas HUP_atlas_final.mat --out <dir>   (the atlas in the same shape)

Writes into <dir>:

  reference_channels.csv  recording_id, channel, x, y, z, space, roi, patient, normative, spike_24h
  features.csv            the cohort's features.csv rows (recording_id, channel, feature, value)
  coherence.csv           the cohort's coherence.csv rows, when ieeg-bandpower wrote one
  provenance.json         ieeg-bandpower's (its spectral settings), plus how this reference was built

A channel is normative when its _channels.tsv says neither soz nor resected is true (unlabelled
channels count as normative), or always with --all-normative (a cohort of controls). spike_24h is
taken from _channels.tsv or the subject's _electrodes.tsv when one of them has it; ieeg-normative
treats spike_24h > 24 as irritative zone (EXCLUDE_IRRITATIVE). Only `good` channels with features
are written. Link <dir> to ieeg-normative's `reference` input.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from processor import atlas as A
from processor import inputs as I
from processor import reference as R
from processor import spectral as S

COLUMNS = ["recording_id", "channel", "x", "y", "z", "space", "roi", "patient", "normative", "spike_24h"]


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return str(v)


def from_cohort(ieeg: Path, features: Path, out: Path, all_normative: bool, patient_by: str) -> int:
    files = I.walk_files(ieeg)
    feats = I.walk_files(features)
    fcsv = [f for f in feats if f.name == "features.csv"]
    if not fcsv:
        raise SystemExit(f"no features.csv under {features} (run ieeg-bandpower first)")
    have = {(r["recording_id"], r["channel"]) for f in fcsv for r in I.read_table(f)}
    subjects = {}
    for p in [f for f in files if f.name == "recordings.csv"]:
        for r in I.read_table(p):
            subjects[r.get("recording_id", "")] = r.get("subject", "")
    spikes: dict[tuple[str, str], float] = {}
    for p in [f for f in files if f.name.endswith("_electrodes.tsv")]:
        sub = I.subject_of(p.name)
        for r in I.read_table(p):
            v = I.fnum(r.get("spike_24h"))
            if np.isfinite(v):
                spikes[(sub, r.get("name", ""))] = v
    rows = []
    for p in sorted(f for f in files if f.name.endswith("_channels.tsv")):
        rid = I.recording_id_of_sidecar(p, "_channels.tsv")
        sub = subjects.get(rid) or I.subject_of(rid)
        for r in I.read_table(p):
            name = r.get("name", "")
            if (rid, name) not in have or (r.get("status") or "good").lower() != "good":
                continue
            labelled = [I.fbool(r.get(k)) for k in ("soz", "resected")]
            normative = all_normative or not any(v is True for v in labelled)
            spk = I.fnum(r.get("spike_24h"))
            if not np.isfinite(spk):
                spk = spikes.get((sub, name), spikes.get((sub, name.split("-")[0]), float("nan")))
            rows.append([rid, name, I.fnum(r.get("x")), I.fnum(r.get("y")), I.fnum(r.get("z")), r.get("space", ""),
                         r.get("roi", ""), sub if patient_by == "subject" else rid, normative, spk])
    if not rows:
        raise SystemExit("no channel with both a _channels.tsv row and features")
    out.mkdir(parents=True, exist_ok=True)
    write_channels(out, rows)
    keep = {(r[0], r[1]) for r in rows}
    concat(fcsv, out / "features.csv", lambda r: (r["recording_id"], r["channel"]) in keep)
    coh = [f for f in feats if f.name == "coherence.csv"]
    if coh:
        concat(coh, out / "coherence.csv", lambda r: (r["recording_id"], r["channel_a"]) in keep and
               (r["recording_id"], r["channel_b"]) in keep)
    prov = [f for f in feats if f.name == "provenance.json"]
    spectral = json.loads(prov[0].read_text()).get("spectral") if prov else None
    write_provenance(out, spectral, {"from": "cohort", "ieeg": str(ieeg), "features": str(features),
                                     "all_normative": all_normative, "patient": patient_by})
    return len(rows)


def from_atlas(mat: Path, out: Path) -> int:
    params = S.SpectralParams()
    atlas = A.load(mat)
    ref = R.from_atlas(atlas, params, exclude_iz=False)
    atlas_spk = atlas.spike_24h
    rows = [[ref.recording[i], ref.channel[i], *ref.xyz[i], "MNI152", "", ref.patient[i], ref.normative[i],
             atlas_spk[i] if atlas_spk is not None else float("nan")] for i in range(ref.n)]
    out.mkdir(parents=True, exist_ok=True)
    write_channels(out, rows)
    with open(out / "features.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel", "feature", "value"])
        for name, v in ref.features.items():
            for i in range(ref.n):
                w.writerow([ref.recording[i], ref.channel[i], name, fmt(v[i])])
    with open(out / "coherence.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "channel_a", "channel_b", "band", "coherence"])
        for e in range(len(ref.edge_a)):
            a, b = ref.edge_a[e], ref.edge_b[e]
            for k, band in enumerate(ref.edge_bands):
                w.writerow([ref.recording[a], ref.channel[a], ref.channel[b], band, fmt(ref.edge_v[e, k])])
    write_provenance(out, params.as_dict(), {"from": "atlas", "mat": mat.name})
    return ref.n


def write_channels(out: Path, rows):
    with open(out / R.CHANNELS_FILE, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows([[fmt(v) for v in r] for r in rows])


def concat(paths, dest: Path, keep):
    with open(dest, "w", newline="") as fh:
        w = None
        for p in paths:
            for r in I.read_table(p):
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r), lineterminator="\n")
                    w.writeheader()
                if keep(r):
                    w.writerow(r)


def write_provenance(out: Path, spectral, how: dict):
    (out / "provenance.json").write_text(json.dumps({"app": "processor-ieeg-normative scripts/build_reference.py",
                                                     "spectral": spectral, "built": how}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ieeg", type=Path)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--from-atlas", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--all-normative", action="store_true", help="every channel is normative (a control cohort)")
    ap.add_argument("--patient", choices=("subject", "recording"), default="subject",
                    help="what counts as one patient for leave-one-patient-out (default: the BIDS subject)")
    a = ap.parse_args(argv)
    if a.from_atlas:
        n = from_atlas(a.from_atlas, a.out)
    elif a.ieeg and a.features:
        n = from_cohort(a.ieeg, a.features, a.out, a.all_normative, a.patient)
    else:
        ap.error("--ieeg and --features, or --from-atlas")
    print(f"{n} reference channels written to {a.out}")


if __name__ == "__main__":
    main()
