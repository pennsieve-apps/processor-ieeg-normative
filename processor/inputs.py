"""
Finding the three inputs under INPUT_DIR (subfolders features/, ieeg/, reference/ when linked;
anywhere when everything is merged at the root):

- features: every features.csv (recording_id, channel, feature, value; ieeg-bandpower or any app
  writing the shared shape), with coherence.csv (channel pairs, for edge-level connectivity) and
  the provenance.json that says how the spectra were computed beside it. A folder with only a
  bandpower.csv (older ieeg-bandpower) is read too: relpow_, normpow_, logpow_ are derived from it.
  A features.csv beside a reference_channels.csv belongs to the reference, not to the input;
- ieeg: every *_channels.tsv (coordinates x/y/z/space, roi, soz/resected labels), every
  *_electrodes.tsv (atlas_patient), recordings.csv (subject of each recording);
- reference: a reference_channels.csv folder (Format A) or the atlas .mat (Format B); see
  processor/reference.py.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("processor-ieeg-normative")

NA = {"", "n/a", "na", "nan", "none"}
MNI_SPACE = re.compile(r"^(mni|ixi549space|icbm|fsaverage)", re.IGNORECASE)
ATLAS_SUBJECT = re.compile(r"^HUPatlas0*(\d+)$")


def walk_files(root: Path) -> list[Path]:
    """Every file under root, following symlinked folders and files (INPUT_DIR entries are symlinks)."""
    found: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                p = Path(dirpath) / name
                if p.is_file():
                    found.append(p)
    return found


def read_table(path: Path) -> list[dict[str, str]]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = []
        for raw in csv.DictReader(fh, delimiter=delim):
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                v = (v or "").strip()
                row[k.strip()] = "" if v.lower() in NA else v
            rows.append(row)
        return rows


def fnum(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def fbool(v: str | None) -> bool | None:
    s = (v or "").strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def subject_of(recording_id: str) -> str:
    m = re.match(r"^sub-([A-Za-z0-9]+)", recording_id)
    return m.group(1) if m else ""


@dataclass
class FeatureFile:
    path: Path
    spectral: dict | None                       # provenance.json "spectral" block of ieeg-bandpower
    # recording_id -> channel -> feature -> value (channel order as first seen)
    rows: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    # recording_id -> [(channel_a, channel_b, band, coherence)] from coherence.csv
    coherence: dict[str, list[tuple]] = field(default_factory=dict)
    legacy: bool = False                        # derived from bandpower.csv


@dataclass
class ChannelInfo:
    xyz: tuple[float, float, float] = (float("nan"),) * 3
    space: str = ""
    soz: bool | None = None
    resected: bool | None = None
    type: str = ""
    roi: str = ""


@dataclass
class Ieeg:
    channels: dict[str, dict[str, ChannelInfo]] = field(default_factory=dict)   # recording_id -> name -> info
    subjects: dict[str, str] = field(default_factory=dict)                     # recording_id -> subject
    atlas_patient: dict[str, int] = field(default_factory=dict)                # subject -> atlas patient_no


def read_coherence(path: Path):
    """coherence.csv -> (recording_id, channel_a, channel_b, band, coherence) tuples, names interned.
    Compact on purpose: a few hundred channels make ~10^6 rows."""
    names: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.reader(fh)
        head = [h.strip() for h in next(rd, [])]
        try:
            ix = [head.index(k) for k in ("recording_id", "channel_a", "channel_b", "band", "coherence")]
        except ValueError:
            log.warning("%s: not a coherence table (columns %s)", path, ",".join(head))
            return
        for row in rd:
            if len(row) < len(head):
                continue
            rid, a, b, band, v = (row[i].strip() for i in ix)
            if rid and a and b and band:
                yield (names.setdefault(rid, rid), names.setdefault(a, a), names.setdefault(b, b),
                       names.setdefault(band, band), fnum(v))


def legacy_features(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """bandpower.csv rows -> recording -> channel -> {relpow_, normpow_, logpow_<band>} (the same
    definitions as spectral.channel_features)."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows:
        band = r.get("band", "")
        if not (r.get("recording_id") and r.get("channel") and band) or band == "broadband":
            continue
        d = out.setdefault(r["recording_id"], {}).setdefault(r["channel"], {})
        d[f"relpow_{band}"] = fnum(r.get("rel_power"))
        d[f"normpow_{band}"] = fnum(r.get("norm_median"))
        a = fnum(r.get("abs_power"))
        d[f"logpow_{band}"] = float(np.log10(a)) if np.isfinite(a) and a > 0 else float("nan")
    return out


def load_features(files: list[Path]) -> list[FeatureFile]:
    reference_dirs = {f.parent for f in files if f.name == "reference_channels.csv"}
    out = []
    by_dir: dict[Path, dict[str, Path]] = {}
    for f in files:
        if f.name in ("features.csv", "bandpower.csv", "coherence.csv", "provenance.json") and \
                f.parent not in reference_dirs:
            by_dir.setdefault(f.parent, {})[f.name] = f
    for d, named in sorted(by_dir.items()):
        main = named.get("features.csv") or named.get("bandpower.csv")
        if main is None:
            continue
        spectral = None
        if "provenance.json" in named:
            try:
                spectral = json.loads(named["provenance.json"].read_text()).get("spectral")
            except (OSError, json.JSONDecodeError) as e:
                log.warning("%s: unreadable (%s)", named["provenance.json"], e)
        ff = FeatureFile(main, spectral, legacy=main.name == "bandpower.csv")
        if ff.legacy:
            ff.rows = legacy_features(read_table(main))
        else:
            for r in read_table(main):
                if r.get("recording_id") and r.get("channel") and r.get("feature"):
                    ff.rows.setdefault(r["recording_id"], {}).setdefault(r["channel"], {})[r["feature"]] = \
                        fnum(r.get("value"))
        if "coherence.csv" in named:
            for rid, a, b, band, v in read_coherence(named["coherence.csv"]):
                ff.coherence.setdefault(rid, []).append((a, b, band, v))
        if ff.rows:
            out.append(ff)
    return out


def recording_id_of_sidecar(path: Path, suffix: str) -> str:
    return re.sub(r"_desc-preproc$", "", path.name[: -len(suffix)])


def load_ieeg(files: list[Path]) -> Ieeg:
    ie = Ieeg()
    for p in [f for f in files if f.name.endswith("_channels.tsv")]:
        rid = recording_id_of_sidecar(p, "_channels.tsv")
        chans = {}
        for r in read_table(p):
            if not r.get("name"):
                continue
            chans[r["name"]] = ChannelInfo((fnum(r.get("x")), fnum(r.get("y")), fnum(r.get("z"))),
                                           r.get("space", ""), fbool(r.get("soz")), fbool(r.get("resected")),
                                           r.get("type", "").upper(), r.get("roi", ""))
        if rid in ie.channels:
            log.warning("%s: two _channels.tsv; using the first", rid)
            continue
        ie.channels[rid] = chans
        ie.subjects.setdefault(rid, subject_of(rid))
    for p in [f for f in files if f.name == "recordings.csv"]:
        for r in read_table(p):
            if r.get("recording_id") and r.get("subject"):
                ie.subjects[r["recording_id"]] = r["subject"]
    for p in [f for f in files if f.name.endswith("_electrodes.tsv")]:
        sub = subject_of(p.name)
        for r in read_table(p):
            v = fnum(r.get("atlas_patient"))
            if sub and np.isfinite(v):
                ie.atlas_patient.setdefault(sub, int(v))
                break
    return ie


def atlas_patient_of(subject: str, ie: Ieeg) -> int | None:
    """The atlas patient_no of a subject: sub-HUPatlas<NNN>, or atlas_patient in its _electrodes.tsv."""
    if subject in ie.atlas_patient:
        return ie.atlas_patient[subject]
    m = ATLAS_SUBJECT.match(subject)
    return int(m.group(1)) if m else None


def is_mni(space: str) -> bool:
    return bool(MNI_SPACE.match(space or ""))
