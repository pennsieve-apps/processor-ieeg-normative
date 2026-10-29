"""
Finding the three inputs under INPUT_DIR (subfolders features/, ieeg/, reference/ when linked;
anywhere when everything is merged at the root):

- features: every bandpower.csv (ieeg-bandpower), with the provenance.json beside it that
  says how the spectra were computed;
- ieeg: every *_channels.tsv (coordinates x/y/z/space, soz/resected labels), every
  *_electrodes.tsv (atlas_patient), recordings.csv (subject of each recording);
- reference: the atlas .mat (see atlas.find).
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
    rows: dict[str, list[dict]] = field(default_factory=dict)   # recording_id -> rows


@dataclass
class ChannelInfo:
    xyz: tuple[float, float, float] = (float("nan"),) * 3
    space: str = ""
    soz: bool | None = None
    resected: bool | None = None
    type: str = ""


@dataclass
class Ieeg:
    channels: dict[str, dict[str, ChannelInfo]] = field(default_factory=dict)   # recording_id -> name -> info
    subjects: dict[str, str] = field(default_factory=dict)                     # recording_id -> subject
    atlas_patient: dict[str, int] = field(default_factory=dict)                # subject -> atlas patient_no


def load_features(files: list[Path]) -> list[FeatureFile]:
    out = []
    for p in [f for f in files if f.name == "bandpower.csv"]:
        prov = p.with_name("provenance.json")
        spectral = None
        if prov.is_file():
            try:
                spectral = json.loads(prov.read_text()).get("spectral")
            except (OSError, json.JSONDecodeError) as e:
                log.warning("%s: unreadable (%s)", prov, e)
        ff = FeatureFile(p, spectral)
        for r in read_table(p):
            if r.get("recording_id") and r.get("channel") and r.get("band"):
                ff.rows.setdefault(r["recording_id"], []).append(r)
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
                                           r.get("type", "").upper())
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
