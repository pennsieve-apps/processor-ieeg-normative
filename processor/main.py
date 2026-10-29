"""
processor-ieeg-normative: how abnormal each channel's features are, compared with normative
channels in the same region of a supplied reference.

  INPUT_DIR/features/    features.csv (+ coherence.csv, provenance.json) from ieeg-bandpower, or any
                         table in the shared shape recording_id, channel, feature, value
  INPUT_DIR/ieeg/        the ieeg-prep derivative: _channels.tsv (x, y, z, space, roi, soz, resected),
                         _electrodes.tsv (atlas_patient), recordings.csv
  INPUT_DIR/reference/   Format A: reference_channels.csv + features.csv (+ coherence.csv), or
                         Format B: HUP_atlas_final.mat (Discover 179)
  OUTPUT_DIR[/LABEL]/abnormality.csv          recording_id, channel, feature, z, abs_z, region, n_reference, family
  OUTPUT_DIR[/LABEL]/channel_abnormality.csv  per channel, a summary per feature family
  OUTPUT_DIR[/LABEL]/edge_abnormality.csv     recording_id, channel_a, channel_b, band, coherence, z, abs_z, n_reference
  OUTPUT_DIR[/LABEL]/validation.csv           AUC per family score for soz / resected, when labels are known
  OUTPUT_DIR[/LABEL]/provenance.json
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from processor import atlas as A
from processor import inputs as I
from processor import normative as N
from processor import reference as R
from processor import score as Sc
from processor import spectral as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("processor-ieeg-normative")
log.setLevel(logging.INFO)

APP_NAME = "processor-ieeg-normative"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE = ("1", "true", "yes", "on")

DEFAULTS = {
    "RADIUS_MM": "20", "MIN_REFERENCE": "10", "REGION": "auto", "LEAVE_PATIENT_OUT": "true",
    "EXCLUDE_IRRITATIVE": "true", "SPECTRAL_FEATURES": "normpow", "EDGES": "true", "LABEL": "",
}
REGIONS = ("auto", "radius", "roi")


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    out["spectral"] = S.SPECTRAL_VERSION
    return out


def read_params(env=os.environ) -> tuple[dict, dict, Sc.Params]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p = {"lopo": str(raw["LEAVE_PATIENT_OUT"]).strip().lower() in TRUE,
         "exclude_iz": str(raw["EXCLUDE_IRRITATIVE"]).strip().lower() in TRUE, "label": raw["LABEL"].strip()}
    spectral = tuple(x.strip() for x in str(raw["SPECTRAL_FEATURES"]).split(",") if x.strip())
    sp = Sc.Params(radius=float(raw["RADIUS_MM"]), min_reference=int(float(raw["MIN_REFERENCE"])),
                   region=str(raw["REGION"]).strip().lower(), edges=str(raw["EDGES"]).strip().lower() in TRUE,
                   spectral=spectral)
    if sp.radius <= 0:
        raise ValueError("RADIUS_MM must be > 0")
    if sp.min_reference < 2:
        raise ValueError("MIN_REFERENCE must be >= 2")
    if sp.region not in REGIONS:
        raise ValueError(f"REGION must be one of {', '.join(REGIONS)}")
    if not spectral or any(x not in Sc.SPECTRAL_PREFIXES for x in spectral):
        raise ValueError(f"SPECTRAL_FEATURES: comma-separated, from {', '.join(Sc.SPECTRAL_PREFIXES)}")
    if p["label"] and not all(c.isalnum() or c in "-_" for c in p["label"]):
        raise ValueError("LABEL: letters, digits, - and _ only")
    return raw, p, sp


def spectral_params(block: dict | None) -> S.SpectralParams:
    """The SpectralParams ieeg-bandpower used (its provenance.json), else the defaults."""
    if not block:
        return S.SpectralParams()
    bands = tuple((n, float(e[0]), float(e[1])) for n, e in block.get("bands", {}).items()) or S.BAND_PRESETS["conventions"]
    coh = block.get("coherence") or {}
    return S.SpectralParams(resample_hz=float(block.get("resample_hz", 200)),
                            window_s=float(block.get("welch_window_s", 2)),
                            overlap=float(block.get("welch_overlap", 0.5)), bands=bands,
                            coh_window_s=float(coh.get("window_s", 1.0)), coh_overlap=float(coh.get("overlap", 0.0)))


class References:
    """The reference: one Format A table, or the atlas converted once per spectral setting."""

    def __init__(self, kind: str, path: Path, p: dict):
        self.kind, self.path, self.p, self.cache, self.seconds = kind, path, p, {}, 0.0
        self.atlas = None
        t = time.monotonic()
        if kind == "table":
            self.table = R.load_table(path, p["exclude_iz"])
        else:
            self.table = None
            self.atlas = A.load(path)
        self.seconds += time.monotonic() - t

    def get(self, params: S.SpectralParams) -> R.RefSet:
        if self.table is not None:
            return self.table
        key = (params.resample_hz, params.window_s, params.overlap, params.bands, params.coh_window_s,
               params.coh_overlap)
        if key not in self.cache:
            t = time.monotonic()
            self.cache[key] = R.from_atlas(self.atlas, params, self.p["exclude_iz"])
            dt = time.monotonic() - t
            self.seconds += dt
            ref = self.cache[key]
            log.info("reference (atlas features): %d normative of %d channels, %d features, %d edges, %.1fs",
                     int(ref.normative.sum()), ref.n, len(ref.features), len(ref.edge_a), dt)
        return self.cache[key]

    def summary(self) -> dict:
        refs = [self.table] if self.table is not None else list(self.cache.values())
        out = refs[0].summary() if refs else {"kind": self.kind}
        out["normative_rule"] = ("normative column" if self.kind == "table" else "soz_ch == 0 and resected_ch == 0") + \
                                (" and not irritative (spike_24h > 24)" if self.p["exclude_iz"] else "")
        out["seconds"] = round(self.seconds, 2)
        return out


def fmt(x) -> str:
    try:
        return "" if not np.isfinite(x) else f"{x:.4g}"
    except TypeError:
        return str(x)


def test_record(rid: str, ff: I.FeatureFile, ie: I.Ieeg, notes: list[str]) -> Sc.TestRec:
    chans = ff.rows[rid]
    names = list(chans)
    feats = sorted({f for c in chans.values() for f in c})
    features = {f: np.array([chans[c].get(f, np.nan) for c in names], float) for f in feats}
    chinfo = ie.channels.get(rid, {})
    if not chinfo:
        notes.append("no _channels.tsv for this recording: no coordinates")
    xyz = np.full((len(names), 3), np.nan)
    roi = np.full(len(names), "", dtype=object)
    non_mni = assumed = 0
    for i, c in enumerate(names):
        info = chinfo.get(c)
        if info is None:
            continue
        roi[i] = info.roi
        if not np.all(np.isfinite(info.xyz)):
            continue
        if info.space and not I.is_mni(info.space):
            non_mni += 1
            continue
        if not info.space:
            assumed += 1
        xyz[i] = info.xyz
    if non_mni:
        notes.append(f"{non_mni} channel(s) with coordinates in a non-MNI space: no radius region")
    if assumed:
        notes.append(f"{assumed} channel(s) with coordinates but no space: taken as MNI")
    t = Sc.TestRec(rid, names, xyz, roi, features)
    if rid in ff.coherence:
        index = {(rid, c): i for i, c in enumerate(names)}
        t.pair_i, t.pair_j, t.pair_bands, t.pair_v = R.pair_table(index, ((rid, *x) for x in ff.coherence[rid]))
    return t


def leave_out(subject: str, ie: I.Ieeg, ref: R.RefSet) -> set[str]:
    """The reference patients that are this subject: the same label, or its atlas patient."""
    keys = {subject}
    n = I.atlas_patient_of(subject, ie)
    if n is not None:
        keys.add(R.atlas_patient_key(n))
    return keys & set(ref.patient.tolist())


def process(rid: str, ff: I.FeatureFile, refs: References, ie: I.Ieeg, p: dict, sp: Sc.Params, writers: dict,
            labels: dict) -> dict:
    t0 = time.monotonic()
    params = spectral_params(ff.spectral)
    ref = refs.get(params)
    notes: list[str] = []
    test = test_record(rid, ff, ie, notes)
    if not set(test.features) & set(ref.features):
        raise ValueError(f"no feature in common with the reference (input has {', '.join(sorted(test.features)[:6])})")
    subject = ie.subjects.get(rid) or I.subject_of(rid)
    exclude = leave_out(subject, ie, ref) if p["lopo"] else set()
    if exclude:
        notes.append(f"reference patient {', '.join(sorted(exclude))}: left out of the reference")
    s = Sc.score(test, ref, sp, exclude)
    summ = Sc.summarise(s, sp)
    ab, ch, ed = writers["abnormality"], writers["channel"], writers["edges"]
    for i, c in enumerate(test.channels):
        for k, f in enumerate(s.features):
            ab.writerow([rid, c, f, fmt(s.z[i, k]), fmt(abs(s.z[i, k])), s.region[i], s.n_ref[i, k], Sc.family(f)])
        for b, band in enumerate(s.conn_bands):
            ab.writerow([rid, c, f"conn_{band}", "", fmt(s.conn[i, b]), s.region[i], s.conn_edges[i, b], "connectivity"])
        ch.writerow([rid, c] + [fmt(summ[k][i]) for k in ("max_abs_z", "max_band")] + [s.region[i], s.n_region[i]] +
                    [fmt(summ[k][i]) for k in Sc.SUMMARY_COLUMNS[2:]])
        info = ie.channels.get(rid, {}).get(c)
        for lab in ("soz", "resected"):
            v = getattr(info, lab) if info else None
            if v is not None:
                labels.setdefault(lab, []).append((rid, {k: summ[k][i] for k in Sc.VALIDATION_SCORES}, v))
    for e in range(len(test.pair_i)):
        for b, band in enumerate(s.conn_bands):
            ed.writerow([rid, test.channels[test.pair_i[e]], test.channels[test.pair_j[e]], band,
                         fmt(test.pair_v[e, test.pair_bands.index(band)]), fmt(s.edge_z[e, b]),
                         fmt(abs(s.edge_z[e, b])), s.edge_n[e, b]])
    counts = {"scored": int(np.isin(s.region, ["insufficient", "unknown"], invert=True).sum()),
              "insufficient": int((s.region == "insufficient").sum()), "unknown": int((s.region == "unknown").sum())}
    n_conn = int(np.isfinite(s.conn).any(axis=1).sum()) if s.conn.shape[1] else 0
    secs = time.monotonic() - t0
    log.info("%s: %d channels, %d scored, %d insufficient, %d unknown, %d features, %d pairs, %d with edge "
             "connectivity%s, %.2fs", rid, len(test.channels), counts["scored"], counts["insufficient"],
             counts["unknown"], len(s.features), len(test.pair_i), n_conn,
             f", {','.join(sorted(exclude))} left out" if exclude else "", secs)
    return {"status": "ok", "subject": subject, "left_out": sorted(exclude), "n_channels": len(test.channels),
            **counts, "features": s.features, "pairs": int(len(test.pair_i)), "edge_bands": s.conn_bands,
            "channels_with_edge_connectivity": n_conn, "notes": notes, "seconds": round(secs, 3)}


def validation_rows(labels: dict) -> list[list]:
    rows = []
    for lab, items in sorted(labels.items()):
        rids = sorted({x[0] for x in items})
        for score in Sc.VALIDATION_SCORES:
            for scope in rids + ["all"]:
                if scope == "all" and len(rids) == 1:
                    continue
                sel = [x for x in items if scope == "all" or x[0] == scope]
                s = np.array([x[1][score] for x in sel], dtype=float)
                if not np.isfinite(s).any():
                    continue
                y = np.array([x[2] for x in sel], dtype=bool)
                ok = np.isfinite(s)
                rows.append([scope, lab, score, int((y & ok).sum()), int((~y & ok).sum()), int((~ok).sum()),
                             fmt(N.auc(s, y))])
    return rows


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw_params, p, sp = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 1
    out = output_dir / p["label"] if p["label"] else output_dir
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s region=%s radius=%g mm min_reference=%d irritative %s, edges %s",
             APP_NAME, app_version(), input_dir, out, sp.region, sp.radius, sp.min_reference,
             "excluded" if p["exclude_iz"] else "kept", "on" if sp.edges else "off")
    files = I.walk_files(input_dir) if input_dir.is_dir() else []

    def rel(x: Path) -> str:
        try:
            return str(x.relative_to(input_dir))
        except ValueError:
            return str(x)

    found = R.find(files)
    if found is None:
        log.error("no reference under %s: a reference_channels.csv folder (Format A) or the atlas .mat "
                  "(HUP_atlas_final.mat, Format B)", input_dir)
        return 1
    try:
        refs = References(*found, p)
        log.info("reference %s (%s), %.1fs", rel(found[1]), "Format A table" if found[0] == "table" else "atlas .mat",
                 refs.seconds)
    except Exception as e:  # noqa: BLE001
        log.error("reference %s: %s", rel(found[1]), e)
        return 1
    features = I.load_features(files)
    n_recs = sum(len(f.rows) for f in features)
    if not n_recs:
        log.error("no features.csv (or bandpower.csv) with rows under %s", input_dir)
        return 1
    ie = I.load_ieeg(files)

    per_rec: dict[str, dict] = {}
    labels: dict[str, list] = {}
    n_ok = 0
    seen: set[str] = set()
    with open(out / "abnormality.csv", "w", newline="") as ab_fh, \
            open(out / "channel_abnormality.csv", "w", newline="") as ch_fh, \
            open(out / "edge_abnormality.csv", "w", newline="") as ed_fh:
        writers = {"abnormality": csv.writer(ab_fh), "channel": csv.writer(ch_fh), "edges": csv.writer(ed_fh)}
        writers["abnormality"].writerow(["recording_id", "channel", "feature", "z", "abs_z", "region", "n_reference",
                                         "family"])
        writers["channel"].writerow(["recording_id", "channel", "max_abs_z", "max_band", "region", "n_reference"] +
                                    Sc.SUMMARY_COLUMNS[2:])
        writers["edges"].writerow(["recording_id", "channel_a", "channel_b", "band", "coherence", "z", "abs_z",
                                   "n_reference"])
        for ff in features:
            if not ff.spectral:
                log.warning("%s: no provenance.json from ieeg-bandpower beside it; assuming default spectral settings",
                            rel(ff.path))
            if ff.legacy:
                log.warning("%s: bandpower.csv without features.csv (older ieeg-bandpower): spectral features only",
                            rel(ff.path))
            for rid in sorted(ff.rows):
                if rid in seen:
                    log.warning("%s: in more than one features table; using the first", rid)
                    continue
                seen.add(rid)
                try:
                    per_rec[rid] = {"features_file": rel(ff.path),
                                    **process(rid, ff, refs, ie, p, sp, writers, labels)}
                    n_ok += 1
                except Exception as e:  # noqa: BLE001 - a recording that fails is logged and skipped
                    log.error("%s: failed: %s", rid, e)
                    log.debug(traceback.format_exc())
                    per_rec[rid] = {"status": "failed", "error": str(e), "features_file": rel(ff.path)}
    vrows = validation_rows(labels)
    outputs = ["abnormality.csv", "channel_abnormality.csv", "edge_abnormality.csv"]
    if vrows:
        with open(out / "validation.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scope", "label", "score", "n_positive", "n_negative", "n_unscored", "auc"])
            w.writerows(vrows)
        outputs.append("validation.csv")
        for row in vrows:
            if row[0] == "all" or len({r[0] for r in vrows}) <= 2:
                log.info("validation %s %s %s: AUC %s (%d positive, %d negative)", row[0], row[1], row[2], row[6],
                         row[3], row[4])
    provenance = {
        "app": APP_NAME, "appVersion": app_version(), "tools": tool_versions(),
        "startedAt": started.isoformat(timespec="seconds"),
        "finishedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t_start, 2), "parameters": raw_params,
        "reference": {"path": rel(found[1]), **refs.summary()},
        "inputsFound": {"features": [rel(f.path) for f in features], "channels": sorted(ie.channels),
                        "reference": rel(found[1])},
        "outputs": outputs, "recordings": per_rec,
    }
    with open(out / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
    log.info("%d of %d recording(s) scored, %.1fs; wrote %s", n_ok, n_recs, time.monotonic() - t_start, out)
    if n_ok == 0:
        log.error("every recording failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
