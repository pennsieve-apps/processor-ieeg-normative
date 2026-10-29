"""
processor-ieeg-normative: how abnormal each channel's band power is, compared with the HUP
interictal iEEG atlas at the same place in MNI space.

  INPUT_DIR/features/    bandpower.csv (+ provenance.json) from ieeg-bandpower
  INPUT_DIR/ieeg/        the ieeg-prep derivative: _channels.tsv (x, y, z, space, soz, resected),
                         _electrodes.tsv (atlas_patient), recordings.csv
  INPUT_DIR/reference/   HUP_atlas_final.mat (Discover 179)
  OUTPUT_DIR[/LABEL]/abnormality.csv          recording_id, channel, band, z, abs_z, region, n_reference
  OUTPUT_DIR[/LABEL]/channel_abnormality.csv  recording_id, channel, max_abs_z, max_band, region, n_reference
  OUTPUT_DIR[/LABEL]/validation.csv           AUC of max_abs_z for soz / resected, when labels are known
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
from processor import spectral as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("processor-ieeg-normative")
log.setLevel(logging.INFO)

APP_NAME = "processor-ieeg-normative"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE = ("1", "true", "yes", "on")

DEFAULTS = {
    "RADIUS_MM": "20", "MIN_REFERENCE": "10", "FEATURE": "norm_median", "LEAVE_PATIENT_OUT": "true",
    "EXCLUDE_IRRITATIVE": "false", "LABEL": "",
}


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


def read_params(env=os.environ) -> tuple[dict, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p = {"radius": float(raw["RADIUS_MM"]), "min_reference": int(float(raw["MIN_REFERENCE"])),
         "feature": raw["FEATURE"].strip(), "lopo": str(raw["LEAVE_PATIENT_OUT"]).strip().lower() in TRUE,
         "exclude_iz": str(raw["EXCLUDE_IRRITATIVE"]).strip().lower() in TRUE, "label": raw["LABEL"].strip()}
    if p["radius"] <= 0:
        raise ValueError("RADIUS_MM must be > 0")
    if p["min_reference"] < 2:
        raise ValueError("MIN_REFERENCE must be >= 2")
    if p["feature"] not in N.FEATURES:
        raise ValueError(f"FEATURE must be one of {', '.join(N.FEATURES)}")
    if p["label"] and not all(c.isalnum() or c in "-_" for c in p["label"]):
        raise ValueError("LABEL: letters, digits, - and _ only")
    return raw, p


def spectral_params(block: dict | None) -> S.SpectralParams:
    """The SpectralParams ieeg-bandpower used (its provenance.json), else the defaults."""
    if not block:
        return S.SpectralParams()
    bands = tuple((n, float(e[0]), float(e[1])) for n, e in block.get("bands", {}).items()) or S.BAND_PRESETS["conventions"]
    return S.SpectralParams(resample_hz=float(block.get("resample_hz", 200)),
                            window_s=float(block.get("welch_window_s", 2)),
                            overlap=float(block.get("welch_overlap", 0.5)), bands=bands)


class ReferenceCache:
    """Reference band tables per spectral setting, computed once per run and kept in memory."""

    def __init__(self, atlas: A.Atlas, p: dict):
        self.atlas, self.p, self.tables, self.seconds = atlas, p, {}, 0.0
        norm = ~atlas.soz & ~atlas.resected
        if p["exclude_iz"] and atlas.spike_24h is not None:
            norm &= ~(atlas.spike_24h > A.IRRITATIVE_SPIKES_24H)
        self.normative = norm

    def get(self, params: S.SpectralParams) -> tuple[list[str], N.Reference]:
        key = (params.resample_hz, params.window_s, params.overlap, params.bands)
        if key not in self.tables:
            t = time.monotonic()
            table = A.reference_features(self.atlas, params)
            nb = len(table.bands) - 1
            vals = N.transform(self.p["feature"], table.abs_power[:, :nb], table.rel_power[:, :nb],
                               table.norm_median[:, :nb])
            m = self.normative
            self.tables[key] = (table.bands[:nb], N.Reference(self.atlas.xyz[m], self.atlas.patient[m], vals[m]))
            self.seconds += time.monotonic() - t
            log.info("reference: %d normative of %d atlas channels, bands %s, %.1fs", int(m.sum()),
                     self.atlas.n_channels, ",".join(table.bands[:nb]), time.monotonic() - t)
        return self.tables[key]


def fmt(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.4g}"


def process(rid: str, rows: list[dict], params: S.SpectralParams, cache: ReferenceCache, ie: I.Ieeg, p: dict,
            ab_w, ch_w, labels: dict) -> dict:
    t0 = time.monotonic()
    bands, ref = cache.get(params)
    by_ch: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_ch.setdefault(r["channel"], {})[r["band"]] = r
    names = list(by_ch)
    missing_bands = [b for b in bands if any(b not in by_ch[c] for c in names)]
    if missing_bands:
        raise ValueError(f"bandpower.csv lacks band(s) {', '.join(missing_bands)} for some channels")
    col = {k: np.array([[I.fnum(by_ch[c][b].get(k)) for b in bands] for c in names])
           for k in ("abs_power", "rel_power", "norm_median")}
    if p["feature"] == "norm_median" and np.isnan(col["norm_median"]).all():
        raise ValueError("bandpower.csv has no norm_median column (FEATURE=norm_median needs ieeg-bandpower >= 0.1.0)")
    values = N.transform(p["feature"], col["abs_power"], col["rel_power"], col["norm_median"])
    chinfo = ie.channels.get(rid, {})
    notes = []
    if not chinfo:
        notes.append("no _channels.tsv for this recording: no coordinates")
    xyz = np.full((len(names), 3), np.nan)
    non_mni = 0
    assumed = 0
    for i, c in enumerate(names):
        info = chinfo.get(c)
        if info is None or not np.all(np.isfinite(info.xyz)):
            continue
        if info.space and not I.is_mni(info.space):
            non_mni += 1
            continue
        if not info.space:
            assumed += 1
        xyz[i] = info.xyz
    if non_mni:
        notes.append(f"{non_mni} channel(s) with coordinates in a non-MNI space: region unknown")
    if assumed:
        notes.append(f"{assumed} channel(s) with coordinates but no space: taken as MNI")
    subject = ie.subjects.get(rid) or I.subject_of(rid)
    patient = I.atlas_patient_of(subject, ie)
    exclude = patient if (p["lopo"] and patient is not None) else None
    if exclude is not None:
        notes.append(f"atlas patient {exclude}: left out of the reference")
    r = N.zscore(ref, xyz, values, exclude, p["radius"], p["min_reference"])
    region_ok = f"within-{p['radius']:g}mm"
    maxz = np.full(len(names), np.nan)
    for i, c in enumerate(names):
        region = region_ok if r.status[i] == "ok" else r.status[i]
        for b, band in enumerate(bands):
            z = r.z[i, b]
            ab_w.writerow([rid, c, band, fmt(z), fmt(abs(z)), region, r.n_reference[i]])
        if r.status[i] == "ok" and np.isfinite(r.z[i]).any():
            j = int(np.nanargmax(np.abs(r.z[i])))
            maxz[i] = abs(r.z[i, j])
            ch_w.writerow([rid, c, fmt(maxz[i]), bands[j], region, r.n_reference[i]])
        else:
            ch_w.writerow([rid, c, "", "", region, r.n_reference[i]])
        info = chinfo.get(c)
        for lab in ("soz", "resected"):
            v = getattr(info, lab) if info else None
            if v is not None:
                labels.setdefault(lab, []).append((rid, maxz[i], v))
    counts = {s: int((r.status == s).sum()) for s in ("ok", "insufficient", "unknown")}
    secs = time.monotonic() - t0
    log.info("%s: %d channels, %d scored, %d insufficient, %d unknown%s, %.2fs", rid, len(names), counts["ok"],
             counts["insufficient"], counts["unknown"], f", atlas patient {exclude} left out" if exclude else "", secs)
    return {"status": "ok", "subject": subject, "atlas_patient_left_out": exclude, "n_channels": len(names),
            "scored": counts["ok"], "insufficient": counts["insufficient"], "unknown": counts["unknown"],
            "notes": notes, "seconds": round(secs, 3)}


def validation_rows(labels: dict) -> list[list]:
    rows = []
    for lab, items in sorted(labels.items()):
        rids = sorted({x[0] for x in items})
        for scope in rids + ["all"]:
            sel = [x for x in items if scope == "all" or x[0] == scope]
            s = np.array([x[1] for x in sel], dtype=float)
            y = np.array([x[2] for x in sel], dtype=bool)
            ok = np.isfinite(s)
            if scope == "all" and len(rids) == 1:
                continue
            rows.append([scope, lab, int((y & ok).sum()), int((~y & ok).sum()), int((~ok).sum()), fmt(N.auc(s, y))])
    return rows


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw_params, p = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 1
    out = output_dir / p["label"] if p["label"] else output_dir
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s feature=%s radius=%g mm min_reference=%d", APP_NAME, app_version(),
             input_dir, out, p["feature"], p["radius"], p["min_reference"])
    files = I.walk_files(input_dir) if input_dir.is_dir() else []

    def rel(x: Path) -> str:
        try:
            return str(x.relative_to(input_dir))
        except ValueError:
            return str(x)

    atlas_path = A.find(input_dir, files)
    if atlas_path is None:
        log.error("no reference atlas (.mat, e.g. HUP_atlas_final.mat) under %s", input_dir)
        return 1
    try:
        t = time.monotonic()
        atlas = A.load(atlas_path)
        log.info("atlas %s: %d channels, %d patients, %.1fs", rel(atlas_path), atlas.n_channels,
                 len(np.unique(atlas.patient)), time.monotonic() - t)
    except Exception as e:  # noqa: BLE001
        log.error("reference %s: %s", rel(atlas_path), e)
        return 1
    features = I.load_features(files)
    n_recs = sum(len(f.rows) for f in features)
    if not n_recs:
        log.error("no bandpower.csv with rows under %s", input_dir)
        return 1
    ie = I.load_ieeg(files)
    cache = ReferenceCache(atlas, p)

    per_rec: dict[str, dict] = {}
    labels: dict[str, list] = {}
    n_ok = 0
    seen: set[str] = set()
    with open(out / "abnormality.csv", "w", newline="") as ab_fh, \
            open(out / "channel_abnormality.csv", "w", newline="") as ch_fh:
        ab_w, ch_w = csv.writer(ab_fh), csv.writer(ch_fh)
        ab_w.writerow(["recording_id", "channel", "band", "z", "abs_z", "region", "n_reference"])
        ch_w.writerow(["recording_id", "channel", "max_abs_z", "max_band", "region", "n_reference"])
        for ff in features:
            if not ff.spectral:
                log.warning("%s: no provenance.json from ieeg-bandpower beside it; assuming default spectral settings",
                            rel(ff.path))
            try:
                params = spectral_params(ff.spectral)
            except ValueError as e:
                log.error("%s: spectral settings: %s", rel(ff.path), e)
                for rid in ff.rows:
                    per_rec[rid] = {"status": "failed", "error": str(e), "features": rel(ff.path)}
                continue
            for rid in sorted(ff.rows):
                if rid in seen:
                    log.warning("%s: in more than one bandpower.csv; using the first", rid)
                    continue
                seen.add(rid)
                try:
                    per_rec[rid] = {"features": rel(ff.path),
                                    **process(rid, ff.rows[rid], params, cache, ie, p, ab_w, ch_w, labels)}
                    n_ok += 1
                except Exception as e:  # noqa: BLE001 - a recording that fails is logged and skipped
                    log.error("%s: failed: %s", rid, e)
                    log.debug(traceback.format_exc())
                    per_rec[rid] = {"status": "failed", "error": str(e), "features": rel(ff.path)}
    vrows = validation_rows(labels)
    outputs = ["abnormality.csv", "channel_abnormality.csv"]
    if vrows:
        with open(out / "validation.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scope", "label", "n_positive", "n_negative", "n_unscored", "auc_max_abs_z"])
            w.writerows(vrows)
        outputs.append("validation.csv")
        for row in vrows:
            if row[0] == "all" or len({r[0] for r in vrows}) <= 2:
                log.info("validation %s %s: AUC %s (%d positive, %d negative)", row[0], row[1], row[5], row[2], row[3])
    provenance = {
        "app": APP_NAME, "appVersion": app_version(), "tools": tool_versions(),
        "startedAt": started.isoformat(timespec="seconds"),
        "finishedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t_start, 2), "parameters": raw_params,
        "reference": {"path": rel(atlas_path), "channels": atlas.n_channels,
                      "patients": int(len(np.unique(atlas.patient))), "normative_channels": int(cache.normative.sum()),
                      "normative_rule": "soz_ch == 0 and resected_ch == 0" +
                                        (" and spike_24h <= 24" if p["exclude_iz"] else ""),
                      "seconds": round(cache.seconds, 2)},
        "inputsFound": {"features": [rel(f.path) for f in features], "channels": sorted(ie.channels),
                        "reference": rel(atlas_path)},
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
