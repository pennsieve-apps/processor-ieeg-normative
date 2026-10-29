"""
Spectral band power of iEEG channels, after Bernabei et al. 2022 (Brain, doi:10.1093/brain/awab480).

This module is shared: processor-ieeg-bandpower computes the band powers of the input
recordings with it, and processor-ieeg-normative computes the band powers of the normative
atlas with it (a vendored, byte-identical copy in processor-ieeg-normative/processor/spectral.py;
`make check-vendored` there compares the two). Change it here first, then copy it over.

Method (Bernabei et al. 2022, Methods, "Univariate features"; reimplemented from the paper,
no code copied from the authors' unlicensed repository):

1. band-pass 0.5-80 Hz with a first-order Butterworth filter (zero-phase), unless the data
   are already filtered (the HUP atlas `wake_clip` is);
2. resample to 200 Hz (polyphase, anti-aliased), so that recordings at 512, 1024 or 2048 Hz
   and the 200 Hz atlas are computed on the same grid;
3. Welch PSD with a 2 s Hamming window and 1 s overlap (0.5 Hz bins), averaged over every
   window of the selected interval;
4. per band:
   - `abs_power`: the PSD integrated over the band (trapezoid, both edges included), µV²;
   - `rel_power`: abs_power divided by the 1-80 Hz broadband power (the band's share);
   - `norm_median`: the paper's feature. The PSD is normalised to sum to 1 over 0.5-80 Hz
     and the median of the normalised PSD over the band's bins is taken (bins lo <= f < hi;
     the highest band also includes its upper edge).

Long recordings are streamed in chunks (CHUNK_S seconds plus padding that absorbs the
filter and resampler transients), so memory does not grow with recording length; Welch
windows run across chunk boundaries as if the interval had been processed in one piece.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Sequence

import numpy as np
from scipy import signal

SPECTRAL_VERSION = "1.0"

# The bands of conventions.md (Bernabei 2022's Fig. 4 legend), and the edges the paper's
# Methods text gives. Names are fixed (downstream steps key on them); edges can change.
BAND_PRESETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "conventions": (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                    ("beta", 13.0, 30.0), ("gamma", 30.0, 80.0)),
    "bernabei2022": (("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
                     ("beta", 12.0, 30.0), ("gamma", 30.0, 80.0)),
}
BROADBAND = ("broadband", 1.0, 80.0)  # rel_power denominator (conventions.md: share of 1-80 Hz)
NORM_RANGE = (0.5, 80.0)              # the paper normalises the 0.5-80 Hz PSD to sum to 1
BAND_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def parse_bands(value: str | None) -> tuple[tuple[str, float, float], ...]:
    """A preset name (conventions, bernabei2022) or a JSON object {"delta": [1, 4], ...}."""
    v = (value or "conventions").strip()
    if v in BAND_PRESETS:
        return BAND_PRESETS[v]
    try:
        obj = json.loads(v)
    except json.JSONDecodeError as e:
        raise ValueError(f"BANDS must be one of {', '.join(BAND_PRESETS)} or a JSON object: {e}") from None
    if not isinstance(obj, dict) or not obj:
        raise ValueError("BANDS JSON must be a non-empty object of name: [low_hz, high_hz]")
    out = []
    for name, edges in obj.items():
        if not BAND_NAME_RE.match(name) or name == "broadband":
            raise ValueError(f"band name {name!r}: lowercase letters, digits and -, not 'broadband'")
        if not (isinstance(edges, (list, tuple)) and len(edges) == 2):
            raise ValueError(f"band {name}: [low_hz, high_hz]")
        lo, hi = float(edges[0]), float(edges[1])
        if not (0 <= lo < hi):
            raise ValueError(f"band {name}: need 0 <= low < high, got {lo}, {hi}")
        out.append((name, lo, hi))
    return tuple(sorted(out, key=lambda b: (b[1], b[2])))


def parse_bandpass(value: str | None) -> tuple[float, float] | None:
    """'0.5-80' -> (0.5, 80.0); 'none', 'off' or '' -> None."""
    v = (value or "").strip().lower()
    if v in ("", "none", "off", "false", "0"):
        return None
    m = re.match(r"^\s*([0-9.]+)\s*[-,:]\s*([0-9.]+)\s*$", v)
    if not m:
        raise ValueError(f"BANDPASS must look like 0.5-80 or be none, got {value!r}")
    lo, hi = float(m.group(1)), float(m.group(2))
    if not 0 < lo < hi:
        raise ValueError(f"BANDPASS: need 0 < low < high, got {value!r}")
    return lo, hi


@dataclass(frozen=True)
class SpectralParams:
    resample_hz: float = 200.0
    window_s: float = 2.0
    overlap: float = 0.5
    bandpass: tuple[float, float] | None = (0.5, 80.0)
    bands: tuple[tuple[str, float, float], ...] = BAND_PRESETS["conventions"]
    chunk_s: float = 60.0
    pad_s: float = 4.0

    def __post_init__(self):
        top = max([b[2] for b in self.bands] + [BROADBAND[2], NORM_RANGE[1]])
        if self.resample_hz < 2 * top:
            raise ValueError(f"RESAMPLE_HZ {self.resample_hz} is below twice the highest band edge ({top} Hz)")
        if not 0 <= self.overlap < 1:
            raise ValueError("WELCH_OVERLAP must be in [0, 1)")
        if self.window_s * self.resample_hz < 8:
            raise ValueError("WELCH_WINDOW_S too short")

    def as_dict(self) -> dict:
        return {"resample_hz": self.resample_hz, "welch_window_s": self.window_s, "welch_overlap": self.overlap,
                "welch_window": "hamming (symmetric)", "bandpass_hz": list(self.bandpass) if self.bandpass else None,
                "bandpass_filter": "Butterworth order 1, zero-phase (sosfiltfilt)",
                "bands": {n: [lo, hi] for n, lo, hi in self.bands}, "broadband_hz": list(BROADBAND[1:]),
                "norm_range_hz": list(NORM_RANGE), "spectral_version": SPECTRAL_VERSION}


def resample_ratio(sfreq: float, target: float) -> tuple[int, int]:
    """(up, down) with sfreq * up / down == target (to 1e-6 relative); (1, 1) when equal."""
    if abs(sfreq - target) <= 1e-9 * target:
        return 1, 1
    frac = Fraction(target / sfreq).limit_denominator(10000)
    if abs(sfreq * frac.numerator / frac.denominator - target) > 1e-6 * target:
        raise ValueError(f"cannot resample {sfreq} Hz to {target} Hz with a rational ratio")
    return frac.numerator, frac.denominator


def preprocess(x: np.ndarray, sfreq: float, params: SpectralParams, prefiltered: bool = False) -> np.ndarray:
    """Band-pass (unless prefiltered) and resample (n_ch, n) to params.resample_hz."""
    x = np.asarray(x, dtype=np.float64)
    if not prefiltered and params.bandpass is not None:
        lo, hi = params.bandpass
        nyq = sfreq / 2
        if hi < nyq * 0.99:
            sos = signal.butter(1, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        else:
            sos = signal.butter(1, lo, btype="highpass", fs=sfreq, output="sos")
        x = signal.sosfiltfilt(sos, x, axis=-1)
    up, down = resample_ratio(sfreq, params.resample_hz)
    if (up, down) != (1, 1):
        x = signal.resample_poly(x, up, down, axis=-1)
    return x


class Accumulator:
    """
    Welch averaging (and optionally the cross-spectral matrix for coherence) over windows
    fed in contiguous pieces. `add(x)` continues the current stretch (samples left over
    from the last piece are prepended); `gap()` starts a new stretch.
    """

    def __init__(self, n_ch: int, fs: float, params: SpectralParams, cross: bool = False, max_hz: float | None = None):
        self.n_ch, self.fs, self.params, self.cross = n_ch, fs, params, cross
        self.nper = int(round(params.window_s * fs))
        self.nover = int(round(self.nper * params.overlap))
        self.step = self.nper - self.nover
        self.win = signal.windows.hamming(self.nper, sym=True)
        self.freqs = np.fft.rfftfreq(self.nper, 1 / fs)
        self.nf = len(self.freqs) if max_hz is None else int(np.searchsorted(self.freqs, max_hz, side="right"))
        self.pxx = np.zeros((n_ch, self.nf))
        self.sxy = np.zeros((n_ch, n_ch, self.nf), dtype=np.complex128) if cross else None
        self.n_windows = 0
        self.tail = np.zeros((n_ch, 0))

    def gap(self):
        self.tail = np.zeros((self.n_ch, 0))

    def add(self, x: np.ndarray):
        buf = np.concatenate([self.tail, x], axis=1) if self.tail.shape[1] else x
        n = buf.shape[1]
        if n < self.nper:
            self.tail = buf
            return
        k = 1 + (n - self.nper) // self.step
        # frames in batches to bound the (n_ch, batch, nper) temporaries
        batch = max(1, int(2e7 // max(1, self.n_ch * self.nper)))
        for a in range(0, k, batch):
            b = min(k, a + batch)
            idx = (np.arange(a, b) * self.step)[:, None] + np.arange(self.nper)[None, :]
            frames = buf[:, idx]                                  # (n_ch, m, nper)
            frames = frames - frames.mean(axis=-1, keepdims=True)  # detrend='constant'
            spec = np.fft.rfft(frames * self.win, axis=-1)[..., : self.nf]
            self.pxx += np.einsum("cmf,cmf->cf", spec, spec.conj()).real
            if self.cross:
                self.sxy += np.einsum("imf,jmf->ijf", spec, spec.conj())
        self.n_windows += k
        self.tail = buf[:, k * self.step:]

    def _scale(self) -> np.ndarray:
        s = np.full(self.nf, 2.0 / (self.fs * np.sum(self.win ** 2)))
        s[0] /= 2
        if self.nper % 2 == 0 and self.nf == len(self.freqs):
            s[-1] /= 2
        return s

    def psd(self) -> tuple[np.ndarray, np.ndarray]:
        """(freqs, psd (n_ch, nf)) one-sided density, units²/Hz; same as scipy.signal.welch."""
        if self.n_windows == 0:
            raise ValueError("no complete Welch window")
        return self.freqs[: self.nf], self.pxx / self.n_windows * self._scale()

    def coherence(self) -> np.ndarray:
        """Magnitude-squared coherence (n_ch, n_ch, nf)."""
        if not self.cross:
            raise ValueError("accumulator built without cross spectra")
        d = np.einsum("iif->if", self.sxy).real
        denom = d[:, None, :] * d[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.abs(self.sxy) ** 2 / denom
        return np.where(denom > 0, c, np.nan)


@dataclass
class BandTable:
    bands: list[str]                 # band names, then "broadband"
    abs_power: np.ndarray            # (n_ch, n_bands + 1), µV²
    rel_power: np.ndarray            # (n_ch, n_bands + 1), share of 1-80 Hz
    norm_median: np.ndarray          # (n_ch, n_bands + 1), median of the sum-1-normalised 0.5-80 Hz PSD
    extra: dict = field(default_factory=dict)


def band_masks(freqs: np.ndarray, bands: Sequence[tuple[str, float, float]]):
    """Per band: (closed mask for integration, half-open mask for the median; top band closed)."""
    top = max(hi for _, _, hi in bands)
    out = []
    for name, lo, hi in list(bands) + [BROADBAND]:
        closed = (freqs >= lo - 1e-9) & (freqs <= hi + 1e-9)
        if name == "broadband" or abs(hi - top) < 1e-9:
            half = closed
        else:
            half = (freqs >= lo - 1e-9) & (freqs < hi - 1e-9)
        out.append((name, closed, half))
    return out


def band_table(freqs: np.ndarray, psd: np.ndarray, bands: Sequence[tuple[str, float, float]]) -> BandTable:
    psd = np.atleast_2d(psd)
    masks = band_masks(freqs, bands)
    nmask = (freqs >= NORM_RANGE[0] - 1e-9) & (freqs <= NORM_RANGE[1] + 1e-9)
    total = psd[:, nmask].sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(total > 0, psd / total, np.nan)
    names, absp, med = [], [], []
    for name, closed, half in masks:
        if closed.sum() < 2:
            raise ValueError(f"band {name} has fewer than two frequency bins; lengthen WELCH_WINDOW_S")
        names.append(name)
        absp.append(np.trapezoid(psd[:, closed], freqs[closed], axis=1))
        med.append(np.median(normed[:, half & nmask], axis=1) if (half & nmask).any() else np.full(psd.shape[0], np.nan))
    absp = np.stack(absp, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(absp[:, -1:] > 0, absp / absp[:, -1:], np.nan)
    return BandTable(names, absp, rel, np.stack(med, axis=1))


def coherence_bands(freqs: np.ndarray, coh: np.ndarray, bands: Sequence[tuple[str, float, float]]) -> tuple[list[str], np.ndarray]:
    """Median magnitude-squared coherence per band: (names, (n_ch, n_ch, n_bands))."""
    names, out = [], []
    for name, _, half in band_masks(freqs[: coh.shape[-1]], bands)[:-1]:
        names.append(name)
        out.append(np.nanmedian(coh[..., half], axis=-1))
    return names, np.stack(out, axis=-1)


def stream(read: Callable[[int, int], np.ndarray], intervals: Sequence[tuple[int, int]], sfreq: float, n_ch: int,
           params: SpectralParams, prefiltered: bool = False, cross: bool = False) -> tuple[Accumulator, dict]:
    """
    Feed the sample intervals [(start, stop), ...] of a recording at sfreq through
    preprocess() and a Welch Accumulator, CHUNK_S at a time. read(start, stop) returns
    (n_ch, stop - start) in µV. Returns the accumulator and counts for provenance.
    """
    up, down = resample_ratio(sfreq, params.resample_hz)
    fs = sfreq * up / down
    top = max([NORM_RANGE[1], BROADBAND[2]] + [hi for _, _, hi in params.bands])
    acc = Accumulator(n_ch, fs, params, cross=cross, max_hz=top + 1e-6)
    # chunk and pad lengths in input samples, multiples of `down` so that output samples line up
    chunk = max(down, int(params.chunk_s * sfreq) // down * down)
    pad = 0 if prefiltered and (up, down) == (1, 1) else int(math.ceil(params.pad_s * sfreq / down)) * down
    used_s = 0.0
    skipped = 0
    for start, stop in intervals:
        if (stop - start) * up / down < acc.nper:
            skipped += 1
            continue
        acc.gap()
        pos = start
        while pos < stop:
            end = min(stop, pos + chunk)
            a = max(start, pos - pad)
            b = min(stop, end + pad)
            # align the padded read start to the chunk start in `down` steps (within the interval)
            a = pos - (pos - a) // down * down
            x = read(a, b)
            y = preprocess(x, sfreq, params, prefiltered=prefiltered)
            lead = (pos - a) * up // down
            keep = int(round((end - pos) * up / down))
            acc.add(y[:, lead: lead + keep])
            pos = end
        used_s += (stop - start) / sfreq
    return acc, {"seconds_used": round(used_s, 3), "intervals_too_short": skipped, "welch_windows": acc.n_windows,
                 "fs": fs}
