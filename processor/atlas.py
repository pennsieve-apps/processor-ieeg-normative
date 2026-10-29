"""
The normative reference: the HUP Interictal iEEG Atlas (Bernabei et al. 2022; Pennsieve Discover
179, doi:10.26275/lzhy-ox20, ODC-By), HUP_atlas_final.mat, a MATLAB 5 (v7) file read with
scipy.io (not v7.3/HDF5):

  wake_clip    12000 x 3431   60 s at 200 Hz per bipolar channel, filtered 0.5-80 Hz (µV)
  mni_coords   3431 x 3       MNI-152 coordinates of each bipolar centroid
  patient_no   3431 x 1       patient number (1-60)
  soz_ch       3431 x 1       1 = in the seizure onset zone
  resected_ch  3431 x 1       1 = resected/ablated
  spike_24h    3431 x 1       estimated spikes per 24 h (irritative zone: > 24)

The reference band powers are computed from wake_clip with processor/spectral.py, the same
module processor-ieeg-bandpower uses for the input recordings, and kept in memory for the run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from processor import spectral as S

log = logging.getLogger("processor-ieeg-normative")

ATLAS_SFREQ = 200.0
REQUIRED = ("wake_clip", "mni_coords", "patient_no", "soz_ch", "resected_ch")
IRRITATIVE_SPIKES_24H = 24.0  # the atlas's own definition of the irritative zone (spike_24h > 24)


@dataclass
class Atlas:
    path: Path
    wake_clip: np.ndarray        # (n_samples, n_channels)
    xyz: np.ndarray              # (n_channels, 3)
    patient: np.ndarray          # (n_channels,) int
    soz: np.ndarray              # bool
    resected: np.ndarray         # bool
    spike_24h: np.ndarray | None  # float or None

    @property
    def n_channels(self) -> int:
        return self.xyz.shape[0]


def is_hdf5(path: Path) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(520)
    return b"\x89HDF\r\n\x1a\n" in head[:8] or head[512:520] == b"\x89HDF\r\n\x1a\n"


def load(path: Path) -> Atlas:
    """Read the atlas .mat. MATLAB v7.3 (HDF5) files are read with h5py when it is installed."""
    if is_hdf5(path):
        try:
            import h5py
        except ImportError:
            raise ValueError(f"{path.name} is a MATLAB v7.3 (HDF5) file; this image reads MATLAB 5/7 files "
                             "(save it with -v7)") from None
        with h5py.File(path, "r") as f:
            missing = [k for k in REQUIRED if k not in f]
            if missing:
                raise ValueError(f"{path.name}: missing {', '.join(missing)}")
            # h5py sees MATLAB arrays transposed
            d = {k: np.array(f[k]).T for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    else:
        from scipy import io as sio
        names = {n for n, _, _ in sio.whosmat(str(path))}
        missing = [k for k in REQUIRED if k not in names]
        if missing:
            raise ValueError(f"{path.name}: missing {', '.join(missing)}")
        want = [k for k in REQUIRED + ("spike_24h",) if k in names]
        d = sio.loadmat(str(path), variable_names=want)
    clip = np.asarray(d["wake_clip"], dtype=np.float64)
    xyz = np.asarray(d["mni_coords"], dtype=np.float64)
    n = xyz.shape[0]
    if clip.shape[1] != n and clip.shape[0] == n:
        clip = clip.T
    if clip.shape[1] != n:
        raise ValueError(f"wake_clip {clip.shape} does not match mni_coords {xyz.shape}")

    def vec(k):
        return np.asarray(d[k], dtype=np.float64).ravel()

    spikes = vec("spike_24h") if "spike_24h" in d else None
    return Atlas(path, clip, xyz, vec("patient_no").astype(int), vec("soz_ch") == 1, vec("resected_ch") == 1, spikes)


def find(input_dir: Path, files: list[Path]) -> Path | None:
    """The reference .mat: HUP_atlas_final.mat preferred, else the only .mat file."""
    mats = [p for p in files if p.suffix.lower() == ".mat"]
    if not mats:
        return None
    named = [p for p in mats if p.name == "HUP_atlas_final.mat"]
    if named:
        return named[0]
    if len(mats) > 1:
        log.warning("%d .mat files; using %s", len(mats), mats[0])
    return mats[0]


def reference_features(atlas: Atlas, params: S.SpectralParams, batch: int = 512) -> S.BandTable:
    """Band table of every atlas channel (the wake clip is already filtered 0.5-80 Hz at 200 Hz)."""
    n_s, n_ch = atlas.wake_clip.shape
    parts = []
    for a in range(0, n_ch, batch):
        cols = slice(a, min(n_ch, a + batch))

        def read(s0, s1, cols=cols):
            return atlas.wake_clip[s0:s1, cols].T

        acc, _ = S.stream(read, [(0, n_s)], ATLAS_SFREQ, cols.stop - cols.start, params, prefiltered=True)
        f, p = acc.psd()
        parts.append(S.band_table(f, p, params.bands))
    return S.BandTable(parts[0].bands, np.concatenate([t.abs_power for t in parts]),
                       np.concatenate([t.rel_power for t in parts]), np.concatenate([t.norm_median for t in parts]))
