"""Windowed datasets over the cached RealPDE trajectories.

Two things here are easy to get wrong, and one of them already cost the
organizers a whole Development Phase.

WINDOW LEAKAGE
    A window is 20 input frames followed by 20 target frames. Cut windows at
    stride s from one trajectory and window i covers [t, t+40) while window i+1
    covers [t+s, t+s+40). With s < 40 the two overlap, and with s = 20 window
    i+1's *input* is exactly window i's *target*.

    That is harmless as training augmentation — it is all training data either
    way. It is not harmless for evaluation: the scorer hands the model many
    windows, so a model that reads across them can return an answer it was
    already given. That is precisely the bug that forced the 5 August restart.

    So: `train_stride` may be small (more samples), but `val_stride` defaults to
    40, which makes validation windows fully disjoint and the local score honest.

SIM/REAL SCALE
    The simulation is non-dimensionalised by the freestream velocity — its mean
    |u| sits near 0.86 at every Reynolds number — while the real data is in m/s
    and its magnitude grows linearly with Re (measured corr = 0.995). Two
    consequences:

      * a model pretrained on raw sim sees a distribution up to 7x off the real one;
      * more subtly, in sim the velocity magnitude carries NO information about
        Re, while in real it carries almost all of it. Since `metadata` is empty
        at inference, magnitude is exactly how a model can infer the regime — and
        a model pretrained on raw sim never learns to use it.

    Both are fixed by redimensionalising sim into m/s before normalising:

        U_inf = Re * U_PER_RE,    U_PER_RE = 1.394e-5 m/s per unit Re

    measured here by regressing the real/sim mean-|u| ratio on Re across 18 cases
    (implied chord Re*nu/U_inf = 7.2 cm, constant across cases as it must be).
    After that both splits are normalised with the official real statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

T_IN = 20
T_OUT = 20

# Official normalization statistics, from the Track 2 starting kit's
# example_data/mean_std_real.pt. The std entries are the two numbers scoring.py
# averages into SIGMA_GLOBAL = 0.0563870259, which confirms their provenance.
OFFICIAL_MEAN_REAL = np.array([0.1550, -0.0005, 0.0], dtype=np.float32)
OFFICIAL_STD_REAL = np.array([0.0968, 0.0160, 1.0], dtype=np.float32)  # 1.0 on p: never divide by 0

# Freestream velocity per unit Reynolds number, measured in scripts/, used to put
# the non-dimensional simulation back into m/s.
U_PER_RE = 1.394e-5

# Held out from training to mimic the private test set, which uses unseen
# parameter regimes. Whole Reynolds numbers, spread across the range, so
# validation measures interpolation to Re values never trained on.
DEFAULT_VAL_RE = (6300, 13950, 22875)

# The same idea taken literally. DEFAULT_VAL_RE holds out three values from the
# middle of the trained range, so a model only ever has to interpolate to reach
# them, and every score measured on it inherits that. The private test says
# "unseen parameter regimes" without promising they lie inside ours, and the
# quantity that matters -- the velocity magnitude -- is proportional to Re, so a
# value outside the range is an extrapolation in the input distribution itself.
# Holding out both ends measures that. Training then spans 6300..24150 and
# validation sits below and above it on both sides.
EXTRAPOLATION_VAL_RE = (3750, 5025, 25425, 26700)


# Cache arrays per split so repeated WindowDataset construction (train + val)
# does not read the same gigabyte twice.
_ARRAY_CACHE: dict[tuple[str, bool], np.ndarray] = {}
_TENSOR_CACHE: dict[tuple[str, bool, bool], torch.Tensor] = {}


def load_split(split: str, in_ram: bool = True):
    """Load a cached split. `in_ram=True` reads the whole array into memory.

    real is 1.05 GB and sim 2.29 GB, against 32 GB of RAM, so holding them
    resident is comfortably affordable and removes the I/O bottleneck: memory
    mapping made an epoch spend most of its time re-reading 3.6 GB of windows
    from disk for a model that only needs a few milliseconds per batch.
    """
    meta = json.loads((PROCESSED / f"{split}.json").read_text())
    key = (split, in_ram)
    arr = _ARRAY_CACHE.get(key)
    if arr is None:
        path = PROCESSED / f"{split}.npy"
        arr = np.load(path) if in_ram else np.load(path, mmap_mode="r")
        _ARRAY_CACHE[key] = arr
    return arr, meta


def prepared_tensor(split: str, normalize: bool, redimensionalize_sim: bool) -> torch.Tensor:
    """The whole split as one normalized float32 tensor, built once and shared.

    Doing the scaling and normalization per sample cost more than everything else
    combined: 5523 windows an epoch, each copying and transforming 655 KB in
    Python, which pinned an epoch at ~44 s for a model that needs milliseconds a
    batch. Applied once to the full array instead, __getitem__ becomes a slice.
    """
    key = (split, normalize, redimensionalize_sim)
    cached = _TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    arr, meta = load_split(split)
    # Copy before touching it: torch.from_numpy shares storage with the cached
    # numpy array, so the in-place rescaling below would otherwise corrupt the
    # cache for every other caller.
    data = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
    channels = len(meta["channels"])

    if split == "sim" and redimensionalize_sim:
        # Per trajectory: the simulation is non-dimensional, so multiply by its
        # own freestream velocity to put it in m/s alongside the real split.
        for t in meta["trajectories"]:
            o, L = t["offset"], t["length"]
            data[o:o + L] *= t["re_nominal"] * U_PER_RE

    if normalize:
        mean = torch.from_numpy(OFFICIAL_MEAN_REAL[:channels])
        std = torch.from_numpy(OFFICIAL_STD_REAL[:channels])
        data = (data - mean) / std

    _TENSOR_CACHE[key] = data
    return data


class WindowDataset(Dataset):
    """(T_IN, H, W, C) -> (T_OUT, H, W, C) windows cut from cached trajectories.

    Windows never cross a trajectory boundary: the index is built per trajectory
    from its own offset and length.
    """

    def __init__(
        self,
        split: str = "real",
        cases: list[str] | None = None,
        stride: int = 10,
        normalize: bool = True,
        redimensionalize_sim: bool = True,
        in_ram: bool = True,
    ):
        self.split = split
        _, self.meta = load_split(split, in_ram=in_ram)
        self.normalize = normalize
        self.redimensionalize_sim = redimensionalize_sim
        self.channels = self.meta["channels"]
        # Scaling and normalization are baked in once, so __getitem__ only slices.
        self.data = prepared_tensor(split, normalize, redimensionalize_sim)

        traj = self.meta["trajectories"]
        if cases is not None:
            wanted = set(cases)
            traj = [t for t in traj if t["case"] in wanted]
        self.trajectories = traj

        # (trajectory_position_in_list, start_frame) for every window.
        self.index: list[tuple[int, int]] = []
        span = T_IN + T_OUT
        for ti, t in enumerate(traj):
            last_start = t["length"] - span
            if last_start < 0:
                continue  # trajectory shorter than one window
            for s in range(0, last_start + 1, stride):
                self.index.append((ti, s))

        self.mean = torch.from_numpy(OFFICIAL_MEAN_REAL[: len(self.channels)])
        self.std = torch.from_numpy(OFFICIAL_STD_REAL[: len(self.channels)])

    def __len__(self) -> int:
        return len(self.index)

    def case_of(self, i: int) -> dict:
        return self.trajectories[self.index[i][0]]

    def __getitem__(self, i: int):
        ti, s = self.index[i]
        traj = self.trajectories[ti]
        o = traj["offset"] + s
        return {
            "input": self.data[o: o + T_IN],                     # (T_IN, H, W, C)
            "target": self.data[o + T_IN: o + T_IN + T_OUT],     # (T_OUT, H, W, C)
            "re": float(traj["re"]),
            "aoa": float(traj["aoa"]),
            "case": traj["case"],
        }


def split_cases(split: str = "real", val_re: tuple[int, ...] = DEFAULT_VAL_RE):
    """Partition trajectories by Reynolds number, so no Re appears in both sides."""
    _, meta = load_split(split)
    train, val = [], []
    for t in meta["trajectories"]:
        (val if t["re_nominal"] in val_re else train).append(t["case"])
    return train, val


def build_datasets(
    split: str = "real",
    val_re: tuple[int, ...] = DEFAULT_VAL_RE,
    train_stride: int = 10,
    val_stride: int = T_IN + T_OUT,   # 40: fully disjoint windows, no cross-window leak
    **kwargs,
):
    train_cases, val_cases = split_cases(split, val_re)
    train = WindowDataset(split, cases=train_cases, stride=train_stride, **kwargs)
    val = WindowDataset(split, cases=val_cases, stride=val_stride, **kwargs)
    return train, val


def denormalize(x: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Undo normalization, back into m/s — what the scorer compares against."""
    mean = torch.from_numpy(OFFICIAL_MEAN_REAL[:channels]).to(x.device)
    std = torch.from_numpy(OFFICIAL_STD_REAL[:channels]).to(x.device)
    return x * std + mean


def to_submission_format(pred: torch.Tensor | np.ndarray) -> np.ndarray:
    """(N, T_OUT, H, W, 2) in m/s -> (N, T_OUT, H, W, 3) with a zero pressure channel."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape[-1] == 3:
        return pred
    pad = np.zeros(pred.shape[:-1] + (3 - pred.shape[-1],), dtype=np.float32)
    return np.concatenate([pred, pad], axis=-1)
