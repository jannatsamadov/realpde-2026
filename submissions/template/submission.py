"""RealPDE Track 1 submission: predict(input_array, metadata) -> (N, 20, 32, 64, 3).

Runs inside pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime with no network and
nothing installed at evaluation time. Everything it needs is in this archive.

Constraints this file is written against, each one a documented failure mode:

  * NO h5py, scipy, pandas, sklearn, einops. They are not in the image. Only
    torch 2.2.2, numpy 1.26, and the standard library.
  * `metadata` is an empty dict on scored calls, so Reynolds number and angle of
    attack are NOT available. Anything the model needs about the regime it has
    to read out of the input window itself.
  * `predict` may be called more than once, each call in a fresh subprocess.
    Nothing carries between calls, so the model is loaded lazily and cached only
    within one call's process.
  * Bounds are all-or-nothing across the whole run. Either every call returns
    lower/upper or none does; mixing them makes the scorer discard all of them.
  * Non-finite outputs, or a shape other than (N, 20, 32, 64, 3), score zero on
    every subscore. Both are guarded at the end.
  * DataLoader, if ever used here, must have num_workers=0.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from model_def import UNetForecaster

# Official normalization statistics for the real split, from the Track 2 starting
# kit's example_data/mean_std_real.pt. The std entries are the two numbers that
# scoring.py averages into SIGMA_GLOBAL, which is how they were confirmed.
MEAN = np.array([0.1550, -0.0005, 0.0], dtype=np.float32)
STD = np.array([0.0968, 0.0160, 1.0], dtype=np.float32)  # 1.0 on p: never divide by zero

T_OUT = 20
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "model.pth")

# Set True only when the model also produces calibrated interval bounds.
RETURN_BOUNDS = False

_MODEL = None
_DEVICE = None


def _load_model():
    """Build the network and load weights. Cached for the life of this process."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location="cpu")
    cfg = state.get("config", {})
    model = UNetForecaster(
        t_in=cfg.get("t_in", 20),
        t_out=cfg.get("t_out", T_OUT),
        channels=cfg.get("channels", 2),
        base=cfg.get("base", 64),
        residual=cfg.get("residual", True),
    )
    model.load_state_dict(state["model"])
    model.eval().to(_DEVICE)
    _MODEL = model
    return _MODEL, _DEVICE


@torch.no_grad()
def predict(input_array, metadata=None):
    """input_array: (N, T_in, H, W, C) in m/s -> (N, 20, H, W, 3) in m/s."""
    x = np.asarray(input_array, dtype=np.float32)
    n, t_in, h, w, c = x.shape

    # Only u and v are measured or scored; pressure is unused on real data.
    xn = (x[..., :2] - MEAN[:2]) / STD[:2]

    model, device = _load_model()
    xt = torch.from_numpy(np.ascontiguousarray(xn)).to(device)

    # Chunked so a large evaluation batch cannot exhaust GPU memory; the loop is
    # cheap because the model is small.
    outs = []
    chunk = 64
    for i in range(0, n, chunk):
        outs.append(model(xt[i:i + chunk]).float().cpu())
    yn = torch.cat(outs).numpy()

    y = yn * STD[:2] + MEAN[:2]

    # Pad the pressure channel back with zeros: the scorer ignores it, but the
    # shape must be (N, 20, H, W, 3) exactly or the submission scores zero.
    pred = np.zeros((n, T_OUT, h, w, 3), dtype=np.float32)
    pred[..., :2] = y

    # A non-finite value zeroes every subscore, so never let one escape.
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    if not RETURN_BOUNDS:
        return pred

    lower, upper = _bounds(pred)
    return {"prediction": pred, "lower": lower, "upper": upper}


def _bounds(pred: np.ndarray):
    """Per-element interval bounds for sps_score.

    Placeholder: a fixed fraction of |pred|, i.e. the scorer's own default band.
    Replaced by calibrated widths once the bound model is trained; SPS is the
    largest single lever on the board and the default band scores about 17.
    """
    half = 0.05 * np.abs(pred)
    lower = (pred - half).astype(np.float32)
    upper = (pred + half).astype(np.float32)
    return lower, upper
