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

from model_def import build_model

# Official normalization statistics for the real split, from the Track 2 starting
# kit's example_data/mean_std_real.pt. The std entries are the two numbers that
# scoring.py averages into SIGMA_GLOBAL, which is how they were confirmed.
MEAN = np.array([0.1550, -0.0005, 0.0], dtype=np.float32)
STD = np.array([0.0968, 0.0160, 1.0], dtype=np.float32)  # 1.0 on p: never divide by zero

T_OUT = 20
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "model.pth")
SIGMA_FILE = os.path.join(HERE, "sps_sigma.npz")

# Set True only when sps_sigma.npz is bundled alongside this file. Bounds are
# all-or-nothing across the whole run: once one call returns them every later
# call must too, so this flag is fixed at build time and never toggled at runtime.
RETURN_BOUNDS = False

# Width multiplier on the analytic optimum, applied after the sigma solve.
# 0.8 is the empirical best on validation and is what submission adv_v2 used.
BOUND_SCALE = 0.8

# BOTH CORRECTIONS BELOW ARE OFF, AND THAT IS THE MEASURED RESULT — NOT AN OVERSIGHT.
#
# Sigma is fitted on our training residuals, and the hidden set's residuals looked
# about 1.49x larger — inferred from submission 2, where the same bands covered
# 87.6% locally and only 69.8% there. Likewise an MSE-trained model returns the
# conditional mean and keeps only ~55% of the true fluctuation, so inflating it by
# ALPHA = 1.15 raised tke from 76.32 to 77.51 locally.
#
# Both were shipped together as adv_v3. Local prediction: +1.9 and +1.2.
# Leaderboard: +0.16 and +0.17 — and adv_v3 finished BELOW adv_v2 overall
# (78.744 vs 78.766). Post-hoc corrections tuned on our own split do not transfer.
# Two independent attempts, same outcome. Turn these back on only with a new
# argument, not a new tuning.
SIGMA_INFLATE = 1.0
ALPHA = 1.0

_MODEL = None
_DEVICE = None
_SIGMA = None


def _load_model():
    """Build the network and load weights. Cached for the life of this process."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location="cpu")
    cfg = state.get("config", {})
    # The architecture is recorded in the checkpoint. Hardcoding one here silently
    # loads the wrong class and fails on a shape mismatch — which is what the
    # packaging check caught before this ever reached the platform.
    model = build_model(
        cfg.get("arch", "unet"),
        t_in=cfg.get("t_in", 20),
        t_out=cfg.get("t_out", T_OUT),
        channels=cfg.get("channels", 2),
        base=cfg.get("base", 64),
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

    # Amplitude correction about each window's own temporal mean.
    if ALPHA != 1.0:
        mean_t = y.mean(axis=1, keepdims=True)
        y = mean_t + ALPHA * (y - mean_t)

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


SIGMA_GLOBAL = 0.0563870259   # frozen for the season; scoring.py divides widths by it


def _erf(x):
    """Vectorised erf (Abramowitz & Stegun 7.1.26, max error 1.5e-7).

    numpy has no erf, scipy is not in the evaluation image, and math.erf is
    scalar-only — wrapping it in np.vectorize over a full prediction array is a
    Python loop that never finishes.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
        1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1.0 - poly * np.exp(-ax * ax))


def _optimal_half_width_ratio(k):
    """Solve phi(r) / (2 Phi(r) - 1) = k for r, by interpolation on a fixed grid.

    Maximising exp(-w/SIGMA_GLOBAL) * P(|residual| <= w/2) for a Gaussian
    residual of standard deviation s gives that condition with k = s/SIGMA_GLOBAL,
    and the optimal total width is w = 2*s*r. The root depends on nothing but k,
    so it is tabulated once and looked up.
    """
    r = np.geomspace(1e-4, 40.0, 4000)
    phi = np.exp(-0.5 * r * r) / np.sqrt(2.0 * np.pi)
    ratio = phi / np.clip(1.0 + _erf(r / np.sqrt(2.0)) - 1.0, 1e-12, None)
    return np.interp(np.asarray(k, dtype=np.float64), ratio[::-1], r[::-1])


def _load_sigma():
    """Per-element residual standard deviation, fitted on the training split."""
    global _SIGMA
    if _SIGMA is None:
        _SIGMA = np.load(SIGMA_FILE)["sigma"].astype(np.float32)   # (1, T, H, W, 2)
    return _SIGMA


def _bounds(pred: np.ndarray):
    """Per-element interval bounds for sps_score.

    The scorer's fallback band is 0.05*|pred| — a fraction of the prediction's
    magnitude, unrelated to how wrong the prediction actually is. It is therefore
    narrowest exactly in the wake, where the velocity is small and the error is
    largest: 60% of scored elements fall outside it and score zero, for an
    sps_score near 23.

    These bounds instead use the model's own measured error per (frame, cell,
    channel) and the width that maximises coverage * exp(-width/SIGMA_GLOBAL).
    Measured on validation: 48.74 against 22.96 for the default.
    """
    # The half-width depends on sigma alone, and sigma is one (1, T, H, W, 2)
    # map shared by every window. Solving it on the map and letting numpy
    # broadcast gives the same numbers for 1/N of the work: broadcasting first
    # ran the interpolation over N x 20 x 32 x 64 x 2 points -- 22 million on our
    # validation split -- in float64, which was seconds, while the forward pass
    # it was attached to takes 0.13 s. time_score is per sample, so that came
    # straight off the score.
    sigma = _load_sigma()[..., :2] * SIGMA_INFLATE
    s = np.clip(sigma, 1e-12, None)
    r = _optimal_half_width_ratio(s / SIGMA_GLOBAL)
    half = (0.5 * BOUND_SCALE * 2.0 * s * r).astype(np.float32)

    lower = pred.copy()
    upper = pred.copy()
    lower[..., :2] -= half
    upper[..., :2] += half
    # Pressure is unmeasured, so its target is zero and it is never scored; a
    # zero-width interval there is correct and costs nothing.
    return lower.astype(np.float32), upper.astype(np.float32)
