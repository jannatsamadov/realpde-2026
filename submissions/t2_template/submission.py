"""RealPDE Track 2 (LTTTA) submission: the Track 1 advective U-Net, adapted online.

Runs inside w3nhao/realpde-track2 (PyTorch 2.2.2, CUDA 12.1, Python 3.10) with no
network and nothing installed at evaluation time. Everything it needs is in this
archive.

WHAT TRACK 2 ACTUALLY ADDS

The stream has stride 20, so window k's target frames ARE window k+1's input
frames. `prev_target_norm` therefore carries no information the current input
does not already contain — it cannot improve the current prediction directly.

What it does give is a genuine supervised pair (prev_input -> prev_target) at
test time, which Track 1 never had. That matters here specifically: the regime
(Reynolds number, angle of attack) is never handed to the model, and measured
performance varies strongly across regimes (tke 67.9 to 86.4). Online adaptation
is the only mechanism available for specialising to the trajectory being scored.

WHY ADAPTATION IS NEARLY FREE HERE

A forward pass is dominated by the semi-Lagrangian advection prior: a 40-step
grid_sample loop, about 11.5 ms, against a sub-millisecond U-Net. But the prior
depends on no learnable parameter, so the adaptation step does not need to
recompute it — `features()` output for the previous window is cached from the
step that already predicted it. Adaptation therefore costs one U-Net forward and
backward, not a second full forward.

Timing is charged for every ttt_step and every reset_ttt_state, so this matters:
time_score is 100/(1+sqrt(t/0.72896)), which is steep at our scale.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from model_def import build_model

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(HERE, "model.pth")
SIGMA_FILE = os.path.join(HERE, "sps_sigma.npz")

T_OUT = 20

# --------------------------------------------------------------------------- #
# Normalization.
#
# The evaluator normalizes with the official train_real statistics; the model was
# trained against those same numbers rounded to four decimals and carries the
# rounded pair as buffers, which it uses to turn normalized values into m/s for
# the advection. The gap is small (2.5e-3 relative on v) but it is free to remove
# exactly, so the two spaces are bridged rather than assumed equal.
#
# These constants are confirmed, not guessed: mean(STD_TGT[:2]) is
# 0.056387025..., exactly the SIGMA_GLOBAL the scorer divides interval widths by
# and which the metrics page defines as the mean of the u, v standard deviations
# on the official train_real split. Season-constant.
# --------------------------------------------------------------------------- #
MEAN_IN = np.array([0.15496085584163666, -0.0005139928543940187, 0.0], dtype=np.float64)
STD_IN = np.array([0.09680565446615219, 0.015960684046149254, 1.0], dtype=np.float64)
MEAN_TGT = np.array([0.15496256947517395, -0.0005177936982363462, 0.0], dtype=np.float64)
STD_TGT = np.array([0.09681040793657303, 0.015963643789291382, 1.0], dtype=np.float64)

# What the checkpoint's own buffers use.
MEAN_OURS = np.array([0.1550, -0.0005], dtype=np.float64)
STD_OURS = np.array([0.0968, 0.0160], dtype=np.float64)

SIGMA_GLOBAL = 0.0563870259

# --------------------------------------------------------------------------- #
# Policy knobs. Every one of these is measured in scripts/t2_eval.py before it is
# changed here; the defaults are the configuration that scored best there.
# --------------------------------------------------------------------------- #

# TEST-TIME WEIGHT ADAPTATION IS OFF, AND THAT IS A MEASUREMENT.
#
# Swept on the real validation stream, adapting the head and adapting everything,
# at learning rates from 1e-4 to 10:
#
#   lr <= 0.01   weights move < 1e-4 relative; every subscore unchanged
#   lr = 0.1     selection 89.684 -> 89.678 (head), 89.634 (all)  — worse
#   lr >= 1      diverges, every subscore 0
#
# There is no window where it helps: it goes from doing nothing straight to doing
# harm, while costing about 14 ms a step. The likely reason is that there is no
# distribution shift to correct — same rig, same preprocessing, only an unseen
# Reynolds number, and the model already infers the regime from the input window
# (measured corr 0.969 between true and predicted TKE). A single-window gradient
# is then just a noisy sample of the objective it was already trained on.
#
# Kept as knobs rather than deleted so the measurement can be re-run.
ADAPT_STEPS = 0
ADAPT_LR = 3e-4
# "head" : the 1x1 output convolution only. "all": every parameter. "none": off.
ADAPT_SCOPE = "none"

# Interval bounds for sps_score. On Track 1 these took sps from 16.6 to 36.2 on
# the leaderboard — by far the largest single gain in the project — so they are
# on by default. 0.8 is the empirical best width multiplier on validation.
RETURN_BOUNDS = True
BOUND_SCALE = 0.8

# Online recalibration of the interval width. This is the one thing Track 2 can
# do that Track 1 structurally cannot.
#
# Track 1 had to size intervals from residuals measured on the TRAINING split,
# because a scored call never sees a target. That mis-sized them badly: the same
# bands covered 87.6% of elements locally and only 69.8% on the hidden set, so
# the residuals there were about 1.49x larger than the map assumed. Widening by
# a factor tuned on our own split did not transfer (+0.16 on the leaderboard
# against +1.9 predicted) — because it was still guessing.
#
# Here the previous window's target arrives one step later, so the model can
# measure its OWN residual on the trajectory being scored and rescale from that.
# No tuning, no transfer assumption: it observes the answer.
#
#   "off"    : the fixed training-fitted map, i.e. the Track 1 behaviour.
#   "scalar" : one multiplier on the whole map, estimated online. One parameter,
#              so it stabilises within a few steps, and it is exactly the shape
#              of the error the leaderboard revealed.
SIGMA_ONLINE = "scalar"

# Where the interval width comes from before any online rescaling.
#
#   "gaussian"  : solve phi(r)/(2Phi(r)-1) = s/SIGMA_GLOBAL for a fitted per-
#                 element sigma, then multiply by BOUND_SCALE. Two approximations
#                 stacked: the residuals are not Gaussian, and BOUND_SCALE = 0.8
#                 is the correction for that, tuned back on validation.
#   "empirical" : the objective exp(-w/SIGMA_GLOBAL) * P(|e| <= w/2) maximised
#                 directly over the EMPIRICAL residual distribution, from ~1200
#                 training residuals per element. No distribution assumed and no
#                 fudge factor — measured on validation, the best multiplier on
#                 it is 1.0, which is what dropping the wrong assumption should
#                 look like. sps 49.63 against the Gaussian route's 48.74.
#
# The empirical file stores residual QUANTILES per element, not just the chosen
# width, because the online rescaling has to re-maximise the objective: if the
# residuals are a times larger, the optimal width is not a times wider —
# exp(-w/SIGMA_GLOBAL) has a fixed scale and the coverage term does not.
SIGMA_SOURCE = "empirical"
# Pseudo-count shrinking the online estimate toward the prior, so the first
# window of a trajectory is not calibrated off a single noisy sample.
SIGMA_PRIOR_COUNT = 4.0
# TEST KNOB, 1.0 in any real submission. Multiplies the loaded prior to simulate
# a mis-fitted sigma map. Our own validation residuals match our own prior by
# construction, so the online estimate has nothing to correct there and the
# mechanism cannot be told apart from the fixed map; the leaderboard revealed a
# prior that was too NARROW by about 1.49x. Setting this to 1/1.49 reproduces
# that failure locally, which is the only way to test whether the online
# estimate actually repairs it.
SIGMA_PRIOR_SCALE = 1.0


def _erf(x):
    """Vectorised erf (Abramowitz & Stegun 7.1.26, max error 1.5e-7).

    numpy has no erf and scipy is not in the evaluation image; math.erf is
    scalar-only. Used once at construction, which is not timed.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
        1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return np.sign(x) * (1.0 - poly * np.exp(-ax * ax))


def _optimal_half_width_ratio(k):
    """Solve phi(r) / (2 Phi(r) - 1) = k for r by interpolation on a fixed grid.

    Maximising exp(-w/SIGMA_GLOBAL) * P(|residual| <= w/2) for a Gaussian
    residual of standard deviation s gives that condition with k = s/SIGMA_GLOBAL,
    and the optimal total width is w = 2*s*r.
    """
    r = np.geomspace(1e-4, 40.0, 4000)
    phi = np.exp(-0.5 * r * r) / np.sqrt(2.0 * np.pi)
    ratio = phi / np.clip(_erf(r / np.sqrt(2.0)), 1e-12, None)
    return np.interp(np.asarray(k, dtype=np.float64), ratio[::-1], r[::-1])


class AdvectiveTTT:
    """The Track 1 forecaster wrapped in the LTTTA streaming contract."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 sigma_file: str, empirical_file: str = ""):
        self.model = model
        self.device = device
        self.sigma_file = sigma_file
        self.empirical_file = empirical_file

        # Normalization bridge, precomputed as tensors so a step is two affine
        # ops rather than any numpy work.
        def t(a):
            return torch.tensor(a, dtype=torch.float32, device=device)

        self.in_scale = t(STD_IN[:2] / STD_OURS)
        self.in_shift = t((MEAN_IN[:2] - MEAN_OURS) / STD_OURS)
        self.out_scale = t(STD_OURS / STD_TGT[:2])
        self.out_shift = t((MEAN_OURS - MEAN_TGT[:2]) / STD_TGT[:2])

        # Weights to restore at every trajectory boundary. Kept on the device so
        # the reset — which IS timed — is a device-to-device copy.
        self._init_state = {k: v.detach().clone()
                            for k, v in model.state_dict().items()}

        self._params = self._select_params()
        self._opt = self._make_opt()

        self._prev_feats: Optional[torch.Tensor] = None
        self._prev_adv: Optional[torch.Tensor] = None

        self._half: Optional[torch.Tensor] = None
        self._half_table: Optional[torch.Tensor] = None
        self._alpha_grid: Optional[torch.Tensor] = None
        self._sigma_norm: Optional[torch.Tensor] = None
        self._quant: Optional[np.ndarray] = None
        self._levels: Optional[np.ndarray] = None
        if RETURN_BOUNDS:
            if SIGMA_SOURCE == "empirical":
                self._half = self._load_empirical()
            else:
                self._half = self._precompute_half_widths()
            if SIGMA_ONLINE == "scalar":
                self._build_alpha_table()

        # Online interval calibration state, reset per trajectory.
        self._prev_pred: Optional[torch.Tensor] = None
        self._sum_r2 = 0.0
        self._sum_s2 = 0.0
        self._n_obs = 0

    # ---------------- construction helpers (not timed) ---------------- #

    def _select_params(self):
        if ADAPT_SCOPE == "none" or ADAPT_STEPS <= 0:
            return []
        if ADAPT_SCOPE == "head":
            return list(self.model.head.parameters())
        return list(self.model.parameters())

    def _make_opt(self):
        if not self._params:
            return None
        # Plain SGD: no optimizer state, so a trajectory reset costs nothing and
        # cannot leak adaptation across trajectory boundaries.
        return torch.optim.SGD(self._params, lr=ADAPT_LR)

    def _precompute_half_widths(self) -> torch.Tensor:
        """Interval half-widths, in the evaluator's normalized space.

        sigma is fixed for the season and the width solve depends only on sigma,
        so the whole map is solved once here — construction is not timed — and a
        scored step is then a single add and subtract. Solving it per step would
        run a 4000-point interpolation over the prediction array every call.
        """
        sigma_ms = np.load(self.sigma_file)["sigma"].astype(np.float64)  # (1,T,H,W,2) m/s
        s = np.clip(sigma_ms[..., :2] * SIGMA_PRIOR_SCALE, 1e-12, None)
        r = _optimal_half_width_ratio(s / SIGMA_GLOBAL)
        half_ms = 0.5 * BOUND_SCALE * 2.0 * s * r
        # Bounds live in the same normalized space as the prediction, and
        # denormalization is affine, so a width in m/s divides by the std.
        half_norm = half_ms / STD_TGT[:2]
        # Keep sigma itself in the same space: the online calibration rescales
        # sigma and re-solves the width, which is not the same as rescaling the
        # width — optimal_width is concave, so a small sigma needs proportionally
        # more widening than a large one.
        self._sigma_norm = torch.tensor(s / STD_TGT[:2], dtype=torch.float32,
                                        device=self.device)
        return torch.tensor(half_norm, dtype=torch.float32, device=self.device)

    def _load_empirical(self) -> torch.Tensor:
        """Half-widths read off the empirical residual quantiles, at alpha = 1."""
        z = np.load(self.empirical_file)
        self._quant = z["quantiles"].astype(np.float64) * SIGMA_PRIOR_SCALE
        self._levels = z["levels"].astype(np.float64)
        # A per-element scale for the online estimator. The level nearest 0.6827
        # is the empirical stand-in for one standard deviation, which is all the
        # alpha estimate needs — it is a ratio, so the constant cancels.
        j = int(np.argmin(np.abs(self._levels - 0.6827)))
        self._sigma_norm = torch.tensor(self._quant[j][None] / STD_TGT[:2],
                                        dtype=torch.float32, device=self.device)
        half_ms = self._empirical_half_at(1.0)
        return torch.tensor(half_ms / STD_TGT[:2], dtype=torch.float32,
                            device=self.device)

    def _empirical_half_at(self, alpha: float) -> np.ndarray:
        """Maximise exp(-2*alpha*q/SIGMA_GLOBAL) * level over the stored levels.

        The empirical CDF is a step function, so the objective only rises at a
        jump: the maximiser is one of the tabulated quantiles, and this is an
        exact argmax rather than an interpolation.
        """
        q = alpha * self._quant
        obj = np.exp(-2.0 * q / SIGMA_GLOBAL) * self._levels[:, None, None, None, None]
        return np.take_along_axis(q, obj.argmax(axis=0)[None], axis=0)[0]

    def _build_alpha_table(self) -> None:
        """Half-width maps for a grid of sigma multipliers, solved up front.

        The width solve is a 4000-point interpolation over every element; run
        per step it would cost more than the forward pass it is protecting. The
        online estimate only ever picks a single multiplier, so the whole family
        is tabulated here (48 x 20 x 32 x 64 x 2 floats, about 16 MB) and a step
        becomes one index.
        """
        sig = self._sigma_norm.detach().cpu().numpy().astype(np.float64)
        grid = np.geomspace(0.25, 6.0, 48)
        table = np.empty((len(grid),) + sig.shape[1:], dtype=np.float32)
        for i, a in enumerate(grid):
            if self._quant is not None:
                table[i] = self._empirical_half_at(float(a)) / STD_TGT[:2]
                continue
            s_ms = a * sig[0] * STD_TGT[:2]          # back to m/s for the solve
            r = _optimal_half_width_ratio(np.clip(s_ms, 1e-12, None) / SIGMA_GLOBAL)
            table[i] = (0.5 * BOUND_SCALE * 2.0 * s_ms * r) / STD_TGT[:2]
        self._alpha_grid = torch.tensor(grid, dtype=torch.float32, device=self.device)
        self._half_table = torch.tensor(table, dtype=torch.float32, device=self.device)

    def _current_half(self) -> torch.Tensor:
        """Interval half-widths for this step, after any online rescaling."""
        if self._half_table is None or self._sum_s2 <= 0.0:
            return self._half
        # alpha^2 = observed residual energy / energy the prior predicted,
        # shrunk toward 1 by a pseudo-count so a single window cannot swing it.
        a2_hat = self._sum_r2 / self._sum_s2
        n = self._n_obs
        a2 = (SIGMA_PRIOR_COUNT + n * a2_hat) / (SIGMA_PRIOR_COUNT + n)
        alpha = float(max(a2, 1e-8) ** 0.5)
        idx = int(torch.argmin((self._alpha_grid - alpha).abs()))
        return self._half_table[idx]

    # ---------------- the timed interface ---------------- #

    def reset_ttt_state(self) -> None:
        """New trajectory: forget the adaptation and the cached window."""
        self.model.load_state_dict(self._init_state)
        self._opt = self._make_opt()
        self._prev_feats = None
        self._prev_adv = None
        # Calibration is per trajectory: a new trajectory can be a different
        # Reynolds number, where the residual scale is different.
        self._prev_pred = None
        self._sum_r2 = 0.0
        self._sum_s2 = 0.0
        self._n_obs = 0

    def ttt_step(
        self,
        input_norm: torch.Tensor,
        prev_target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = torch.as_tensor(input_norm).to(self.device, dtype=torch.float32)

        # (1) Adapt on the previous pair, reusing that window's cached features.
        #     The advection prior inside them depends on no parameter, so this is
        #     the same computation the previous step already paid for.
        adapt_loss: Optional[float] = None
        y_prev_eval = None
        if prev_target_norm is not None:
            y_prev_eval = torch.as_tensor(prev_target_norm).to(self.device,
                                                              dtype=torch.float32)

        # (0) Calibrate the interval width on the residual just revealed. This
        #     needs no gradient and no extra forward pass — the prediction for
        #     that window was returned last step and only has to be remembered.
        if (self._half_table is not None and y_prev_eval is not None
                and self._prev_pred is not None):
            with torch.no_grad():
                tgt = y_prev_eval[..., :2]
                # Elements outside the PIV field of view and inside the airfoil
                # are zero in the target and are not scored; including them
                # would drag the estimate toward zero.
                mask = (tgt != 0.0).to(tgt.dtype)
                r = (self._prev_pred - tgt) * mask
                self._sum_r2 += float((r * r).sum())
                self._sum_s2 += float(((self._sigma_norm * mask) ** 2).sum())
                self._n_obs += 1

        if (self._opt is not None and y_prev_eval is not None
                and self._prev_feats is not None):
            y_prev = self._to_ours(y_prev_eval[..., :2])
            self.model.train()
            for _ in range(ADAPT_STEPS):
                self._opt.zero_grad(set_to_none=True)
                pred_prev = self.model.forward_from_features(self._prev_feats,
                                                             self._prev_adv)
                loss = torch.mean((pred_prev - y_prev) ** 2)
                loss.backward()
                self._opt.step()
                adapt_loss = float(loss.detach())

        # (2) Predict the current window.
        self.model.eval()
        xo = self._to_ours(x[..., :2])
        with torch.no_grad():
            feats, adv = self.model.features(xo)
            pred_ours = self.model.forward_from_features(feats, adv)

        # (3) Cache this window's parameter-free half for the next adaptation.
        self._prev_feats = feats.detach()
        self._prev_adv = adv.detach()

        pred = torch.zeros(x.shape[0], T_OUT, x.shape[2], x.shape[3], 3,
                           dtype=torch.float32, device=self.device)
        pred[..., :2] = pred_ours * self.out_scale + self.out_shift
        # A non-finite value zeroes every subscore, so never let one escape.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Remember this window's prediction: its target is revealed next step,
        # and the pair is what calibrates the interval width.
        self._prev_pred = pred[..., :2].detach()

        info: Dict[str, Any] = {"adapt_loss": adapt_loss}
        if self._half is not None:
            half = self._current_half()
            lower = pred.clone()
            upper = pred.clone()
            lower[..., :2] -= half
            upper[..., :2] += half
            # Pressure keeps a zero-width interval: it is never scored, and a
            # width there would only cost exp(-width/sigma) for nothing.
            info["lower"] = lower
            info["upper"] = upper
        return pred, info

    # ---------------- normalization bridge ---------------- #

    def _to_ours(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluator's normalized space -> the checkpoint's own."""
        return x * self.in_scale + self.in_shift


def get_ttt_model(submission_dir: str, device: str):
    """Entry point, called once before the stream. Not timed."""
    ckpt = os.path.join(submission_dir, "model.pth")
    sigma = os.path.join(submission_dir, "sps_sigma.npz")
    empirical = os.path.join(submission_dir, "sps_empirical.npz")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    state = torch.load(ckpt, map_location="cpu")
    cfg = state.get("config", {})
    # The architecture travels with the weights: hardcoding one here silently
    # loads the wrong class and fails on a shape mismatch.
    model = build_model(
        cfg.get("arch", "advective"),
        t_in=cfg.get("t_in", 20),
        t_out=cfg.get("t_out", T_OUT),
        channels=cfg.get("channels", 2),
        base=cfg.get("base", 64),
    )
    model.load_state_dict(state["model"])
    model.to(dev)
    return AdvectiveTTT(model, dev, sigma, empirical)
