"""Losses that target the subscores we can actually move.

Measured headroom on the local validation split:

    rel_l2   94.0 -> 100   (6 points)
    mvpe     94.4 -> 100   (6 points)
    tke      66.7 -> 100   (33 points)

Plain MSE optimises the conditional mean of the output. A conditional mean over a
chaotic wake is smooth, and a smooth prediction has no temporal variance, so its
TKE error is exactly 1.0 — the same score as predicting zeros. Both trivial
baselines land there. Optimising MSE alone therefore buys a few points on an
already-saturated metric while leaving the 33-point one at the floor.

So the loss carries an explicit TKE term, written to match scoring.py's
definition so that improving it improves the leaderboard number rather than a
proxy for it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tke_map(field: torch.Tensor) -> torch.Tensor:
    """0.5 * (var_t(u) + var_t(v)) over the time axis.

    scoring.py's kinetic_energy(): the biased (population) variance across the
    20 frames, per grid cell, averaged over the u and v components.

    (B, T, H, W, C) -> (B, H, W)
    """
    u, v = field[..., 0], field[..., 1]
    du = u - u.mean(dim=1, keepdim=True)
    dv = v - v.mean(dim=1, keepdim=True)
    return 0.5 * ((du ** 2).mean(dim=1) + (dv ** 2).mean(dim=1))


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample ||pred - target|| / ||target||, as the scorer computes it."""
    b = pred.shape[0]
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    return (p - t).norm(dim=1) / t.norm(dim=1).clamp(min=eps)


def tke_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 between the predicted and true TKE maps — the scored quantity."""
    return relative_l2(tke_map(pred), tke_map(target)).mean()


# scoring.py's MVPE probes on the 32x64 grid, hard-coded here so the loss can
# weight them. Derived in scripts/probe_mask_check.py; 36 points, 1.8% of the field.
MVPE_PROBE_Y = (8, 10, 12, 14, 16, 18, 20, 22, 24)
MVPE_PROBE_X = (13, 21, 29, 37)


def mvpe_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 of the time-averaged velocity at the 36 scored probe points."""
    ys = torch.as_tensor(MVPE_PROBE_Y, device=pred.device)
    xs = torch.as_tensor(MVPE_PROBE_X, device=pred.device)
    p = pred[:, :, ys][:, :, :, xs].mean(dim=1)      # (B, 9, 4, C)
    t = target[:, :, ys][:, :, :, xs].mean(dim=1)
    return relative_l2(p, t).mean()


def sigma_weights(sigma: torch.Tensor, power: float = 1.3,
                  clip: float = 8.0) -> torch.Tensor:
    """Per-element weights for the squared error, from a residual sigma map.

    WHY THE SQUARED ERROR SHOULD NOT BE WEIGHTED EQUALLY

    sps_score is 100 * acc * Q, and Q is a per-element mean of exp(-nil) when
    the target is covered. Measured on the validation split (scripts/q_deficit.py),
    shrinking the residual by 20% in one region at a time and refitting sigma
    gives, per unit of squared error removed:

        freestream   6.47        (sigma 0.0034)
        middle       1.76        (sigma 0.0075)
        wake         1.00        (sigma 0.0137)

    Accurate cells pay roughly six times what wake cells pay. The reason is in
    the metric: where sigma is small the band is already tight and nearly all
    of the loss is coverage, so a little more accuracy flips many elements from
    zero to almost one. Where sigma is large the optimal band is one that
    mostly misses, and a 20% improvement does not rescue it.

    Plain MSE does the opposite -- it spends its gradient where the errors are
    biggest, which is the wake. Weighting by sigma**-power tilts it back;
    power = 1.3 reproduces the measured ratios (6.08 : 2.19 : 1), and it is
    measured rather than chosen. power = 0 is plain MSE and power = 2 fully
    standardises the residual, which overshoots.

    Weights are normalised to mean one so the loss keeps its scale, and clipped
    because the map's smallest entries are 1e-4 -- a few masked or near-static
    cells would otherwise take over the gradient.
    """
    w = (sigma.clamp_min(1e-5) / sigma.mean()) ** (-power)
    w = w.clamp(max=clip)
    return w / w.mean()


class CompositeLoss(torch.nn.Module):
    """w_mse * MSE + w_tke * TKE-relative-L2 + w_mvpe * MVPE-relative-L2.

    Weights are deliberately explicit rather than tuned in advance: the MSE and
    TKE terms pull against each other (sharpening a prediction raises its MSE),
    and where the balance sits is an empirical question for the first sweep.

    `weight_map`, if given, replaces the plain MSE with a per-element weighted
    one -- see sigma_weights for what it is and why.
    """

    def __init__(self, w_mse: float = 1.0, w_tke: float = 0.0, w_mvpe: float = 0.0,
                 weight_map: torch.Tensor | None = None):
        super().__init__()
        self.w_mse, self.w_tke, self.w_mvpe = w_mse, w_tke, w_mvpe
        if weight_map is None:
            self.weight_map = None
        else:
            self.register_buffer("weight_map", weight_map)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.weight_map is None:
            mse = F.mse_loss(pred, target)
        else:
            # Masked cells read exactly zero in both components and are not
            # scored. They also sit at the bottom of the sigma map -- 2.7% of
            # the grid at sigma 0.0009 against 0.0083 elsewhere -- so weighting
            # by sigma**-power would hand them the largest weight in the loss,
            # which is precisely backwards. They cannot be excluded from a fixed
            # map either, because which cells are masked moves with the angle of
            # attack. So they are found per sample, from the target, and held at
            # weight one: the model still learns to output zero there, but no
            # gradient is spent making it more zero.
            valid = (target != 0).any(dim=-1, keepdim=True)
            w = torch.where(valid, self.weight_map, torch.ones_like(self.weight_map))
            mse = (w * (pred - target) ** 2).mean() / w.mean()
        parts = {"mse": mse.detach()}
        total = self.w_mse * mse

        if self.w_tke > 0:
            lt = tke_loss(pred, target)
            total = total + self.w_tke * lt
            parts["tke"] = lt.detach()
        if self.w_mvpe > 0:
            lm = mvpe_loss(pred, target)
            total = total + self.w_mvpe * lm
            parts["mvpe"] = lm.detach()

        parts["total"] = total.detach()
        # Reported for monitoring only; not optimised directly.
        with torch.no_grad():
            parts["rel_l2"] = relative_l2(pred, target).mean()
        return total, parts
