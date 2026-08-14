"""Forecasting models: 20 input frames -> 20 output frames on a 32x64 grid.

Design follows three measurements rather than taste.

RESIDUAL PARAMETERISATION
    99.2% of a target window's L2 energy is its steady temporal mean, which
    barely moves in the 0.4 s we forecast; only 0.8% is the fluctuation. Asking a
    network to reproduce the whole field wastes its capacity on the part that
    copying the last input frame already gets right. So the network predicts a
    per-frame delta and the last input frame is added back. What it has to learn
    is exactly the part tke_score measures.

DIRECT, NOT AUTOREGRESSIVE
    All 20 output frames come out of one forward pass. Rolling a one-step model
    forward 20 times accumulates error and costs 20x the time, and time_score is
    100/(1+sqrt(t/0.72896)) — a millisecond-scale model scores in the mid nineties.

TIME AS CHANNELS
    (B, 20, 32, 64, 2) is folded to (B, 40, 32, 64). At this grid size a small
    U-Net sees the whole field within its receptive field, and 2D convolutions
    are far cheaper than any attention over 2048 points.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Imported both as part of the `realpde` package and as a flat `model_def.py`
# vendored beside submission.py, where there is no package to be relative to.
try:  # package context
    from .advection import (advect_sequence, coordinate_channels,
                            derived_channels)
except ImportError:  # vendored flat inside a submission archive
    from advection import (advect_sequence, coordinate_channels,
                           derived_channels)


def fold_time(x: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, T*C, H, W)."""
    b, t, h, w, c = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)


def unfold_time(x: torch.Tensor, t: int, c: int) -> torch.Tensor:
    """(B, T*C, H, W) -> (B, T, H, W, C)."""
    b, _, h, w = x.shape
    return x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and GELU."""

    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, c_out)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(g1, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(g1, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetForecaster(nn.Module):
    """Compact U-Net over a 32x64 grid, folded time in the channel axis.

    Three downsamples take 32x64 -> 4x8, which puts the whole field inside the
    receptive field of the bottleneck.
    """

    def __init__(
        self,
        t_in: int = 20,
        t_out: int = 20,
        channels: int = 2,
        base: int = 64,
        residual: bool = True,
    ):
        super().__init__()
        self.t_in, self.t_out, self.channels = t_in, t_out, channels
        self.residual = residual

        c_in = t_in * channels
        c_out = t_out * channels
        w = [base, int(base * 1.5), base * 2, int(base * 2.5)]

        self.enc1 = ConvBlock(c_in, w[0])
        self.enc2 = ConvBlock(w[0], w[1])
        self.enc3 = ConvBlock(w[1], w[2])
        self.bottleneck = ConvBlock(w[2], w[3])

        self.dec3 = ConvBlock(w[3] + w[2], w[2])
        self.dec2 = ConvBlock(w[2] + w[1], w[1])
        self.dec1 = ConvBlock(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], c_out, 1)

        # Start near the identity: with a zero-initialised head the model begins
        # as exact persistence, which already scores ~93 on rel_l2. Training then
        # only has to improve on that rather than first rediscover it.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, t_in, H, W, C) -> (B, t_out, H, W, C)."""
        last = x[:, -1:]                              # (B, 1, H, W, C)
        h = fold_time(x)

        e1 = self.enc1(h)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))

        out = unfold_time(self.head(d1), self.t_out, self.channels)
        return last + out if self.residual else out


class AdvectiveUNet(nn.Module):
    """UNetForecaster with a semi-Lagrangian advection prior as its baseline.

    UNetForecaster predicts a residual on top of the last input frame, i.e. on
    top of persistence. That baseline is wrong by construction at high Reynolds
    number, where the flow crosses half the field during the forecast horizon.
    Here the baseline is instead the field transported along its own velocity,
    so the residual the network must learn is what advection alone cannot
    explain — dissipation, pressure coupling, and the growth of new structure.

    The network additionally receives, as input channels:
      * the advected estimate for every output frame,
      * speed, vorticity and in-plane divergence of the last input frame,
      * absolute x and y coordinates, since the airfoil never moves and
        convolutions cannot otherwise know where they are.

    Everything runs in normalized space except the advection itself, which needs
    metres per second to convert a velocity into a displacement in cells.
    """

    def __init__(
        self,
        t_in: int = 20,
        t_out: int = 20,
        channels: int = 2,
        base: int = 64,
        substeps: int = 2,
        mean=(0.1550, -0.0005),
        std=(0.0968, 0.0160),
    ):
        super().__init__()
        self.t_in, self.t_out, self.channels = t_in, t_out, channels
        self.substeps = substeps
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        c_in = t_in * channels + t_out * channels + 3 + 2
        c_out = t_out * channels
        w = [base, int(base * 1.5), base * 2, int(base * 2.5)]

        self.enc1 = ConvBlock(c_in, w[0])
        self.enc2 = ConvBlock(w[0], w[1])
        self.enc3 = ConvBlock(w[1], w[2])
        self.bottleneck = ConvBlock(w[2], w[3])
        self.dec3 = ConvBlock(w[3] + w[2], w[2])
        self.dec2 = ConvBlock(w[2] + w[1], w[1])
        self.dec1 = ConvBlock(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], c_out, 1)

        # Begin as pure advection: a zero head means the first forward pass
        # returns the physics prior untouched, and training improves on it.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _advected(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Advected estimate for each output frame, in normalized space.

        Self-advection: the field is transported along its own velocity, which is
        re-read at every step, so vortices rotate the pattern without a rotation
        parameter. It is inherently sequential and therefore slow — measured at
        about 11.5 ms per sample, which costs roughly 7 points of time_score.

        `advect_frozen` is a single batched warp and much faster, but as a network
        input it is a markedly weaker signal: swapping it in dropped the selection
        score from 89.3 to 85.4 at the same epoch, because pure translation
        carries none of the rotation. Accuracy is the current priority, so the
        slow, stronger signal stays.
        """
        phys = x_norm * self.std + self.mean          # (B, T, H, W, C) m/s
        u0, v0 = phys[:, -1, ..., 0], phys[:, -1, ..., 1]
        adv = advect_sequence(u0, v0, self.t_out, substeps=self.substeps)
        return (adv - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w_, c = x.shape
        adv = self._advected(x)                        # (B, t_out, H, W, C)

        phys_last = x[:, -1] * self.std + self.mean
        der = derived_channels(phys_last[..., 0], phys_last[..., 1])
        # Scale the derived fields to roughly unit variance so they do not
        # dominate the first convolution: vorticity is O(10) in 1/s.
        der = der / torch.tensor([0.1, 10.0, 10.0], device=der.device)
        coords = coordinate_channels(b, h, w_, x.device, x.dtype)

        feats = torch.cat([
            fold_time(x),
            fold_time(adv),
            der.permute(0, 3, 1, 2),
            coords.permute(0, 3, 1, 2),
        ], dim=1)

        e1 = self.enc1(feats)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        bo = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(bo, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))

        return adv + unfold_time(self.head(d1), self.t_out, self.channels)


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    if name == "unet":
        return UNetForecaster(**kwargs)
    if name == "advective":
        kwargs.pop("residual", None)
        return AdvectiveUNet(**kwargs)
    raise ValueError(f"unknown model {name!r}")


def count_parameters(model: nn.Module) -> tuple[int, float]:
    n = sum(p.numel() for p in model.parameters())
    return n, n * 4 / 1024**2  # fp32 megabytes


if __name__ == "__main__":
    m = UNetForecaster()
    n, mb = count_parameters(m)
    x = torch.randn(2, 20, 32, 64, 2)
    y = m(x)
    print(f"UNetForecaster: {n:,} parameters, {mb:.1f} MB fp32")
    print(f"  {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  zero-init head => output equals persistence at step 0: "
          f"{torch.allclose(y, x[:, -1:].expand_as(y))}")
