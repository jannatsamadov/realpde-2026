"""Flow-matching generator for the fluctuation only.

WHY ONLY THE FLUCTUATION

99.2% of a target window's L2 energy is its steady temporal mean, and the
deterministic model already predicts that well — relative L2 error 0.078, with
rel_l2_score sitting at 96 where the ceiling is 100. Every point still missing on
tke_score lives in the other 0.8%: the fluctuation about that mean, which an
MSE-trained model shrinks to about 55% of its true amplitude because the
conditional mean of a chaotic wake is smooth.

So the generator is given the hard 0.8% and nothing else:

    prediction = temporal_mean(deterministic) + sampled_fluctuation

The deterministic mean is untouched, which protects rel_l2 and mvpe, and the
generative target has smaller amplitude and far more homogeneous statistics than
the raw field, which makes it much easier to learn.

WHY FLOW MATCHING RATHER THAN DDPM

time_score is 100/(1+sqrt(t/0.72896)), so sampling cost is scored directly. DDPM
needs tens of denoising steps; flow matching learns a straight-line velocity field
and produces usable samples in 4-8 Euler steps. Training is also simpler — one
regression loss, no noise schedule to tune.

    z ~ N(0, I),  t ~ U(0, 1)
    x_t = (1 - t) * z + t * fluctuation
    target velocity = fluctuation - z
    loss = || v_theta(x_t, t, condition) - (fluctuation - z) ||^2

Sampling integrates dx/dt = v_theta from z at t=0 to a sample at t=1.

THE RISK THIS IS TESTING

tke_score is a relative L2 on the TKE *map*, not on total fluctuation energy. A
sample with correct statistics but misplaced structure scores worse than a smooth
one — that is exactly how the advection prior failed, driving tke error from 1.0
to 2.4. Conditioning on the input window is what should prevent it, since the
deterministic model already locates the fluctuation well (window-level TKE
correlation 0.969). Whether that carries over to sampled fluctuation is the
question the experiment answers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .advection import coordinate_channels, derived_channels
    from .models import ConvBlock, fold_time, unfold_time
except ImportError:
    from advection import coordinate_channels, derived_channels
    from models import ConvBlock, fold_time, unfold_time


def temporal_mean(x: torch.Tensor) -> torch.Tensor:
    """(B, T, H, W, C) -> (B, 1, H, W, C)."""
    return x.mean(dim=1, keepdim=True)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1))
    )
    ang = t.reshape(-1, 1) * freqs.reshape(1, -1) * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FluctuationFlow(nn.Module):
    """Velocity field v(x_t, t | condition) over the fluctuation.

    The condition is the input window plus the deterministic model's temporal
    mean, so the generator never has to rediscover the steady flow.
    """

    def __init__(self, t_in: int = 20, t_out: int = 20, channels: int = 2,
                 base: int = 64, t_embed: int = 128, fluct_scale: float = 0.3546,
                 start_noise: float = 0.5):
        super().__init__()
        self.t_in, self.t_out, self.channels = t_in, t_out, channels
        self.t_embed = t_embed
        # Noise added to the deterministic starting point, in units of
        # fluct_scale. It is what makes the map stochastic — with 0 the flow is a
        # deterministic refinement and every sample is identical.
        self.start_noise = float(start_noise)
        # The flow runs on the fluctuation divided by this, so its target has unit
        # variance and matches the noise it interpolates from. Without it the
        # fluctuation's standard deviation is 0.355 against the noise's 1.0, so at
        # t = 0.5 the signal is only 11% of x_t's energy and below t = 0.5 the
        # network is effectively asked to predict the noise draw itself. Measured
        # on the training split; stored so inference uses the same constant.
        self.register_buffer("fluct_scale", torch.tensor(float(fluct_scale)))

        # noisy fluctuation + input window + deterministic temporal mean
        # + the deterministic fluctuation the flow is refining
        # + speed / vorticity / divergence of the last input frame
        # + absolute x, y
        c_in = (t_out * channels          # x_t
                + t_in * channels         # conditioning window
                + channels                # deterministic temporal mean
                + t_out * channels        # deterministic fluctuation
                + 3                       # speed, vorticity, divergence
                + 2)                      # coordinates
        c_out = t_out * channels
        w = [base, int(base * 1.5), base * 2, int(base * 2.5)]

        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed, base * 2), nn.GELU(), nn.Linear(base * 2, base * 2))

        self.enc1 = ConvBlock(c_in, w[0])
        self.enc2 = ConvBlock(w[0], w[1])
        self.enc3 = ConvBlock(w[1], w[2])
        self.bottleneck = ConvBlock(w[2], w[3])
        # Flow time is injected at the bottleneck, where it is cheapest and where
        # the whole field is in view.
        self.t_proj = nn.Linear(base * 2, w[3])
        self.dec3 = ConvBlock(w[3] + w[2], w[2])
        self.dec2 = ConvBlock(w[2] + w[1], w[1])
        self.dec1 = ConvBlock(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], c_out, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_window: torch.Tensor,
                cond_mean: torch.Tensor, cond_fluct: torch.Tensor,
                mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """x_t, output: (B, t_out, H, W, C). t: (B,). cond_mean: (B, 1, H, W, C).

        `cond_fluct` is the deterministic model's own fluctuation — the estimate
        this flow refines. `mean`/`std` denormalise the last input frame so the
        derived physics channels are computed in m/s.
        """
        b, _, h, w, _ = x_t.shape
        phys_last = cond_window[:, -1] * std + mean
        der = derived_channels(phys_last[..., 0], phys_last[..., 1])
        der = der / torch.tensor([0.1, 10.0, 10.0], device=der.device)
        coords = coordinate_channels(b, h, w, x_t.device, x_t.dtype)

        feats = torch.cat([
            fold_time(x_t),
            fold_time(cond_window),
            cond_mean.squeeze(1).permute(0, 3, 1, 2),
            fold_time(cond_fluct),
            der.permute(0, 3, 1, 2),
            coords.permute(0, 3, 1, 2),
        ], dim=1)

        e1 = self.enc1(feats)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        emb = self.t_proj(self.t_mlp(timestep_embedding(t, self.t_embed)))
        b = b + emb.reshape(b.shape[0], -1, 1, 1)

        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return unfold_time(self.head(d1), self.t_out, self.channels)


def flow_loss(model: FluctuationFlow, fluct: torch.Tensor, cond_window: torch.Tensor,
              cond_mean: torch.Tensor, cond_fluct: torch.Tensor,
              mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Rectified-flow regression between the deterministic estimate and the truth.

    The first design interpolated from pure noise to the true fluctuation, and it
    plateaued far short of the deterministic model — it was being asked to invent
    the whole structure from nothing, while the deterministic fluctuation already
    correlates 0.63 with the truth.

    So the path now starts from that estimate plus a modest noise term rather than
    from noise alone:

        x_0 = deterministic_fluctuation + START_NOISE * z
        x_1 = true_fluctuation

    The velocity field only has to supply what the deterministic model misses —
    amplitude and fine structure — and the starting point already protects rel_l2.
    Everything is in units of `fluct_scale`.
    """
    target = fluct / model.fluct_scale
    start = cond_fluct / model.fluct_scale + model.start_noise * torch.randn_like(target)
    b = target.shape[0]
    t = torch.rand(b, device=target.device)
    tb = t.reshape(-1, 1, 1, 1, 1)
    x_t = (1.0 - tb) * start + tb * target
    v = model(x_t, t, cond_window, cond_mean, cond_fluct, mean, std)
    return F.mse_loss(v, target - start)


@torch.no_grad()
def sample_fluctuation(model: FluctuationFlow, cond_window: torch.Tensor,
                       cond_mean: torch.Tensor, cond_fluct: torch.Tensor,
                       mean: torch.Tensor, std: torch.Tensor, steps: int = 6,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate from the deterministic estimate to a refined sample.

    Starting from the deterministic fluctuation rather than from noise means even
    zero steps returns something usable, so the step count trades refinement
    against time_score instead of trading away validity. 4-8 is the useful range.
    """
    x = cond_fluct / model.fluct_scale
    if model.start_noise > 0:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = x + model.start_noise * noise
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t, cond_window, cond_mean, cond_fluct, mean, std)
    # Back out of the unit-variance space the flow was trained in.
    return x * model.fluct_scale
