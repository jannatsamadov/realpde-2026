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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Imported both as part of the `realpde` package and as a flat `model_def.py`
# vendored beside submission.py, where there is no package to be relative to.
try:  # package context
    from .advection import (advect_one_step, advect_sequence,
                            coordinate_channels, derived_channels)
except ImportError:  # vendored flat inside a submission archive
    from advection import (advect_one_step, advect_sequence,
                           coordinate_channels, derived_channels)


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

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Everything upstream of the network: (feats, advected estimate).

        None of it depends on a single learnable parameter — the advection is a
        fixed transport of the input and the derived channels are finite
        differences of it. Split out so test-time adaptation can cache it: the
        advection is the expensive part of a forward pass (a 40-step
        grid_sample loop), and re-running it to compute a gradient on a window
        already seen would double the per-step cost for nothing.
        """
        adv = self._advected(x)                        # (B, t_out, H, W, C)
        return self._pack(x, adv), adv

    def _pack(self, x: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """Assemble the network's input channels from a window and a prior.

        Split out from `features` so a model that recomputes the prior can reuse
        the packing without duplicating it; `features` is unchanged by it.
        """
        b, _, h, w_, c = x.shape
        phys_last = x[:, -1] * self.std + self.mean
        der = derived_channels(phys_last[..., 0], phys_last[..., 1])
        # Scale the derived fields to roughly unit variance so they do not
        # dominate the first convolution: vorticity is O(10) in 1/s.
        der = der / torch.tensor([0.1, 10.0, 10.0], device=der.device)
        coords = coordinate_channels(b, h, w_, x.device, x.dtype)

        return torch.cat([
            fold_time(x),
            fold_time(adv),
            der.permute(0, 3, 1, 2),
            coords.permute(0, 3, 1, 2),
        ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, adv = self.features(x)
        return self.forward_from_features(feats, adv)

    def forward_from_features(self, feats: torch.Tensor,
                              adv: torch.Tensor) -> torch.Tensor:
        """The learnable half, given what `features` produced."""
        e1 = self.enc1(feats)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        bo = self.bottleneck(F.avg_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(bo, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))

        return adv + unfold_time(self.head(d1), self.t_out, self.channels)


# Regime label normalisation. Reynolds number spans 3750 to 27975 — a factor of
# 7.5 — so it is regressed in log space, where the spacing between the released
# regimes is roughly uniform. Both targets land in [0, 1].
RE_LOG_LO, RE_LOG_HI = math.log(3750.0), math.log(27975.0)
AOA_HI = 20.0


def encode_regime(re: torch.Tensor, aoa: torch.Tensor) -> torch.Tensor:
    """(B,), (B,) -> (B, 2) in [0, 1]."""
    r = (torch.log(re.clamp_min(1.0)) - RE_LOG_LO) / (RE_LOG_HI - RE_LOG_LO)
    return torch.stack([r, aoa / AOA_HI], dim=-1)


class TemporalRegimeEncoder(nn.Module):
    """Per-frame embedding, attention over the 20 frames, and a regime code.

    WHY ATTENTION AND NOT RECURRENCE

    The per-step cost of anything sequential is what hurts here: Track 2 is
    scored at batch size 1, where a 20-step loop is dominated by kernel launches
    rather than arithmetic — measured on the advection prior, whose 40-step loop
    was 90% of a forward pass. Attention over 20 tokens is a single batched
    matmul with no sequential dependency, so it buys temporal structure without
    paying that cost. Frames are folded into the BATCH for the stem, so even the
    per-frame encoding runs in parallel.

    WHAT THE REGIME CODE IS FOR

    Reynolds number and angle of attack are never available at scoring time, so
    they cannot be inputs. But the error is strongly regime-dependent — measured
    against the PIV noise floor, the model is 2.46x the floor at 0 degrees and
    3.11x at 20, and worst of all at low Reynolds number with high incidence,
    which is the separated regime. One set of weights is covering attached and
    fully separated flow at once.

    So the regime is ESTIMATED from the window instead, and the estimate
    conditions the forecast. An auxiliary loss on the code makes it actually
    encode Re and angle of attack rather than whatever else would minimise the
    forecast loss. Training and inference use the same self-produced code, so
    there is no teacher-forcing mismatch to leak.
    """

    def __init__(self, channels: int = 2, d: int = 48, heads: int = 4,
                 t_in: int = 20):
        super().__init__()
        self.d, self.t_in = d, t_in
        g = min(8, d)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, d, 3, stride=2, padding=1),   # 32x64 -> 16x32
            nn.GroupNorm(g, d), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1),          # -> 8x16
            nn.GroupNorm(g, d), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1),          # -> 4x8
            nn.GroupNorm(g, d), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, t_in, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.regime = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor):
        """(B, T, H, W, C) -> (context (B, d, 4, 8), regime (B, 2), code (B, d))."""
        b, t, h, w, c = x.shape
        z = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        z = self.stem(z)                                   # (B*T, d, 4, 8)
        _, d, hh, ww = z.shape
        # (B, T, d, P) -> (B*P, T, d): one sequence per bottleneck cell, so the
        # attention asks "how does this location evolve", not "which frame".
        z = z.reshape(b, t, d, hh * ww).permute(0, 3, 1, 2).reshape(b * hh * ww, t, d)
        z = z + self.pos
        a, _ = self.attn(z, z, z, need_weights=False)
        z = self.norm(z + a)

        ctx = z.mean(dim=1).reshape(b, hh * ww, d).permute(0, 2, 1).reshape(b, d, hh, ww)
        code = ctx.mean(dim=(2, 3))                        # (B, d)
        return ctx, self.regime(code), code


class RegimeAttentiveUNet(AdvectiveUNet):
    """AdvectiveUNet plus temporal attention and self-estimated regime conditioning.

    Everything the base model does is unchanged — the advection prior, the
    residual on top of it, the zero-initialised head that starts as pure physics.
    The additions are a temporal-attention context concatenated at the
    bottleneck, and FiLM modulation of the bottleneck from the regime code.

    `forward` still returns only the prediction, so every existing script,
    checkpoint loader and submission path keeps working. Training calls
    `forward_with_regime` to get the auxiliary output as well.
    """

    def __init__(self, *args, d_time: int = 48, heads: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        w_bott = self.bottleneck.net[0].out_channels
        self.temporal = TemporalRegimeEncoder(channels=self.channels, d=d_time,
                                              heads=heads, t_in=self.t_in)
        self.merge = nn.Conv2d(w_bott + d_time, w_bott, 1)
        # Zero-initialised FiLM: at step 0 the modulation is the identity, so the
        # model starts exactly where the base model starts rather than having to
        # recover from a random perturbation of a working bottleneck.
        self.film = nn.Linear(d_time, 2 * w_bott)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def _trunk(self, feats: torch.Tensor, adv: torch.Tensor, x: torch.Tensor):
        e1 = self.enc1(feats)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        bo = self.bottleneck(F.avg_pool2d(e3, 2))

        ctx, regime, code = self.temporal(x)
        bo = self.merge(torch.cat([bo, ctx], dim=1))
        gamma, beta = self.film(code).chunk(2, dim=-1)
        bo = bo * (1.0 + gamma[..., None, None]) + beta[..., None, None]

        d3 = self.dec3(torch.cat([F.interpolate(bo, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
        return adv + unfold_time(self.head(d1), self.t_out, self.channels), regime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, adv = self.features(x)
        return self._trunk(feats, adv, x)[0]

    def forward_with_regime(self, x: torch.Tensor):
        feats, adv = self.features(x)
        return self._trunk(feats, adv, x)

    def forward_from_features(self, feats: torch.Tensor, adv: torch.Tensor):
        raise RuntimeError(
            "RegimeAttentiveUNet needs the raw window for its temporal branch, so "
            "the features/trunk split the Track 2 adapter uses does not apply. "
            "Call forward(x) instead."
        )


# Reference velocity scale: the median 90th-percentile speed over the real
# split's disjoint windows, measured in scripts/check_scale_stat.py. A window at
# this scale passes through ScaleInvariantUNet exactly as it would through
# AdvectiveUNet, so the reference regime is the one the base model was tuned on.
U_REF = 0.208734   # m/s


def window_scale(phys: torch.Tensor, q: float = 0.90) -> torch.Tensor:
    """Velocity scale of each window, from the window alone. (B,T,H,W,C) -> (B,).

    The 90th percentile of speed over unmasked cells. Measured against the
    freestream on 1672 disjoint windows: corr(.,Re) = 0.9959, log-log slope
    0.961, residual scatter 3.8% -- the tightest of the three candidates tried,
    and stable to ~1% between windows of one trajectory, so it is an estimate of
    the regime rather than of the turbulence.

    Masked cells read exactly zero in both components and are excluded through
    nanquantile; including them would make the statistic track the airfoil's
    projected area, which grows with angle of attack, instead of the flow.

    Detached on purpose: this is a measurement of the input used to choose a
    change of units, not a quantity the loss should push on.
    """
    speed = torch.sqrt(phys[..., 0] ** 2 + phys[..., 1] ** 2)
    speed = speed.flatten(1)
    speed = torch.where(speed > 0, speed, torch.nan)
    s = torch.nanquantile(speed, q, dim=1)
    return s.detach().clamp_min(1e-6)


class ScaleInvariantUNet(AdvectiveUNet):
    """AdvectiveUNet made homogeneous of degree one in velocity.

    WHY
        The private test uses unseen Reynolds numbers, and the input is in m/s,
        so |u| is proportional to Re: across our windows the field magnitude
        spans 8x. An unseen Re is therefore an unseen *magnitude*, which a
        convolution has to extrapolate to. Navier-Stokes is invariant under
        u -> u/U, x -> x/c, t -> t*U/c, so dividing each window by its own
        velocity scale collapses every regime onto one non-dimensional problem
        and turns that extrapolation into an interpolation.

        This is not the same idea as regressing Re as an auxiliary output, which
        was measured and did nothing: the network could already read Re off the
        magnitude, so telling it what it knew changed nothing. Here the problem
        itself changes -- the network never sees the magnitude at all.

    WHY THE SCALE IS ALSO FED BACK IN
        Only two of the three invariance factors are applied. The frame interval
        is fixed at 0.02 s, so the non-dimensional step dt*U/c grows with Re, and
        measured on all 81 trajectories the shedding frequency does follow
        Strouhal: d ln f / d ln Re = +1.07, giving a period between 5.6 and 145
        frames -- a 26x spread, or 0.14 to 3.55 shedding cycles inside one
        20-frame window.

        Dividing by the velocity scale and stopping there would therefore be
        strictly worse than doing nothing: the magnitude is exactly how the
        network currently knows how fast the movie is playing, and normalising
        it away destroys that. So log(scale ratio) comes back as an explicit
        conditioning scalar -- one smooth number to extrapolate in, instead of a
        whole field of magnitudes.

        Full temporal non-dimensionalisation, resampling the window in time,
        is not done here: at low Re twenty real frames span about three
        non-dimensional ones and at high Re more than twenty, which a fixed
        20 -> 20 interface cannot express.

    INTERFACE
        features()/forward_from_features() keep the base model's two-tensor
        signature, so the Track 2 adapter, the submission templates and every
        script work unchanged. The scale travels inside `feats` as a constant
        last channel, which forward_from_features reads back exactly.
    """

    LOG_R_CHANNEL = -1   # the constant plane appended by features()

    def load_advective_state(self, state: dict) -> None:
        """Warm-start from an AdvectiveUNet checkpoint of the same width.

        Only enc1's first convolution differs, by the one log-scale plane; its
        column is zero-padded so the loaded model begins by ignoring the scale
        exactly as the checkpoint did. Everything else is shape-identical.
        """
        state = dict(state)
        k = "enc1.net.0.weight"
        w = state[k]
        want = self.enc1.net[0].weight
        if w.shape[1] == want.shape[1] - 1:
            state[k] = torch.cat(
                [w, torch.zeros_like(w[:, :1])], dim=1)
        self.load_state_dict(state, strict=False)

    def __init__(self, *args, d_cond: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        # One extra input plane: log of the scale ratio, broadcast over the grid.
        old = self.enc1.net[0]
        new = nn.Conv2d(old.in_channels + 1, old.out_channels,
                        kernel_size=old.kernel_size, padding=old.padding)
        with torch.no_grad():
            new.weight[:, :old.in_channels] = old.weight
            new.weight[:, old.in_channels:] = 0.0   # starts as the base model
            new.bias.copy_(old.bias)
        self.enc1.net[0] = new

        w_bott = self.bottleneck.net[0].out_channels
        self.cond = nn.Sequential(nn.Linear(1, d_cond), nn.GELU(),
                                  nn.Linear(d_cond, 2 * w_bott))
        # Zero-initialised FiLM: the modulation is the identity at step 0, so
        # training starts from the base model rather than from a perturbation.
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def _rescale_norm(self, x_norm: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Normalized field -> normalized field of the same flow at the reference
        scale. Physical round trip, so the official constants stay the only
        normalization in play."""
        shape = (-1,) + (1,) * (x_norm.dim() - 1)
        phys = (x_norm * self.std + self.mean) / r.view(shape)
        return (phys - self.mean) / self.std

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w_, c = x.shape
        phys = x * self.std + self.mean
        r = window_scale(phys) / U_REF                    # (B,)

        # The advection prior is transport at the true velocity and is already
        # correct at any scale; it is the *output* baseline and stays physical.
        adv = self._advected(x)
        # What the network looks at is the same flow rescaled to the reference.
        x_sn = self._rescale_norm(x, r)
        adv_sn = self._rescale_norm(adv, r)

        phys_last = x_sn[:, -1] * self.std + self.mean
        der = derived_channels(phys_last[..., 0], phys_last[..., 1])
        der = der / torch.tensor([0.1, 10.0, 10.0], device=der.device)
        coords = coordinate_channels(b, h, w_, x.device, x.dtype)

        log_r = torch.log(r).view(-1, 1, 1, 1).expand(b, 1, h, w_)
        feats = torch.cat([
            fold_time(x_sn),
            fold_time(adv_sn),
            der.permute(0, 3, 1, 2),
            coords.permute(0, 3, 1, 2),
            log_r.to(x.dtype),
        ], dim=1)
        return feats, adv

    def forward_from_features(self, feats: torch.Tensor,
                              adv: torch.Tensor) -> torch.Tensor:
        # The plane is constant over the grid, so any cell recovers it exactly.
        log_r = feats[:, self.LOG_R_CHANNEL, 0, 0]
        r = torch.exp(log_r)

        e1 = self.enc1(feats)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        bo = self.bottleneck(F.avg_pool2d(e3, 2))

        gamma, beta = self.cond(log_r[:, None]).chunk(2, dim=-1)
        bo = bo * (1.0 + gamma[..., None, None]) + beta[..., None, None]

        d3 = self.dec3(torch.cat([F.interpolate(bo, scale_factor=2, mode="nearest"), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))

        # The residual comes out at the reference scale; multiplying by r sends
        # it back to this window's, which is what makes the whole map
        # homogeneous of degree one in velocity. Normalized space is a linear
        # map of physical with a fixed std, so the factor is the same here.
        res = unfold_time(self.head(d1), self.t_out, self.channels)
        return adv + res * r.view(-1, 1, 1, 1, 1)


class SolverLoopUNet(AdvectiveUNet):
    """AdvectiveUNet whose transport prior is recomputed from its own forecast.

    THE PROBLEM WITH A PRIOR COMPUTED ONCE
        The base model advects the last input frame twenty steps forward and
        hands the network all twenty at once. Nothing corrects the transport
        along the way: by frame twenty the prior is the result of twenty
        unassisted advection steps, with no pressure term and no viscosity, and
        its error has compounded the whole way. That matches where our error
        actually is -- 1.3x the measurement noise floor at lead 1 and 3.6x at
        lead 20, so essentially all of the remaining room is at long range.

    WHAT CHANGES
        One refinement pass. After the first forecast, frame k's prior is
        rebuilt as the *predicted* frame k-1 advanced a single step, so the
        transport starts from a field the network has already corrected instead
        of from a twenty-step extrapolation. The same U-Net is then applied
        again to that better prior. Weights are shared, so parameter count is
        unchanged and the comparison against the base model is clean.

    WHY IT IS AFFORDABLE
        The twenty single-step advections are independent of each other, so they
        run as one batched call rather than a sequential loop -- see
        advect_one_step. The added cost is close to one extra U-Net pass, not a
        second advection prior, which matters because the sequential prior is
        about 90% of a forward pass at Track 2's batch size of one.

    NOT COMBINED WITH SCALE CONDITIONING YET, on purpose: one change at a time,
    or a gain cannot be attributed.
    """

    def __init__(self, *args, refine: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.refine = refine

    def _prior_from_forecast(self, x: torch.Tensor,
                             pred: torch.Tensor) -> torch.Tensor:
        """Frame k's prior = corrected frame k-1, advanced one step.

        Frame 0 has no predecessor among the outputs, so it uses the last input
        frame -- which is exactly what the base prior does for its first step.
        """
        prev = torch.cat([x[:, -1:], pred[:, :-1]], dim=1)   # (B, t_out, H, W, C)
        phys = prev * self.std + self.mean
        adv = advect_one_step(phys, substeps=self.substeps)
        return (adv - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, adv = self.features(x)
        pred = self.forward_from_features(feats, adv)
        for _ in range(self.refine):
            adv = self._prior_from_forecast(x, pred)
            pred = self.forward_from_features(self._pack(x, adv), adv)
        return pred


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    if name == "unet":
        return UNetForecaster(**kwargs)
    if name == "advective":
        kwargs.pop("residual", None)
        return AdvectiveUNet(**kwargs)
    if name == "regime":
        kwargs.pop("residual", None)
        return RegimeAttentiveUNet(**kwargs)
    if name == "scaleinv":
        kwargs.pop("residual", None)
        return ScaleInvariantUNet(**kwargs)
    if name == "solverloop":
        kwargs.pop("residual", None)
        return SolverLoopUNet(**kwargs)
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
