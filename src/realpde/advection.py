"""Semi-Lagrangian advection: transport the field along its own velocity.

WHY THIS AND NOT MORE CONVOLUTIONS

Over the 20-frame horizon the flow does not sit still. Measured on the real
split, in 0.4 s it travels:

    Re  3750    5.9 evaluation cells   ( 9% of the field width)
    Re 10125   14.9                    (23%)
    Re 20325   28.3                    (44%)
    Re 26700   34.2                    (54%)

So the value at a cell 20 frames from now originates up to 34 cells upstream.
Transport, not local dynamics, is the dominant term. A network can learn to
transport, but it has to spend capacity doing so and has to relearn it at every
Reynolds number; handing it an advected estimate makes that part free and leaves
the residual — the part that is genuinely hard — to be learned.

HOW ROTATION COMES OUT FOR FREE

Advecting once by a frozen velocity is pure translation. Advecting *iteratively*,
transporting the velocity field along with the scalars and re-reading it each
step, reproduces rotation without a rotation parameter: inside a vortex the
velocity field itself turns the pattern. That is what `advect_sequence` does.

This is a prior, not a solver. There is no pressure projection and no viscosity,
so the estimate degrades over the horizon; the network corrects it.

GRID CONVENTION

y DECREASES with row index (dy is negative in the source data), so a positive v
(physically upward) moves the pattern to a SMALLER row index. Getting this sign
wrong advects the wake the wrong way and is silent — it just trains worse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DT = 0.02            # seconds per frame
DX_EVAL = 0.003422   # metres per cell at the 32x64 evaluation resolution


def _base_grid(h: int, w: int, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx, yy


def warp(field: torch.Tensor, disp_col: torch.Tensor, disp_row: torch.Tensor) -> torch.Tensor:
    """Backward-warp `field` by a displacement given in cells.

    field    : (B, C, H, W)
    disp_col : (B, H, W) displacement along the column axis (+x)
    disp_row : (B, H, W) displacement along the row axis

    Semi-Lagrangian: the value arriving at p came from p - displacement, so the
    source is sampled at the departure point.
    """
    b, c, h, w = field.shape
    xx, yy = _base_grid(h, w, field.device, field.dtype)
    src_x = xx.unsqueeze(0) - disp_col
    src_y = yy.unsqueeze(0) - disp_row

    gx = 2.0 * src_x / max(w - 1, 1) - 1.0
    gy = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def displacement_cells(u: torch.Tensor, v: torch.Tensor, steps: float = 1.0):
    """Physical velocities (m/s) -> displacement in grid cells over `steps` frames.

    The row displacement is negated because y decreases with row index.
    """
    k = steps * DT / DX_EVAL
    return u * k, -v * k


def advect_sequence(u0: torch.Tensor, v0: torch.Tensor, n_steps: int,
                    substeps: int = 1, clamp: float | None = 3.0):
    """Iteratively transport the velocity field along itself.

    u0, v0 : (B, H, W) physical velocities of the frame to advect from
    returns : (B, n_steps, H, W, 2), the advected estimate for each future frame

    `substeps` splits each frame into smaller advection steps, which keeps the
    per-step displacement near or below one cell and limits interpolation
    smearing. `clamp` bounds the per-substep displacement in cells: without it a
    self-advecting field with no pressure term can run away.
    """
    u, v = u0, v0
    outs = []
    for _ in range(n_steps):
        for _ in range(substeps):
            dc, dr = displacement_cells(u, v, steps=1.0 / substeps)
            if clamp is not None:
                dc = dc.clamp(-clamp, clamp)
                dr = dr.clamp(-clamp, clamp)
            stacked = torch.stack([u, v], dim=1)          # (B, 2, H, W)
            moved = warp(stacked, dc, dr)
            u, v = moved[:, 0], moved[:, 1]
        outs.append(torch.stack([u, v], dim=-1))          # (B, H, W, 2)
    return torch.stack(outs, dim=1)                        # (B, n, H, W, 2)


def advect_frozen(field_u: torch.Tensor, field_v: torch.Tensor,
                  trans_u: torch.Tensor, trans_v: torch.Tensor,
                  n_steps: int, valid: torch.Tensor | None = None):
    """Transport a frame by a FIXED velocity field — Taylor's frozen hypothesis.

    `advect_sequence` lets the velocity advect itself, which is the nonlinear
    term of Navier-Stokes with no pressure projection and no viscosity. Measured
    on the validation split that is unstable: it scores below persistence and its
    TKE error exceeds 1, meaning it destroys the field rather than forecasting it.

    Transporting instead by a fixed, smooth field — in practice the temporal mean
    of the input window — is the classical frozen-turbulence approximation:
    eddies are carried by the mean flow while keeping their shape. It cannot
    create or destroy structure, so it cannot blow up.

    `valid` optionally marks cells that hold a measurement. Advection otherwise
    drags the zero-valued laser shadow out into the flow; where the transported
    validity falls below a half, the original frame is kept instead.

    Every output frame is an independent warp of the same source frame by a
    displacement that scales with the frame index, so all n_steps are issued as a
    single batched grid_sample instead of a Python loop. Looping cost about
    11.5 ms per sample on the evaluation platform and dropped time_score from
    95.8 to 88.3, which more than cancelled the accuracy the prior bought.
    """
    b, h, w = field_u.shape
    steps = torch.arange(1, n_steps + 1, device=field_u.device,
                         dtype=field_u.dtype).view(1, n_steps, 1, 1)
    k = steps * (DT / DX_EVAL)
    dc = (trans_u.unsqueeze(1) * k).reshape(b * n_steps, h, w)
    dr = (-trans_v.unsqueeze(1) * k).reshape(b * n_steps, h, w)

    stacked = torch.stack([field_u, field_v], dim=1)                  # (B, 2, H, W)
    src = stacked.unsqueeze(1).expand(b, n_steps, 2, h, w).reshape(b * n_steps, 2, h, w)
    moved = warp(src, dc, dr)

    if valid is not None:
        v_src = valid.reshape(b, 1, 1, h, w).expand(b, n_steps, 1, h, w) \
                     .reshape(b * n_steps, 1, h, w).to(moved.dtype)
        keep = (warp(v_src, dc, dr) >= 0.5).to(moved.dtype)
        moved = moved * keep + src * (1.0 - keep)

    return moved.reshape(b, n_steps, 2, h, w).permute(0, 1, 3, 4, 2)  # (B, n, H, W, 2)


def derived_channels(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Physically meaningful fields the network would otherwise have to derive.

    speed, vorticity and in-plane divergence, from centred finite differences.
    Divergence is NOT zero here — the plane is a slice of a three-dimensional
    flow, and it measures about half the vorticity magnitude on the real split —
    so it carries real information rather than being a constant zero.

    u, v : (B, H, W) -> (B, H, W, 3)
    """
    dudx = torch.gradient(u, spacing=DX_EVAL, dim=-1)[0]
    dvdx = torch.gradient(v, spacing=DX_EVAL, dim=-1)[0]
    # Row index runs opposite to y, hence the negative spacing.
    dudy = torch.gradient(u, spacing=-DX_EVAL, dim=-2)[0]
    dvdy = torch.gradient(v, spacing=-DX_EVAL, dim=-2)[0]

    speed = torch.sqrt(u * u + v * v + 1e-12)
    vorticity = dvdx - dudy
    divergence = dudx + dvdy
    return torch.stack([speed, vorticity, divergence], dim=-1)


def coordinate_channels(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalised x and y in [-1, 1].

    Convolutions are translation-equivariant, but the airfoil sits at a fixed
    place in every sample, so absolute position is informative and the network
    cannot represent it from convolutions alone.
    """
    xx, yy = _base_grid(h, w, device, dtype)
    gx = (2.0 * xx / max(w - 1, 1) - 1.0).expand(b, h, w)
    gy = (2.0 * yy / max(h - 1, 1) - 1.0).expand(b, h, w)
    return torch.stack([gx, gy], dim=-1)
