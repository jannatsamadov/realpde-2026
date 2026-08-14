"""Animate RealPDE trajectories as GIFs: vorticity field + velocity arrows.

Renders what the model actually sees, so the flow can be judged by eye:

  * background  — vorticity omega_z = dv/dx - du/dy, diverging colormap
  * arrows      — the (u, v) velocity field, subsampled so it stays readable
  * grey cells  — masked cells (u == v == 0): airfoil body, and PIV dropouts,
                  which move from frame to frame rather than being static
  * red squares — the 36 probe points MVPE scores, computed exactly as
                  scoring.py computes them

The grid is uniform: dx = dy = 1.711 mm, dt = 0.02 s (50 Hz), field of view
21.9 x 10.8 cm. Note y DECREASES with row index, so dy is negative and the
vorticity sign depends on getting that right.

Usage:
    python scripts/visualize.py --case 10125_10
    python scripts/visualize.py --case 10125_10 --eval-res     # 32x64, what the model sees
    python scripts/visualize.py --case 5025_20 --frames 200 --stride 2
    python scripts/visualize.py --aoa-sweep 10125              # static PNG, all AoA
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Derived from this file's location, so moving the project does not break it.
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "NeurIPS 2026 RealPDE Competition"
OUT_DIR = ROOT / "figures"


def mvpe_probe_indices(h: int, w: int, sub_s_real: int = 2):
    """The probe grid from scoring.py:113-143, reproduced exactly.

    Returns (probe_y, probe_x) as index lists on an (h, w) grid.
    """
    d, center_x, center_y, n_probe = 16, 10, 32, 9
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [
        probe_center_y + interval_y * j
        for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)
    ]
    probe_y = [y for y in probe_y if 0 <= y < h]

    probe_x = []
    for i in range(4):
        if int((2 * d + center_x) / sub_s_real) < w:
            px = int(((i + 1) * d + center_x) / sub_s_real)
        else:
            px = int((0.5 * (i + 2) * d + center_x) / sub_s_real)
        if 0 <= px < w:
            probe_x.append(px)
    return probe_y, probe_x


def load_case(case: str, split: str, n_frames: int, stride: int):
    path = DATA_ROOT / f"train_{split}" / f"train_{split}" / f"{case}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        total = f["u"].shape[0]
        idx = slice(0, min(n_frames * stride, total), stride)
        u = np.asarray(f["u"][idx], dtype=np.float64)
        v = np.asarray(f["v"][idx], dtype=np.float64)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        re = int(f["re"][()])
        aoa = int(f["aoa"][()])
        t = np.asarray(f["t"][idx], dtype=np.float64)
    return u, v, x, y, re, aoa, t, total


def vorticity(u, v, dx, dy):
    """omega_z = dv/dx - du/dy, with signed spacings (dy is negative here)."""
    dvdx = np.gradient(v, dx, axis=-1)
    dudy = np.gradient(u, dy, axis=-2)
    return dvdx - dudy


def animate(case: str, split: str, n_frames: int, stride: int, eval_res: bool,
            arrow_every: int, fps: int) -> Path:
    u, v, x, y, re, aoa, t, total = load_case(case, split, n_frames, stride)

    if eval_res:
        u, v, x, y = u[:, ::2, ::2], v[:, ::2, ::2], x[::2, ::2], y[::2, ::2]

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]  # negative: y decreases down the array
    h, w = u.shape[1], u.shape[2]

    mask = (u == 0) & (v == 0)
    vort = vorticity(u, v, dx, dy)
    vort = np.where(mask, np.nan, vort)
    speed = np.hypot(u, v)

    # Symmetric colour limits from a robust percentile of the finite vorticity.
    finite = vort[np.isfinite(vort)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0

    extent = [x.min(), x.max(), y.min(), y.max()]
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=110)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")  # masked cells render grey

    im = ax.imshow(vort[0], extent=extent, origin="upper", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal")

    # Arrows on a coarse subsample, masked cells dropped.
    ys, xs = np.mgrid[0:h:arrow_every, 0:w:arrow_every]
    qx, qy = x[ys, xs], y[ys, xs]

    def arrows(k):
        uu = np.where(mask[k][ys, xs], np.nan, u[k][ys, xs])
        vv = np.where(mask[k][ys, xs], np.nan, v[k][ys, xs])
        return uu, vv

    u0, v0 = arrows(0)
    # A fixed scale keeps arrow length comparable between frames and cases.
    qscale = max(float(np.nanpercentile(speed, 99)) * 25, 1e-6)
    quiv = ax.quiver(qx, qy, u0, v0, color="k", scale=qscale, width=0.0016,
                     headwidth=3.5, headlength=4.5, alpha=0.75)

    # MVPE probes, always evaluated on the 32x64 grid then mapped to this one.
    py, px = mvpe_probe_indices(32, 64)
    scale_i = 1 if eval_res else 2
    pyy = [p * scale_i for p in py if p * scale_i < h]
    pxx = [p * scale_i for p in px if p * scale_i < w]
    gx, gy = np.meshgrid([x[0, i] for i in pxx], [y[j, 0] for j in pyy])
    ax.scatter(gx, gy, s=26, marker="s", facecolors="none",
               edgecolors="#00c000", linewidths=1.4, zorder=5,
               label=f"MVPE probes ({len(pyy) * len(pxx)})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cb.set_label(r"vorticity  $\omega_z$  [1/s]")
    title = ax.set_title("")

    res_tag = "32x64 (evaluation)" if eval_res else "64x128 (raw)"

    def update(k):
        im.set_data(vort[k])
        uu, vv = arrows(k)
        quiv.set_UVC(uu, vv)
        title.set_text(
            f"{split}  Re={re}  AoA={aoa}°   {res_tag}\n"
            f"frame {k * stride}/{total}   t={t[k]:.2f}s   "
            f"masked {100 * mask[k].mean():.1f}%"
        )
        return im, quiv, title

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_eval32x64" if eval_res else ""
    out = OUT_DIR / f"{split}_{case}{suffix}.gif"
    anim = FuncAnimation(fig, update, frames=len(u), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}  ({len(u)} frames, {out.stat().st_size / 1024**2:.1f} MB)")
    return out


def aoa_sweep(re_tag: str, split: str, frame: int) -> Path:
    """One static row of panels: the same Re at every available angle of attack."""
    # Sort by angle of attack numerically: plain string sort gives 0, 10, 15, 20, 5.
    cases = sorted((DATA_ROOT / f"train_{split}" / f"train_{split}").glob(f"{re_tag}_*.h5"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not cases:
        raise FileNotFoundError(f"no cases for Re tag {re_tag}")

    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 3.1 * len(cases)), dpi=110)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")

    py, px = mvpe_probe_indices(32, 64)

    for ax, path in zip(axes, cases):
        with h5py.File(path, "r") as f:
            k = min(frame, f["u"].shape[0] - 1)
            u = np.asarray(f["u"][k]); v = np.asarray(f["v"][k])
            x = np.asarray(f["x"]); y = np.asarray(f["y"])
            re, aoa = int(f["re"][()]), int(f["aoa"][()])
        m = (u == 0) & (v == 0)
        vort = np.where(m, np.nan, vorticity(u, v, x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]))
        fin = vort[np.isfinite(vort)]
        vmax = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0
        ax.imshow(vort, extent=[x.min(), x.max(), y.min(), y.max()], origin="upper",
                  cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")

        # MVPE probes, mapped from the 32x64 evaluation grid onto this 64x128 one.
        gx, gy = np.meshgrid([x[0, i * 2] for i in px], [y[j * 2, 0] for j in py])
        ax.scatter(gx, gy, s=22, marker="s", facecolors="none",
                   edgecolors="#00b000", linewidths=1.3, zorder=5)

        ax.set_title(f"Re={re}  AoA={aoa}°   (frame {k})   "
                     f"masked {100 * m.mean():.1f}%", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{split}_{re_tag}_aoa_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="e.g. 10125_10  ({Re}_{AoA})")
    ap.add_argument("--split", default="real", choices=["real", "sim"])
    ap.add_argument("--frames", type=int, default=120, help="frames to render")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame")
    ap.add_argument("--eval-res", action="store_true", help="render at 32x64 instead of 64x128")
    ap.add_argument("--arrow-every", type=int, default=4, help="arrow spacing in cells")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--aoa-sweep", help="Re tag, e.g. 10125: static PNG across all AoA")
    args = ap.parse_args()

    if args.aoa_sweep:
        aoa_sweep(args.aoa_sweep, args.split, args.frames)
    elif args.case:
        animate(args.case, args.split, args.frames, args.stride,
                args.eval_res, args.arrow_every, args.fps)
    else:
        ap.error("give --case or --aoa-sweep")


if __name__ == "__main__":
    main()
