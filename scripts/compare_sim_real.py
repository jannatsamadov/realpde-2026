"""Side-by-side simulation vs real PIV for the same operating point.

This is the picture of the gap Track 1 asks us to close. The two splits share the
grid exactly (dx = dy = 1.711 mm, dt = 0.02 s, same x/y extent) but differ in
three ways that matter for training:

  1. SCALE. The simulation is non-dimensionalised by the freestream velocity: its
     mean u sits near 0.86 at every Reynolds number. The real data is in m/s and
     its mean u grows linearly with Re (corr = 0.995). Feeding both to one model
     without rescaling means fighting a factor that reaches 7x.
  2. MASKING. Real PIV loses ~9% of cells to the airfoil shadow and to dropouts
     that move frame to frame; the simulation masks only the body, ~0.3%.
  3. NOISE AND DETAIL. Simulation fields are smooth; PIV fields carry measurement
     noise and coarser effective resolution.

Rows are drawn twice: once in each split's own units, and once with both
normalised by their own mean |u| so the structures can be compared directly.

Usage:
    python scripts/compare_sim_real.py --case 10125_10
    python scripts/compare_sim_real.py --case 20325_20 --frame 400
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Derived from this file's location, so moving the project does not break it.
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "NeurIPS 2026 RealPDE Competition"
OUT_DIR = ROOT / "figures"


def mvpe_probe_indices(h: int = 32, w: int = 64, sub_s_real: int = 2):
    """Reproduced exactly from scoring.py:113-143."""
    d, center_x, center_y, n_probe = 16, 10, 32, 9
    probe_center_y = int(center_y / sub_s_real)
    interval_y = min(2, int(h / (n_probe + 1)))
    probe_y = [probe_center_y + interval_y * j
               for j in range(-(n_probe - 1) // 2, n_probe - (n_probe - 1) // 2)]
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


def vorticity(u, v, dx, dy):
    return np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2)


def load(split: str, case: str, frame: int):
    path = DATA_ROOT / f"train_{split}" / f"train_{split}" / f"{case}.h5"
    with h5py.File(path, "r") as f:
        k = min(frame, f["u"].shape[0] - 1)
        u = np.asarray(f["u"][k], dtype=np.float64)
        v = np.asarray(f["v"][k], dtype=np.float64)
        x = np.asarray(f["x"], dtype=np.float64)
        y = np.asarray(f["y"], dtype=np.float64)
        re, aoa, T = int(f["re"][()]), int(f["aoa"][()]), f["u"].shape[0]
    return u, v, x, y, re, aoa, k, T


def panel(ax, u, v, x, y, title, probes, normalise: bool):
    m = (u == 0) & (v == 0)
    scale = np.abs(u[~m]).mean() if normalise and (~m).any() else 1.0
    uu, vv = u / scale, v / scale
    vort = np.where(m, np.nan, vorticity(uu, vv, x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]))
    fin = vort[np.isfinite(vort)]
    vmax = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")
    ax.imshow(vort, extent=[x.min(), x.max(), y.min(), y.max()], origin="upper",
              cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")

    step = 4
    ys, xs = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
    au = np.where(m[ys, xs], np.nan, uu[ys, xs])
    av = np.where(m[ys, xs], np.nan, vv[ys, xs])
    ax.quiver(x[ys, xs], y[ys, xs], au, av, color="k",
              scale=float(np.nanpercentile(np.hypot(uu, vv), 99)) * 25,
              width=0.0018, headwidth=3.5, alpha=0.7)

    py, px = probes
    gx, gy = np.meshgrid([x[0, i * 2] for i in px], [y[j * 2, 0] for j in py])
    ax.scatter(gx, gy, s=20, marker="s", facecolors="none",
               edgecolors="#00b000", linewidths=1.2, zorder=5)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="10125_10", help="{Re}_{AoA}, e.g. 10125_10")
    ap.add_argument("--frame", type=int, default=300)
    args = ap.parse_args()

    probes = mvpe_probe_indices()
    fig, axes = plt.subplots(2, 2, figsize=(15, 6.4), dpi=115)

    for col, normalise in enumerate([False, True]):
        for row, split in enumerate(["real", "sim"]):
            u, v, x, y, re, aoa, k, T = load(split, args.case, args.frame)
            m = (u == 0) & (v == 0)
            umean = np.abs(u[~m]).mean()
            unit = "normalised by mean|u|" if normalise else ("m/s" if split == "real" else "non-dimensional")
            panel(axes[row, col], u, v, x, y,
                  f"{split.upper()}  Re={re}  AoA={aoa}°  frame {k}/{T}\n"
                  f"mean|u| = {umean:.4f}   masked {100 * m.mean():.1f}%   [{unit}]",
                  probes, normalise)

    fig.suptitle(f"Sim2Real gap — case {args.case}    "
                 f"left: each in its own units    right: both normalised by their own mean|u|",
                 fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sim_vs_real_{args.case}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
