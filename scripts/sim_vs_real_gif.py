"""Side-by-side animation of the simulated and the measured flow, same regime.

This is the picture the whole competition turns on. Both panels are the same
Reynolds number and angle of attack, on the same 32x64 evaluation grid, with the
same colour scale: the left is a clean simulation, the right is what a PIV
camera actually recorded. The simulation is non-dimensional, so it is multiplied
by its own freestream velocity first (U = Re * 1.394e-5 m/s, measured in
scripts/compare_sim_real.py) -- without that the two differ by up to 7x and the
comparison is meaningless.

Deliberately small: this is a README figure, so the frame count and dpi are set
to keep the file a couple of megabytes rather than the 12 MB of the full-
resolution single-field animations in figures/.

Usage:
    python scripts/sim_vs_real_gif.py --case 10125_10
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from realpde.data import U_PER_RE, load_split


def field(split: str, case: str, start: int, n: int):
    arr, meta = load_split(split)
    t = next(x for x in meta["trajectories"] if x["case"] == case)
    o, L = t["offset"], t["length"]
    s = min(start, max(L - n, 0))
    f = np.asarray(arr[o + s:o + s + n, ..., 0], dtype=np.float32)   # streamwise u
    if split == "sim":
        f = f * (t["re_nominal"] * U_PER_RE)       # non-dimensional -> m/s
    return f, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="10125_10")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "figures" / "sim_vs_real_anim.gif"))
    args = ap.parse_args()

    real, t = field("real", args.case, args.start, args.frames)
    sim, _ = field("sim", args.case, args.start, args.frames)
    re, aoa = t["re_nominal"], t["aoa_nominal"]

    # One scale for both panels, from the measurement: rescaling each to its own
    # range would hide exactly the difference this figure exists to show.
    lo, hi = np.percentile(real, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.15), dpi=100)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.875, top=0.84, bottom=0.14, wspace=0.04)
    ims = []
    for ax, data, title in ((axes[0], sim, "Simulation  (clean, non-dimensional)"),
                            (axes[1], real, "PIV measurement  (noisy, masked)")):
        im = ax.imshow(data[0], cmap="viridis", vmin=lo, vmax=hi,
                       interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    cax = fig.add_axes([0.895, 0.16, 0.014, 0.66])
    cb = fig.colorbar(ims[1], cax=cax)
    cb.set_label("streamwise velocity u  [m/s]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    sup = fig.text(0.445, 0.035, "", fontsize=8.5, ha="center", color="0.25")

    def update(i):
        ims[0].set_data(sim[i])
        ims[1].set_data(real[i])
        sup.set_text(f"Re {re}   AoA {aoa}°   32x64 evaluation grid   "
                     f"frame {i + 1}/{len(real)}")
        return (*ims, sup)

    out = Path(args.out)
    FuncAnimation(fig, update, frames=len(real), blit=False).save(
        out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
