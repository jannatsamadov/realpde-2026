"""How much of the MVPE probe grid lands on masked (zero) cells?

MVPE is the one accuracy metric computed from a tiny set of points — 4 x-stations
by 9 y-points on the 32x64 evaluation grid, time-averaged over the 20 output
frames. scoring.py does NOT mask it, unlike SPS: whatever sits at those points
goes into the relative error, zeros included.

If a probe sits inside the airfoil body or in a PIV dropout, its target is
exactly 0. That both shrinks the denominator (norm of the target at the probes)
and hands us free error-free points where predicting 0 is exactly right. Either
way it changes how we should weight the training loss, so it is worth a number
rather than a guess.

Usage:
    python scripts/probe_mask_check.py
"""

from pathlib import Path

import h5py
import numpy as np

# Derived from this file's location, so moving the project does not break it.
DATA_ROOT = Path(__file__).resolve().parent.parent / "NeurIPS 2026 RealPDE Competition"
EXCLUDED = {"7575_0.h5"}


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


def main() -> None:
    py, px = mvpe_probe_indices()
    print(f"probe_y ({len(py)}): {py}")
    print(f"probe_x ({len(px)}): {px}")
    print(f"total probe points: {len(py) * len(px)} of {32 * 64} cells "
          f"({100 * len(py) * len(px) / (32 * 64):.1f}% of the field)\n")

    files = sorted((DATA_ROOT / "train_real" / "train_real").glob("*.h5"),
                   key=lambda p: tuple(int(s) for s in p.stem.split("_")))

    # Physical coordinates of the probes (same for every case: the grid is fixed).
    with h5py.File(files[0], "r") as f:
        x = np.asarray(f["x"])[::2, ::2]
        y = np.asarray(f["y"])[::2, ::2]
    print("probe x [m]:", [f"{x[0, i]:.4f}" for i in px])
    print("probe y [m]:", [f"{y[j, 0]:.4f}" for j in py])
    print()

    rows = []
    print(f"{'case':<14} {'AoA':>4} {'probe cells masked':>19} {'field masked':>13}")
    print("-" * 56)
    for path in files:
        if path.name in EXCLUDED:
            continue
        with h5py.File(path, "r") as f:
            u = np.asarray(f["u"][:, ::2, ::2])
            v = np.asarray(f["v"][:, ::2, ::2])
            aoa = int(f["aoa"][()])
            re = int(f["re"][()])
        mask = (u == 0) & (v == 0)                    # (T, 32, 64)
        probe_mask = mask[:, np.ix_(py, px)[0], np.ix_(py, px)[1]]
        pm = float(probe_mask.mean())
        fm = float(mask.mean())
        rows.append({"case": path.stem, "re": re, "aoa": aoa, "probe": pm, "field": fm})
        print(f"{path.stem:<14} {aoa:>4} {100 * pm:>18.1f}% {100 * fm:>12.1f}%")

    print("-" * 56)
    print(f"{'MEAN':<14} {'':>4} {100 * np.mean([r['probe'] for r in rows]):>18.1f}% "
          f"{100 * np.mean([r['field'] for r in rows]):>12.1f}%")

    by_aoa = {}
    for r in rows:
        by_aoa.setdefault(r["aoa"], []).append(r["probe"])
    print("\nprobe-masked fraction by AoA:")
    for a in sorted(by_aoa):
        print(f"  AoA {a:>2}°: {100 * np.mean(by_aoa[a]):.1f}%")

    # Which individual probe columns are the offenders?
    print("\nper-probe-column masked fraction (averaged over cases and time):")
    per_col = {i: [] for i in px}
    for path in files:
        if path.name in EXCLUDED:
            continue
        with h5py.File(path, "r") as f:
            u = np.asarray(f["u"][:, ::2, ::2]); v = np.asarray(f["v"][:, ::2, ::2])
        mask = (u == 0) & (v == 0)
        for i in px:
            per_col[i].append(float(mask[:, py, :][:, :, i].mean()))
    for i in px:
        print(f"  x index {i:>2} (x={x[0, i]:.4f} m): {100 * np.mean(per_col[i]):.1f}%")


if __name__ == "__main__":
    main()
