"""How strong is the out-of-plane motion, and does it grow with angle of attack?

The measurement is a plane through a three-dimensional flow. For an incompressible
fluid, mass conservation in 3D reads

    du/dx + dv/dy + dw/dz = 0     =>     dw/dz = -(du/dx + dv/dy)

so the in-plane divergence is a direct measurement of how much fluid leaves or
enters the plane. A truly two-dimensional flow would have it at zero; it is not.

That makes it the one quantity in this dataset that carries three-dimensional
information, and the model already receives it as an input channel. The question
worth measuring is whether it grows with angle of attack, which would explain why
the same Reynolds number behaves so differently at 0 and 15 degrees.

Reported against the in-plane vorticity, since both are velocity gradients and the
ratio is dimensionless — an absolute divergence would just track Reynolds number.

Usage:
    python scripts/out_of_plane.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "NeurIPS 2026 RealPDE Competition" / "train_real" / "train_real"
SIM = ROOT / "NeurIPS 2026 RealPDE Competition" / "train_sim" / "train_sim"
DX = 0.001711
EXCLUDED = {"7575_0.h5"}


def ratios(path, frames=200):
    with h5py.File(path, "r") as f:
        n = min(frames, f["u"].shape[0])
        u = np.asarray(f["u"][:n], dtype=np.float64)
        v = np.asarray(f["v"][:n], dtype=np.float64)
        aoa, re = int(f["aoa"][()]), int(f["re"][()])
    m = ~((u == 0) & (v == 0))
    div = np.gradient(u, DX, axis=-1) + np.gradient(v, -DX, axis=-2)
    vort = np.gradient(v, DX, axis=-1) - np.gradient(u, -DX, axis=-2)
    d = np.sqrt((div[m] ** 2).mean())
    w = np.sqrt((vort[m] ** 2).mean())
    return re, aoa, d, w, d / w


def survey(root, label, limit_aoa=None):
    rows = []
    for p in sorted(root.glob("*.h5")):
        if p.name in EXCLUDED:
            continue
        rows.append(ratios(p))
    if not rows:
        return None
    arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows])

    print(f"\n{label}: {len(rows)} trajectories")
    print(f"  {'AoA':>5}{'cases':>7}{'rms div':>11}{'rms vort':>11}{'div/vort':>11}")
    print("  " + "-" * 45)
    for a in sorted(set(arr[:, 1].tolist())):
        sel = arr[:, 1] == a
        print(f"  {a:>5.0f}{sel.sum():>7}{arr[sel, 2].mean():>11.3f}"
              f"{arr[sel, 3].mean():>11.3f}{arr[sel, 4].mean():>11.3f}")

    c = np.corrcoef(arr[:, 1], arr[:, 4])[0, 1]
    print(f"  corr(AoA, div/vort) = {c:+.3f}")
    c2 = np.corrcoef(arr[:, 0], arr[:, 4])[0, 1]
    print(f"  corr(Re,  div/vort) = {c2:+.3f}")
    return arr


def main() -> None:
    print("dw/dz = -(du/dx + dv/dy): the in-plane divergence measures flow")
    print("leaving the measurement plane. Reported relative to in-plane vorticity")
    print("so the number is dimensionless.")

    real = survey(REAL, "REAL (PIV)")
    sim = survey(SIM, "SIMULATION (CFD)")

    if real is not None and sim is not None:
        print("\n" + "=" * 60)
        print("REAL versus SIMULATION")
        print("=" * 60)
        print(f"  mean div/vort   real {real[:, 4].mean():.3f}   "
              f"sim {sim[:, 4].mean():.3f}")
        print("\n  The real split's ratio also contains PIV noise, which inflates any")
        print("  derivative. The simulation's ratio is the cleaner estimate of how")
        print("  three-dimensional the flow genuinely is; the gap between them is")
        print("  roughly what measurement adds.")


if __name__ == "__main__":
    main()
