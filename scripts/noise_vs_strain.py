"""Does PIV noise track the local strain rate?

PIV recovers velocity by correlating particle images between two frames inside a
small interrogation window. When the particles inside one window move at
different speeds — that is, where the strain rate is high — the correlation peak
broadens and the measurement degrades. This is a documented PIV error source,
usually called velocity-gradient bias.

If it holds in this data, the measurement noise is predictable from quantities
the model already sees, and the SPS bounds could be sized per window instead of
from one fixed map. That fixed map is the weakest part of the current submission:
the hidden set's residuals ran about 1.49x larger than ours and the bands were
too narrow there.

An earlier measurement looked like evidence AGAINST this: the noise share of
variance is 13.9% in the wake and 35.3% in the freestream — lower where the
strain is higher. But that is a share, not an amount. The freestream has almost
no real fluctuation, so even a small absolute noise dominates its variance. This
script measures the ABSOLUTE noise instead, which is the quantity that matters
for sizing an interval.

Noise is estimated per cell from the lag-1 correlation of the fluctuation: a
physical signal at 50 Hz is correlated between consecutive frames, measurement
noise is not, so

    var_noise = var_total * (1 - r1)

Usage:
    python scripts/noise_vs_strain.py
    python scripts/noise_vs_strain.py --cases 10125_10 20325_15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "NeurIPS 2026 RealPDE Competition" / "train_real" / "train_real"
DX = 0.001711


def gradients(u, v, dx=DX):
    return (np.gradient(u, dx, axis=-1), np.gradient(u, -dx, axis=-2),
            np.gradient(v, dx, axis=-1), np.gradient(v, -dx, axis=-2))


def strain_and_q(u, v):
    """Frobenius norm of the strain-rate tensor, and the Q-criterion."""
    a, b, c, d = gradients(u, v)
    s12 = 0.5 * (b + c)
    s_norm2 = a ** 2 + d ** 2 + 2.0 * s12 ** 2
    w_norm2 = 2.0 * (0.5 * (c - b)) ** 2
    return np.sqrt(s_norm2), 0.5 * (w_norm2 - s_norm2)


def noise_variance(f):
    """Per-cell noise variance from the lag-1 correlation. f is (T, H, W)."""
    x = f - f.mean(axis=0, keepdims=True)
    a, b = x[:-1], x[1:]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    r1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-14)
    return x.var(axis=0) * np.clip(1.0 - r1, 0.0, 1.0), x.var(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+",
                    default=["6300_10", "10125_10", "13950_15", "20325_15", "26700_20"])
    ap.add_argument("--frames", type=int, default=500)
    args = ap.parse_args()

    all_noise, all_strain, all_q, all_var = [], [], [], []

    print(f"{'case':<12}{'corr(noise,strain)':>20}{'corr(noise,|Q|)':>18}"
          f"{'noise wake':>12}{'noise free':>12}{'ratio':>8}")
    print("-" * 82)

    for case in args.cases:
        path = DATA / f"{case}.h5"
        if not path.exists():
            print(f"{case:<12}  (yoxdur)")
            continue
        with h5py.File(path, "r") as f:
            n = min(args.frames, f["u"].shape[0])
            u = np.asarray(f["u"][:n], dtype=np.float64)
            v = np.asarray(f["v"][:n], dtype=np.float64)

        measured = ~(((u == 0) & (v == 0)).any(axis=0))
        nv_u, var_u = noise_variance(u)
        nv_v, var_v = noise_variance(v)
        noise = np.sqrt(0.5 * (nv_u + nv_v))          # noise sd, m/s
        total = np.sqrt(0.5 * (var_u + var_v))

        s, q = strain_and_q(u, v)
        s_mean = s.mean(axis=0)                        # time-averaged strain rate
        q_abs = np.abs(q).mean(axis=0)

        m = measured & np.isfinite(noise) & np.isfinite(s_mean)
        c_s = np.corrcoef(noise[m], s_mean[m])[0, 1]
        c_q = np.corrcoef(noise[m], q_abs[m])[0, 1]

        # Wake versus freestream by fluctuation energy, as before.
        hi = total > np.percentile(total[m], 75)
        lo = total < np.percentile(total[m], 25)
        n_wake = noise[m & hi].mean()
        n_free = noise[m & lo].mean()

        print(f"{case:<12}{c_s:>20.3f}{c_q:>18.3f}"
              f"{n_wake:>12.5f}{n_free:>12.5f}{n_wake / n_free:>8.2f}")

        all_noise.append(noise[m]); all_strain.append(s_mean[m])
        all_q.append(q_abs[m]); all_var.append(total[m])

    if not all_noise:
        return
    noise = np.concatenate(all_noise); strain = np.concatenate(all_strain)
    q_abs = np.concatenate(all_q); total = np.concatenate(all_var)

    print("\n" + "=" * 82)
    print("POOLED OVER ALL CASES")
    print("=" * 82)
    print(f"  corr(absolute noise, strain rate) : {np.corrcoef(noise, strain)[0, 1]:+.3f}")
    print(f"  corr(absolute noise, |Q|)         : {np.corrcoef(noise, q_abs)[0, 1]:+.3f}")
    print(f"  corr(absolute noise, total sd)    : {np.corrcoef(noise, total)[0, 1]:+.3f}")

    print(f"\n  absolute noise by strain quintile:")
    edges = np.percentile(strain, [0, 20, 40, 60, 80, 100])
    print(f"  {'strain range [1/s]':<26}{'mean noise sd [m/s]':>22}{'n cells':>10}")
    print("  " + "-" * 58)
    for i in range(5):
        sel = (strain >= edges[i]) & (strain <= edges[i + 1])
        print(f"  {edges[i]:>8.2f} - {edges[i + 1]:<14.2f}"
              f"{noise[sel].mean():>22.5f}{sel.sum():>10,}")

    lo, hi = noise[strain <= edges[1]].mean(), noise[strain >= edges[4]].mean()
    print(f"\n  noise in the top strain quintile is {hi / lo:.2f}x the bottom quintile")
    print("\n  A strong positive correlation would mean the interval width could be")
    print("  computed from the input window instead of read off a fixed map — which")
    print("  is exactly what the hidden set punished.")


if __name__ == "__main__":
    main()
