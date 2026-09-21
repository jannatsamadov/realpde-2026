"""Is a per-window speed statistic a usable stand-in for the freestream velocity?

Scale conditioning divides each window by its own velocity scale, so that
statistic has to be (a) computable from the 20 input frames alone -- metadata is
empty at inference -- and (b) proportional to Re, since U_inf = Re * 1.394e-5.

Three candidates are compared, all over unmasked cells only (roughly 9% of the
field is masked and reads exactly zero, which would drag a plain mean down by a
regime-dependent amount):

    mean_speed  mean of sqrt(u^2+v^2)
    rms_speed   root mean square of the same
    q90_speed   90th percentile, Qwen's suggestion

What matters is not which is closest to the true U_inf -- any consistent,
monotone statistic non-dimensionalises the problem equally well -- but which has
the tightest proportionality to Re, because a noisy scale estimate injects
noise into every channel the network sees.

Usage:
    python scripts/check_scale_stat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from realpde.data import T_IN, T_OUT, U_PER_RE, load_split

WINDOW_STRIDE = 40   # disjoint windows, as validation uses


def stats_for_window(w: np.ndarray) -> dict:
    """w: (T, H, W, 2) physical m/s -> the three candidate scale statistics."""
    speed = np.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2)
    valid = speed > 0
    if valid.sum() == 0:
        return {}
    s = speed[valid]
    return {"mean_speed": float(s.mean()),
            "rms_speed": float(np.sqrt((s ** 2).mean())),
            "q90_speed": float(np.quantile(s, 0.90))}


def main() -> None:
    arr, meta = load_split("real")
    span = T_IN + T_OUT

    re_list, aoa_list, rows = [], [], []
    for t in meta["trajectories"]:
        o, L = t["offset"], t["length"]
        for s in range(0, L - span + 1, WINDOW_STRIDE):
            w = np.asarray(arr[o + s:o + s + T_IN], dtype=np.float32)
            st = stats_for_window(w)
            if not st:
                continue
            rows.append(st)
            re_list.append(float(t["re_nominal"]))
            aoa_list.append(float(t.get("aoa_nominal", t.get("aoa", 0.0))))

    re = np.array(re_list)
    aoa = np.array(aoa_list)
    print(f"{len(rows)} disjoint windows over {len(meta['trajectories'])} trajectories\n")

    print(f"{'statistic':>12} {'corr(.,Re)':>11} {'log-log slope':>14} "
          f"{'scatter':>9} {'range':>9}")
    for key in ("mean_speed", "rms_speed", "q90_speed"):
        v = np.array([r[key] for r in rows])
        c = np.corrcoef(re, v)[0, 1]
        slope, intercept = np.polyfit(np.log(re), np.log(v), 1)
        # Residual scatter about the fit, in per cent -- this is the noise the
        # scale estimate would inject into the normalised input.
        resid = np.log(v) - (slope * np.log(re) + intercept)
        print(f"{key:>12} {c:>11.4f} {slope:>14.4f} "
              f"{100 * resid.std():>8.1f}% {v.max() / v.min():>8.1f}x")

    # The chosen statistic, per regime, and what dividing by it actually buys.
    v = np.array([r["mean_speed"] for r in rows])
    print(f"\nmean_speed vs the measured freestream U_inf = Re * {U_PER_RE}:")
    u_inf = re * U_PER_RE
    ratio = v / u_inf
    print(f"  ratio mean_speed / U_inf : mean {ratio.mean():.3f} "
          f"sd {ratio.std():.3f}  ({100 * ratio.std() / ratio.mean():.1f}% spread)")

    print(f"\nspread of the raw field magnitude across windows : "
          f"{v.max() / v.min():.1f}x")
    print(f"spread after dividing by mean_speed             : 1.0x by construction")

    # Per angle of attack, since the statistic mixes freestream and wake and the
    # wake fraction grows with angle.
    print(f"\n{'AoA':>5} {'n':>5} {'mean_speed/U_inf':>18} {'sd':>7}")
    for a in np.unique(aoa):
        m = aoa == a
        print(f"{a:>5.1f} {m.sum():>5} {ratio[m].mean():>18.3f} {ratio[m].std():>7.3f}")

    # Within one trajectory, how much does the statistic wander window to window?
    # A scale estimate that jitters inside a fixed regime is estimator noise.
    print("\nwithin-trajectory jitter of mean_speed (should be small):")
    jit = []
    idx = 0
    for t in meta["trajectories"]:
        n = len(range(0, t["length"] - span + 1, WINDOW_STRIDE))
        if n >= 3:
            seg = v[idx:idx + n]
            jit.append(seg.std() / seg.mean())
        idx += n
    jit = np.array(jit)
    print(f"  median {100 * np.median(jit):.2f}%   max {100 * jit.max():.2f}%")


if __name__ == "__main__":
    main()
