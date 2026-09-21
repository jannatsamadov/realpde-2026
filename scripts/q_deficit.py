"""Where does the sps interval term lose its points -- which cells, which lead time?

sps = 100 * acc * Q, and 6oo showed that Q is what responds to accuracy: a 20%
smaller residual is worth +4.0 sps, against +0.30 for a perfect per-window sigma.
So the question is no longer how to set the bands but which part of the field is
keeping Q down, because that is where a better forecast has to land.

Q is a mean over elements of exp(-nil) when covered and 0 when not, so every
element's contribution is directly attributable. This splits that mean by lead
time and by region, taking the wake as the cells whose fluctuation energy is in
the top third -- the same split the error maps show.

Reads the cache sps_ceiling.py leaves behind, so it costs nothing to rerun.

Usage:
    python scripts/q_deficit.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from realpde.sps import SIGMA_GLOBAL, bounds_from_sigma

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "checkpoints" / "sps_ceiling_cache"
WIDTHS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)


def q_elements(pred, targ, sigma, width):
    """Per-element interval quality, the quantity Q averages."""
    lo, up = bounds_from_sigma(pred, sigma, scale=width)
    nil = (up - lo) / SIGMA_GLOBAL
    covered = (targ >= lo) & (targ <= up)
    return np.exp(-nil) * covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="adv_finetuned_best_default.npz")
    args = ap.parse_args()

    d = np.load(CACHE / args.cache)
    pt, yt, pv, yv = d["pt"], d["yt"], d["pv"], d["yv"]
    print(f"{args.cache}: {len(pt)} train / {len(pv)} val windows\n")

    fixed = (pt - yt).std(axis=0).astype(np.float32)
    sigma = np.broadcast_to(fixed, pv.shape)

    # Pick the width the submission would pick, then attribute at that width.
    best_w, best_q = None, -1.0
    for w in WIDTHS:
        q = q_elements(pv, yv, sigma, w)
        if q.mean() > best_q:
            best_w, best_q = w, q.mean()
    q = q_elements(pv, yv, sigma, best_w)
    print(f"band width {best_w}, Q = {q.mean():.4f}   "
          f"(perfect would be 1.0, so {1 - q.mean():.4f} is the deficit)\n")

    # Masked cells read exactly zero in the target and are not scored.
    scored = ~((yv[..., 0] == 0) & (yv[..., 1] == 0))
    qs = q.mean(axis=-1)                         # average the two channels
    print(f"{100 * scored.mean():.1f}% of elements are scored\n")

    # --- by lead time -------------------------------------------------------
    t_out = pv.shape[1]
    print("by lead time")
    print(f"{'lead':>5} {'Q':>8} {'coverage':>10} {'mean nil':>10}")
    lo, up = bounds_from_sigma(pv, sigma, scale=best_w)
    nil = ((up - lo) / SIGMA_GLOBAL).mean(axis=-1)
    cov = ((yv >= lo) & (yv <= up)).all(axis=-1)
    for t in range(t_out):
        m = scored[:, t]
        print(f"{t + 1:>5} {qs[:, t][m].mean():>8.4f} {cov[:, t][m].mean():>10.4f} "
              f"{nil[:, t][m].mean():>10.4f}")

    # --- by region ----------------------------------------------------------
    # Wake = cells whose target fluctuation energy is in the top third.
    fluct = yv - yv.mean(axis=1, keepdims=True)
    energy = (fluct ** 2).sum(axis=-1).mean(axis=(0, 1))       # (H, W)
    flat = energy.ravel()
    lo_cut, hi_cut = np.quantile(flat[flat > 0], [1 / 3, 2 / 3])
    band = np.digitize(energy, [lo_cut, hi_cut])               # 0,1,2
    names = {0: "freestream", 1: "middle", 2: "wake"}

    print("\nby region (target fluctuation energy terciles)")
    print(f"{'region':>12} {'cells':>7} {'share':>7} {'Q':>8} {'coverage':>10} "
          f"{'mean nil':>10} {'lost Q':>8}")
    total_deficit = 1.0 - qs[scored].mean()
    for b in (0, 1, 2):
        cell = band == b
        m = scored & cell[None, None, :, :]
        if m.sum() == 0:
            continue
        share = m.sum() / scored.sum()
        qb = qs[m].mean()
        # How much of the overall deficit this region accounts for.
        lost = share * (1.0 - qb) / total_deficit
        print(f"{names[b]:>12} {cell.sum():>7} {share:>7.3f} {qb:>8.4f} "
              f"{cov[m].mean():>10.4f} {nil[m].mean():>10.4f} {100 * lost:>7.1f}%")

    print(f"\ntotal Q deficit {total_deficit:.4f}; the 'lost Q' column says which")
    print("region a better forecast has to improve to move sps at all.")

    # --- is a spatially weighted loss worth it? (idea C5) -------------------
    # A loss weighted toward the wake buys wake accuracy by giving up accuracy
    # elsewhere. Whether that trade pays depends on how much sps each region's
    # error is worth at the margin, which is measurable directly: shrink the
    # residual in ONE region, refit sigma, rescore, and divide the gain by the
    # total error reduction it took. Equal ratios mean a uniform loss is already
    # right and C5 is dead; a large ratio means the metric pays more there.
    from realpde.local_score import score_arrays

    def sps_after(mask_hw, g):
        """Shrink the residual by g inside mask_hw only, refit sigma, rescore."""
        m = mask_hw[None, None, :, :, None]
        p2 = np.where(m, yt + (pt - yt) * g, pt)
        q2 = np.where(m, yv + (pv - yv) * g, pv)
        sig = np.broadcast_to((p2 - yt).std(axis=0).astype(np.float32), q2.shape)
        best = -1.0
        for w in WIDTHS:
            b_lo, b_up = bounds_from_sigma(q2, sig, scale=w)
            r = score_arrays(q2, yv, mean_t_neural_s=1e-3, lower=b_lo, upper=b_up)
            best = max(best, r["sps_score"])
        err = np.sqrt((((q2 - yv) ** 2)).mean())
        return best, err

    base_sps, base_err = sps_after(np.ones_like(band, dtype=bool), 1.0)
    print(f"\nmarginal value of accuracy BY REGION (residual x0.8 in one region)")
    print(f"{'region':>12} {'d sps':>9} {'d rms err':>11} {'sps per 1% err':>16}")
    for b in (0, 1, 2):
        s, e = sps_after(band == b, 0.8)
        drop = 100 * (base_err - e) / base_err
        print(f"{names[b]:>12} {s - base_sps:>+9.4f} {base_err - e:>11.6f} "
              f"{(s - base_sps) / max(drop, 1e-9):>16.4f}")
    s, e = sps_after(np.ones_like(band, dtype=bool), 0.8)
    drop = 100 * (base_err - e) / base_err
    print(f"{'everywhere':>12} {s - base_sps:>+9.4f} {base_err - e:>11.6f} "
          f"{(s - base_sps) / max(drop, 1e-9):>16.4f}")
    print("\nIf one region's 'sps per 1% err' clearly beats 'everywhere', a loss")
    print("weighted toward it is worth training; if they are close, C5 is dead.")


if __name__ == "__main__":
    main()
