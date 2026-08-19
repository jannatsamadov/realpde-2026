"""Interval widths chosen from the observed residuals, with no distribution assumed.

The current bounds solve the Gaussian optimality condition

    phi(r) / (2 Phi(r) - 1) = s / SIGMA_GLOBAL,   w = 2 s r

and then multiply the result by 0.8. That 0.8 is not a physical constant: it is
an admission that the residuals are heavier-tailed than the Gaussian the
derivation assumes, tuned back on validation. Two approximations stacked.

But the objective the scorer actually pays is

    g(w) = exp(-w / SIGMA_GLOBAL) * P(|e| <= w / 2)

and we have ~1200 training residuals per element, which is plenty to use the
EMPIRICAL distribution instead of a fitted one. With the empirical CDF, F is a
step function, so g only ever increases at a jump: the maximiser is exactly one
of the sorted |e| values. For element residuals sorted ascending,

    g_k = exp(-2 |e|_(k) / SIGMA_GLOBAL) * (k / N)

and the optimal half-width is |e|_(k*). No Gaussian, no fudge factor, and the
argmax is exact rather than interpolated.

Masked cells (outside the PIV field of view, inside the airfoil) are stored as
exact zeros in m/s and are not scored, so they are dropped per element rather
than counted as tiny residuals — counting them would pull every width down.

Usage:
    python scripts/fit_empirical_bounds.py
    python scripts/fit_empirical_bounds.py --train-windows 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from realpde.data import build_datasets, denormalize      # noqa: E402
from realpde.local_score import score_arrays              # noqa: E402
from realpde.models import build_model                    # noqa: E402
from realpde.sps import SIGMA_GLOBAL, bounds_from_sigma   # noqa: E402
from gpu_guard import Governor                            # noqa: E402
from gpu_lock import GpuLock                              # noqa: E402


def predict(model, ds, idx, device, batch=32):
    xs = torch.stack([ds[int(i)]["input"] for i in idx])
    ys = torch.stack([ds[int(i)]["target"] for i in idx])
    outs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            outs.append(model(xs[i:i + batch].to(device)).float().cpu())
    return (denormalize(torch.cat(outs), 2).numpy(),
            denormalize(ys, 2).numpy())


# Coverage levels at which the residual quantiles are stored. The optimum sits
# near 0.85 with a correct prior, but Track 2 rescales the residual estimate
# online and the optimum then moves, so the grid has to span the range it can
# move through rather than just bracket the offline answer.
LEVELS = np.linspace(0.30, 0.999, 48)


def empirical_half_width(resid: np.ndarray, valid: np.ndarray,
                         chunk: int = 4096):
    """Per-element half-width maximising exp(-2a/SIGMA) * F_hat(a), exactly.

    resid : (N, T, H, W, C) prediction minus target, m/s
    valid : (N, T, H, W, C) bool, elements that are actually scored

    Returns (half_width, quantiles):
      half_width : (1, T, H, W, C) the optimum at the residual scale as fitted
      quantiles  : (len(LEVELS), T, H, W, C) |residual| at each level in LEVELS

    The quantile table is what makes the width recomputable later. If the true
    residuals turn out a times larger than these — which is exactly what the
    hidden set did to us, by about 1.49x — the optimal width is NOT a times the
    old one, because exp(-w/SIGMA_GLOBAL) has a fixed scale while the coverage
    term does not. Keeping the quantiles lets the objective be re-maximised at
    any a, for 10 MB instead of the 400 MB the raw residuals would cost.
    """
    shape = resid.shape[1:]
    a = np.abs(resid).reshape(len(resid), -1)
    v = valid.reshape(len(valid), -1)
    n_el = a.shape[1]
    out = np.zeros(n_el, dtype=np.float64)
    quant = np.zeros((len(LEVELS), n_el), dtype=np.float32)

    for s in range(0, n_el, chunk):
        e = s + min(chunk, n_el - s)
        col = a[:, s:e].astype(np.float64)
        vc = v[:, s:e]
        # Invalid samples must not occupy a rank, so push them to +inf: they
        # sort last and the running count never reaches them.
        col = np.where(vc, col, np.inf)
        col.sort(axis=0)
        counts = vc.sum(axis=0)                       # valid samples per element
        k = np.arange(1, len(col) + 1, dtype=np.float64)[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            g = np.exp(-2.0 * col / SIGMA_GLOBAL) * (k / np.maximum(counts, 1))
        g[~np.isfinite(col)] = -1.0                   # never pick a padded rank
        best = np.argmax(g, axis=0)
        cols = np.arange(col.shape[1])
        chosen = col[best, cols]
        # An element with no valid sample anywhere gets a zero-width interval,
        # which is correct: it is never scored.
        out[s:e] = np.where(counts > 0, chosen, 0.0)

        # Quantiles read off the same sorted column, by rank among the VALID
        # samples only, so a masked element does not shift the levels.
        ranks = np.clip(np.ceil(LEVELS[:, None] * counts[None, :]).astype(int) - 1,
                        0, max(len(col) - 1, 0))
        q = col[ranks, cols[None, :]]
        quant[:, s:e] = np.where(counts[None, :] > 0, np.nan_to_num(q, posinf=0.0), 0.0)

    return out.reshape((1,) + shape), quant.reshape((len(LEVELS),) + shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--train-windows", type=int, default=1200)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "checkpoints" / "sps_empirical.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "advective"), base=targs.get("base", 64),
                        channels=2)
    model.load_state_dict(state["model"])
    model.eval().to(device)

    train, val = build_datasets("real")
    idx_t = np.linspace(0, len(train) - 1,
                        min(args.train_windows, len(train))).astype(int)
    print(f"{args.checkpoint.name}: fitting on {len(idx_t)} train windows, "
          f"testing on {len(val)} val windows\n")

    lock = GpuLock("realpde", "empirical_bounds").acquire() if device.type == "cuda" else None
    gov = Governor(target_util=72, pause_temp=78, resume_temp=70) if device.type == "cuda" else None
    try:
        step = gov.step() if gov is not None else None
        if step is not None:
            with step:
                pt, yt = predict(model, train, idx_t, device)
        else:
            pt, yt = predict(model, train, idx_t, device)
        if lock is not None:
            lock.progress("val")
        pv, yv = predict(model, val, range(len(val)), device)
    finally:
        if lock is not None:
            lock.release()

    valid_t = yt != 0.0
    scored_v = yv != 0.0

    # Reference: the Gaussian solve currently in the submission.
    sigma = (pt - yt).std(axis=0, keepdims=True)
    print(f"{'method':<34}{'sps':>8}{'coverage':>11}{'mean width':>12}")
    print("-" * 65)
    base = score_arrays(pv, yv, mean_t_neural_s=0.005)
    print(f"{'default band':<34}{base['sps_score']:>8.2f}"
          f"{base['_sps_coverage']:>11.3f}{0.1 * np.abs(pv)[scored_v].mean():>12.5f}")

    best_gauss = None
    for scale in (0.8, 1.0):
        lo, hi = bounds_from_sigma(pv, np.broadcast_to(sigma, pv.shape), scale=scale)
        r = score_arrays(pv, yv, mean_t_neural_s=0.005, lower=lo, upper=hi)
        tag = f"gaussian sigma x{scale}"
        print(f"{tag:<34}{r['sps_score']:>8.2f}{r['_sps_coverage']:>11.3f}"
              f"{(hi - lo)[scored_v].mean():>12.5f}")
        if scale == 0.8:
            best_gauss = r

    half, quant = empirical_half_width(pt - yt, valid_t)
    print(f"\nempirical half-width: mean {half[half > 0].mean():.5f} m/s, "
          f"vs gaussian x0.8 {0.5 * (bounds_from_sigma(pv, np.broadcast_to(sigma, pv.shape), scale=0.8)[1] - bounds_from_sigma(pv, np.broadcast_to(sigma, pv.shape), scale=0.8)[0])[scored_v].mean():.5f}\n")

    results = {}
    for scale in (0.8, 0.9, 1.0, 1.1, 1.25):
        h = np.broadcast_to(half * scale, pv.shape).astype(np.float32)
        lo, hi = pv - h, pv + h
        r = score_arrays(pv, yv, mean_t_neural_s=0.005, lower=lo, upper=hi)
        results[scale] = r
        tag = f"EMPIRICAL x{scale}"
        print(f"{tag:<34}{r['sps_score']:>8.2f}{r['_sps_coverage']:>11.3f}"
              f"{(hi - lo)[scored_v].mean():>12.5f}")

    best_scale = max(results, key=lambda k: results[k]["sps_score"])
    gain = results[best_scale]["sps_score"] - best_gauss["sps_score"]
    print(f"\nbest empirical x{best_scale} -> sps {results[best_scale]['sps_score']:.2f}, "
          f"against gaussian x0.8 {best_gauss['sps_score']:.2f}  ({gain:+.2f})")
    print(f"on the fitted marginal weight w_sps = 0.30 that is {0.30 * gain:+.3f} final")

    # Sanity check the quantile table reproduces the direct argmax at a = 1:
    # if it does not, the shipped bounds would silently differ from what was
    # measured here.
    obj = np.exp(-2.0 * quant / SIGMA_GLOBAL) * LEVELS[:, None, None, None, None]
    from_table = np.take_along_axis(quant, obj.argmax(axis=0)[None], axis=0)
    h_tab = np.broadcast_to(from_table, pv.shape).astype(np.float32)
    r_tab = score_arrays(pv, yv, mean_t_neural_s=0.005,
                         lower=pv - h_tab, upper=pv + h_tab)
    print(f"\nquantile table at a=1 reproduces sps {r_tab['sps_score']:.2f} "
          f"(direct argmax gave {results[1.0]['sps_score']:.2f}) — "
          f"{len(LEVELS)} levels, {quant.nbytes / 1024**2:.1f} MB uncompressed")

    np.savez_compressed(args.out, half_width=half.astype(np.float32),
                        quantiles=quant, levels=LEVELS.astype(np.float32),
                        best_scale=np.array([best_scale], dtype=np.float32))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
