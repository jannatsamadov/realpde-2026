"""Why sps_score is 16.6, and what calibrated bounds would give instead.

The submission returned no lower/upper, so the scorer fell back to its default
band of +/- 5% of |prediction|. That width is a property of the prediction's
magnitude, not of how wrong the prediction actually is — so it is far too narrow
where the model is uncertain and needlessly wide where the model is confident.

This measures the residual the model actually makes, compares it to the default
band, and computes what the analytically optimal width from realpde.sps would
score instead.

Usage:
    python scripts/sps_diagnose.py --checkpoint checkpoints/tke0_best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.local_score import score_arrays
from realpde.models import build_model
from realpde.sps import SIGMA_GLOBAL, bounds_from_sigma, optimal_width


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tke0_best.pt"))
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, val = build_datasets("real")
    n = len(val)
    X = torch.stack([val[i]["input"] for i in range(n)])
    Y = torch.stack([val[i]["target"] for i in range(n)])
    with torch.no_grad():
        P = torch.cat([model(X[i:i + 32]) for i in range(0, n, 32)])
    p = denormalize(P, 2).numpy()
    t = denormalize(Y, 2).numpy()

    resid = p - t
    scored = t != 0.0                       # exactly the elements SPS counts

    print(f"{n} validation windows, {100 * scored.mean():.1f}% of elements are scored\n")

    print("1. THE MODEL'S ACTUAL ERROR")
    r = np.abs(resid[scored])
    print(f"  |residual|  mean {r.mean():.5f}   median {np.median(r):.5f}   "
          f"p90 {np.percentile(r, 90):.5f}   p99 {np.percentile(r, 99):.5f}")
    print(f"  residual std                      {resid[scored].std():.5f}")
    print(f"  SIGMA_GLOBAL (the scorer's scale) {SIGMA_GLOBAL:.5f}")

    print("\n2. THE DEFAULT BAND THE SCORER USED")
    half_def = 0.05 * np.abs(p)
    print(f"  half-width  mean {half_def[scored].mean():.5f}   "
          f"median {np.median(half_def[scored]):.5f}")
    cov_def = (np.abs(resid) <= half_def)[scored].mean()
    print(f"  coverage: {100 * cov_def:.1f}% of scored elements fall inside")
    print(f"  -> {100 * (1 - cov_def):.1f}% of elements score exactly ZERO")
    print("  The band is a fraction of |prediction|, so it is narrow exactly where")
    print("  the velocity is small — which is the wake, where the error is largest.")

    print("\n3. WHAT CALIBRATED BOUNDS WOULD GIVE")
    # A per-element sigma from the model's own error statistics, estimated per
    # (frame, cell) across the validation windows. On a submission this has to
    # come from the training split instead, since targets are not available.
    sigma_cell = resid.std(axis=0, keepdims=True)          # (1, T, H, W, C)
    sigma_cell = np.broadcast_to(sigma_cell, resid.shape)

    results = {}
    lo, hi = bounds_from_sigma(p, sigma_cell)
    results["optimal width (analytic)"] = (lo, hi)
    for scale in (0.6, 0.8, 1.2, 1.6, 2.0, 3.0):
        lo_s, hi_s = bounds_from_sigma(p, sigma_cell, scale=scale)
        results[f"optimal x {scale}"] = (lo_s, hi_s)
    # A single global width, for reference: does per-element actually matter?
    g = float(resid[scored].std())
    w_g = float(optimal_width(np.array([g]))[0])
    results["single global width"] = (p - w_g / 2, p + w_g / 2)

    base = score_arrays(p, t, mean_t_neural_s=0.005)
    print(f"  {'bounds':<28}{'sps':>8}{'coverage':>11}{'mean width':>12}")
    print("  " + "-" * 59)
    print(f"  {'default (no bounds sent)':<28}{base['sps_score']:>8.2f}"
          f"{base['_sps_coverage']:>11.3f}{2 * half_def[scored].mean():>12.5f}")
    for name, (lo_, hi_) in results.items():
        s = score_arrays(p, t, mean_t_neural_s=0.005, lower=lo_, upper=hi_)
        print(f"  {name:<28}{s['sps_score']:>8.2f}{s['_sps_coverage']:>11.3f}"
              f"{(hi_ - lo_)[scored].mean():>12.5f}")

    print("\n  Note: sigma_cell here is fitted on the validation split itself, so these")
    print("  numbers are optimistic. A submission must estimate it from training data.")


if __name__ == "__main__":
    main()
