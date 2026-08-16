"""Fit the SPS interval widths on TRAINING residuals, then test them on validation.

sps_diagnose.py fitted the per-element sigma on the validation split itself and
reached SPS 48.97. That number is not deliverable: at submission time there are no
targets, so sigma has to come from data the model was fitted on and then survive
the move to unseen Reynolds numbers.

Two estimators, both computable at inference:

  fixed      one sigma per (output frame, cell, channel), averaged over the whole
             training split. The error field is strongly structured in space --
             it lives in the wake -- and grows with lead time, so a static map
             already captures most of it.

  scaled     the same map, multiplied per sample by how energetic that window is
             relative to the training average. Absolute error grows with Reynolds
             number, and the input window's own velocity spread is a proxy for Re
             that is available at inference (metadata is empty on scored calls).

The gap between the validation-fitted number and these is the honest cost of not
knowing the answer in advance.

Usage:
    python scripts/calibrate_sps.py --checkpoint checkpoints/adv_finetuned_best.pt
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
from realpde.sps import bounds_from_sigma

ROOT = Path(__file__).resolve().parent.parent


def predict_all(model, ds, batch=32, max_windows=None):
    n = len(ds) if max_windows is None else min(len(ds), max_windows)
    idx = np.linspace(0, len(ds) - 1, n).astype(int) if max_windows else range(n)
    xs = torch.stack([ds[i]["input"] for i in idx])
    ys = torch.stack([ds[i]["target"] for i in idx])
    with torch.no_grad():
        ps = torch.cat([model(xs[i:i + batch]) for i in range(0, len(xs), batch)])
    return (denormalize(xs, 2).numpy(), denormalize(ps, 2).numpy(),
            denormalize(ys, 2).numpy())


def energy(x):
    """Per-sample scalar tracking how energetic a window is — a stand-in for Re.

    The input's velocity spread, which the model always has. Reynolds number
    itself is not available at inference: metadata is empty on scored calls.
    """
    return x.reshape(len(x), -1).std(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/adv_finetuned_best.pt"))
    ap.add_argument("--train-windows", type=int, default=1200,
                    help="training windows sampled to fit sigma (they overlap heavily)")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "sps_sigma.npz")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="amplitude correction the submission applies; sigma must "
                         "be fitted to the corrected prediction, not the raw one")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    train, val = build_datasets("real")
    print(f"checkpoint {args.checkpoint.name}")
    print(f"fitting sigma on {args.train_windows} of {len(train)} training windows\n")

    def alpha_correct(p):
        if args.alpha == 1.0:
            return p
        m = p.mean(axis=1, keepdims=True)
        return m + args.alpha * (p - m)

    xt, pt, yt = predict_all(model, train, max_windows=args.train_windows)
    pt = alpha_correct(pt)
    sigma_map = (pt - yt).std(axis=0, keepdims=True)        # (1, T, H, W, C)
    train_energy = float(energy(xt).mean())

    xv, pv, yv = predict_all(model, val)
    pv = alpha_correct(pv)
    scored = yv != 0.0
    if args.alpha != 1.0:
        print(f"amplitude correction alpha = {args.alpha} applied before fitting sigma\n")

    print(f"{'estimator':<34}{'sps':>8}{'coverage':>11}{'mean width':>12}")
    print("-" * 65)

    base = score_arrays(pv, yv, mean_t_neural_s=0.005)
    print(f"{'default band (what we submitted)':<34}{base['sps_score']:>8.2f}"
          f"{base['_sps_coverage']:>11.3f}{0.1 * np.abs(pv)[scored].mean():>12.5f}")

    results = {}
    for scale in (0.8, 1.0, 1.2, 1.5):
        s = np.broadcast_to(sigma_map, pv.shape)
        lo, hi = bounds_from_sigma(pv, s, scale=scale)
        r = score_arrays(pv, yv, mean_t_neural_s=0.005, lower=lo, upper=hi)
        results[f"fixed map x{scale}"] = r
        print(f"{f'train-fitted, fixed x{scale}':<34}{r['sps_score']:>8.2f}"
              f"{r['_sps_coverage']:>11.3f}{(hi - lo)[scored].mean():>12.5f}")

    ev = energy(xv) / train_energy
    for scale in (0.8, 1.0, 1.2, 1.5):
        s = sigma_map * ev.reshape(-1, 1, 1, 1, 1)
        lo, hi = bounds_from_sigma(pv, s, scale=scale)
        r = score_arrays(pv, yv, mean_t_neural_s=0.005, lower=lo, upper=hi)
        results[f"scaled x{scale}"] = r
        print(f"{f'train-fitted, energy-scaled x{scale}':<34}{r['sps_score']:>8.2f}"
              f"{r['_sps_coverage']:>11.3f}{(hi - lo)[scored].mean():>12.5f}")

    # The cheating reference: sigma fitted on validation itself.
    s_val = np.broadcast_to((pv - yv).std(axis=0, keepdims=True), pv.shape)
    lo, hi = bounds_from_sigma(pv, s_val, scale=0.8)
    r = score_arrays(pv, yv, mean_t_neural_s=0.005, lower=lo, upper=hi)
    print(f"\n{'val-fitted x0.8 (NOT deliverable)':<34}{r['sps_score']:>8.2f}"
          f"{r['_sps_coverage']:>11.3f}{(hi - lo)[scored].mean():>12.5f}")

    best = max(results.items(), key=lambda kv: kv[1]["sps_score"])
    print(f"\nbest deliverable: {best[0]} -> sps {best[1]['sps_score']:.2f} "
          f"(default was {base['sps_score']:.2f}, gain {best[1]['sps_score'] - base['sps_score']:+.2f})")

    np.savez_compressed(args.out, sigma=sigma_map.astype(np.float32),
                        train_energy=np.array([train_energy], dtype=np.float32))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
