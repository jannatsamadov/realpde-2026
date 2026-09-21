"""Where is sps_score actually won -- in the bands, or in the forecast?

sps = 100 * acc * Q, with acc set by the forecast's accuracy and Q = mean over
elements of exp(-nil) when covered. The bands only touch Q, so the whole
calibration programme has been an attempt to raise Q.

Two oracles have now been measured and both came back nearly empty:

    per-window sigma, known exactly    +0.30 sps   (test_f1_regime_bands.py)
    per-lead-time sigma, known exactly  +0.01 sps   (test_a1_leadtime.py)

So if there is a large gap left in sps, it cannot be reached by estimating
sigma better. This script tests the alternative directly: shrink the residual
by a factor, refit sigma to the shrunken residual, rescore, and read off how
much sps a given improvement in accuracy is worth. That converts "forecast
better" into a number, and says whether the remaining sps gap is an accuracy
gap or simply does not exist.

Predictions are cached to an npz on first run: the model runs on CPU here so it
can share the machine with a training job, and that costs about ten minutes.

Usage:
    python scripts/sps_ceiling.py --checkpoint adv_finetuned_best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import (DEFAULT_VAL_RE, EXTRAPOLATION_VAL_RE, WindowDataset,
                          denormalize, split_cases)
from realpde.local_score import score_arrays
from realpde.models import build_model
from realpde.sps import bounds_from_sigma

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "checkpoints" / "sps_ceiling_cache"
WIDTHS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
GAINS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


@torch.no_grad()
def run_model(model, cases, device, batch=8):
    ds = WindowDataset("real", cases=cases, stride=40)
    xs = torch.stack([ds[i]["input"] for i in range(len(ds))])[..., :2]
    ys = torch.stack([ds[i]["target"] for i in range(len(ds))])[..., :2]
    ps = torch.cat([model(xs[i:i + batch].to(device)).cpu()
                    for i in range(0, len(xs), batch)])
    return denormalize(ps).numpy(), denormalize(ys).numpy()


def best_sps(pred, targ, sigma):
    """Best sps over the band-width grid -- never compare against an untuned one."""
    best = -1.0
    for w in WIDTHS:
        lo, up = bounds_from_sigma(pred, sigma, scale=w)
        best = max(best, score_arrays(pred, targ, mean_t_neural_s=1e-3,
                                      lower=lo, upper=up)["sps_score"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="adv_finetuned_best.pt")
    ap.add_argument("--val-re", default="default")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    val_re = (EXTRAPOLATION_VAL_RE if args.val_re == "extrap"
              else DEFAULT_VAL_RE if args.val_re == "default"
              else tuple(int(v) for v in args.val_re.split(",")))

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.checkpoint).stem}_{args.val_re}"
    cf = CACHE / f"{tag}.npz"
    if cf.exists():
        d = np.load(cf)
        pt, yt, pv, yv = d["pt"], d["yt"], d["pv"], d["yv"]
        print(f"loaded cached predictions from {cf.name}")
    else:
        ck = torch.load(ROOT / "checkpoints" / args.checkpoint, map_location="cpu")
        a = ck["args"]
        model = build_model(a["model"], base=a["base"], channels=2)
        model.load_state_dict(ck["model"])
        model = model.to(torch.device(args.device)).eval()
        train_cases, val_cases = split_cases("real", val_re)
        pt, yt = run_model(model, train_cases, torch.device(args.device))
        pv, yv = run_model(model, val_cases, torch.device(args.device))
        np.savez_compressed(cf, pt=pt, yt=yt, pv=pv, yv=yv)
        print(f"cached predictions to {cf.name}")

    print(f"checkpoint {args.checkpoint}  val Re {val_re}")
    print(f"{len(pt)} train windows / {len(pv)} val windows\n")

    print("what a given improvement in forecast accuracy is worth, with sigma")
    print("refitted to the improved residual each time\n")
    rows = []
    for g in GAINS:
        p2 = yt + (pt - yt) * g
        q2 = yv + (pv - yv) * g
        sigma = np.broadcast_to((p2 - yt).std(axis=0).astype(np.float32), q2.shape)
        r = score_arrays(q2, yv, mean_t_neural_s=1e-3)
        rows.append((g, r["_rel_l2_error"], r["rel_l2_score"],
                     best_sps(q2, yv, sigma)))

    base = next(r for r in rows if r[0] == 1.0)
    print(f"{'residual x':>11} {'rel_l2 err':>11} {'rel_l2':>9} {'sps':>9} "
          f"{'d sps':>9} {'d final':>9}")
    for g, err, rel, sps in rows:
        d_final = 0.19 * (rel - base[2]) + 0.30 * (sps - base[3])
        print(f"{g:>11.2f} {err:>11.5f} {rel:>9.4f} {sps:>9.4f} "
              f"{sps - base[3]:>+9.4f} {d_final:>+9.4f}")

    print("\nd final uses the validated marginal weights: 0.19 on rel_l2, 0.30 on sps.")
    print("Compare against the two calibration oracles, which are the whole of")
    print("what better bands can ever buy:  per-window +0.30 sps, per-lead +0.01 sps.")


if __name__ == "__main__":
    main()
