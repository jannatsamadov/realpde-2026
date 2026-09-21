"""A1: should the online sigma correction vary with lead time instead of being one scalar?

WHY THIS IS NOT THE IDEA F1 KILLED
    F1 scaled sigma per WINDOW and died: once the global band width is tuned,
    even an oracle that knows each window's true error buys +0.30 sps. A1 scales
    per LEAD TIME, which is a different axis. The fixed sigma map already varies
    with lead time, because it is fitted per output frame -- so A1 only has room
    if the RATIO of true to fitted error changes with lead time on regimes the
    map was not fitted on. That is a specific, falsifiable claim.

    The mechanism it would capture: error grows with lead time, and on unseen
    regimes it may grow FASTER than on the fitted ones. A single online scalar
    corrects the average and cannot tilt.

THE LESSON FROM F1 IS BUILT IN
    The baseline is the fixed map at its own best width AND its own best global
    scalar -- not an uncalibrated one. Then the oracle per-lead-time correction,
    fitted on the scored data itself, gives the ceiling. If the ceiling is small
    the idea is dead regardless of how it would be estimated online.

Usage:
    python scripts/test_a1_leadtime.py --checkpoint adv_finetuned_best.pt
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
WIDTHS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)


@torch.no_grad()
def run_model(model, cases, device, batch=8):
    ds = WindowDataset("real", cases=cases, stride=40)
    xs = torch.stack([ds[i]["input"] for i in range(len(ds))])[..., :2]
    ys = torch.stack([ds[i]["target"] for i in range(len(ds))])[..., :2]
    ps = torch.cat([model(xs[i:i + batch].to(device)).cpu()
                    for i in range(0, len(xs), batch)])
    return denormalize(ps).numpy(), denormalize(ys).numpy()


def sps_of(pred, targ, sigma, width):
    lo, up = bounds_from_sigma(pred, sigma, scale=width)
    return score_arrays(pred, targ, mean_t_neural_s=1e-3,
                        lower=lo, upper=up)["sps_score"]


def best_width(pred, targ, sigma):
    return max((sps_of(pred, targ, sigma, w), w) for w in WIDTHS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="adv_finetuned_best.pt")
    ap.add_argument("--val-re", default="default")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default so this can run beside a training job")
    args = ap.parse_args()

    val_re = (EXTRAPOLATION_VAL_RE if args.val_re == "extrap"
              else DEFAULT_VAL_RE if args.val_re == "default"
              else tuple(int(v) for v in args.val_re.split(",")))

    device = torch.device(args.device)
    ck = torch.load(ROOT / "checkpoints" / args.checkpoint, map_location="cpu")
    a = ck["args"]
    model = build_model(a["model"], base=a["base"], channels=2)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    train_cases, val_cases = split_cases("real", val_re)
    pt, yt = run_model(model, train_cases, device)
    pv, yv = run_model(model, val_cases, device)
    print(f"checkpoint {args.checkpoint}  val Re {val_re}")
    print(f"{len(pt)} train windows / {len(pv)} val windows  [{device}]\n")

    fixed = (pt - yt).std(axis=0).astype(np.float32)          # (T, H, W, C)

    # --- the falsifiable claim, stated as a number per lead time ------------
    # How far off is the fitted map, frame by frame, on held-out regimes?
    t_out = fixed.shape[0]
    fitted = np.sqrt((fixed ** 2).reshape(t_out, -1).mean(axis=1))
    actual = np.sqrt(((pv - yv) ** 2).reshape(len(pv), t_out, -1)
                     .mean(axis=(0, 2)))
    ratio = actual / fitted

    print("does the map's error grow with lead time differently on unseen Re?")
    print(f"{'lead':>5} {'fitted sd':>11} {'actual sd':>11} {'ratio':>8}")
    for t in range(t_out):
        print(f"{t + 1:>5} {fitted[t]:>11.5f} {actual[t]:>11.5f} {ratio[t]:>8.4f}")
    print(f"\nratio: mean {ratio.mean():.4f}  sd {ratio.std():.4f}  "
          f"({100 * ratio.std() / ratio.mean():.1f}% of mean)")
    slope = np.polyfit(np.arange(t_out), ratio, 1)[0]
    print(f"trend across lead time: {slope:+.5f} per frame "
          f"({100 * slope * t_out / ratio.mean():+.1f}% end to end)")
    print("  a flat ratio means one global scalar already captures it -> A1 dead")

    # --- the honest baseline: best width AND best global scalar -------------
    sig = np.broadcast_to(fixed, pv.shape)
    base, bw = best_width(pv, yv, sig)
    best_g, bg = base, 1.0
    for g in np.linspace(0.6, 1.6, 21):
        v = sps_of(pv, yv, sig * g, bw)
        if v > best_g:
            best_g, bg = v, g
    print(f"\nfixed map, best width {bw}                 : sps {base:.4f}")
    print(f"  + best GLOBAL scalar {bg:.2f}              : sps {best_g:.4f}"
          f"   <- the honest baseline")

    # --- the ceiling: per-lead-time scalars fitted on the scored data -------
    per_t = (ratio / ratio.mean()).astype(np.float32)
    sig_t = np.broadcast_to(fixed, pv.shape) * per_t[None, :, None, None, None]
    best_t = max(sps_of(pv, yv, sig_t * g, bw)
                 for g in np.linspace(0.6, 1.6, 21))
    print(f"  + ORACLE per-lead-time scalars           : sps {best_t:.4f}"
          f"   {best_t - best_g:+.4f}  <- A1's ceiling")

    # And the version that keeps the global level free too, for completeness.
    sig_full = np.broadcast_to(fixed, pv.shape) * ratio[None, :, None, None, None]
    best_f = max(sps_of(pv, yv, sig_full * g, bw)
                 for g in np.linspace(0.6, 1.6, 21))
    print(f"  + ORACLE per-lead-time, level free       : sps {best_f:.4f}"
          f"   {best_f - best_g:+.4f}")


if __name__ == "__main__":
    main()
