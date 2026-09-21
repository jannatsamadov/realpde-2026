"""F1: should the interval width depend on how fast the flow is evolving?

THE CLAIM
    Measured on all 81 trajectories (scripts/measure_shedding.py), the shedding
    period ranges from 5.6 to 145 frames, so a 20-frame window covers between
    0.14 and 3.55 shedding cycles. Forecast difficulty should follow that.
    The sigma behind the bounds is currently a fixed per-element map times one
    global online scalar, and a global scalar cannot express a per-window
    difference by construction.

WHY THIS IS NOT THE ENERGY PROXY THAT ALREADY FAILED
    calibrate_sps.py scaled sigma by the window's velocity spread and came out
    slightly worse (48.56 against 48.79). Velocity spread tracks Reynolds number
    and nothing else. Shedding rate does not: at Re 10125 the peak frequency is
    0.063 cycles/frame at 0 degrees and 0.017 at 20 degrees -- a factor of 3.6
    at identical energy. So the two proxies are not interchangeable and the
    earlier negative result does not settle this one.

WHAT IS COMPUTABLE FROM 20 FRAMES
    Not the frequency itself: a 20-frame window resolves 0.05 cycles/frame and
    most of our periods are longer than that. Four proxies are compared, all
    from the input window alone, with the energy proxy included as the control:

      q90        90th-percentile speed -- the control
      decorr     1 - lag-1 correlation of the fluctuation
      dfrac      mean |x[t+1]-x[t]| over the fluctuation's sd
      fft20      the coarse 20-frame spectral peak

HOW THE EXPONENT IS CHOSEN
    sigma_i = fixed_map * (p_i / mean p)**beta. The best beta is not searched
    for: the multiplier should make sigma track the window's actual residual
    spread, so beta is the least-squares slope of log(residual sd) on
    log(predictor), fitted on TRAIN windows and then applied unchanged to VAL.
    An earlier version grid-searched beta by rescoring the whole training set 31
    times per predictor; it ran 26 minutes and 10 GB before being killed, and
    the closed form is both faster and the thing the grid was approximating.

Usage:
    python scripts/test_f1_regime_bands.py --checkpoint x_adv_extrap_best.pt
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


def predictors(x: np.ndarray) -> dict:
    """x: (N, T, H, W, C) input windows in m/s -> {name: (N,)}, all positive."""
    speed = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    flat = speed.reshape(len(x), -1)
    q90 = np.array([np.quantile(s[s > 0], 0.90) if (s > 0).any() else 1e-6
                    for s in flat])

    fluct = x - x.mean(axis=1, keepdims=True)
    sd = fluct.reshape(len(x), -1).std(axis=1) + 1e-12

    a, b = fluct[:, :-1], fluct[:, 1:]
    num = (a * b).reshape(len(x), -1).sum(axis=1)
    den = np.sqrt((a ** 2).reshape(len(x), -1).sum(axis=1)
                  * (b ** 2).reshape(len(x), -1).sum(axis=1)) + 1e-12
    decorr = np.maximum(1.0 - num / den, 1e-6)

    dfrac = np.abs(np.diff(x, axis=1)).reshape(len(x), -1).mean(axis=1) / sd

    v = fluct[..., 1]
    spec = np.abs(np.fft.rfft(v, axis=1)) ** 2
    power = spec.reshape(spec.shape[0], spec.shape[1], -1).sum(axis=2)
    power[:, 0] = 0.0
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0)
    fft20 = np.maximum(freqs[power.argmax(axis=1)], 1e-6)

    # The statistic calibrate_sps used when per-window scaling was first tried
    # and came out slightly worse. Kept here so the two are compared directly:
    # it is the sd of the raw signed components, so it mixes the u channel's
    # large mean with v's near-zero one and counts masked cells as real zeros.
    old_energy = x.reshape(len(x), -1).std(axis=1)

    return {"q90": q90, "old_std": old_energy,
            "decorr": decorr, "dfrac": dfrac, "fft20": fft20}


def window_error(pred: np.ndarray, targ: np.ndarray) -> np.ndarray:
    r = (pred - targ).reshape(len(pred), -1)
    return r.std(axis=1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def run_model(model, cases, device, batch=16):
    ds = WindowDataset("real", cases=cases, stride=40)
    xs = torch.stack([ds[i]["input"] for i in range(len(ds))])[..., :2]
    ys = torch.stack([ds[i]["target"] for i in range(len(ds))])[..., :2]
    ps = torch.cat([model(xs[i:i + batch].to(device)).cpu()
                    for i in range(0, len(xs), batch)])
    return (denormalize(xs).numpy(), denormalize(ps).numpy(),
            denormalize(ys).numpy())


WIDTHS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)


def sps_with(pred, targ, fixed, scale=None, width=1.0):
    """SPS for sigma = fixed map, optionally times a per-window scalar.

    `width` is the global band multiplier calibrate_sps grid-searched. It has to
    be in the comparison: a per-window multiplier that also moves the mean width
    can look better or worse purely through that, and the submission already
    calibrates the global level online. Comparing each estimator at its own best
    width is the apples-to-apples version.
    """
    sigma = np.broadcast_to(fixed, pred.shape)
    if scale is not None:
        sigma = sigma * scale[:, None, None, None, None].astype(np.float32)
    lo, up = bounds_from_sigma(pred, sigma, scale=width)
    return score_arrays(pred, targ, mean_t_neural_s=1e-3,
                        lower=lo, upper=up)["sps_score"]


def best_over_width(pred, targ, fixed, scale=None):
    vals = [(sps_with(pred, targ, fixed, scale, w), w) for w in WIDTHS]
    return max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="x_adv_extrap_best.pt")
    ap.add_argument("--val-re", default="extrap",
                    help="must match the split the checkpoint was TRAINED with, "
                         "or the 'val' windows are training data")
    args = ap.parse_args()
    val_re = (EXTRAPOLATION_VAL_RE if args.val_re == "extrap"
              else DEFAULT_VAL_RE if args.val_re == "default"
              else tuple(int(v) for v in args.val_re.split(",")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "checkpoints" / args.checkpoint, map_location=device)
    a = ck["args"]
    model = build_model(a["model"], base=a["base"], channels=2)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    train_cases, val_cases = split_cases("real", val_re)
    xt, pt, yt = run_model(model, train_cases, device)
    xv, pv, yv = run_model(model, val_cases, device)
    print(f"checkpoint {args.checkpoint}  model={a['model']}  val Re {val_re}")
    print(f"{len(xt)} train windows / {len(xv)} val windows\n")

    et, ev = window_error(pt, yt), window_error(pv, yv)
    Pt, Pv = predictors(xt), predictors(xv)

    # --- gate: does anything beat the energy control at predicting error? ----
    print("how well each input-window statistic predicts that window's error sd")
    print(f"{'predictor':>10} {'spearman(train)':>16} {'spearman(val)':>14} "
          f"{'beta':>7} {'R2(train)':>10}")
    betas = {}
    for k in Pt:
        lp, le = np.log(Pt[k]), np.log(et)
        beta, c0 = np.polyfit(lp, le, 1)
        r2 = 1.0 - ((le - (beta * lp + c0)) ** 2).sum() / ((le - le.mean()) ** 2).sum()
        betas[k] = beta
        print(f"{k:>10} {spearman(Pt[k], et):>16.4f} {spearman(Pv[k], ev):>14.4f} "
              f"{beta:>7.3f} {r2:>10.4f}")

    # --- the fixed sigma map, fitted on TRAIN residuals ----------------------
    fixed = (pt - yt).std(axis=0).astype(np.float32)      # (T, H, W, C)
    base1 = sps_with(pv, yv, fixed)
    base, base_w = best_over_width(pv, yv, fixed)
    print(f"\nfixed sigma map, width 1.0                 : sps {base1:.4f}")
    print(f"fixed sigma map, best width {base_w:<4}           : sps {base:.4f}"
          f"   <- the honest baseline")

    # Ceiling: the per-window sd known exactly. Not an upper bound in the strict
    # sense -- the SPS optimum is not exactly at sigma = residual sd -- but it is
    # the natural reference for what a per-window multiplier is trying to do.
    oracle, ow = best_over_width(pv, yv, fixed, ev / ev.mean())
    print(f"oracle per-window scaling (NOT deliverable): sps {oracle:.4f} "
          f"(width {ow})  {oracle - base:+.4f}")

    print(f"\neach estimator at ITS OWN best width, against the baseline above")
    print(f"{'predictor':>10} {'beta':>7} {'sps':>9} {'width':>7} {'delta':>9} "
          f"{'% of oracle':>12}")
    for k in Pt:
        scale = (Pv[k] / Pt[k].mean()) ** betas[k]
        got, w = best_over_width(pv, yv, fixed, scale / scale.mean())
        frac = (got - base) / (oracle - base) * 100 if oracle > base else float("nan")
        print(f"{k:>10} {betas[k]:>7.3f} {got:>9.4f} {w:>7} {got - base:>+9.4f} "
              f"{frac:>11.1f}%")


if __name__ == "__main__":
    main()
