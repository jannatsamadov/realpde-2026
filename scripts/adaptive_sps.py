"""Size the SPS interval per cell AND per window, from the input's local strain.

The submitted bounds use one sigma map fitted on training residuals. It captures
where the error lives, because the map is per (frame, cell), but it cannot tell a
high-Reynolds window from a low-Reynolds one — every window gets the width the
average window deserved. The hidden set punished exactly that: its residuals ran
about 1.49x larger than ours and coverage fell from 87.6% to 69.8%.

noise_vs_strain.py measured why the map has the shape it does: absolute PIV noise
tracks the local strain rate, correlation +0.638, and the top strain quintile
carries 9.9x the noise of the bottom. Strain is computable from the input window
at inference, so it can size the interval per window as well as per cell.

An earlier attempt scaled the whole map by one scalar per window (the input's
velocity spread) and did not help: 48.54 against 48.74. This is the finer
version — a per-cell factor from that window's own strain field, which is a
different quantity from a single number per sample.

Four estimators are compared on the same validation windows, all scored with the
organizers' scoring.py:

    fixed          the submitted map, unchanged
    scalar         map x one scalar per window            (already known to fail)
    strain         map x per-cell strain of this window
    fluctuation    map x per-cell input fluctuation of this window

Usage:
    python scripts/adaptive_sps.py --checkpoint checkpoints/adv_finetuned_best.pt
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
DX = 0.003422          # metres per cell at 32x64


def strain_field(u, v, dx=DX):
    """Frobenius norm of the strain-rate tensor. (N, T, H, W) -> (N, H, W)."""
    a = np.gradient(u, dx, axis=-1)
    b = np.gradient(u, -dx, axis=-2)
    c = np.gradient(v, dx, axis=-1)
    d = np.gradient(v, -dx, axis=-2)
    s12 = 0.5 * (b + c)
    return np.sqrt(a ** 2 + d ** 2 + 2.0 * s12 ** 2).mean(axis=1)


def normalise_factor(f, floor=0.25, ceil=4.0):
    """Turn a per-window field into a multiplier centred on 1 across the split."""
    g = f / np.mean(f)
    return np.clip(g, floor, ceil)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--sigma", type=Path, default=ROOT / "checkpoints" / "sps_sigma.npz")
    ap.add_argument("--train-windows", type=int, default=1200)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    train, val = build_datasets("real")

    def run(ds, n=None):
        idx = range(len(ds)) if n is None else np.linspace(0, len(ds) - 1, n).astype(int)
        xs = torch.stack([ds[int(i)]["input"] for i in idx])
        ys = torch.stack([ds[int(i)]["target"] for i in idx])
        with torch.no_grad():
            ps = torch.cat([model(xs[i:i + 32]) for i in range(0, len(xs), 32)])
        return (denormalize(xs, 2).numpy(), denormalize(ps, 2).numpy(),
                denormalize(ys, 2).numpy())

    xt, pt, yt = run(train, args.train_windows)
    xv, pv, yv = run(val)
    scored = yv != 0.0
    sigma_map = np.load(args.sigma)["sigma"].astype(np.float64)     # (1,T,H,W,C)

    # Per-window fields, from the INPUT only — available at inference.
    def fields(x):
        s = strain_field(x[..., 0], x[..., 1])                       # (N,H,W)
        f = np.sqrt((x - x.mean(axis=1, keepdims=True)).var(axis=1).mean(axis=-1))
        return s, f

    s_tr, f_tr = fields(xt)
    s_v, f_v = fields(xv)

    print(f"train {len(xt)} windows, val {len(xv)} windows")
    print(f"strain: train mean {s_tr.mean():.3f}, val mean {s_v.mean():.3f}")
    print(f"input fluctuation: train mean {f_tr.mean():.5f}, "
          f"val mean {f_v.mean():.5f}\n")

    def expand(fac):
        """(N,H,W) -> (N,1,H,W,1) so it multiplies the sigma map."""
        return fac[:, None, :, :, None]

    variants = {"fixed": np.ones_like(s_v)}
    variants["scalar"] = np.broadcast_to(
        normalise_factor(s_v.mean(axis=(1, 2)) / s_tr.mean())[:, None, None],
        s_v.shape).copy()
    variants["strain"] = normalise_factor(s_v / s_tr.mean())
    variants["fluctuation"] = normalise_factor(f_v / f_tr.mean())
    # Square root damps the factor — the analytic width is concave in sigma, so a
    # raw multiplier over-corrects where the strain is extreme.
    variants["strain^0.5"] = normalise_factor(np.sqrt(s_v / s_tr.mean()))

    print(f"{'estimator':<16}{'scale':>7}{'sps':>9}{'coverage':>11}{'mean width':>12}")
    print("-" * 55)
    best = ("", -1.0, None)
    for name, fac in variants.items():
        for scale in (0.8, 1.0, 1.2):
            s = np.broadcast_to(sigma_map, pv.shape) * expand(fac)
            lo, hi = bounds_from_sigma(pv, s, scale=scale)
            r = score_arrays(pv, yv, mean_t_neural_s=0.0176, lower=lo, upper=hi)
            if r["sps_score"] > best[1]:
                best = (f"{name} x{scale}", r["sps_score"], r["_sps_coverage"])
            print(f"{name:<16}{scale:>7.1f}{r['sps_score']:>9.2f}"
                  f"{r['_sps_coverage']:>11.3f}{(hi - lo)[scored].mean():>12.5f}")
        print()

    print(f"best: {best[0]} -> sps {best[1]:.2f} (coverage {best[2]:.3f})")
    print("\nThe number to beat is the submitted fixed map at scale 0.8, whose")
    print("validation sps was 48.74. A gain here only matters if it also survives")
    print("the move to the hidden set, which is where the fixed map lost 12 points.")


if __name__ == "__main__":
    main()
