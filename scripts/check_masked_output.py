"""What does the model actually output where the PIV measurement is missing?

The TKE figure masked the prediction panel using the ground truth's mask, which
made it look as though the model reproduces the laser shadow. It does not — that
grey was drawn on. This measures what the model really puts there.

It matters beyond the picture: rel_l2, tke and mvpe are NOT masked by scoring.py,
so whatever the model emits in shadowed cells goes straight into the error. Only
SPS skips them (its `scored = target != 0`).

Two regions are compared:
    masked   cells where the target is exactly zero (airfoil body, PIV dropout)
    valid    everything else

Usage:
    python scripts/check_masked_output.py --checkpoint checkpoints/tke0_best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL, build_datasets, denormalize
from realpde.models import UNetForecaster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=64)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = UNetForecaster(base=state.get("args", {}).get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, val = build_datasets("real")
    n = min(args.windows, len(val))
    x = torch.stack([val[i]["input"] for i in range(n)])
    y = torch.stack([val[i]["target"] for i in range(n)])

    with torch.no_grad():
        pred = model(x)

    p = denormalize(pred, 2).numpy()
    t = denormalize(y, 2).numpy()
    xin = denormalize(x, 2).numpy()

    masked = (t == 0.0)                      # elementwise, per channel
    valid = ~masked
    print(f"{n} validation windows, {100 * masked.mean():.1f}% of target elements are zero\n")

    print(f"{'region':<10} {'|pred| mean':>12} {'|pred| max':>12} {'|target| mean':>14}")
    print("-" * 52)
    for name, m in (("masked", masked), ("valid", valid)):
        print(f"{name:<10} {np.abs(p[m]).mean():>12.6f} {np.abs(p[m]).max():>12.6f} "
              f"{np.abs(t[m]).mean():>14.6f}")

    # How much of the total squared error comes from cells that hold no measurement?
    err2 = (p - t) ** 2
    frac = err2[masked].sum() / err2.sum()
    print(f"\nshare of total squared error contributed by masked cells: {100 * frac:.1f}%")

    # The residual parameterisation adds the last input frame, which is itself
    # zero in shadowed cells, so the model can only put something there by
    # actively predicting a non-zero delta.
    last_in = xin[:, -1:]
    last_zero = (last_in == 0.0)
    both = last_zero & masked[:, :1]
    print(f"input's last frame is zero in {100 * last_zero.mean():.1f}% of elements; "
          f"agreement with target mask: {100 * (both.sum() / max(last_zero.sum(), 1)):.1f}%")

    # What a zero-in-shadow prediction would gain, as an upper bound on the fix.
    p_forced = p.copy()
    p_forced[masked] = 0.0
    b = p.shape[0]
    rel = lambda a: (np.linalg.norm((a - t).reshape(b, -1), axis=1)
                     / np.linalg.norm(t.reshape(b, -1), axis=1)).mean()
    print(f"\nrel_l2 error as-is                     : {rel(p):.5f}")
    print(f"rel_l2 error if masked cells forced to 0: {rel(p_forced):.5f}")
    print("(the second uses the target mask, which we do NOT have at inference —")
    print(" it is an upper bound on what predicting the mask could buy)")

    # The mask is inferable from the input, which we DO have.
    input_zero = (xin == 0.0).all(axis=1)                      # (N, H, W, C) over time
    target_zero_any = masked.any(axis=1)
    agree = (input_zero == target_zero_any).mean()
    print(f"\ninput-derived mask agrees with target mask on {100 * agree:.2f}% of cells")
    p_inferred = p.copy()
    p_inferred[np.broadcast_to(input_zero[:, None], p.shape)] = 0.0
    print(f"rel_l2 error using the input-derived mask: {rel(p_inferred):.5f}")


if __name__ == "__main__":
    main()
