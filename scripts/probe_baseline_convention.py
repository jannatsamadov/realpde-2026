"""Which input convention do the released baselines expect?

Feeding them raw m/s produced a relative L2 error of 1.56 for FNO — worse than
predicting zeros. The organizers' own released checkpoint cannot be that bad, so
the fault is in how it is being called, not in the checkpoint.

The likely culprit is normalization: the starting kit ships mean_std_real.pt, and
the Track 2 interface says the evaluator normalizes before handing tensors to the
model. This tries the plausible conventions on a small batch and reports the
relative L2 for each; the right one should land near the persistence baseline
(0.139) or better, not above 1.

Usage:
    python scripts/probe_baseline_convention.py --model fno
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
KIT = (ROOT / "NeurIPS 2026 RealPDE Competition"
       / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9")
sys.path.insert(0, str(KIT))

import numpy as np
import torch

from realpde.data import OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL, build_datasets, denormalize


def rel_l2(pred, target):
    b = pred.shape[0]
    p = pred[..., :2].reshape(b, -1)
    t = target[..., :2].reshape(b, -1)
    return float(np.mean(np.linalg.norm(p - t, axis=1)
                         / np.clip(np.linalg.norm(t, axis=1), 1e-8, None)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fno")
    ap.add_argument("--windows", type=int, default=24)
    args = ap.parse_args()

    from load_baseline import load_baseline

    _, val = build_datasets("real")
    n = min(args.windows, len(val))
    xn = torch.stack([val[i]["input"] for i in range(n)])        # normalized, 2ch
    yn = torch.stack([val[i]["target"] for i in range(n)])
    x_ms = denormalize(xn, 2).numpy()
    y_ms = denormalize(yn, 2).numpy()

    def to3(a):
        out = np.zeros(a.shape[:-1] + (3,), dtype=np.float32)
        out[..., :2] = a
        return out

    mean3 = np.zeros(3, dtype=np.float32); mean3[:2] = OFFICIAL_MEAN_REAL[:2]
    std3 = np.ones(3, dtype=np.float32); std3[:2] = OFFICIAL_STD_REAL[:2]

    model, meta = load_baseline(str(ROOT / "baselines" / f"sim_real_{args.model}.pth"))
    print(f"{args.model}: {meta.get('format')}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"persistence on these {n} windows: rel_l2 error "
          f"{rel_l2(np.repeat(to3(x_ms)[:, -1:], 20, axis=1), to3(y_ms)):.4f}\n")

    conventions = {
        "raw m/s in, raw out": (to3(x_ms), None),
        "normalized in, normalized out": (to3((x_ms - mean3[:2] * 0 - OFFICIAL_MEAN_REAL[:2])
                                              / OFFICIAL_STD_REAL[:2]), "denorm"),
        "normalized in, raw out": (to3((x_ms - OFFICIAL_MEAN_REAL[:2])
                                       / OFFICIAL_STD_REAL[:2]), None),
    }

    print(f"{'convention':<34}{'rel_l2 err':>12}{'pred mean':>12}{'pred std':>11}")
    print("-" * 69)
    for name, (xin, post) in conventions.items():
        with torch.no_grad():
            out = model(torch.from_numpy(np.ascontiguousarray(xin, dtype=np.float32)))
        p = out.float().numpy()
        if post == "denorm":
            p = p * std3 + mean3
        print(f"{name:<34}{rel_l2(p, to3(y_ms)):>12.4f}"
              f"{p[..., :2].mean():>12.5f}{p[..., :2].std():>11.5f}")

    print(f"\n{'target (m/s)':<34}{'':>12}{y_ms.mean():>12.5f}{y_ms.std():>11.5f}")
    print("The convention whose prediction statistics match the target's is the")
    print("one the checkpoint was trained under.")


if __name__ == "__main__":
    main()
