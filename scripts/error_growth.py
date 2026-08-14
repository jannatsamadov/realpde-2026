"""Does the error grow with lead time, and is the model only tracking the mean flow?

Two claims to test, both raised from watching the difference animation:

  1. the error grows frame by frame across the 20-step horizon;
  2. the model captures the global flow but not the local vortices.

For (1) the relative L2 is computed per output frame rather than over the window.
For (2) each field is split into its temporal mean and its fluctuation, and the
error is attributed to each part separately: a model that has the mean right and
the eddies wrong shows a small steady-part error and a fluctuation error near 1.

Usage:
    python scripts/error_growth.py --checkpoint checkpoints/tke0_best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.models import UNetForecaster


def rel(a, b, axis=None):
    num = np.linalg.norm((a - b).reshape(a.shape[0], -1), axis=1)
    den = np.linalg.norm(b.reshape(b.shape[0], -1), axis=1)
    return num / np.clip(den, 1e-12, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=273)
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
        pred = torch.cat([model(x[i:i + 32]) for i in range(0, n, 32)])

    p = denormalize(pred, 2).numpy()
    t = denormalize(y, 2).numpy()
    xin = denormalize(x, 2).numpy()
    persist = np.repeat(xin[:, -1:], p.shape[1], axis=1)

    print(f"{n} validation windows, checkpoint {args.checkpoint.name}\n")

    print("1. ERROR VERSUS LEAD TIME")
    print(f"  {'frame':>6} {'t ahead':>9} {'model':>9} {'persistence':>13} {'model/persist':>14}")
    print("  " + "-" * 56)
    for k in range(p.shape[1]):
        em = rel(p[:, k], t[:, k]).mean()
        ep = rel(persist[:, k], t[:, k]).mean()
        if k < 3 or k % 4 == 3 or k == p.shape[1] - 1:
            print(f"  {k + 1:>6} {0.02 * (k + 1):>8.2f}s {em:>9.4f} {ep:>13.4f} "
                  f"{em / max(ep, 1e-12):>14.3f}")
    e_first = rel(p[:, 0], t[:, 0]).mean()
    e_last = rel(p[:, -1], t[:, -1]).mean()
    print(f"\n  error grows {e_last / e_first:.2f}x from frame 1 to frame 20")

    print("\n2. STEADY PART VERSUS FLUCTUATION")
    tm, pm_ = t.mean(axis=1, keepdims=True), p.mean(axis=1, keepdims=True)
    tf, pf = t - tm, p - pm_
    e_mean = rel(pm_, tm).mean()
    e_fluc = rel(pf, tf).mean()
    print(f"  temporal mean  (the global flow) : rel L2 = {e_mean:.4f}")
    print(f"  fluctuation    (the local eddies): rel L2 = {e_fluc:.4f}")
    print(f"  fluctuation amplitude retained   : "
          f"{np.linalg.norm(pf) / np.linalg.norm(tf):.4f}")
    print(f"  correlation of the fluctuations  : "
          f"{float((pf * tf).sum() / (np.linalg.norm(pf) * np.linalg.norm(tf))):.4f}")

    print("\n  A model that only tracked the global flow would show a small mean")
    print("  error and a fluctuation error at or above 1.0 with correlation near 0.")

    print("\n3. FLUCTUATION CORRELATION VERSUS LEAD TIME")
    print(f"  {'frame':>6} {'corr':>8} {'amplitude kept':>16}")
    print("  " + "-" * 32)
    for k in range(p.shape[1]):
        a, b = pf[:, k], tf[:, k]
        c = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        amp = float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-12))
        if k < 3 or k % 4 == 3 or k == p.shape[1] - 1:
            print(f"  {k + 1:>6} {c:>8.4f} {amp:>16.4f}")


if __name__ == "__main__":
    main()
