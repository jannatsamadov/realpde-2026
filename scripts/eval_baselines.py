"""Score trivial baselines on the local validation split.

These set the floor. Any model we build has to beat persistence, and the numbers
here calibrate how much each subscore can actually move:

  persistence  repeat the last input frame for all 20 output frames
  last_window  repeat the whole 20-frame input window as the output
  time_mean    the temporal mean of the input window, held constant
  zero         all zeros (a sanity check on the metric definitions)

Everything is evaluated in m/s, which is what the scorer compares against.

Usage:
    python scripts/eval_baselines.py
    python scripts/eval_baselines.py --t-neural 0.002
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import T_OUT, build_datasets, denormalize
from realpde.local_score import format_scores, score_arrays


def collect_val(max_windows: int | None = None):
    _, val = build_datasets("real")
    n = len(val) if max_windows is None else min(len(val), max_windows)
    xs, ys = [], []
    for i in range(n):
        b = val[i]
        xs.append(b["input"])
        ys.append(b["target"])
    x = torch.stack(xs)                      # (N, 20, 32, 64, 2) normalized
    y = torch.stack(ys)
    # The scorer works in m/s, so undo normalization on both sides.
    return denormalize(x, 2).numpy(), denormalize(y, 2).numpy()


def baselines(x: np.ndarray) -> dict:
    return {
        "persistence": np.repeat(x[:, -1:], T_OUT, axis=1),
        "last_window": x.copy(),
        "time_mean": np.repeat(x.mean(axis=1, keepdims=True), T_OUT, axis=1),
        "zero": np.zeros_like(x),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-neural", type=float, default=0.005,
                    help="assumed per-sample inference seconds, for time_score")
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    x, y = collect_val(args.max_windows)
    print(f"validation set: {x.shape[0]} windows, input {x.shape}, target {y.shape}")
    print(f"loaded in {time.perf_counter() - t0:.1f}s\n")

    for name, pred in baselines(x).items():
        print(f"=== {name} ===")
        scores = score_arrays(pred, y, mean_t_neural_s=args.t_neural)
        print(format_scores(scores))
        print()

    # What the default SPS band costs versus a perfect prediction, for reference.
    print("=== oracle (prediction == target) ===")
    print(format_scores(score_arrays(y, y, mean_t_neural_s=args.t_neural)))
    print("\nNote: even the oracle's sps_score is bounded by the default +/-5%|pred|")
    print("band, which is why supplying calibrated lower/upper bounds matters.")


if __name__ == "__main__":
    main()
