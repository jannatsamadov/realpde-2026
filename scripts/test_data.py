"""Sanity-check the datasets: shapes, splits, normalization, and leak-freedom.

The leak check is the important one. It asserts that on the validation side no
window's target frames appear inside any other window's input frames, which is
exactly the property the organizers' evaluation set lacked before 5 August.

Usage:
    python scripts/test_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import (
    T_IN, T_OUT, WindowDataset, build_datasets, denormalize, split_cases,
    to_submission_format,
)


def check_split_disjoint():
    train_cases, val_cases = split_cases("real")
    overlap = set(train_cases) & set(val_cases)
    print(f"train cases {len(train_cases)}, val cases {len(val_cases)}, overlap {len(overlap)}")
    assert not overlap, "a case appears in both splits"

    train_re = {int(c.split('_')[0]) for c in train_cases}
    val_re = {int(c.split('_')[0]) for c in val_cases}
    print(f"  train Re: {sorted(train_re)}")
    print(f"  val   Re: {sorted(val_re)}")
    assert not (train_re & val_re), "a Reynolds number appears in both splits"
    print("  OK: splits are disjoint by case and by Reynolds number\n")


def check_no_window_leak(ds: WindowDataset, label: str):
    """No window's target range may intersect any other window's input range."""
    per_traj: dict[int, list[int]] = {}
    for ti, s in ds.index:
        per_traj.setdefault(ti, []).append(s)

    violations = 0
    for ti, starts in per_traj.items():
        starts = sorted(starts)
        inputs = [(s, s + T_IN) for s in starts]
        targets = [(s + T_IN, s + T_IN + T_OUT) for s in starts]
        for i, (ta, tb) in enumerate(targets):
            for j, (ia, ib) in enumerate(inputs):
                if i == j:
                    continue
                if ta < ib and ia < tb:          # half-open interval overlap
                    violations += 1
    print(f"{label}: {len(ds)} windows, cross-window target-in-input overlaps: {violations}")
    return violations


def main() -> None:
    check_split_disjoint()

    train, val = build_datasets("real", train_stride=10)
    print(f"real  train windows {len(train)}, val windows {len(val)}")

    b = train[0]
    print(f"\nsample: input {tuple(b['input'].shape)}  target {tuple(b['target'].shape)}  "
          f"case {b['case']}  Re {b['re']:.0f}  AoA {b['aoa']:.0f}")
    assert b["input"].shape == (T_IN, 32, 64, 2)
    assert b["target"].shape == (T_OUT, 32, 64, 2)

    # Normalized data should sit near zero mean / unit variance on u.
    xs = torch.stack([train[i]["input"] for i in range(0, len(train), max(1, len(train) // 200))])
    print(f"normalized train inputs: mean {xs.mean():+.4f}  std {xs.std():.4f}")
    print(f"  per channel mean {[f'{v:+.3f}' for v in xs.mean(dim=(0,1,2,3)).tolist()]}"
          f"  std {[f'{v:.3f}' for v in xs.std(dim=(0,1,2,3)).tolist()]}")

    # Round-trip through denormalize must return the original m/s values.
    raw = WindowDataset("real", cases=[b["case"]], stride=1000, normalize=False)[0]["input"]
    norm = train[0]["input"]
    back = denormalize(norm, channels=2)
    err = (back - raw).abs().max().item()
    print(f"denormalize round-trip max abs error: {err:.3e}")
    assert err < 1e-5, "normalization is not invertible"

    print()
    v_train = check_no_window_leak(train, "train (stride 10, overlap expected)")
    v_val = check_no_window_leak(val, "val   (stride 40, must be clean)")
    assert v_val == 0, "validation windows leak targets into other windows' inputs"

    # Simulation split: redimensionalised, it should land in the same range as real.
    sim = WindowDataset("sim", stride=200)
    ss = torch.stack([sim[i]["input"] for i in range(0, len(sim), max(1, len(sim) // 100))])
    sim_raw = WindowDataset("sim", stride=200, normalize=False, redimensionalize_sim=False)
    sr = torch.stack([sim_raw[i]["input"] for i in range(0, len(sim_raw), max(1, len(sim_raw) // 100))])
    print(f"\nsim windows {len(sim)}  (channels {sim.channels})")
    print(f"  raw sim         u mean {sr[..., 0].mean():+.4f}  std {sr[..., 0].std():.4f}")
    print(f"  normalized sim  u mean {ss[..., 0].mean():+.4f}  std {ss[..., 0].std():.4f}")
    print(f"  normalized real u mean {xs[..., 0].mean():+.4f}  std {xs[..., 0].std():.4f}")

    out = to_submission_format(torch.zeros(2, T_OUT, 32, 64, 2))
    print(f"\nsubmission format: {out.shape} dtype {out.dtype}, "
          f"p channel all zero: {bool((out[..., 2] == 0).all())}")
    assert out.shape == (2, T_OUT, 32, 64, 3)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
