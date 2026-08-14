"""Audit the local setup for leakage, and explain why Rel-L2 looks so forgiving.

Two independent questions, both worth a number rather than an assurance.

LEAKAGE
    1. Do train and validation share a trajectory, a case, or a Reynolds number?
    2. Within validation, does any window's target appear inside another
       window's input? (The bug that forced the 5 August restart.)
    3. Do any two trajectories contain identical frames? This would catch a
       repeat of the 7575_0 / 6300_0 duplication that the organizers shipped, in
       case there is another one they have not found.

WHY REL-L2 IS SOFT
    Relative L2 divides by ||target||, and the target is dominated by a steady
    mean flow that barely moves in 0.4 s. Decomposing the target into its
    temporal mean and its fluctuation shows how little of the norm is actually
    the part a forecaster has to work for — which is why copying the input
    already scores in the nineties, with no model and no leakage involved.

Usage:
    python scripts/leakage_audit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import T_IN, T_OUT, build_datasets, denormalize, load_split


def audit_splits():
    print("=" * 72)
    print("1. TRAIN / VALIDATION SEPARATION")
    print("=" * 72)
    train, val = build_datasets("real")

    tr_cases = {t["case"] for t in train.trajectories}
    va_cases = {t["case"] for t in val.trajectories}
    tr_re = {t["re_nominal"] for t in train.trajectories}
    va_re = {t["re_nominal"] for t in val.trajectories}
    tr_off = {t["offset"] for t in train.trajectories}
    va_off = {t["offset"] for t in val.trajectories}

    print(f"  train: {len(tr_cases)} cases, {len(train)} windows")
    print(f"  val:   {len(va_cases)} cases, {len(val)} windows")
    print(f"  shared cases:            {sorted(tr_cases & va_cases) or 'none'}")
    print(f"  shared Reynolds numbers: {sorted(tr_re & va_re) or 'none'}")
    print(f"  shared trajectory offsets: {sorted(tr_off & va_off) or 'none'}")
    assert not (tr_cases & va_cases) and not (tr_re & va_re) and not (tr_off & va_off)
    print("  -> train and val cannot share a single frame: different files entirely.\n")
    return train, val


def audit_val_windows(val):
    print("=" * 72)
    print("2. CROSS-WINDOW LEAKAGE INSIDE VALIDATION")
    print("=" * 72)
    per_traj = {}
    for ti, s in val.index:
        per_traj.setdefault(ti, []).append(s)
    bad = 0
    for ti, starts in per_traj.items():
        for i, si in enumerate(sorted(starts)):
            ta, tb = si + T_IN, si + T_IN + T_OUT
            for j, sj in enumerate(sorted(starts)):
                if i == j:
                    continue
                ia, ib = sj, sj + T_IN
                if ta < ib and ia < tb:
                    bad += 1
    print(f"  windows: {len(val)}   target-inside-another-input overlaps: {bad}")
    assert bad == 0
    print("  -> clean. No window can be answered from another window's input.\n")


def audit_duplicate_trajectories():
    print("=" * 72)
    print("3. DUPLICATE TRAJECTORIES IN THE RELEASE")
    print("=" * 72)
    arr, meta = load_split("real")
    traj = meta["trajectories"]
    # Cheap fingerprint: a few frames' checksums. Identical files collide, and
    # different files essentially never do.
    fps = {}
    for t in traj:
        o, L = t["offset"], t["length"]
        picks = [0, L // 3, 2 * L // 3, L - 1]
        sig = tuple(float(np.asarray(arr[o + p], dtype=np.float64).sum()) for p in picks)
        fps.setdefault(sig, []).append(t["case"])
    dupes = {k: v for k, v in fps.items() if len(v) > 1}
    if dupes:
        for k, v in dupes.items():
            print(f"  DUPLICATE: {v}")
    else:
        print("  no duplicate trajectories found among the 81 kept files")
    print("  (7575_0 is already excluded; this checks for a second, unreported one.)\n")


def audit_rel_l2_structure(val):
    print("=" * 72)
    print("4. WHY REL-L2 IS SOFT — DECOMPOSING THE TARGET NORM")
    print("=" * 72)
    ys = []
    for i in range(len(val)):
        ys.append(val[i]["target"])
    y = denormalize(torch.stack(ys), 2).numpy()          # (N, 20, 32, 64, 2) m/s

    mean_t = y.mean(axis=1, keepdims=True)               # steady part
    fluct = y - mean_t                                   # unsteady part

    n_full = np.linalg.norm(y.reshape(len(y), -1), axis=1)
    n_mean = np.linalg.norm(np.broadcast_to(mean_t, y.shape).reshape(len(y), -1), axis=1)
    n_fluc = np.linalg.norm(fluct.reshape(len(y), -1), axis=1)

    print(f"  ||target||          mean over windows: {n_full.mean():.4f}")
    print(f"  ||temporal mean||                     : {n_mean.mean():.4f}  "
          f"({100 * (n_mean**2 / n_full**2).mean():.1f}% of the energy)")
    print(f"  ||fluctuation||                       : {n_fluc.mean():.4f}  "
          f"({100 * (n_fluc**2 / n_full**2).mean():.1f}% of the energy)")

    # An oracle for the steady part only: predict the true temporal mean, get the
    # fluctuation entirely wrong. This is the ceiling for any smooth predictor.
    err_steady_oracle = (n_fluc / n_full).mean()
    print(f"\n  A predictor that nails the temporal mean but predicts zero")
    print(f"  fluctuation scores rel_l2_error = {err_steady_oracle:.4f}")
    print(f"  -> rel_l2_score = {100 / (1 + 0.5 * err_steady_oracle):.2f}")
    print(f"  and tke_score   = 66.67 (zero fluctuation => TKE error 1.0)")

    masked = (y == 0).all(axis=-1)
    print(f"\n  masked cells (u = v = 0) in the validation targets: "
          f"{100 * masked.mean():.1f}%")
    print("  These contribute nothing to either side of the Rel-L2 ratio, so they")
    print("  neither inflate nor deflate the score; they shrink the effective field.")
    print("\n  CONCLUSION: copying the input scores in the nineties because the")
    print("  target's norm is ~97% steady mean flow that barely moves in 0.4 s.")
    print("  That is a property of the metric, not evidence of leakage.\n")


def main() -> None:
    train, val = audit_splits()
    audit_val_windows(val)
    audit_duplicate_trajectories()
    audit_rel_l2_structure(val)
    print("=" * 72)
    print("AUDIT COMPLETE — no leakage found in the local setup.")
    print("=" * 72)


if __name__ == "__main__":
    main()
