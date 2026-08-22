"""How far is the forecast from the limit, and is the gap chaos or generalisation?

Every result since 17 August has come from calibration, not from forecasting: the
three accuracy subscores are identical to six decimals across three submissions.
Before changing the model it is worth knowing whether there is anything left to
win in the forecast at all, and if so, where.

Three quantities, on the same real trajectories:

  NOISE FLOOR    PIV noise is a fraction of the target's own temporal variance,
                 estimated per cell from the lag-1 correlation of the
                 fluctuation: a 50 Hz physical signal is correlated between
                 consecutive frames, measurement noise is not. A perfect
                 forecast still carries this, so it is a hard lower bound on
                 rel_l2 that no model can go below.

  SEEN Re        Model error on trajectories whose Reynolds number IS in the
                 training set. Optimistic — those windows were trained on — so
                 it brackets from below what the architecture can do when the
                 regime is familiar.

  UNSEEN Re      Model error on the held-out Reynolds numbers, which is what the
                 leaderboard actually measures.

The split between them is the point. If SEEN and UNSEEN are close, the model is
near whatever limit the flow imposes and architecture work is the only way up.
If UNSEEN is much worse, the gap is generalisation to unfamiliar regimes, and
the fix is invariance and coverage rather than capacity.

Usage:
    python scripts/skill_ceiling.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from realpde.data import (DEFAULT_VAL_RE, T_IN, T_OUT,      # noqa: E402
                          WindowDataset, denormalize, load_split)
from realpde.models import build_model                      # noqa: E402


def rel_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window relative L2 over u, v, exactly as the scorer computes it."""
    p = pred.reshape(len(pred), -1)
    t = target.reshape(len(target), -1)
    return np.linalg.norm(p - t, axis=1) / np.clip(np.linalg.norm(t, axis=1), 1e-8, None)


def noise_floor(arr: np.ndarray, offset: int, length: int) -> float:
    """rel_l2 a perfect forecast would still incur, from PIV noise alone.

    Noise variance per cell is var_total * (1 - r1) with r1 the lag-1
    correlation of the fluctuation. Expressed as a fraction of the field's own
    L2 energy, its square root is the floor on relative L2.
    """
    f = arr[offset:offset + length].astype(np.float64)      # (T, H, W, 2)
    x = f - f.mean(axis=0, keepdims=True)
    a, b = x[:-1], x[1:]
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = (a * b).sum(axis=0)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    r1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-14)
    var_noise = x.var(axis=0) * np.clip(1.0 - r1, 0.0, 1.0)
    measured = ~((f == 0).all(axis=0))
    # Both sides must be per frame. var_noise is one value per cell, while the
    # energy below sums over every frame as well, so the denominator is divided
    # by the frame count rather than left as a whole-trajectory total — without
    # that the floor comes out sqrt(n_frames) too small.
    energy_per_frame = (f[:, measured] ** 2).sum() / len(f)
    return float(np.sqrt(var_noise[measured].sum() / energy_per_frame))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-case", type=int, default=12,
                    help="windows sampled per trajectory")
    args = ap.parse_args()
    device = torch.device(args.device)

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "advective"), base=targs.get("base", 64),
                        channels=2)
    model.load_state_dict(state["model"])
    model.eval().to(device)

    arr, meta = load_split("real")
    traj = meta["trajectories"]
    val_re = set(DEFAULT_VAL_RE)

    rows = []
    for t in traj:
        seen = t["re_nominal"] not in val_re
        ds = WindowDataset("real", cases=[t["case"]], stride=1)
        if len(ds) < 2:
            continue
        idx = np.linspace(0, len(ds) - 1, min(args.per_case, len(ds))).astype(int)
        xs = torch.stack([ds[int(i)]["input"] for i in idx])
        ys = torch.stack([ds[int(i)]["target"] for i in idx])
        with torch.no_grad():
            ps = torch.cat([model(xs[i:i + 8].to(device)).cpu()
                            for i in range(0, len(xs), 8)])
        p = denormalize(ps, 2).numpy()
        y = denormalize(ys, 2).numpy()

        e_model = float(rel_l2(p, y).mean())
        # Persistence: repeat the last input frame. The reference every forecast
        # has to beat before any of this means anything.
        last = denormalize(xs[:, -1:], 2).numpy()
        e_pers = float(rel_l2(np.repeat(last, T_OUT, axis=1), y).mean())
        floor = noise_floor(arr[..., :2], t["offset"], min(t["length"], 400))

        # Error per lead time, to see whether the gap is at short or long range.
        per_frame = np.array([rel_l2(p[:, k:k + 1], y[:, k:k + 1]).mean()
                              for k in range(T_OUT)])
        rows.append({"case": t["case"], "re": t["re_nominal"], "seen": seen,
                     "model": e_model, "pers": e_pers, "floor": floor,
                     "f1": per_frame[0], "f20": per_frame[-1]})

    if not rows:
        raise SystemExit("no trajectories scored")

    def agg(sel, key):
        v = [r[key] for r in rows if r["seen"] == sel]
        return float(np.mean(v)) if v else float("nan")

    print(f"{len(rows)} trajectories, {args.per_case} windows each, "
          f"held-out Re = {sorted(val_re)}\n")
    hdr = f"{'group':<14}{'n':>4}{'noise floor':>13}{'model':>9}{'persistence':>13}{'frame 1':>10}{'frame 20':>10}"
    print(hdr)
    print("-" * len(hdr))
    for sel, name in ((True, "SEEN Re"), (False, "UNSEEN Re")):
        n = sum(1 for r in rows if r["seen"] == sel)
        print(f"{name:<14}{n:>4}{agg(sel, 'floor'):>13.4f}{agg(sel, 'model'):>9.4f}"
              f"{agg(sel, 'pers'):>13.4f}{agg(sel, 'f1'):>10.4f}{agg(sel, 'f20'):>10.4f}")

    m_seen, m_unseen = agg(True, "model"), agg(False, "model")
    fl = agg(False, "floor")
    print(f"\ngeneralisation gap  : {m_unseen - m_seen:+.4f} "
          f"({100 * (m_unseen / m_seen - 1):+.1f}% error on unfamiliar Re)")
    print(f"headroom over floor : {m_unseen - fl:+.4f} "
          f"(unseen-Re error is {m_unseen / fl:.2f}x the noise floor)")
    print("\nIf the generalisation gap is the larger of the two, capacity and")
    print("temporal architecture are not the constraint — regime invariance is.")

    print(f"\n{'case':<12}{'Re':>7}{'seen':>6}{'floor':>9}{'model':>9}{'pers':>9}")
    print("-" * 52)
    for r in sorted(rows, key=lambda r: (r["seen"], r["re"])):
        print(f"{r['case']:<12}{r['re']:>7}{'y' if r['seen'] else 'n':>6}"
              f"{r['floor']:>9.4f}{r['model']:>9.4f}{r['pers']:>9.4f}")


if __name__ == "__main__":
    main()
