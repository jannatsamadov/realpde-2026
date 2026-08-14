"""Same scoring code, different data: how much do the subscores move?

The submitted model scored lower on the hidden set than on our validation split,
which invites the explanation that the platform computes the metrics differently.
It does not — local scoring imports the organizers' scoring.py and calls it, so
the formulas are identical by construction.

What differs is the data. This quantifies that directly by scoring one model on
several disjoint slices of our own validation split. If the spread across our own
Reynolds groups is comparable to the local-vs-hidden gap, the gap needs no
explanation beyond "a different sample of windows".

Runs on CPU so it does not compete with a training job for the GPU.

Usage:
    python scripts/score_spread.py --checkpoint checkpoints/tke0_best.pt
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

HIDDEN = {"rel_l2_score": 94.634455, "tke_score": 73.676616,
          "mvpe_score": 93.159743, "sps_score": 16.644090}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tke0_best.pt"))
    ap.add_argument("--model", default="unet")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(args.model, base=state.get("args", {}).get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, val = build_datasets("real")
    n = len(val)
    X = torch.stack([val[i]["input"] for i in range(n)])
    Y = torch.stack([val[i]["target"] for i in range(n)])
    cases = [val[i]["case"] for i in range(n)]
    aoas = np.array([val[i]["aoa"] for i in range(n)])
    res = np.array([int(c.split("_")[0]) for c in cases])

    with torch.no_grad():
        P = torch.cat([model(X[i:i + 32]) for i in range(0, n, 32)])
    p = denormalize(P, 2).numpy()
    t = denormalize(Y, 2).numpy()

    def row(label, m):
        if m.sum() < 5:
            return None
        s = score_arrays(p[m], t[m], mean_t_neural_s=0.005)
        print(f"  {label:<26}{m.sum():>6}{s['rel_l2_score']:>10.2f}{s['tke_score']:>9.2f}"
              f"{s['mvpe_score']:>9.2f}{s['sps_score']:>8.2f}")
        return s

    hdr = f"  {'slice':<26}{'n':>6}{'rel_l2':>10}{'tke':>9}{'mvpe':>9}{'sps':>8}"
    print(f"model {args.checkpoint.name}, {n} validation windows\n")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    allrow = row("ALL (what we reported)", np.ones(n, bool))

    print("\n  by Reynolds number:")
    per_re = [row(f"Re {r}", res == r) for r in sorted(set(res.tolist()))]

    print("\n  by angle of attack:")
    per_aoa = [row(f"AoA {a:.0f}", aoas == a) for a in sorted(set(aoas.tolist()))]

    print("\n  random halves (same distribution, different sample):")
    rng = np.random.default_rng(0)
    for i in range(3):
        m = np.zeros(n, bool)
        m[rng.choice(n, n // 2, replace=False)] = True
        row(f"half #{i + 1}", m)

    groups = [g for g in (per_re + per_aoa) if g]
    print("\n" + "=" * 68)
    print("SPREAD ACROSS OUR OWN DATA versus THE LOCAL-TO-HIDDEN GAP")
    print("=" * 68)
    print(f"  {'metric':<14}{'all':>9}{'min':>9}{'max':>9}{'range':>9}"
          f"{'hidden':>9}{'gap':>8}")
    print("  " + "-" * 66)
    for k in ("rel_l2_score", "tke_score", "mvpe_score", "sps_score"):
        vals = [g[k] for g in groups]
        gap = HIDDEN[k] - allrow[k]
        print(f"  {k:<14}{allrow[k]:>9.2f}{min(vals):>9.2f}{max(vals):>9.2f}"
              f"{max(vals) - min(vals):>9.2f}{HIDDEN[k]:>9.2f}{gap:>+8.2f}")
    print("\n  If 'range' is comparable to or larger than |gap|, the hidden-set")
    print("  difference is ordinary sampling variation, not a different formula.")


if __name__ == "__main__":
    main()
