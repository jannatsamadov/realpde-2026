"""Score checkpoints per Reynolds number on the extrapolation validation split.

A single selection number hides the thing we actually want to know. Scale
conditioning is supposed to help where the velocity magnitude is outside the
trained range, and to do nothing in particular elsewhere. If a gain is spread
evenly across regimes it is not the mechanism working, it is noise or a generic
regularisation effect.

Usage:
    python scripts/score_by_regime.py --checkpoints a.pt b.pt --val-re extrap
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde.data import (DEFAULT_VAL_RE, EXTRAPOLATION_VAL_RE, WindowDataset,
                          denormalize, split_cases)
from realpde.local_score import score_arrays
from realpde.models import build_model

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict(model, ds, device, batch=8):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    preds, targs, times = [], [], []
    for b in loader:
        x = b["input"][..., :2].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / x.shape[0])
        preds.append(p.cpu())
        targs.append(b["target"][..., :2])
    return (denormalize(torch.cat(preds)).numpy(),
            denormalize(torch.cat(targs)).numpy(),
            float(np.mean(times)))


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck.get("args", {})
    m = build_model(a.get("model", "advective"), base=a.get("base", 64), channels=2)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--val-re", default="extrap")
    args = ap.parse_args()

    val_re = (EXTRAPOLATION_VAL_RE if args.val_re == "extrap"
              else DEFAULT_VAL_RE if args.val_re == "default"
              else tuple(int(v) for v in args.val_re.split(",")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_cases = split_cases("real", val_re)
    by_re = {}
    for c in val_cases:
        by_re.setdefault(int(c.split("_")[0]), []).append(c)

    print(f"val Re {val_re}   {len(val_cases)} cases\n")

    results = {}
    for cp in args.checkpoints:
        name = Path(cp).stem.replace("_best", "")
        model, a = load(ROOT / "checkpoints" / Path(cp).name
                        if not Path(cp).is_absolute() else cp, device)
        row = {}
        for re, cases in sorted(by_re.items()):
            ds = WindowDataset("real", cases=cases, stride=40)
            pred, targ, t = predict(model, ds, device)
            s = score_arrays(pred, targ, mean_t_neural_s=t)
            s["_selection"] = float(np.mean([s["rel_l2_score"], s["tke_score"],
                                             s["mvpe_score"]]))
            s["_n"] = len(ds)
            row[re] = s
        results[name] = (row, a.get("model"))

    # Per-Reynolds table, one block per metric that moved.
    names = list(results)
    for key, label in [("rel_l2_score", "rel_l2"), ("tke_score", "tke"),
                       ("mvpe_score", "mvpe"), ("sps_score", "sps"),
                       ("_selection", "SELECTION")]:
        print(f"--- {label}")
        print(f"{'Re':>7} {'n':>5} " + " ".join(f"{n:>22}" for n in names)
              + f" {'delta':>8}")
        for re in sorted(by_re):
            vals = [results[n][0][re][key] for n in names]
            d = vals[-1] - vals[0]
            n_win = results[names[0]][0][re]["_n"]
            print(f"{re:>7} {n_win:>5} " + " ".join(f"{v:>22.4f}" for v in vals)
                  + f" {d:>+8.4f}")
        # Window-count weighted mean across regimes.
        w = np.array([results[names[0]][0][re]["_n"] for re in sorted(by_re)],
                     dtype=float)
        means = [float(np.average([results[n][0][re][key] for re in sorted(by_re)],
                                  weights=w)) for n in names]
        print(f"{'all':>7} {int(w.sum()):>5} " + " ".join(f"{m:>22.4f}" for m in means)
              + f" {means[-1] - means[0]:>+8.4f}\n")

    print("model types:", {n: results[n][1] for n in names})


if __name__ == "__main__":
    main()
