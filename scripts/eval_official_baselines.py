"""Score the organizers' own FNO / CNO / Transolver baselines on our validation split.

Until now every comparison has been against trivial baselines or against our own
earlier models, which says how far we have come but not where we stand. These are
the checkpoints the organizers released, trained by them with the same two-stage
recipe we arrived at independently: pretrain on the simulated split, fine-tune on
the real one.

Scoring them through the same harness on the same 273 Reynolds-disjoint windows
makes the comparison apples to apples — same data, same scoring.py, same masking.

The checkpoints come from the competition data repository:
    huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data
      baseline_checkpoints/sim_real_ft/{sim_real_fno,sim_real_cno,sim_real_transolver}.pth

The loader is the starting kit's own load_baseline.py, so construction kwargs match
the released checkpoint shapes rather than the repo's stale yaml configs.

Usage:
    python scripts/eval_official_baselines.py
    python scripts/eval_official_baselines.py --models fno cno
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# The kit vendors its own einops and rpde_baselines subtree; load_baseline.py
# prepends them to sys.path itself when imported.
KIT = (ROOT / "NeurIPS 2026 RealPDE Competition"
       / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9")
sys.path.insert(0, str(KIT))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.local_score import format_scores, score_arrays

BASELINE_DIR = ROOT / "baselines"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fno", "cno", "transolver"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="cpu keeps this off a GPU with no working fan")
    args = ap.parse_args()

    from load_baseline import load_baseline   # noqa: E402  (needs KIT on sys.path)

    from realpde.data import OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL

    _, val = build_datasets("real")
    n = len(val)
    y = denormalize(torch.stack([val[i]["target"] for i in range(n)]), 2).numpy()

    # These checkpoints work in NORMALIZED space, in and out — which is why the
    # kit ships mean_std_real.pt. Feeding raw m/s instead gives FNO a relative L2
    # error of 2.37 against persistence's 0.14; with the right convention it is
    # 0.058. Verified in scripts/probe_baseline_convention.py by matching the
    # prediction's mean and standard deviation against the target's.
    xn = torch.stack([val[i]["input"] for i in range(n)]).numpy()   # already normalized
    x3 = np.zeros(xn.shape[:-1] + (3,), dtype=np.float32)
    x3[..., :2] = xn

    mean3 = np.zeros(3, dtype=np.float32); mean3[:2] = OFFICIAL_MEAN_REAL[:2]
    std3 = np.ones(3, dtype=np.float32); std3[:2] = OFFICIAL_STD_REAL[:2]
    print(f"validation: {n} windows, input {x3.shape} (normalized)\n")

    rows = []
    for name in args.models:
        ckpt = BASELINE_DIR / f"sim_real_{name}.pth"
        if not ckpt.exists():
            print(f"{name}: checkpoint missing at {ckpt} — skipped")
            continue
        try:
            model, meta = load_baseline(str(ckpt), device=args.device)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name}: failed to load — {type(exc).__name__}: {exc}")
            continue
        print(f"{name}: loaded ({meta.get('format')}), "
              f"{sum(p.numel() for p in model.parameters()):,} params")

        preds, times = [], []
        with torch.no_grad():
            for i in range(0, n, args.batch):
                xb = torch.from_numpy(x3[i:i + args.batch]).to(args.device)
                t0 = time.perf_counter()
                out = model(xb)
                times.append((time.perf_counter() - t0) / xb.shape[0])
                preds.append(out.float().cpu().numpy())
        # Back to m/s, which is what the scorer compares against.
        pred = np.concatenate(preds) * std3 + mean3
        t_neural = float(np.mean(times))

        s = score_arrays(pred, y, mean_t_neural_s=t_neural)
        s["_selection"] = float(np.mean([s["rel_l2_score"], s["tke_score"],
                                         s["mvpe_score"]]))
        rows.append((name, s, t_neural))
        print(format_scores(s))
        print()

    if not rows:
        return

    # Ours, measured the same way, for the side-by-side.
    OURS = {
        "tke0 (submission 1)": dict(rel_l2_score=96.06, tke_score=75.37,
                                    mvpe_score=96.10, sps_score=22.07,
                                    _selection=89.17),
        "adv_finetuned (submission 2)": dict(rel_l2_score=96.25, tke_score=76.32,
                                             mvpe_score=96.54, sps_score=22.96,
                                             _selection=89.70),
    }
    hdr = (f"{'model':<32}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'sps*':>7}"
           f"{'ms/sample':>11}{'selection':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, s, t in rows:
        print(f"{'official ' + name:<32}{s['rel_l2_score']:>9.2f}{s['tke_score']:>8.2f}"
              f"{s['mvpe_score']:>8.2f}{s['sps_score']:>7.2f}"
              f"{1000 * t:>11.1f}{s['_selection']:>11.2f}")
    for name, s in OURS.items():
        print(f"{'ours: ' + name:<32}{s['rel_l2_score']:>9.2f}{s['tke_score']:>8.2f}"
              f"{s['mvpe_score']:>8.2f}{s['sps_score']:>7.2f}{'—':>11}{s['_selection']:>11.2f}")
    print("\n* sps with the scorer's default band, no bounds supplied — comparable")
    print("  across rows. Our submitted bounds lift it to 48.7 locally.")
    print("  ms/sample is CPU here; the platform runs on GPU.")


if __name__ == "__main__":
    main()
