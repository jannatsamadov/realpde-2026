"""Run the best model across five simulated regimes and show what it does.

The model was fine-tuned on real PIV, so the simulated split is out of
distribution for it: no laser shadow, no measurement noise, cleaner structure.
That makes it a useful test rather than a mismatch, for two reasons.

  * Regime coverage. Five cases spanning Reynolds 3750 to 27975 and angle of
    attack 0 to 20 degrees show whether behaviour degrades smoothly with regime
    or falls off somewhere. Two of these Reynolds numbers (15225, 27975) exist
    only in simulation, so the model has never seen them in any form.

  * The noise ceiling. On real data 22% of the target's temporal variance is PIV
    noise and cannot be predicted, which caps tke_score near 90. Simulation has
    no such noise. If tke_score is markedly higher here, that ceiling is real and
    the remaining gap on real data is smaller than it looks.

Usage:
    python scripts/sim_regime_report.py
    python scripts/sim_regime_report.py --cases 3750_0 20325_20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from realpde.data import WindowDataset, denormalize
from realpde.local_score import score_arrays
from realpde.models import build_model

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
DX = 0.003422

# A diagonal through the parameter space: Reynolds and angle both rise together,
# so five rows cover the corners rather than five neighbours.
DEFAULT = ["3750_0", "8850_5", "15225_10", "21600_15", "27975_20"]


def vorticity(u, v, dx=DX):
    return np.gradient(v, dx, axis=-1) - np.gradient(u, -dx, axis=-2)


def tke(f):
    return 0.5 * (f[..., 0].var(axis=0) + f[..., 1].var(axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--cases", nargs="+", default=DEFAULT)
    ap.add_argument("--windows", type=int, default=24,
                    help="windows per case used for the scores")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    for case in args.cases:
        ds = WindowDataset("sim", cases=[case], stride=40)
        if len(ds) == 0:
            print(f"{case}: no windows"); continue
        n = min(args.windows, len(ds))
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        x = torch.stack([ds[int(i)]["input"][..., :2] for i in idx])
        y = torch.stack([ds[int(i)]["target"][..., :2] for i in idx])
        with torch.no_grad():
            p = torch.cat([model(x[i:i + 16]) for i in range(0, len(x), 16)])
        pred = denormalize(p, 2).numpy()
        targ = denormalize(y, 2).numpy()
        s = score_arrays(pred, targ, mean_t_neural_s=0.0176)
        meta = ds.trajectories[0]
        rows.append(dict(case=case, re=meta["re"], aoa=meta["aoa"],
                         pred=pred, targ=targ, scores=s))
        print(f"{case:<12} Re={meta['re']:<6} AoA={meta['aoa']:<3} "
              f"rel_l2 {s['rel_l2_score']:.2f}  tke {s['tke_score']:.2f}  "
              f"mvpe {s['mvpe_score']:.2f}")

    if not rows:
        return

    # ---------- figure ----------
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.0 * len(rows)), dpi=110)
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("RdBu_r")
    hot = plt.get_cmap("inferno")

    for r, row in enumerate(rows):
        t, p = row["targ"][0], row["pred"][0]      # first window of this case
        vt = vorticity(t[-1, ..., 0], t[-1, ..., 1])
        vp = vorticity(p[-1, ..., 0], p[-1, ..., 1])
        lim = float(np.percentile(np.abs(vt), 99))
        kt, kp = tke(t), tke(p)
        kmax = float(np.percentile(kt, 99))

        for c, (data, cm, vmin, vmax, label) in enumerate([
            (vt, cmap, -lim, lim, "truth: vorticity, frame 20"),
            (vp, cmap, -lim, lim, "model: vorticity, frame 20"),
            (kt, hot, 0, kmax, "truth: TKE"),
            (kp, hot, 0, kmax, "model: TKE"),
        ]):
            ax = axes[r, c]
            ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax,
                      aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=10)

        s = row["scores"]
        axes[r, 0].set_ylabel(
            f"Re {row['re']}\nAoA {row['aoa']}°\n\n"
            f"rel_l2 {s['rel_l2_score']:.1f}\ntke {s['tke_score']:.1f}\n"
            f"mvpe {s['mvpe_score']:.1f}", fontsize=9, rotation=0,
            labelpad=42, va="center")

    fig.suptitle(
        f"{args.checkpoint.stem} on SIMULATED data — five regimes\n"
        f"trained on real PIV; simulation has no measurement noise and no laser shadow",
        fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "sim_regimes.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")

    # ---------- summary ----------
    print(f"\n{'case':<12}{'Re':>7}{'AoA':>5}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'tke_err':>9}")
    print("-" * 58)
    for row in rows:
        s = row["scores"]
        print(f"{row['case']:<12}{row['re']:>7}{row['aoa']:>5}"
              f"{s['rel_l2_score']:>9.2f}{s['tke_score']:>8.2f}"
              f"{s['mvpe_score']:>8.2f}{s['_tke_error']:>9.4f}")
    mt = np.mean([r["scores"]["tke_score"] for r in rows])
    print(f"\nmean tke on simulation      : {mt:.2f}")
    print(f"tke on real validation      : 76.32")
    print("\nSimulation carries no PIV noise. A clearly higher tke here would say the")
    print("noise ceiling measured on real data is what limits that subscore; a similar")
    print("or lower one would say the limit is the forecasting itself.")


if __name__ == "__main__":
    main()
