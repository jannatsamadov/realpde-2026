"""Animate one operating point four ways: real, model, error, and simulation.

Four panels over the 20 forecast frames:

    real truth        what the model is scored against
    model             its prediction for the same window
    difference        where it goes wrong
    simulation        the same nominal Reynolds and angle, computed rather than
                      measured — no laser shadow, no PIV noise

The simulation panel is a different realisation of the same operating point, not
the same flow instance, so it is a reference for what the structure looks like
when nothing is missing rather than something the model should reproduce
frame-for-frame. Comparing it against the real panel shows how much of what the
model faces is measurement rather than physics.

Only the real truth is masked; the model panel is raw output.

Usage:
    python scripts/regime_gif.py --case 26700_15
    python scripts/regime_gif.py --case 3750_0 --window 20
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
from matplotlib.animation import FuncAnimation, PillowWriter

from realpde.data import T_IN, T_OUT, WindowDataset, denormalize
from realpde.local_score import score_arrays
from realpde.models import build_model

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
DX = 0.003422


def vorticity(u, v, dx=DX):
    return np.gradient(v, dx, axis=-1) - np.gradient(u, -dx, axis=-2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="26700_15")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "adv_finetuned_best.pt")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    real = WindowDataset("real", cases=[args.case], stride=T_IN + T_OUT)
    if len(real) == 0:
        raise SystemExit(f"no real windows for {args.case}")
    w = min(args.window, len(real) - 1)
    b = real[w]
    x = b["input"][..., :2][None]
    with torch.no_grad():
        p = model(x)
    pred = denormalize(p, 2).numpy()[0]
    targ = denormalize(b["target"][..., :2], 2).numpy()

    s = score_arrays(pred[None], targ[None], mean_t_neural_s=0.0176)

    # The same nominal operating point, simulated.
    try:
        sim_ds = WindowDataset("sim", cases=[args.case], stride=T_IN + T_OUT)
        sim = denormalize(sim_ds[min(w, len(sim_ds) - 1)]["target"][..., :2], 2).numpy()
    except Exception:
        sim = None

    mask = (targ == 0).all(axis=-1)
    diff = pred - targ
    v_real = np.where(mask, np.nan, vorticity(targ[..., 0], targ[..., 1]))
    v_pred = vorticity(pred[..., 0], pred[..., 1])
    v_diff = np.where(mask, np.nan, vorticity(diff[..., 0], diff[..., 1]))
    v_sim = vorticity(sim[..., 0], sim[..., 1]) if sim is not None else None

    fin = v_real[np.isfinite(v_real)]
    lim = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")

    panels = [(v_real, "REAL — measured (grey = no PIV data)"),
              (v_pred, "MODEL — prediction, raw"),
              (v_diff, "DIFFERENCE — model minus real")]
    if v_sim is not None:
        panels.append((v_sim, "SIMULATION — same Re and AoA, computed"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.5 * len(panels)), dpi=110)
    ims = []
    for ax, (data, label) in zip(axes, panels):
        im = ax.imshow(data[0], cmap=cmap, vmin=-lim, vmax=lim,
                       aspect="equal", interpolation="nearest")
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    sup = fig.suptitle("")

    def update(k):
        for im, (data, _) in zip(ims, panels):
            im.set_data(data[k])
        sup.set_text(
            f"{args.case}   Re={b['re']:.0f}  AoA={b['aoa']:.0f}°   "
            f"window {w}\n"
            f"forecast frame {k + 1}/{T_OUT}  (t = {0.02 * (k + 1):.2f}s ahead)   "
            f"rel_l2 {s['rel_l2_score']:.1f}  tke {s['tke_score']:.1f}  "
            f"mvpe {s['mvpe_score']:.1f}")
        return ims

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"regime_{args.case}.gif"
    FuncAnimation(fig, update, frames=T_OUT, blit=False).save(
        out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Re={b['re']:.0f} AoA={b['aoa']:.0f}  "
          f"rel_l2 {s['rel_l2_score']:.2f}  tke {s['tke_score']:.2f}  "
          f"mvpe {s['mvpe_score']:.2f}")


if __name__ == "__main__":
    main()
