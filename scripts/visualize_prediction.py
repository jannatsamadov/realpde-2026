"""Animate a prediction against the ground truth, the way the data itself is drawn.

Two outputs per run:

  *_pred.gif   three panels over the 20 output frames — truth, prediction, and
               their difference — so a prediction that is merely blurry is
               immediately distinguishable from one that is wrong.
  *_tke.png    the turbulent kinetic energy map of truth vs prediction, which is
               the quantity tke_score actually measures. A smooth prediction has
               almost no TKE and the panel goes blank; that is what a TKE error
               of 1.0 looks like.

Works on any predictor. The built-in baselines are there to show the failure
mode; `--checkpoint` will load a trained model once we have one.

Usage:
    python scripts/visualize_prediction.py --predictor time_mean
    python scripts/visualize_prediction.py --predictor persistence --window 40
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

from realpde.data import T_OUT, build_datasets, denormalize
from realpde.local_score import score_arrays
from realpde.models import build_model

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
DX = 0.001711


def load_checkpoint_predictor(path: Path, device: str = "cpu"):
    """Wrap a trained checkpoint as a predictor over denormalized m/s windows.

    Defaults to CPU: the model is 1.76M parameters, so a handful of windows costs
    milliseconds, and it keeps this off a GPU that may be busy training on a
    machine whose fan does not work.
    """
    from realpde.data import OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL

    state = torch.load(path, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval().to(device)

    mean = OFFICIAL_MEAN_REAL[:2]
    std = OFFICIAL_STD_REAL[:2]

    @torch.no_grad()
    def predict(x_ms: np.ndarray) -> np.ndarray:
        """(T_IN, H, W, 2) in m/s -> (T_OUT, H, W, 2) in m/s.

        Normalize, run, denormalize — the same path submission.py takes, so what
        is drawn here is what would be submitted.
        """
        xn = (x_ms - mean) / std
        xt = torch.from_numpy(np.ascontiguousarray(xn, dtype=np.float32))[None].to(device)
        yn = model(xt)[0].cpu().numpy()
        return yn * std + mean

    return predict, state


def vorticity(u, v, dx=DX):
    """omega_z with the array's own row direction (y decreases down the array)."""
    return np.gradient(v, dx, axis=-1) - np.gradient(u, -dx, axis=-2)


def tke(field):
    """0.5 * (var_t(u) + var_t(v)) over the time axis — scoring.py's definition."""
    u, v = field[..., 0], field[..., 1]
    return 0.5 * (u.var(axis=0) + v.var(axis=0))


# These are NOT models. They are arithmetic on the input window, included to show
# the floor and to make the TKE failure mode visible. `last_window` in particular
# returns the input unchanged, so of course it reproduces the laser shadow — that
# is the definition of the baseline, not a symptom of leakage or overfitting.
PREDICTORS = {
    "persistence": lambda x: np.repeat(x[-1:], T_OUT, axis=0),
    "last_window": lambda x: x.copy(),
    "time_mean": lambda x: np.repeat(x.mean(axis=0, keepdims=True), T_OUT, axis=0),
}

# Anything not in PREDICTORS is a trained model; label the figures accordingly.
BASELINE_NOTE = "TRIVIAL BASELINE (no model, no training)"


def animate(truth, pred, title, out_path, fps=6):
    # Only the ground truth is masked. Greying the prediction with the truth's
    # mask would draw the laser shadow onto the model's panel and make it look
    # as though the model reproduced it — the prediction panel shows exactly what
    # the model emits, shadow region included.
    mask = (truth == 0).all(axis=-1)
    diff = pred - truth

    vt = np.where(mask, np.nan, vorticity(truth[..., 0], truth[..., 1]))
    vp = vorticity(pred[..., 0], pred[..., 1])
    vd = np.where(mask, np.nan, vorticity(diff[..., 0], diff[..., 1]))

    fin = vt[np.isfinite(vt)]
    vmax = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#9a9a9a")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), dpi=110)
    ims = []
    for ax, data, name in zip(axes, [vt, vp, vd],
                              ["ground truth", "prediction", "difference"]):
        im = ax.imshow(data[0], cmap=cmap, vmin=-vmax, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    sup = fig.suptitle("")

    def update(k):
        for im, data in zip(ims, [vt, vp, vd]):
            im.set_data(data[k])
        sup.set_text(f"{title}\noutput frame {k + 1}/{T_OUT}   "
                     f"(t = {(k + 1) * 0.02:.2f}s ahead)   vorticity")
        return ims

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=T_OUT, blit=False).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out_path}")


def tke_figure(truth, pred, title, out_path):
    # As in animate(): mask only the ground truth. The prediction panel is the
    # model's raw output, so a blank shadow region there means the model really
    # learned to emit nothing where the PIV measured nothing.
    mask = (truth == 0).all(axis=-1).all(axis=0)
    kt = np.where(mask, np.nan, tke(truth))
    kp = tke(pred)

    fin = kt[np.isfinite(kt)]
    vmax = float(np.percentile(fin, 99)) if fin.size else 1.0

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#4a4a4a")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), dpi=115)
    names = ["ground truth (grey = no PIV measurement)", "prediction (raw model output, unmasked)"]
    for ax, data, name in zip(axes, [kt, kp], names):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        total = np.nansum(data)
        ax.set_title(f"TKE — {name}   (sum = {total:.3e})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)

    # Compare like with like: the ratio is taken over the measured cells only.
    ratio = np.nansum(np.where(mask, np.nan, kp)) / max(np.nansum(kt), 1e-30)
    fig.suptitle(f"{title}\nprediction retains {100 * ratio:.1f}% of the true "
                 f"turbulent kinetic energy", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="time_mean",
                    help=f"one of {sorted(PREDICTORS)}, or ignored when --checkpoint is given")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="a trained checkpoint; overrides --predictor")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--window", type=int, default=0, help="index into the validation set")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    _, val = build_datasets("real")
    b = val[args.window]
    case = b["case"]
    x = denormalize(b["input"], 2).numpy()
    y = denormalize(b["target"], 2).numpy()

    if args.checkpoint:
        fn, state = load_checkpoint_predictor(args.checkpoint, args.device)
        pred = fn(x)
        name = args.checkpoint.stem
        ep = state.get("epoch", "?")
        label = f"MODEL {name} (epoch {ep}, {args.device})"
    else:
        if args.predictor not in PREDICTORS:
            ap.error(f"unknown predictor {args.predictor}; choose from {sorted(PREDICTORS)}")
        pred = PREDICTORS[args.predictor](x)
        name = args.predictor
        label = f"{args.predictor}  [{BASELINE_NOTE}]"

    scores = score_arrays(pred[None], y[None], mean_t_neural_s=0.005)
    title = (f"{label}\n"
             f"case {case}   Re={b['re']:.0f}  AoA={b['aoa']:.0f}°\n"
             f"rel_l2 {scores['rel_l2_score']:.1f}   tke {scores['tke_score']:.1f}   "
             f"mvpe {scores['mvpe_score']:.1f}   sps {scores['sps_score']:.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"pred_{name}_{case}_w{args.window}"
    animate(y, pred, title, OUT_DIR / f"{stem}.gif", fps=args.fps)
    tke_figure(y, pred, title, OUT_DIR / f"{stem}_tke.png")


if __name__ == "__main__":
    main()
