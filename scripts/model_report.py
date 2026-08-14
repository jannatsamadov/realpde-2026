"""One figure that says what a model gets right and where it fails.

Six panels, chosen so that the next improvement is visible rather than inferred:

  row 1  vorticity at the LAST output frame — truth, prediction, difference.
         The last frame is the hardest one; showing the first would flatter.
  row 2  TKE maps, truth and prediction, plus the spatial error map.
         The error map is the actionable one: it says WHERE the model fails,
         which is what a targeted fix needs.
  row 3  error against lead time, and fluctuation correlation against lead time.

Only the ground truth is masked. The prediction panels are raw model output, so
a blank shadow means the model really learned to emit nothing there.

Usage:
    python scripts/model_report.py --checkpoint checkpoints/tke0_best.pt
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

from realpde.data import build_datasets, denormalize
from realpde.local_score import score_arrays
from realpde.models import build_model

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
DX = 0.003422


def vorticity(u, v, dx=DX):
    return np.gradient(v, dx, axis=-1) - np.gradient(u, -dx, axis=-2)


def tke(f):
    """(T, H, W, C) -> (H, W): variance over the time axis, per scoring.py."""
    u, v = f[..., 0], f[..., 1]           # each (T, H, W)
    return 0.5 * (u.var(axis=0) + v.var(axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    targs = state.get("args", {})
    # The checkpoint records which architecture produced it; older ones predate
    # the --model flag and are plain U-Nets.
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, val = build_datasets("real")
    n = len(val)
    X = torch.stack([val[i]["input"] for i in range(n)])
    Y = torch.stack([val[i]["target"] for i in range(n)])
    with torch.no_grad():
        P = torch.cat([model(X[i:i + 32]) for i in range(0, n, 32)])
    p_all = denormalize(P, 2).numpy()
    y_all = denormalize(Y, 2).numpy()
    x_all = denormalize(X, 2).numpy()

    scores = score_arrays(p_all, y_all, mean_t_neural_s=0.0012)

    k = args.window
    y, p = y_all[k], p_all[k]
    b = val[k]
    mask_t = (y == 0).all(axis=-1)

    fig = plt.figure(figsize=(16, 10.5), dpi=110)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85], hspace=0.35, wspace=0.18)

    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#9a9a9a")
    hot = plt.get_cmap("inferno").copy(); hot.set_bad("#4a4a4a")

    # --- row 1: vorticity at the final, hardest output frame ---
    last = -1
    vt = np.where(mask_t[last], np.nan, vorticity(y[last, ..., 0], y[last, ..., 1]))
    vp = vorticity(p[last, ..., 0], p[last, ..., 1])
    vd = np.where(mask_t[last], np.nan,
                  vorticity((p - y)[last, ..., 0], (p - y)[last, ..., 1]))
    lim = float(np.nanpercentile(np.abs(vt), 99))
    for j, (d, t) in enumerate([(vt, "truth"), (vp, "prediction (raw)"), (vd, "difference")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(d, cmap=cmap, vmin=-lim, vmax=lim, aspect="equal", interpolation="nearest")
        ax.set_title(f"vorticity, output frame 20 of 20 — {t}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # --- row 2: TKE truth / prediction, and where the error lives ---
    kt = np.where(mask_t.all(axis=0), np.nan, tke(y))
    kp = tke(p)
    kmax = float(np.nanpercentile(kt, 99))
    for j, (d, t) in enumerate([(kt, "TKE truth"), (kp, "TKE prediction (raw)")]):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(d, cmap=hot, vmin=0, vmax=kmax, aspect="equal", interpolation="nearest")
        ax.set_title(f"{t}   (sum {np.nansum(d):.3e})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Spatial error, averaged over every validation window: where to fix next.
    err_map = np.sqrt(((p_all - y_all) ** 2).sum(axis=-1)).mean(axis=(0, 1))
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(err_map, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title(f"mean |error| over all {n} windows — WHERE it fails", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # --- row 3: how the error and the eddies decay with lead time ---
    T = p_all.shape[1]
    persist = np.repeat(x_all[:, -1:], T, axis=1)
    rel = lambda a, t_: (np.linalg.norm((a - t_).reshape(len(a), -1), axis=1)
                         / np.linalg.norm(t_.reshape(len(t_), -1), axis=1)).mean()
    e_m = [rel(p_all[:, i:i + 1], y_all[:, i:i + 1]) for i in range(T)]
    e_p = [rel(persist[:, i:i + 1], y_all[:, i:i + 1]) for i in range(T)]

    ax = fig.add_subplot(gs[2, 0])
    frames = np.arange(1, T + 1)
    ax.plot(frames, e_m, "o-", label="model", lw=1.6, ms=3)
    ax.plot(frames, e_p, "s--", label="persistence", lw=1.4, ms=3, color="#888")
    ax.set_xlabel("output frame"); ax.set_ylabel("relative L2 error")
    ax.set_title("error vs lead time", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ym, pm = y_all.mean(axis=1, keepdims=True), p_all.mean(axis=1, keepdims=True)
    yf, pf = y_all - ym, p_all - pm
    corr, amp = [], []
    for i in range(T):
        a, c_ = pf[:, i], yf[:, i]
        corr.append(float((a * c_).sum() / max(np.linalg.norm(a) * np.linalg.norm(c_), 1e-12)))
        amp.append(float(np.linalg.norm(a) / max(np.linalg.norm(c_), 1e-12)))
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(frames, corr, "o-", label="correlation", lw=1.6, ms=3)
    ax.plot(frames, amp, "s-", label="amplitude kept", lw=1.6, ms=3)
    ax.axhline(1.0, color="#bbb", ls=":", lw=1)
    ax.set_xlabel("output frame"); ax.set_ylim(0, 1.15)
    ax.set_title("eddies: correlation and amplitude vs lead time", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    rows = [("rel_l2", scores["rel_l2_score"], 94.0), ("tke", scores["tke_score"], 66.7),
            ("mvpe", scores["mvpe_score"], 94.4), ("sps", scores["sps_score"], 17.5)]
    txt = f"{'subscore':<10}{'model':>9}{'best trivial':>14}{'gain':>8}\n" + "-" * 41 + "\n"
    for name, v, base in rows:
        txt += f"{name:<10}{v:>9.2f}{base:>14.2f}{v - base:>+8.2f}\n"
    txt += "-" * 41 + "\n"
    txt += f"{'fluct corr':<10}{np.mean(corr):>9.3f}\n{'amp kept':<10}{np.mean(amp):>9.3f}\n"
    txt += f"{'tke err':<10}{scores['_tke_error']:>9.4f}\n"
    ax.text(0, 1, txt, family="monospace", fontsize=10, va="top")

    fig.suptitle(
        f"{args.checkpoint.stem}   —   window {k}, case {b['case']} "
        f"(Re={b['re']:.0f}, AoA={b['aoa']:.0f}°)   —   scored on all {n} validation windows",
        fontsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"report_{args.checkpoint.stem}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
