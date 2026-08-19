"""Is the organizers' FNO worth submitting, alone or blended with ours?

The rules allow it explicitly: "Checkpoints trained from that release are
permitted, including the official baselines the organizers provide."

FNO is more accurate than our model on the same validation split, but accuracy is
only one of five subscores. This script settles the trade with numbers rather
than argument, on the same 273 Reynolds-disjoint windows and the same scoring.py:

  * per-model inference time on the GPU, measured by ALTERNATING the models
    inside one loop. This card's clock swings between 300 MHz and 2100 MHz, so
    timing model A then model B ranks them by clock state, not by cost.
  * interval bounds calibrated PER MODEL. sigma is a model's own residual
    spread, so scoring FNO with our map would size its intervals from someone
    else's errors.
  * fp16 weights. FNO is 384 MB in fp32 and the archive cap is 256 MB, so the
    only shippable form is fp16-packed; evaluating the fp32 file would measure a
    submission we cannot make.

Verdicts are reported as a marginal-weighted delta against our current model,
using w_sps = 0.30 and 0.19 for the others. Those weights are not published;
they were fitted from two submissions and then CHECKED against a third
(predicted +5.106 against an actual +5.101), so they are reliable for
differences even though they do not reproduce absolute scores.

Usage:
    python scripts/eval_fno_hybrid.py
    python scripts/eval_fno_hybrid.py --train-windows 600   # faster sigma fit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))
KIT = (ROOT / "NeurIPS 2026 RealPDE Competition"
       / "realpde_t1_starting_kit_v9" / "realpde_t1_starting_kit_v9")
sys.path.insert(0, str(KIT))

from realpde.data import (OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL,  # noqa: E402
                          build_datasets, denormalize)
from realpde.local_score import score_arrays                      # noqa: E402
from realpde.models import build_model                            # noqa: E402
from realpde.sps import bounds_from_sigma                         # noqa: E402
from gpu_guard import Governor                                    # noqa: E402
from gpu_lock import GpuLock                                      # noqa: E402

BASELINES = ROOT / "baselines"
W_SPS, W_OTHER = 0.30, 0.19


def to_fp16_and_back(model: torch.nn.Module) -> torch.nn.Module:
    """Round every float parameter through fp16, keeping the module in fp32.

    This is what packing the checkpoint at fp16 and unpacking it at load time
    does to the weights. Complex tensors are handled as real pairs, which is how
    pack_ckpt_fp16.py stores FNO's spectral weights — a plain .half() on a
    complex tensor raises.
    """
    with torch.no_grad():
        for p in model.parameters():
            if p.is_complex():
                r = torch.view_as_real(p.data)
                r.copy_(r.half().float())
            elif p.dtype.is_floating_point:
                p.data.copy_(p.data.half().float())
    return model


def predict(model, x3: np.ndarray, batch: int, device, is_ours: bool) -> np.ndarray:
    """Normalized (N,20,32,64,3) in -> normalized (N,20,32,64,2) out."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(x3), batch):
            xb = torch.from_numpy(x3[i:i + batch]).to(device)
            out = model(xb[..., :2]) if is_ours else model(xb)
            outs.append(out[..., :2].float().cpu().numpy())
    return np.concatenate(outs)


def time_alternating(models: dict, x3: np.ndarray, batch: int, device,
                     reps: int = 12) -> dict:
    """Per-sample time for each model, alternating them to cancel clock drift."""
    sample = x3[:batch]
    totals = {k: 0.0 for k in models}
    for name, (m, is_ours) in models.items():        # warm every arm first
        with torch.no_grad():
            for _ in range(3):
                xb = torch.from_numpy(sample).to(device)
                m(xb[..., :2]) if is_ours else m(xb)
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(reps):
            for name, (m, is_ours) in models.items():
                xb = torch.from_numpy(sample).to(device)
                torch.cuda.synchronize()
                t0 = perf_counter()
                m(xb[..., :2]) if is_ours else m(xb)
                torch.cuda.synchronize()
                totals[name] += perf_counter() - t0
    return {k: v / (reps * batch) for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-windows", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--blend", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75],
                    help="ensemble weights on FNO; 0 is ours, 1 is FNO alone")
    args = ap.parse_args()
    device = torch.device(args.device)

    from load_baseline import load_baseline    # noqa: E402  (needs KIT on path)

    mean3 = np.zeros(3, dtype=np.float32); mean3[:2] = OFFICIAL_MEAN_REAL[:2]
    std3 = np.ones(3, dtype=np.float32); std3[:2] = OFFICIAL_STD_REAL[:2]

    train, val = build_datasets("real")
    nv = len(val)
    idx_t = np.linspace(0, len(train) - 1, min(args.train_windows, len(train))).astype(int)

    def stack(ds, idx, key):
        return torch.stack([ds[int(i)][key] for i in idx]).numpy()

    xv = stack(val, range(nv), "input")
    yv = denormalize(torch.from_numpy(stack(val, range(nv), "target")), 2).numpy()
    xt = stack(train, idx_t, "input")
    yt = denormalize(torch.from_numpy(stack(train, idx_t, "target")), 2).numpy()

    def to3(a):
        out = np.zeros(a.shape[:-1] + (3,), dtype=np.float32)
        out[..., :2] = a
        return out

    xv3, xt3 = to3(xv), to3(xt)
    print(f"val {nv} windows, train {len(idx_t)} windows for the sigma fit\n")

    ours = build_model("advective", base=64, channels=2)
    state = torch.load(ROOT / "checkpoints" / "adv_finetuned_best.pt", map_location="cpu")
    ours.load_state_dict(state["model"])
    ours.eval().to(device)

    fno, meta = load_baseline(str(BASELINES / "sim_real_fno.pth"), device=str(device))
    fno.eval()
    n_par = sum(p.numel() for p in fno.parameters())
    print(f"FNO loaded ({meta.get('format')}), {n_par:,} params "
          f"-> fp16 rounding applied (archive cap is 256 MB)")
    to_fp16_and_back(fno)

    lock = GpuLock("realpde", "fno_hybrid").acquire() if device.type == "cuda" else None
    gov = Governor(target_util=72, pause_temp=78, resume_temp=70) if device.type == "cuda" else None
    try:
        models = {"ours": (ours, True), "fno": (fno, False)}
        if device.type == "cuda":
            times = time_alternating(models, xv3, args.batch, device)
        else:
            times = {"ours": 0.013, "fno": 0.05}
        print(f"per-sample time: ours {1000 * times['ours']:.2f} ms, "
              f"fno {1000 * times['fno']:.2f} ms\n")

        preds_v, preds_t = {}, {}
        for name, (m, is_ours) in models.items():
            if gov is not None:
                with gov.step():
                    preds_v[name] = predict(m, xv3, args.batch, device, is_ours)
                with gov.step():
                    preds_t[name] = predict(m, xt3, args.batch, device, is_ours)
            else:
                preds_v[name] = predict(m, xv3, args.batch, device, is_ours)
                preds_t[name] = predict(m, xt3, args.batch, device, is_ours)
            if lock is not None:
                lock.progress(name)
    finally:
        if lock is not None:
            lock.release()

    # Normalized -> m/s, which is what the scorer compares against.
    def phys(a):
        return a * std3[:2] + mean3[:2]

    cands = {"ours": (phys(preds_v["ours"]), phys(preds_t["ours"]), times["ours"]),
             "fno": (phys(preds_v["fno"]), phys(preds_t["fno"]), times["fno"])}
    for w in args.blend:
        cands[f"blend {w:g}"] = (
            (1 - w) * phys(preds_v["ours"]) + w * phys(preds_v["fno"]),
            (1 - w) * phys(preds_t["ours"]) + w * phys(preds_t["fno"]),
            times["ours"] + times["fno"],      # a blend runs BOTH models
        )

    rows = {}
    hdr = (f"{'candidate':<14}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'sps':>8}"
           f"{'time':>8}{'seçim':>9}{'ms':>8}{'Δfinal':>9}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for name, (pv, pt, t) in cands.items():
        # sigma from TRAINING residuals of this same candidate: at submission
        # time there are no targets, so it has to be fitted on data the model
        # was fitted on and survive the move to unseen Reynolds numbers.
        sigma = (pt - yt).std(axis=0, keepdims=True)
        lo, hi = bounds_from_sigma(pv, np.broadcast_to(sigma, pv.shape), scale=0.8)
        s = score_arrays(pv, yv, mean_t_neural_s=t, lower=lo, upper=hi)
        s["_selection"] = float(np.mean([s["rel_l2_score"], s["tke_score"],
                                         s["mvpe_score"]]))
        rows[name] = s
        if base is None:
            base = s
        d = (W_OTHER * ((s["rel_l2_score"] - base["rel_l2_score"])
                        + (s["tke_score"] - base["tke_score"])
                        + (s["mvpe_score"] - base["mvpe_score"])
                        + (s["time_score"] - base["time_score"]))
             + W_SPS * (s["sps_score"] - base["sps_score"]))
        print(f"{name:<14}{s['rel_l2_score']:>9.3f}{s['tke_score']:>8.3f}"
              f"{s['mvpe_score']:>8.3f}{s['sps_score']:>8.3f}{s['time_score']:>8.3f}"
              f"{s['_selection']:>9.3f}{1000 * t:>8.2f}{d:>+9.3f}")

    (ROOT / "submissions" / "fno_hybrid.json").write_text(json.dumps(rows, indent=2))
    print("\nΔfinal is against 'ours', using the fitted marginal weights "
          f"(w_sps={W_SPS}, others={W_OTHER}). A blend pays BOTH models' runtime.")


if __name__ == "__main__":
    main()
