"""Train the fluctuation flow-matching generator on top of a frozen forecaster.

The deterministic checkpoint supplies the temporal mean and is not updated. Only
the velocity field is trained, on the fluctuation the deterministic model cannot
produce.

Evaluation composes them the way a submission would:

    prediction = temporal_mean(deterministic) + sampled_fluctuation

and scores it with the organizers' scoring.py, alongside the deterministic model
alone so the trade is visible on every subscore rather than just tke.

Same thermal discipline as train.py: this machine's GPU fan does not work.

Usage:
    python scripts/train_flow.py --epochs 40
    python scripts/train_flow.py --epochs 40 --steps 4 --tag flow_s4
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from realpde.data import build_datasets, denormalize
from realpde.flow import FluctuationFlow, flow_loss, sample_fluctuation, temporal_mean
from realpde.local_score import format_scores, score_arrays
from realpde.models import build_model, count_parameters

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"


def gpu_query():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return tuple(float(s) for s in out.split(","))
    except Exception:
        return None


def gpu_status():
    q = gpu_query()
    return "gpu n/a" if q is None else f"gpu {q[0]:.0f}%  {q[1]:.0f}C  {q[2]:.0f}W"


def thermal_guard(max_temp, resume_temp):
    q = gpu_query()
    if q is None or q[1] < max_temp:
        return False
    print(f"\n  !! GPU at {q[1]:.0f}C (limit {max_temp:.0f}C) — pausing to cool", flush=True)
    while True:
        time.sleep(5.0)
        q = gpu_query()
        if q is None or q[1] <= resume_temp:
            break
    print(f"  resumed at {q[1] if q else float('nan'):.0f}C\n", flush=True)
    return True


@torch.no_grad()
def evaluate(det, flow, val_loader, device, steps, norm_mean, norm_std, seed=0):
    det.eval(); flow.eval()
    det_preds, gen_preds, targets, times = [], [], [], []
    g = torch.Generator(device=device).manual_seed(seed)
    for batch in val_loader:
        x = batch["input"][..., :2].to(device)
        y = batch["target"][..., :2]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = det(x)
        mean = temporal_mean(d)
        f = sample_fluctuation(flow, x, mean, d - mean, norm_mean, norm_std,
                               steps=steps, generator=g)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / x.shape[0])
        det_preds.append(d.cpu())
        gen_preds.append((mean + f).cpu())
        targets.append(y)

    tgt = denormalize(torch.cat(targets), 2).numpy()
    dp = denormalize(torch.cat(det_preds), 2).numpy()
    gp = denormalize(torch.cat(gen_preds), 2).numpy()
    t_neural = float(np.mean(times))
    return (score_arrays(dp, tgt, mean_t_neural_s=0.0012),
            score_arrays(gp, tgt, mean_t_neural_s=t_neural), t_neural)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-checkpoint", type=Path,
                    default=CKPT_DIR / "adv_finetuned_best.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6, help="Euler steps when sampling")
    ap.add_argument("--start-noise", type=float, default=0.5,
                    help="noise added to the deterministic starting point, in "
                         "units of the fluctuation scale; 0 makes the refinement "
                         "deterministic")
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--gpu-duty", type=float, default=0.72)
    ap.add_argument("--max-temp", type=float, default=78.0)
    ap.add_argument("--resume-temp", type=float, default=70.0)
    ap.add_argument("--tag", default="flow")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.det_checkpoint, map_location="cpu")
    dargs = state.get("args", {})
    det = build_model(dargs.get("model", "unet"), base=dargs.get("base", 64), channels=2)
    det.load_state_dict(state["model"])
    det.eval().to(device)
    for p in det.parameters():
        p.requires_grad_(False)

    train_ds, val_ds = build_datasets("real", train_stride=args.train_stride)

    # Needed to denormalise the last input frame inside the flow, where the
    # derived physics channels are computed in m/s.
    from realpde.data import OFFICIAL_MEAN_REAL, OFFICIAL_STD_REAL
    norm_mean = torch.from_numpy(OFFICIAL_MEAN_REAL[:2]).to(device)
    norm_std = torch.from_numpy(OFFICIAL_STD_REAL[:2]).to(device)

    # Measure the fluctuation's scale on the training split so the flow's target
    # has unit variance. Both ends of the interpolation then live at the same
    # scale; otherwise the network spends the low-t half of training being asked
    # to predict the noise draw, which it cannot.
    with torch.no_grad():
        idx = np.linspace(0, len(train_ds) - 1, 400).astype(int)
        yt = torch.stack([train_ds[int(i)]["target"][..., :2] for i in idx]).to(device)
        xt = torch.stack([train_ds[int(i)]["input"][..., :2] for i in idx]).to(device)
        mt = torch.cat([temporal_mean(det(xt[i:i + 32])) for i in range(0, len(xt), 32)])
        fluct_scale = float((yt - mt).std())
    print(f"fluctuation scale measured on train: {fluct_scale:.4f}")

    flow = FluctuationFlow(base=args.base, fluct_scale=fluct_scale,
                           start_noise=args.start_noise).to(device)
    n_par, mb = count_parameters(flow)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\ndevice {device}   {gpu_status()}")
    print(f"frozen forecaster: {args.det_checkpoint.name}")
    print(f"flow model: {n_par:,} params, {mb:.1f} MB, base={args.base}")
    print(f"data: {len(train_ds)} train / {len(val_ds)} val windows")
    print(f"sampling with {args.steps} Euler steps")
    print(f"gpu duty {args.gpu_duty:.0%}, pause above {args.max_temp:.0f}C\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sleep_ratio = max(1.0 / max(args.gpu_duty, 1e-3) - 1.0, 0.0)
    best, history, peak, pauses = -1.0, [], 0.0, 0

    for epoch in range(1, args.epochs + 1):
        flow.train()
        tot, nb = 0.0, 0
        t_epoch = time.perf_counter()
        for step, batch in enumerate(train_loader):
            if step % 25 == 0:
                q = gpu_query()
                if q:
                    peak = max(peak, q[1])
                if thermal_guard(args.max_temp, args.resume_temp):
                    pauses += 1
            t0 = time.perf_counter()
            x = batch["input"][..., :2].to(device, non_blocking=True)
            y = batch["target"][..., :2].to(device, non_blocking=True)
            with torch.no_grad():
                d = det(x)
                mean = temporal_mean(d)
            fluct = y - mean

            opt.zero_grad(set_to_none=True)
            loss = flow_loss(flow, fluct, x, mean, d - mean, norm_mean, norm_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if sleep_ratio > 0:
                time.sleep((time.perf_counter() - t0) * sleep_ratio)

        sched.step()
        print(f"epoch {epoch:>3}/{args.epochs}  flow_mse {tot / nb:.5f}  "
              f"[{time.perf_counter() - t_epoch:.1f}s, {gpu_status()}]", flush=True)

        if epoch % 5 == 0 or epoch == args.epochs:
            d_s, g_s, t_n = evaluate(det, flow, val_loader, device, args.steps,
                                     norm_mean, norm_std)
            print(f"  deterministic : rel_l2 {d_s['rel_l2_score']:.2f}  "
                  f"tke {d_s['tke_score']:.2f}  mvpe {d_s['mvpe_score']:.2f}")
            print(f"  + generated   : rel_l2 {g_s['rel_l2_score']:.2f}  "
                  f"tke {g_s['tke_score']:.2f}  mvpe {g_s['mvpe_score']:.2f}  "
                  f"time {g_s['time_score']:.2f} ({1000 * t_n:.1f} ms/sample)")
            sel = float(np.mean([g_s["rel_l2_score"], g_s["tke_score"], g_s["mvpe_score"]]))
            history.append({"epoch": epoch, "det": d_s, "gen": g_s, "selection": sel})
            if sel > best:
                best = sel
                torch.save({"model": flow.state_dict(), "args": vars(args),
                            "scores": g_s, "epoch": epoch},
                           CKPT_DIR / f"{args.tag}_best.pt")
                print(f"  -> new best (selection {best:.3f}), saved")
            print(flush=True)

    (CKPT_DIR / f"{args.tag}_history.json").write_text(json.dumps(history, indent=2, default=float))
    print(f"done. best selection {best:.3f}")
    print(f"thermals: peak {peak:.0f}C, {pauses} pause(s), final {gpu_status()}")


if __name__ == "__main__":
    main()
