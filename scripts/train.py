"""Train a forecaster and score it with the organizers' scoring.py.

GPU DUTY CYCLE
    This machine's GPU fan is not working, so the loop deliberately idles between
    optimizer steps to hold utilization near --gpu-duty (default 0.6). It measures
    each step and sleeps step_time * (1/duty - 1) afterwards, then reports
    temperature and utilization from nvidia-smi as it goes. Training is a few
    minutes at this grid size, so the throttle costs little.

MODEL SELECTION
    final_score's combination rule is unpublished, so the checkpoint is chosen on
    the mean of rel_l2_score, tke_score and mvpe_score. sps_score is reported but
    not selected on: without calibrated bounds it reflects the default band, which
    is a separate piece of work.

Usage:
    python scripts/train.py --epochs 30 --w-tke 0.0          # MSE only
    python scripts/train.py --epochs 30 --w-tke 0.5          # with the TKE term
    python scripts/train.py --split sim --epochs 10 --tag pretrain
    python scripts/train.py --init-from checkpoints/pretrain_best.pt --epochs 30
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# The shared GPU lock lives one level up, beside the other competitions, so every
# project claims the same card through the same file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gpu_guard import Governor, gpu_status
from gpu_lock import GpuLock
from realpde.advection import strain_scaled_noise
from realpde.data import (DEFAULT_VAL_RE, EXTRAPOLATION_VAL_RE, build_datasets,
                          denormalize)
from realpde.local_score import format_scores, score_arrays
from realpde.losses import CompositeLoss, sigma_weights
from realpde.models import build_model, count_parameters, encode_regime

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"


def gpu_line() -> str:
    """One-line card state for the epoch log."""
    u, t, m = gpu_status()
    return "gpu n/a" if t is None else f"gpu {u}%  {t}C  {m}MiB"


@torch.no_grad()
def evaluate(model, val_loader, device, channels: int = 2) -> dict:
    model.eval()
    preds, targets, times = [], [], []
    for batch in val_loader:
        # Slice to the scored channels exactly as the training loop does. The
        # simulated split carries a third (pressure) channel that the model never
        # takes, so omitting this crashes on sim and silently passes on real.
        x = batch["input"][..., :channels].to(device, non_blocking=True)
        y = batch["target"][..., :channels]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / x.shape[0])
        preds.append(p.cpu())
        targets.append(y)

    pred = denormalize(torch.cat(preds), channels).numpy()
    target = denormalize(torch.cat(targets), channels).numpy()
    t_neural = float(np.mean(times))
    scores = score_arrays(pred, target, mean_t_neural_s=t_neural)
    scores["_selection"] = float(np.mean([
        scores["rel_l2_score"], scores["tke_score"], scores["mvpe_score"]
    ]))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real", choices=["real", "sim"])
    ap.add_argument("--w-regime", type=float, default=0.0,
                    help="weight on the auxiliary Reynolds/angle-of-attack head. "
                         "Only the 'regime' architecture has one; the prediction "
                         "it conditions on is its OWN estimate, so nothing about "
                         "this leaks into inference, where metadata is empty.")
    ap.add_argument("--model", default="unet",
                    choices=["unet", "advective", "regime", "scaleinv",
                             "solverloop"],
                    help="'advective' adds the semi-Lagrangian prior, derived "
                         "physics channels and coordinates as network inputs; "
                         "'scaleinv' additionally divides each window by its own "
                         "velocity scale and conditions on it, so the network "
                         "never sees a magnitude and an unseen Reynolds number "
                         "becomes an interpolation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=64, help="U-Net base width")
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-tke", type=float, default=0.0)
    ap.add_argument("--w-mvpe", type=float, default=0.0)
    ap.add_argument("--sigma-power", type=float, default=0.0,
                    help="tilt the squared error toward accurate cells by "
                         "weighting it with sigma**-power, sigma being the "
                         "residual map in checkpoints/sigma_map_train.npz. "
                         "Measured: a unit of error removed from the freestream "
                         "is worth 6.5x one removed from the wake, because the "
                         "metric's loss there is coverage rather than width. "
                         "0 disables it (plain MSE); 1.3 matches the measurement.")
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--val-re", type=str, default=None,
                    help="comma-separated Reynolds numbers to hold out, e.g. "
                         "'3750,5025,25425,26700'. The default holds out three "
                         "INTERIOR values, which measures interpolation; holding "
                         "out the extremes measures extrapolation, which is what "
                         "the private test actually asks for. Use 'extrap' for "
                         "the four extreme regimes.")
    ap.add_argument("--noise-aug", type=float, default=0.0,
                    help="strain-scaled noise added to the INPUT during training, "
                         "as a fraction of the input's fluctuation sd. Teaches the "
                         "model to denoise: most useful on the simulated split, "
                         "whose targets are clean. 0 disables it.")
    ap.add_argument("--gpu-duty", type=float, default=0.72,
                    help="target GPU duty cycle; the fan on this machine is dead")
    ap.add_argument("--max-temp", type=float, default=78.0,
                    help="pause training above this GPU temperature (C)")
    ap.add_argument("--resume-temp", type=float, default=70.0,
                    help="resume once the GPU has cooled back to this temperature (C)")
    ap.add_argument("--init-from", type=str, default=None)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait-for-gpu", action="store_true",
                    help="queue behind whatever holds the shared GPU lock instead "
                         "of refusing to start")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The private test uses unseen regimes. Whether "unseen" means between the
    # trained values or outside them changes the question entirely, and our
    # default split only ever asked the easier one.
    if args.val_re is None:
        val_re = DEFAULT_VAL_RE
    elif args.val_re == "extrap":
        val_re = EXTRAPOLATION_VAL_RE
    else:
        val_re = tuple(int(v) for v in args.val_re.split(","))

    train_ds, val_ds = build_datasets(args.split, val_re=val_re,
                                      train_stride=args.train_stride)
    # The simulation split carries pressure as a third channel; only u and v are
    # ever scored, so both splits are trained on the same two.
    use_channels = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = build_model(args.model, base=args.base, channels=use_channels).to(device)
    n_par, mb = count_parameters(model)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        model.load_state_dict(state["model"])
        print(f"initialised from {args.init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    weight_map = None
    if args.sigma_power > 0:
        sig = np.load(ROOT / "checkpoints" / "sigma_map_train.npz")["sigma"]
        weight_map = sigma_weights(torch.from_numpy(sig).to(device),
                                   power=args.sigma_power)
        print(f"loss   sigma-weighted, power {args.sigma_power}, "
              f"weights {weight_map.min():.3f}..{weight_map.max():.3f}")
    criterion = CompositeLoss(args.w_mse, args.w_tke, args.w_mvpe,
                              weight_map=weight_map).to(device)

    print(f"\ndevice {device}   {gpu_line()}")
    print(f"model  {n_par:,} params, {mb:.1f} MB fp32, base={args.base}")
    print(f"data   {args.split}: {len(train_ds)} train / {len(val_ds)} val windows")
    print(f"val Re {val_re}  "
          f"({'EXTRAPOLATION' if min(val_re) < 6300 or max(val_re) > 24150 else 'interpolation'})")
    print(f"loss   mse={args.w_mse} tke={args.w_tke} mvpe={args.w_mvpe}")
    print(f"gpu duty target {args.gpu_duty:.0%}, pause above {args.max_temp:.0f}C "
          f"(fan is dead — deliberate throttle)\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # The governor duty-cycles and backs off near the ceiling; the lock keeps a
    # second competition's training off the card while this one runs. Both live
    # in the shared folder so every project uses the same policy.
    gov = Governor(target_util=int(args.gpu_duty * 100),
                   pause_temp=args.max_temp, resume_temp=args.resume_temp)
    lock = GpuLock("realpde", args.tag, wait=args.wait_for_gpu).acquire()
    try:
        run(model, opt, sched, criterion, train_loader, val_loader, device,
            use_channels, args, lock, gov)
    finally:
        lock.release()


def run(model, opt, sched, criterion, train_loader, val_loader, device,
        use_channels, args, lock, gov):
    """The training loop proper, so the lock stays a two-line concern above."""
    best, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        lock.progress(f"epoch {epoch}/{args.epochs}")
        model.train()
        agg, nb = {}, 0
        t_epoch = time.perf_counter()
        for batch in train_loader:
            with gov.step():
                x = batch["input"][..., :use_channels].to(device, non_blocking=True)
                y = batch["target"][..., :use_channels].to(device, non_blocking=True)

                # Corrupt the input, keep the target. On the simulated split the
                # target is genuinely clean, so this teaches denoising outright;
                # measured on this data, strain-scaled noise costs roughly twice
                # the forecast skill that white noise of the same energy does, so
                # it is the corruption worth training against.
                if args.noise_aug > 0:
                    x = x + strain_scaled_noise(x, args.noise_aug)

                opt.zero_grad(set_to_none=True)
                if args.w_regime > 0 and hasattr(model, "forward_with_regime"):
                    pred, regime = model.forward_with_regime(x)
                    loss, parts = criterion(pred, y)
                    # The head is supervised here and its output is discarded at
                    # inference; what survives is a bottleneck that had to encode
                    # the regime, which is the point. Re and angle of attack are
                    # never fed in — metadata is empty on scored calls.
                    tgt = encode_regime(batch["re"].to(device).float(),
                                        batch["aoa"].to(device).float())
                    l_reg = torch.mean((regime - tgt) ** 2)
                    loss = loss + args.w_regime * l_reg
                    parts["regime"] = float(l_reg.detach())
                else:
                    loss, parts = criterion(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()   # the governor times real compute

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            nb += 1

        sched.step()
        line = "  ".join(f"{k} {agg[k] / nb:.5f}" for k in sorted(agg))
        print(f"epoch {epoch:>3}/{args.epochs}  {line}  "
              f"[{time.perf_counter() - t_epoch:.1f}s, {gpu_line()}]")

        if epoch % 5 == 0 or epoch == args.epochs:
            scores = evaluate(model, val_loader, device, use_channels)
            history.append({"epoch": epoch, **{k: v for k, v in scores.items()}})
            print(format_scores(scores))
            if scores["_selection"] > best:
                best = scores["_selection"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "scores": scores, "epoch": epoch},
                           CKPT_DIR / f"{args.tag}_best.pt")
                print(f"  -> new best (selection {best:.3f}), checkpoint saved")
            print()

    (CKPT_DIR / f"{args.tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done. best selection score {best:.3f}")
    print(f"checkpoint: {CKPT_DIR / f'{args.tag}_best.pt'}")
    print(f"thermals: {gov.summary()}")


if __name__ == "__main__":
    main()
