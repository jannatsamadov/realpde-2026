"""Score a Track 2 (LTTTA) submission on our held-out real trajectories.

The kit's local_eval.py runs the right loop but only on two synthetic example
trajectories and only on CPU, so it proves plumbing and nothing else. This
mirrors that loop exactly — stride-20 stream, reset at trajectory boundaries,
previous target handed in one step late, reset charged to the following step —
on the real validation cases and on the GPU, and scores the result with the
organizers' own scoring.py.

Two things it is careful about, because both would quietly flatter the result:

  * The evaluator normalizes with the official train_real statistics, not the
    rounded pair the checkpoint was trained against. Data is loaded raw, in m/s,
    and normalized here with the exact constants, so the submission sees exactly
    what the platform will hand it.
  * The current window's target is never in scope when the prediction is made.
    The stream caches the pair only after the timed call returns.

Per-step time is measured on this machine's RTX 2060. The evaluation platform is
faster, so time_score here is a floor, not a forecast.

Usage:
    python scripts/t2_eval.py                                   # defaults
    python scripts/t2_eval.py --adapt-scope none                # no adaptation
    python scripts/t2_eval.py --adapt-scope head --adapt-lr 1e-3 --adapt-steps 2
    python scripts/t2_eval.py --sweep                           # the whole grid
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))          # the shared folder

from realpde.data import DEFAULT_VAL_RE, load_split          # noqa: E402
from realpde.local_score import format_scores, score_arrays  # noqa: E402
from gpu_guard import Governor                               # noqa: E402
from gpu_lock import GpuLock                                 # noqa: E402

IN_STEP, OUT_STEP, INTERVAL = 20, 20, 20
HORIZON = IN_STEP + OUT_STEP

# The exact official statistics the evaluator uses, from the Track 2 kit's
# example_data/mean_std_real.pt. mean(STD_TGT[:2]) reproduces SIGMA_GLOBAL to
# every published digit, which is how these were confirmed.
MEAN_IN = np.array([0.15496085584163666, -0.0005139928543940187, 0.0], dtype=np.float32)
STD_IN = np.array([0.09680565446615219, 0.015960684046149254, 1.0], dtype=np.float32)
MEAN_TGT = np.array([0.15496256947517395, -0.0005177936982363462, 0.0], dtype=np.float32)
STD_TGT = np.array([0.09681040793657303, 0.015963643789291382, 1.0], dtype=np.float32)


def build_stream(val_re=DEFAULT_VAL_RE):
    """The scored stream: windows in time order, grouped by trajectory.

    Same construction as the platform's: stride 20 with a 40-frame horizon, so
    window k's target frames are window k+1's input frames.
    """
    arr, meta = load_split("real")
    wanted = [t for t in meta["trajectories"]
              if any(str(re) in t["case"] for re in val_re)]
    if not wanted:
        raise SystemExit(f"no validation trajectories matched {val_re}")

    stream = []
    for t in wanted:
        o, length = t["offset"], t["length"]
        for time_id in range(0, length - HORIZON + 1, INTERVAL):
            s = o + time_id
            stream.append({
                "input": arr[s:s + IN_STEP],
                "target": arr[s + IN_STEP:s + HORIZON],
                "case": t["case"],
                "is_first": time_id == 0,
            })
    return stream, [t["case"] for t in wanted]


def to3(a: np.ndarray) -> np.ndarray:
    """Pad to the 3 channels the interface specifies; p is zero on real data."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape[-1] == 3:
        return a
    out = np.zeros(a.shape[:-1] + (3,), dtype=np.float32)
    out[..., :a.shape[-1]] = a
    return out


def import_submission(sub_dir: Path):
    sys.path.insert(0, str(sub_dir))
    spec = importlib.util.spec_from_file_location("t2_submission",
                                                  sub_dir / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(sub_dir: Path, device: str, overrides: dict, stream, sync: bool,
        gov=None, warmup: int = 60):
    module = import_submission(sub_dir)
    for key, val in overrides.items():
        if val is not None:
            setattr(module, key, val)

    model = module.get_ttt_model(str(sub_dir), device)
    for attr in ("reset_ttt_state", "ttt_step"):
        if not hasattr(model, attr):
            raise SystemExit(f"get_ttt_model() returned an object lacking {attr}()")

    dev = torch.device(device)

    # This card idles at 300 MHz and boosts to 2100. A cold first config measured
    # 43 ms/step and a warm one 25 ms/step for byte-identical predictions, so a
    # sweep that did not warm up would rank its configurations by clock state
    # rather than by cost. Burn a few steps before anything is timed.
    if warmup and dev.type == "cuda":
        with torch.no_grad():
            for step in stream[:warmup]:
                x = to3(step["input"])[None]
                xn = torch.from_numpy((x - MEAN_IN) / STD_IN).to(dev)
                model.ttt_step(xn, None)
        model.reset_ttt_state()
        torch.cuda.synchronize()

    times, preds, targets, lowers, uppers = [], [], [], [], []
    adapt_losses = []
    bounds_active = None
    prev_target = None

    def tick():
        if sync and dev.type == "cuda":
            torch.cuda.synchronize()
        return perf_counter()

    def one_step(step):
        nonlocal prev_target, bounds_active, adapt_losses
        x_ms = to3(step["input"])[None]
        y_ms = to3(step["target"])[None]
        xn = torch.from_numpy((x_ms - MEAN_IN) / STD_IN).to(dev)
        yn = torch.from_numpy((y_ms - MEAN_TGT) / STD_TGT).to(dev)

        # reset is timed and charged to the step that follows it, exactly as the
        # platform does it, so moving work into the boundary cannot hide it.
        reset_elapsed = 0.0
        if step["is_first"]:
            t_reset = tick()
            model.reset_ttt_state()
            reset_elapsed = tick() - t_reset
            prev_target = None

        t0 = tick()
        pred_norm, info = model.ttt_step(xn, prev_target)
        times.append(tick() - t0 + reset_elapsed)

        prev_target = yn.detach()          # cached only AFTER the timed call

        pred_norm = torch.as_tensor(pred_norm)
        if tuple(pred_norm.shape) != tuple(yn.shape):
            raise SystemExit(f"ttt_step returned {tuple(pred_norm.shape)}, "
                             f"expected {tuple(yn.shape)}")
        if not isinstance(info, dict) or "adapt_loss" not in info:
            raise SystemExit('ttt_step must return (pred, info) with info["adapt_loss"]')
        if info["adapt_loss"] is not None:
            adapt_losses.append(float(info["adapt_loss"]))

        has = "lower" in info
        if has != ("upper" in info):
            raise SystemExit("lower/upper must be supplied together or not at all")
        if bounds_active is None:
            bounds_active = has
        elif has != bounds_active:
            raise SystemExit("bounds must be supplied on every step or on none")

        preds.append((pred_norm.detach().cpu().numpy() * STD_TGT + MEAN_TGT)[0])
        targets.append(y_ms[0])
        if has:
            lowers.append((torch.as_tensor(info["lower"]).detach().cpu().numpy()
                           * STD_TGT + MEAN_TGT)[0])
            uppers.append((torch.as_tensor(info["upper"]).detach().cpu().numpy()
                           * STD_TGT + MEAN_TGT)[0])

    for step in stream:
        # The governor's throttle sleep lands at the END of this block, after the
        # per-step time has already been recorded, so duty cycling this machine's
        # fanless card cannot inflate the very measurement it exists to protect.
        if gov is None:
            one_step(step)
        else:
            with gov.step():
                one_step(step)

    pred = np.stack(preds).astype(np.float32)
    target = np.stack(targets).astype(np.float32)
    lower = np.stack(lowers).astype(np.float32) if bounds_active else None
    upper = np.stack(uppers).astype(np.float32) if bounds_active else None

    # Did adaptation actually do anything? An ineffective policy and a policy
    # that never ran look identical in the subscores, so measure the weights
    # directly rather than inferring from the metric.
    drift = 0.0
    init = getattr(model, "_init_state", None)
    if init is not None:
        num, den = 0.0, 0.0
        for k, v in model.model.state_dict().items():
            if v.dtype.is_floating_point:
                num += float((v - init[k]).pow(2).sum())
                den += float(init[k].pow(2).sum())
        drift = (num / max(den, 1e-12)) ** 0.5

    scores = score_arrays(pred, target, mean_t_neural_s=float(np.mean(times)),
                          lower=lower, upper=upper)
    scores["_weight_drift"] = drift
    scores["_adapt_calls"] = len(adapt_losses)
    # Only when there is something to average: format_scores cannot format None,
    # and with adaptation off there are no losses at all.
    if adapt_losses:
        scores["_adapt_loss_mean"] = float(np.mean(adapt_losses))
    scores["_selection"] = float(np.mean([scores["rel_l2_score"],
                                          scores["tke_score"],
                                          scores["mvpe_score"]]))
    scores["_total_wall_s"] = float(np.sum(times))
    scores["_bounds"] = bool(bounds_active)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path,
                    default=ROOT / "submissions" / "t2_template")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--adapt-scope", default=None, choices=["none", "head", "all"])
    ap.add_argument("--adapt-steps", type=int, default=None)
    ap.add_argument("--adapt-lr", type=float, default=None)
    ap.add_argument("--no-bounds", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip cuda synchronize around the timer (faster, but the "
                         "per-step time is then a lower bound, not a measurement)")
    ap.add_argument("--warmup", type=int, default=60,
                    help="untimed steps before each measured run, so a config is "
                         "not ranked by whether the card had boosted yet")
    ap.add_argument("--sweep", action="store_true", help="run the policy grid")
    ap.add_argument("--out", type=Path, default=ROOT / "submissions" / "t2_eval.json")
    args = ap.parse_args()

    stream, cases = build_stream()
    print(f"{len(stream)} steps over {len(cases)} trajectories: {', '.join(cases)}")
    print(f"device {args.device}\n")

    configs = []
    if args.sweep:
        # Weight adaptation is already settled: it does nothing up to lr 0.01,
        # degrades at 0.1 and diverges at 1. Kept off here. What is being tested
        # now is online interval calibration, and specifically whether it repairs
        # a prior that is wrong in the direction the leaderboard revealed.
        base = {"ADAPT_SCOPE": "none", "ADAPT_STEPS": 0}
        configs.append(("fixed map, scale 0.8",
                        dict(base, SIGMA_ONLINE="off", BOUND_SCALE=0.8)))
        # BOUND_SCALE 0.8 was tuned against the FIXED map, where it was partly
        # compensating for a sigma that did not match the residuals. Once the
        # width is calibrated online that compensation is double counting, so the
        # optimum has to be found again rather than carried over.
        for scale in (0.8, 1.0, 1.2):
            for count in (2.0, 4.0, 16.0):
                configs.append((f"online s={scale} n0={count:g}",
                                dict(base, SIGMA_ONLINE="scalar",
                                     BOUND_SCALE=scale, SIGMA_PRIOR_COUNT=count)))
    else:
        configs.append(("run", {
            "ADAPT_SCOPE": args.adapt_scope,
            "ADAPT_STEPS": args.adapt_steps,
            "ADAPT_LR": args.adapt_lr,
            "RETURN_BOUNDS": False if args.no_bounds else None,
        }))

    lock = GpuLock("realpde", "t2_eval").acquire() if args.device == "cuda" else None
    gov = Governor(target_util=72, pause_temp=78, resume_temp=70) if args.device == "cuda" else None
    results = {}
    try:
        header = (f"{'config':<22}{'rel_l2':>9}{'tke':>8}{'mvpe':>8}{'sps':>8}"
                  f"{'time':>8}{'seçim':>9}{'ms/step':>9}{'|Δw|/|w|':>11}")
        print(header)
        print("-" * len(header))
        for name, over in configs:
            if lock is not None:
                lock.progress(name)
            s = run(args.submission, args.device, over, stream,
                    not args.no_sync, gov, args.warmup)
            results[name] = s
            print(f"{name:<22}{s['rel_l2_score']:>9.3f}{s['tke_score']:>8.3f}"
                  f"{s['mvpe_score']:>8.3f}{s['sps_score']:>8.3f}"
                  f"{s['time_score']:>8.3f}{s['_selection']:>9.3f}"
                  f"{1000 * s['_mean_t_neural_s']:>9.2f}"
                  f"{s['_weight_drift']:>11.2e}")
    finally:
        if lock is not None:
            lock.release()

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    if not args.sweep:
        print(format_scores(next(iter(results.values()))))


if __name__ == "__main__":
    main()
