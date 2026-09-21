"""The full training recipe in one command, so no part of it can be forgotten.

THE STANDARD, and what each part was measured to be worth
    model       scaleinv   advection prior + speed/vorticity/divergence channels
                           + coordinates + scale conditioning. Scale conditioning
                           was +0.49 and +0.51 final on two independent setups
                           against the advective model it extends.
    recipe      sim pretraining, then real finetuning from the sim checkpoint.
    validation  the extrapolation split: Reynolds numbers held out at BOTH ends
                of the range, and held out of the sim pretraining too, or the
                "unseen" regimes have been seen in simulation.

Every new idea is a change ON TOP of this, passed through as extra arguments --
a loss term, a different architecture name. The rule this script exists to
enforce: once a change is confirmed, it is the baseline; measuring the next idea
against something weaker makes the comparison meaningless.

RESUMING
    A stage whose history file exists finished (train.py writes it last), so it
    is skipped. An overnight run that was killed between stages picks up at the
    stage that did not finish instead of redoing the pretraining.

Usage:
    python scripts/train_standard.py --tag std                     # the standard
    python scripts/train_standard.py --tag std_loop --model scaleinv_loop
    python scripts/train_standard.py --tag std_w13 --sigma-power 1.3
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train.py"
CKPT = ROOT / "checkpoints"
# While this file exists no new stage starts; a stage already running finishes.
# It is checked by each new process, so it can pause a queue of commands that
# was launched earlier -- create it to hold, delete it and relaunch to go on.
HOLD = CKPT / "HOLD_QUEUE"

sys.path.insert(0, str(ROOT / "src"))
from realpde.models import STANDARD_ARCH  # noqa: E402


def run_stage(args_list: list[str], tag: str) -> None:
    if (CKPT / f"{tag}_history.json").exists():
        print(f"[standard] {tag}: already complete, skipping")
        return
    if HOLD.exists():
        # Exit cleanly rather than skip: a held sim stage must not let the real
        # stage start from a checkpoint that does not exist yet.
        print(f"[standard] {tag}: queue on hold ({HOLD}), not started", flush=True)
        raise SystemExit(0)
    cmd = [sys.executable, str(TRAIN), *args_list, "--tag", tag, "--wait-for-gpu"]
    print(f"[standard] {tag}: {' '.join(args_list)}", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[standard] {tag} failed with exit code {rc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=STANDARD_ARCH)
    ap.add_argument("--val-re", default="extrap")
    ap.add_argument("--sim-epochs", type=int, default=15)
    ap.add_argument("--real-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args, extra = ap.parse_known_args()

    common = ["--model", args.model, "--val-re", args.val_re,
              "--seed", str(args.seed), *extra]

    sim_tag, real_tag = f"{args.tag}_sim", f"{args.tag}_real"
    run_stage(["--split", "sim", "--epochs", str(args.sim_epochs), *common], sim_tag)
    run_stage(["--split", "real", "--epochs", str(args.real_epochs),
               "--init-from", str(CKPT / f"{sim_tag}_best.pt"), *common], real_tag)
    print(f"[standard] done: {CKPT / f'{real_tag}_best.pt'}")


if __name__ == "__main__":
    main()
