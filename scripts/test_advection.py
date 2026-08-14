"""Is the advection prior, on its own, better than persistence?

Before training anything on top of it, the prior has to earn its place: with a
zero-initialised head, AdvectiveUNet returns pure advection, so this scores that
directly against persistence and against the trained MSE model.

If advection alone does not beat persistence, the idea is wrong and no amount of
network on top will fix the baseline it sits on.

Usage:
    python scripts/test_advection.py
    python scripts/test_advection.py --substeps 1 2 4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.advection import advect_frozen, advect_sequence
from realpde.data import T_OUT, build_datasets, denormalize
from realpde.local_score import score_arrays
from realpde.models import UNetForecaster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--clamp", type=float, default=3.0)
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tke0_best.pt"))
    args = ap.parse_args()

    _, val = build_datasets("real")
    n = len(val)
    x = torch.stack([val[i]["input"] for i in range(n)])
    y = torch.stack([val[i]["target"] for i in range(n)])
    xp = denormalize(x, 2)           # m/s
    yp = denormalize(y, 2).numpy()
    print(f"validation: {n} windows\n")

    results = {}

    persist = np.repeat(xp[:, -1:].numpy(), T_OUT, axis=1)
    results["persistence"] = score_arrays(persist, yp, mean_t_neural_s=0.005)

    u0, v0 = xp[:, -1, ..., 0], xp[:, -1, ..., 1]

    for ss in args.substeps:
        t0 = time.perf_counter()
        adv = advect_sequence(u0, v0, T_OUT, substeps=ss, clamp=args.clamp).numpy()
        dt = (time.perf_counter() - t0) / n
        results[f"self-advect (substeps={ss})"] = score_arrays(adv, yp, mean_t_neural_s=dt)

    # Frozen-turbulence variants: transport by a fixed, smooth velocity field.
    valid = ((xp[..., 0] != 0) | (xp[..., 1] != 0)).all(dim=1)
    um = xp[..., 0].mean(dim=1)
    vm = xp[..., 1].mean(dim=1)
    variants = {
        "frozen: window mean V": (um, vm),
        "frozen: mean u only": (um, torch.zeros_like(vm)),
        "frozen: global mean u": (torch.full_like(um, float(um.mean())),
                                  torch.zeros_like(vm)),
    }
    for name, (tu, tv) in variants.items():
        for masked in (False, True):
            t0 = time.perf_counter()
            adv = advect_frozen(u0, v0, tu, tv, T_OUT,
                                valid=valid if masked else None).numpy()
            dt = (time.perf_counter() - t0) / n
            tag = f"{name}{' +mask' if masked else ''}"
            results[tag] = score_arrays(adv, yp, mean_t_neural_s=dt)

    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        model = UNetForecaster(base=state.get("args", {}).get("base", 64), channels=2)
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[i:i + 32]) for i in range(0, n, 32)])
        results[f"trained MSE ({args.checkpoint.stem})"] = score_arrays(
            denormalize(pred, 2).numpy(), yp, mean_t_neural_s=0.0012)

    hdr = f"{'method':<28}{'rel_l2':>9}{'tke':>9}{'mvpe':>9}{'sps':>8}{'rel_err':>10}{'tke_err':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in results.items():
        print(f"{name:<28}{s['rel_l2_score']:>9.2f}{s['tke_score']:>9.2f}"
              f"{s['mvpe_score']:>9.2f}{s['sps_score']:>8.2f}"
              f"{s['_rel_l2_error']:>10.4f}{s['_tke_error']:>9.4f}")

    print("\nThe advection rows must beat persistence for the prior to be worth")
    print("building on; they need not beat the trained network, which they feed.")


if __name__ == "__main__":
    main()
