"""Controlled test: add PIV-like noise to simulated input and watch the forecast decay.

chaos_vs_noise.py compared real against simulation and found the simulated
correlation holding at 0.717 by frame 20 against real's 0.532. Tempting to read
that as the cost of measurement noise, except the persistence curves point the
other way: the simulated flow decorrelates from its own last frame FASTER than
the real one (0.458 against 0.705). PIV averages over its interrogation windows,
so the measured field is smoother and more persistent than the true one. The two
splits are not the same flow with and without noise.

This removes that confound. One flow, one model, one thing changed: noise is
added to the simulated INPUT only, at levels around what the real split carries
(~22% of temporal variance). The target stays clean, so the curve isolates what a
blurred initial condition costs.

Because the real noise is not white — it scales with the local strain rate, 9.9x
across strain quintiles — it is injected proportionally to strain as well as
uniformly, and the two are compared.

Usage:
    python scripts/noise_injection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.models import build_model

ROOT = Path(__file__).resolve().parent.parent
DX = 0.003422


def strain_rate(u, v, dx=DX):
    a = np.gradient(u, dx, axis=-1)
    b = np.gradient(u, -dx, axis=-2)
    c = np.gradient(v, dx, axis=-1)
    d = np.gradient(v, -dx, axis=-2)
    return np.sqrt(a ** 2 + d ** 2 + 2.0 * (0.5 * (b + c)) ** 2)


def correlation_curve(pred, targ):
    pf = pred - pred.mean(axis=1, keepdims=True)
    tf = targ - targ.mean(axis=1, keepdims=True)
    out = []
    for k in range(pred.shape[1]):
        a, b = pf[:, k], tf[:, k]
        out.append(float((a * b).sum()
                         / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    return np.array(out)


def main() -> None:
    state = torch.load(ROOT / "checkpoints" / "sim_pretrain_best.pt", map_location="cpu")
    a = state.get("args", {})
    model = build_model(a.get("model", "unet"), base=a.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, sim_val = build_datasets("sim")
    n = min(273, len(sim_val))
    idx = np.linspace(0, len(sim_val) - 1, n).astype(int)
    x0 = torch.stack([sim_val[int(i)]["input"][..., :2] for i in idx])
    y0 = torch.stack([sim_val[int(i)]["target"][..., :2] for i in idx])
    targ = denormalize(y0, 2).numpy()

    xn = x0.numpy()
    fluct_sd = float((xn - xn.mean(axis=1, keepdims=True)).std())
    # strain_rate returns (N, T, H, W); the array to weight is (N, T, H, W, C),
    # so the channel axis is added rather than inserted anywhere else.
    s = strain_rate(xn[..., 0], xn[..., 1])
    s_rel = (s / s.mean())[..., None]

    print(f"{n} simulated windows, model sim_pretrain")
    print(f"input fluctuation sd (normalized): {fluct_sd:.4f}")
    print("noise is a fraction of that, added to the INPUT only; target stays clean\n")

    rng = np.random.default_rng(0)

    def run(x_np):
        xt = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32))
        with torch.no_grad():
            p = torch.cat([model(xt[i:i + 32]) for i in range(0, len(xt), 32)])
        return correlation_curve(denormalize(p, 2).numpy(), targ)

    rows = {}
    rows["clean"] = run(xn)
    for frac in (0.15, 0.25, 0.40):
        z = rng.standard_normal(xn.shape).astype(np.float32)
        rows[f"uniform {frac:.0%}"] = run(xn + frac * fluct_sd * z)
    for frac in (0.25, 0.40):
        z = rng.standard_normal(xn.shape).astype(np.float32)
        # Strain-proportional, rescaled so the total injected energy matches the
        # uniform case at the same fraction.
        w = s_rel / np.sqrt((s_rel ** 2).mean())
        rows[f"strain-scaled {frac:.0%}"] = run(xn + frac * fluct_sd * w * z)

    print(f"{'input':<22}" + "".join(f"{f'f{k}':>8}" for k in (1, 4, 8, 12, 16, 20)))
    print("-" * (22 + 8 * 6))
    for name, c in rows.items():
        print(f"{name:<22}" + "".join(f"{c[k - 1]:>8.3f}" for k in (1, 4, 8, 12, 16, 20)))

    clean = rows["clean"]
    print(f"\ncost at frame 20, against clean input:")
    for name, c in rows.items():
        if name == "clean":
            continue
        print(f"  {name:<22}{c[-1] - clean[-1]:+.3f}")

    print(f"\nFor reference, measured earlier:")
    print(f"  real data, frame 20 correlation : 0.532")
    print(f"  clean simulation, frame 20      : {clean[-1]:.3f}")
    print("\n  If a realistic noise level costs far less than the 0.185 gap between")
    print("  real and simulation, then most of that gap is the difference between")
    print("  the two flows rather than the measurement, and denoising the input")
    print("  buys correspondingly little.")


if __name__ == "__main__":
    main()
