"""How much of the forecast's decay is chaos, and how much is measurement noise?

On real data the fluctuation correlation between prediction and truth falls from
0.86 at the first forecast frame to 0.55 at the twentieth. Two different things
produce that curve and they call for opposite responses:

  chaos   the flow amplifies any uncertainty in the initial state, so even a
          perfect model loses skill over 1.23 shedding cycles. Nothing fixes it.

  noise   22% of the measured variance is PIV error, so the model starts from a
          blurred state AND is scored against a blurred target. Cleaning the
          input before forecasting would attack the first half of that.

Simulation has no measurement noise. Running the same comparison there isolates
the chaos term: if the simulated curve decays just as fast, noise is not what
limits us and denoising the input is not worth building. If it holds up much
better, the gap is the noise's contribution and there is something to win.

Both models are used in-distribution — the real model on real data, the
simulation model on simulated data — so the comparison is about the data, not
about domain shift.

Usage:
    python scripts/chaos_vs_noise.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.models import build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def load(path):
    state = torch.load(path, map_location="cpu")
    a = state.get("args", {})
    m = build_model(a.get("model", "unet"), base=a.get("base", 64), channels=2)
    m.load_state_dict(state["model"])
    return m.eval()


def curves(model, ds, n_max=273):
    n = min(n_max, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    x = torch.stack([ds[int(i)]["input"][..., :2] for i in idx])
    y = torch.stack([ds[int(i)]["target"][..., :2] for i in idx])
    with torch.no_grad():
        p = torch.cat([model(x[i:i + 32]) for i in range(0, len(x), 32)])
    pred = denormalize(p, 2).numpy()
    targ = denormalize(y, 2).numpy()

    pf = pred - pred.mean(axis=1, keepdims=True)
    tf = targ - targ.mean(axis=1, keepdims=True)
    corr, amp = [], []
    for k in range(pred.shape[1]):
        a, b = pf[:, k], tf[:, k]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        corr.append(float((a * b).sum() / max(na * nb, 1e-12)))
        amp.append(float(na / max(nb, 1e-12)))
    return np.array(corr), np.array(amp), n


def persistence_curve(ds, n_max=273):
    """How fast the flow decorrelates from itself — the raw predictability limit."""
    n = min(n_max, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    x = torch.stack([ds[int(i)]["input"][..., :2] for i in idx])
    y = torch.stack([ds[int(i)]["target"][..., :2] for i in idx])
    xp = denormalize(x, 2).numpy()
    tp = denormalize(y, 2).numpy()
    last = xp[:, -1:]                              # the final observed frame
    lf = last - xp.mean(axis=1, keepdims=True)
    tf = tp - tp.mean(axis=1, keepdims=True)
    out = []
    for k in range(tp.shape[1]):
        a, b = lf[:, 0], tf[:, k]
        out.append(float((a * b).sum()
                         / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    return np.array(out)


def main() -> None:
    real_model = load(CKPT / "adv_finetuned_best.pt")
    sim_model = load(CKPT / "sim_pretrain_best.pt")

    _, real_val = build_datasets("real")
    _, sim_val = build_datasets("sim")

    c_real, a_real, n_real = curves(real_model, real_val)
    c_sim, a_sim, n_sim = curves(sim_model, sim_val)
    p_real = persistence_curve(real_val)
    p_sim = persistence_curve(sim_val)

    print(f"real: {n_real} windows, model adv_finetuned (trained on real)")
    print(f"sim : {n_sim} windows, model sim_pretrain  (trained on sim)\n")

    print(f"{'frame':>6}{'t ahead':>9}"
          f"{'REAL corr':>11}{'SIM corr':>10}{'gap':>8}"
          f"{'REAL amp':>10}{'SIM amp':>9}")
    print("-" * 63)
    for k in range(len(c_real)):
        if k < 3 or (k + 1) % 4 == 0 or k == len(c_real) - 1:
            print(f"{k + 1:>6}{0.02 * (k + 1):>8.2f}s"
                  f"{c_real[k]:>11.3f}{c_sim[k]:>10.3f}{c_sim[k] - c_real[k]:>8.3f}"
                  f"{a_real[k]:>10.3f}{a_sim[k]:>9.3f}")

    print(f"\ndecay from frame 1 to frame 20:")
    print(f"  real   {c_real[0]:.3f} -> {c_real[-1]:.3f}   "
          f"(lost {c_real[0] - c_real[-1]:.3f})")
    print(f"  sim    {c_sim[0]:.3f} -> {c_sim[-1]:.3f}   "
          f"(lost {c_sim[0] - c_sim[-1]:.3f})")

    print(f"\nthe flow against its own last observed frame (no model at all):")
    print(f"  real   {p_real[0]:.3f} -> {p_real[-1]:.3f}")
    print(f"  sim    {p_sim[0]:.3f} -> {p_sim[-1]:.3f}")

    gap0, gap20 = c_sim[0] - c_real[0], c_sim[-1] - c_real[-1]
    print("\nREADING THIS")
    print(f"  gap at frame 1  : {gap0:+.3f}")
    print(f"  gap at frame 20 : {gap20:+.3f}")
    if gap20 > 0.10:
        print("  The simulated curve holds up markedly better, so a large part of the")
        print("  decay on real data is measurement rather than chaos. Cleaning the")
        print("  input before forecasting is worth building.")
    elif gap20 > 0.03:
        print("  The simulated curve holds up somewhat better. Some of the decay is")
        print("  measurement, but most is the flow itself; denoising would buy a")
        print("  little.")
    else:
        print("  Both decay at nearly the same rate, so the loss is the flow's own")
        print("  chaos and not the measurement. Denoising the input would not help,")
        print("  and effort belongs elsewhere.")


if __name__ == "__main__":
    main()
