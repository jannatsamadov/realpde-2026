"""Two measurements: the PIV noise floor, and whether the model tracks Re and AoA.

NOISE FLOOR
    PIV infers velocity from the displacement of seeding particles, so the target
    carries measurement noise. That matters more than it first appears: tke_score
    is a temporal variance, and noise contributes variance. Part of the "true" TKE
    is therefore unpredictable in principle, and a perfect physical forecaster
    would score below 100 on it.

    Estimated from the lag-1 temporal correlation of the fluctuation. At 50 Hz the
    eddy timescale spans many frames, so a physical signal is strongly correlated
    between consecutive frames while measurement noise is not:

        r1 = var_signal / (var_signal + var_noise)   =>   noise fraction = 1 - r1

    Computed in the freestream and in the wake separately, since the wake's real
    fluctuation is large and the freestream's is nearly all noise.

REGIME DEPENDENCE
    Does the model know that a higher Reynolds number means more turbulence? It is
    never told Re — metadata is empty at inference — so it would have to read the
    regime out of the input window. Comparing true against predicted TKE across Re
    and AoA shows whether it does.

Usage:
    python scripts/noise_and_regime.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from realpde.data import build_datasets, denormalize
from realpde.models import build_model

CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "adv_finetuned_best.pt"


def lag1_noise_fraction(f, mask):
    """1 - lag-1 correlation of the fluctuation, per cell, averaged where mask."""
    a, b = f[:, :-1], f[:, 1:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    r1 = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(np.clip(1.0 - r1[mask], 0.0, 1.0).mean())


def main() -> None:
    state = torch.load(CKPT, map_location="cpu")
    targs = state.get("args", {})
    model = build_model(targs.get("model", "unet"), base=targs.get("base", 64), channels=2)
    model.load_state_dict(state["model"])
    model.eval()

    _, val = build_datasets("real")
    n = len(val)
    x = torch.stack([val[i]["input"] for i in range(n)])
    y = torch.stack([val[i]["target"] for i in range(n)])
    with torch.no_grad():
        p = torch.cat([model(x[i:i + 32]) for i in range(0, n, 32)])
    pred = denormalize(p, 2).numpy()
    targ = denormalize(y, 2).numpy()
    res = np.array([val[i]["re"] for i in range(n)])
    aoa = np.array([val[i]["aoa"] for i in range(n)])

    print("1. PIV NOISE FLOOR (lag-1 estimator)\n")
    tke_true = 0.5 * (targ[..., 0].var(axis=1) + targ[..., 1].var(axis=1))  # (N,H,W)
    measured = (targ != 0).all(axis=-1).all(axis=1)                        # (N,H,W)
    hi = tke_true > np.percentile(tke_true[measured], 75)
    lo = tke_true < np.percentile(tke_true[measured], 25)

    for label, m in (("wake (top-quartile TKE)", measured & hi),
                     ("freestream (bottom quartile)", measured & lo),
                     ("all measured cells", measured)):
        nf_u = lag1_noise_fraction(targ[..., 0], m)
        nf_v = lag1_noise_fraction(targ[..., 1], m)
        print(f"  {label:<32} noise share of variance: "
              f"u {100 * nf_u:>5.1f}%   v {100 * nf_v:>5.1f}%")

    nf = 0.5 * (lag1_noise_fraction(targ[..., 0], measured)
                + lag1_noise_fraction(targ[..., 1], measured))
    print(f"\n  A forecaster that predicted the physics perfectly and the noise not")
    print(f"  at all would still miss ~{100 * nf:.0f}% of the target's temporal variance.")
    print(f"  Rough ceiling on tke_score from that alone: "
          f"{100 / (1 + 0.5 * nf):.1f}  (we are at 76.3)")
    print("  Treat as an upper bound on the noise: the estimator also counts any")
    print("  real fluctuation that decorrelates within one frame as noise.\n")

    print("2. DOES THE MODEL TRACK THE REGIME?\n")
    tke_pred = 0.5 * (pred[..., 0].var(axis=1) + pred[..., 1].var(axis=1))
    t_sum = np.array([tke_true[i][measured[i]].sum() for i in range(n)])
    p_sum = np.array([tke_pred[i][measured[i]].sum() for i in range(n)])

    print(f"  {'Re':>7}{'true TKE':>12}{'pred TKE':>12}{'kept':>8}")
    print("  " + "-" * 39)
    for r in sorted(set(res.tolist())):
        m = res == r
        print(f"  {r:>7.0f}{t_sum[m].mean():>12.4e}{p_sum[m].mean():>12.4e}"
              f"{p_sum[m].mean() / t_sum[m].mean():>8.3f}")

    print(f"\n  {'AoA':>7}{'true TKE':>12}{'pred TKE':>12}{'kept':>8}")
    print("  " + "-" * 39)
    for a in sorted(set(aoa.tolist())):
        m = aoa == a
        print(f"  {a:>7.0f}{t_sum[m].mean():>12.4e}{p_sum[m].mean():>12.4e}"
              f"{p_sum[m].mean() / t_sum[m].mean():>8.3f}")

    c_re = np.corrcoef(res, t_sum)[0, 1], np.corrcoef(res, p_sum)[0, 1]
    c_ao = np.corrcoef(aoa, t_sum)[0, 1], np.corrcoef(aoa, p_sum)[0, 1]
    print(f"\n  corr(Re,  TKE):  truth {c_re[0]:+.3f}   model {c_re[1]:+.3f}")
    print(f"  corr(AoA, TKE):  truth {c_ao[0]:+.3f}   model {c_ao[1]:+.3f}")
    print(f"  corr(true TKE, predicted TKE) across windows: "
          f"{np.corrcoef(t_sum, p_sum)[0, 1]:+.3f}")
    print("\n  Matching signs and a high window-level correlation mean the model")
    print("  reads the regime from the input rather than predicting a fixed amount")
    print("  of turbulence — it is never told Re or AoA.")


if __name__ == "__main__":
    main()
