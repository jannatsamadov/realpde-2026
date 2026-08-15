"""Use the leaderboard's own numbers to retune the SPS interval width.

Submission 2 sent bounds fitted on our training residuals at BOUND_SCALE 0.8 and
scored sps 36.17, against 48.74 locally. The gap is far larger than the 1-3 point
optimism the other subscores show, and it has a single plausible cause: the hidden
set's residuals are bigger than the ones our sigma map was fitted to, so bands
sized for our error miss more often there.

That is measurable rather than guessable. scoring.py's SPS is

    weighted = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
    branch_i = mean over scored of (1 - pm_i) * exp(-nil) * [target inside]

and each pm_i is a published function of the corresponding subscore. Factoring out
the pm terms leaves Q = E[exp(-nil) * inside], which depends only on the bounds.
Comparing Q locally with Q on the hidden set isolates how much coverage was lost,
and under a Gaussian residual that converts into a ratio of standard deviations.

Everything here follows from the five reported subscores. The Gaussian step is an
assumption; the rest is arithmetic.

Usage:
    python scripts/infer_hidden_sigma.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from realpde.sps import _Phi, optimal_width

# What the leaderboard reported for submission 2 (adv_v2).
HIDDEN = dict(rel_l2=94.546636, tke=74.630657, mvpe=93.486546,
              time=87.152894, sps=36.169825, final=78.765833)
# The same model on our 273-window validation split.
LOCAL = dict(rel_l2=96.250, tke=76.318, mvpe=96.542, sps=48.743, coverage=0.87551)
# Submission 1, for the two-point comparison.
PREV = dict(rel_l2=94.634455, tke=73.676616, mvpe=93.159743,
            time=92.302811, sps=16.644090, final=73.664598)

BOUND_SCALE_USED = 0.8


def err_from_score(score):
    """Invert score = 100 / (1 + 0.5 * err)."""
    return 2.0 * (100.0 / score - 1.0)


def pm(err):
    """scoring.py's normalisation: pm = err / (0.5 + err)."""
    return err / (0.5 + err)


def pm_weight(rel_l2, tke, mvpe):
    """0.5*(1-pm_dm) + 0.3*(1-pm_tke) + 0.2*(1-pm_mvpe)."""
    return (0.5 * (1 - pm(err_from_score(rel_l2)))
            + 0.3 * (1 - pm(err_from_score(tke)))
            + 0.2 * (1 - pm(err_from_score(mvpe))))


def main() -> None:
    print("1. HOW THE TWO SUBMISSIONS COMPARE")
    print(f"  {'subscore':<10}{'sub 1':>10}{'sub 2':>10}{'change':>10}")
    print("  " + "-" * 40)
    for k in ("rel_l2", "tke", "mvpe", "time", "sps", "final"):
        print(f"  {k:<10}{PREV[k]:>10.3f}{HIDDEN[k]:>10.3f}{HIDDEN[k] - PREV[k]:>+10.3f}")

    print("\n2. WHAT final_score WEIGHS")
    d = {k: HIDDEN[k] - PREV[k] for k in ("rel_l2", "tke", "mvpe", "time", "sps")}
    eq = sum(d.values()) / 5
    print(f"  equal weighting would have moved final by {eq:+.3f}; it moved {d and HIDDEN['final'] - PREV['final']:+.3f}")
    # Solve for a single sps weight with the other four sharing one weight.
    rest_prev = sum(PREV[k] for k in ("rel_l2", "tke", "mvpe", "time"))
    rest_now = sum(HIDDEN[k] for k in ("rel_l2", "tke", "mvpe", "time"))
    a, b = np.linalg.solve(
        np.array([[rest_prev, PREV["sps"]], [rest_now, HIDDEN["sps"]]]),
        np.array([PREV["final"], HIDDEN["final"]]))
    print(f"  fitting one weight for sps and one shared by the other four:")
    print(f"    sps weight {b:.3f}, each of the others {a:.3f} (they sum to {4 * a + b:.3f})")
    print("  Two submissions are two equations in five unknowns, so this is one")
    print("  consistent solution, not the formula. What it does show is that sps")
    print(f"  carries more than an equal share: {b:.3f} against {1 / 5:.3f}.")

    print("\n3. HOW MUCH BIGGER THE HIDDEN RESIDUALS ARE")
    w_local = pm_weight(LOCAL["rel_l2"], LOCAL["tke"], LOCAL["mvpe"])
    w_hidden = pm_weight(HIDDEN["rel_l2"], HIDDEN["tke"], HIDDEN["mvpe"])
    q_local = (LOCAL["sps"] / 100.0) / w_local
    q_hidden = (HIDDEN["sps"] / 100.0) / w_hidden
    print(f"  accuracy factor  local {w_local:.4f}   hidden {w_hidden:.4f}")
    print(f"  Q = E[exp(-nil)*inside]  local {q_local:.4f}   hidden {q_hidden:.4f}")

    # nil is identical in both runs (same sigma map, same widths), so the whole
    # drop in Q is lost coverage.
    exp_term = q_local / LOCAL["coverage"]
    cov_hidden = q_hidden / exp_term
    print(f"  implied E[exp(-nil) | inside] {exp_term:.4f}  (same bands in both runs)")
    print(f"  coverage   local {LOCAL['coverage']:.3f}   hidden {cov_hidden:.3f}")

    # For a Gaussian residual, coverage = 2*Phi(w/(2s)) - 1. Same w, so the ratio
    # of the inverse-CDF arguments is the inverse ratio of the sigmas.
    def z_for(cov):
        z = np.linspace(0.01, 6.0, 20000)
        return float(np.interp(cov, 2.0 * _Phi(z) - 1.0, z))

    z_local, z_hidden = z_for(LOCAL["coverage"]), z_for(cov_hidden)
    ratio = z_local / z_hidden
    print(f"  z  local {z_local:.3f}   hidden {z_hidden:.3f}")
    print(f"  -> hidden residuals are about {ratio:.2f}x ours")

    print("\n4. THE WIDTH THAT WOULD HAVE BEEN OPTIMAL THERE")
    s = np.array([0.005, 0.0107, 0.02, 0.04])       # representative local sigmas
    grow = optimal_width(s * ratio) / optimal_width(s)
    print(f"  {'local sigma':>12}{'opt width':>12}{'opt width x' + f'{ratio:.2f}':>16}{'ratio':>8}")
    for si, g in zip(s, grow):
        print(f"  {si:>12.4f}{float(optimal_width(np.array([si]))[0]):>12.5f}"
              f"{float(optimal_width(np.array([si * ratio]))[0]):>16.5f}{g:>8.3f}")
    new_scale = BOUND_SCALE_USED * float(np.mean(grow))
    print(f"\n  BOUND_SCALE used: {BOUND_SCALE_USED}")
    print(f"  suggested next  : {new_scale:.2f}")
    print("\n  This assumes Gaussian residuals and that the pm factors move the way")
    print("  the reported subscores say. It is a calibrated guess from one data")
    print("  point, and the next submission is what tests it.")


if __name__ == "__main__":
    main()
