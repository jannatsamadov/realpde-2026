"""Optimal interval bounds for sps_score.

WHAT THE SCORER REWARDS

Per element, with SIGMA = 0.0563870259 frozen for the season:

    nil     = (upper - lower) / SIGMA
    element = (1 - pm) * exp(-nil)   if lower <= target <= upper, else 0

`pm` depends only on the window's accuracy, not on the interval, so choosing
bounds is a pure trade-off between covering the target and staying narrow. A wide
band always covers but `exp(-nil)` collapses; a narrow band keeps `exp(-nil)`
near 1 but usually misses and scores zero.

THE OPTIMUM

Take the residual e = target - pred as symmetric with standard deviation s, and a
symmetric band of total width w. The expected contribution is proportional to

    g(w) = exp(-w / SIGMA) * P(|e| <= w/2)

With e ~ N(0, s^2), writing r = w / (2s), setting d/dw log g = 0 gives

    phi(r) / (2 * Phi(r) - 1) = s / SIGMA

so the optimal half-width in units of s depends only on k = s / SIGMA. The
consequence worth noting: where the model is accurate (s << SIGMA) the optimum
covers nearly everything and still keeps nil small, scoring close to 1; where it
is poor (s >> SIGMA) the optimum is a narrow band that mostly misses. The metric
pays for accuracy twice, which is why per-element widths beat one global width —
freestream cells are easy and should get tight, near-certain intervals, while
wake cells cannot be rescued by widening.

The default band the scorer falls back to is 0.1 * |pred| — unrelated to the
model's actual error — and scores about 17 on our validation split.

Everything here uses numpy and math only, so it can be vendored into a
submission: scipy is not in the evaluation image.
"""

from __future__ import annotations

import math

import numpy as np

SIGMA_GLOBAL = 0.0563870259


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf, Abramowitz & Stegun 7.1.26 (max error 1.5e-7).

    `math.erf` is scalar-only, and wrapping it in np.vectorize is a Python loop:
    applied elementwise to a submission's 22M-element residual array inside a
    bisection it never finishes. numpy has no erf and scipy is absent from the
    evaluation image, so the approximation is written out here.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (
        1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _phi(r: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    return np.exp(-0.5 * np.asarray(r, dtype=np.float64) ** 2) / math.sqrt(2.0 * math.pi)


def _Phi(r: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(np.asarray(r, dtype=np.float64) / math.sqrt(2.0)))


def _objective_ratio(r: np.ndarray) -> np.ndarray:
    """phi(r) / (2 Phi(r) - 1), the left-hand side of the optimality condition.

    Decreasing in r on (0, inf): it diverges as r -> 0 and decays to 0, so any
    positive target k has exactly one root.
    """
    denom = np.clip(2.0 * _Phi(r) - 1.0, 1e-12, None)
    return _phi(r) / denom


# The root r(k) depends on nothing but k, so it is solved once on a grid and
# interpolated thereafter — an O(1) lookup per element instead of a bisection.
_R_GRID = np.geomspace(1e-4, 40.0, 4000)
_RATIO_GRID = _objective_ratio(_R_GRID)
# np.interp needs an increasing x: the ratio decreases in r, so reverse both.
_RATIO_ASC = _RATIO_GRID[::-1]
_R_ASC = _R_GRID[::-1]


def optimal_half_width_ratio(k: np.ndarray) -> np.ndarray:
    """Solve phi(r)/(2Phi(r)-1) = k for r, elementwise.

    `k = s / SIGMA_GLOBAL`. Returns r such that the optimal total width is
    w = 2 * s * r. Outside the tabulated range the endpoints are held, which is
    harmless: both tails are already saturated.
    """
    k = np.asarray(k, dtype=np.float64)
    return np.interp(k, _RATIO_ASC, _R_ASC)


def optimal_width(sigma_pred: np.ndarray, sigma_global: float = SIGMA_GLOBAL) -> np.ndarray:
    """Total interval width maximising exp(-w/sigma_global) * P(|e| <= w/2).

    `sigma_pred` is the per-element predicted residual standard deviation.
    """
    s = np.clip(np.asarray(sigma_pred, dtype=np.float64), 1e-12, None)
    r = optimal_half_width_ratio(s / sigma_global)
    return 2.0 * s * r


def bounds_from_sigma(pred: np.ndarray, sigma_pred: np.ndarray,
                      sigma_global: float = SIGMA_GLOBAL, scale: float = 1.0):
    """Symmetric lower/upper bounds around `pred` at the optimal width.

    `scale` multiplies the width, for sweeping around the analytic optimum: the
    derivation assumes Gaussian residuals, and real residuals are heavier-tailed,
    so the empirical best usually sits slightly wide of it.
    """
    w = optimal_width(sigma_pred, sigma_global) * scale
    half = (0.5 * w).astype(np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    return pred - half, pred + half


def expected_element_score(s: np.ndarray, w: np.ndarray,
                           sigma_global: float = SIGMA_GLOBAL) -> np.ndarray:
    """exp(-w/sigma_global) * P(|e| <= w/2) for Gaussian residuals — the quantity
    optimal_width maximises. Useful for plotting the trade-off."""
    s = np.clip(np.asarray(s, dtype=np.float64), 1e-12, None)
    coverage = 2.0 * _Phi(0.5 * w / s) - 1.0
    return np.exp(-w / sigma_global) * coverage


if __name__ == "__main__":
    print(f"SIGMA_GLOBAL = {SIGMA_GLOBAL}\n")
    print(f"{'s (resid std)':>14} {'s/SIGMA':>9} {'opt width':>10} "
          f"{'w/SIGMA':>9} {'coverage':>9} {'E[score]':>9} {'default w':>10} {'E[def]':>8}")
    print("-" * 88)
    for s in [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        w = float(optimal_width(np.array([s]))[0])
        cov = float(2.0 * _Phi(np.array([0.5 * w / s]))[0] - 1.0)
        e = float(expected_element_score(np.array([s]), np.array([w]))[0])
        # The scorer's fallback: 0.1 * |pred|, here at a typical |pred| ~ 0.15 m/s.
        w_def = 0.1 * 0.15
        e_def = float(expected_element_score(np.array([s]), np.array([w_def]))[0])
        print(f"{s:>14.4f} {s / SIGMA_GLOBAL:>9.3f} {w:>10.5f} {w / SIGMA_GLOBAL:>9.3f} "
              f"{cov:>9.3f} {e:>9.3f} {w_def:>10.5f} {e_def:>8.3f}")
    print("\nRead the last two columns against E[score]: the default band is a fixed")
    print("width regardless of how accurate the model is at that element, so it is")
    print("badly wrong at both ends of the accuracy range.")
