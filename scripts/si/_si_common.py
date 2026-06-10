"""Shared definitions for the Supporting-Information table scripts.

All three SI tables (smoothing-parameter selection, bootstrap CI, and the
verification grid) are generated from the *same* canonical autocatalytic
system, the *same* smoothing rule, and the *same* landmark detector defined
here. Keeping a single pipeline guarantees the three tables are mutually
consistent and that every number printed in the SI is reproducible from a
committed, seeded script.

Canonical system (matches the main text and SI):
    A + B -> 2B,  k = 1.5 1/(M s),  A_tot = 1 M,  [B]_0 = 0.02 M.
True inflection (pos->neg acceleration crossing, = rate maximum, = 50 %
conversion):  t*_true = ln((A_tot - B0)/B0) / (k A_tot) = ln(49)/1.5.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from reaction_acceleration import (
    acceleration_zero_crossing_time,
    estimate_derivatives,
)

# --- Canonical parameters --------------------------------------------------
K = 1.5  # 1/(M s)
A_TOT = 1.0  # M
B0 = 0.02  # M
T0, T1, N_DEFAULT = 0.0, 6.0, 100

#: True inflection time for the canonical system (analytic).
T_STAR_TRUE = float(np.log((A_TOT - B0) / B0) / (K * A_TOT))  # ~2.5945 s


def _rhs(y, t, k):
    a, b = y
    return [-k * a * b, k * a * b]


def clean_signal(t: np.ndarray) -> np.ndarray:
    """Noise-free product concentration [B](t) on the grid ``t``."""
    sol = odeint(_rhs, [A_TOT - B0, B0], t, args=(K,))
    return sol[:, 1]


def grid(n: int = N_DEFAULT) -> np.ndarray:
    return np.linspace(T0, T1, n)


def smoothing_factor(n: int, sigma: float, factor: float = 2.0) -> float:
    """Spline smoothing factor s = factor * n * sigma**2 (default factor 2)."""
    return float(factor * n * sigma**2)


def landmark_fn(t, yhat, dy, d2y):
    """Recommended inflection landmark: pos->neg acceleration crossing
    nearest the rate maximum (robust to spurious early sign flips)."""
    return acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")


def estimate_inflection(t, y, s):
    """Smooth + differentiate + extract the inflection landmark (fixed-s rule)."""
    yhat, dy, d2y, _ = estimate_derivatives(t, y, method="spline", s=s)
    return landmark_fn(t, yhat, dy, d2y)


def estimate_inflection_gcv(t, y):
    """Smooth + differentiate + extract the inflection landmark (GCV P-spline).

    The recommended data-driven pipeline: the smoothing penalty is chosen by
    generalized cross-validation rather than fixed at s = 2 n sigma**2.
    """
    yhat, dy, d2y, _ = estimate_derivatives(t, y, method="gcv")
    return landmark_fn(t, yhat, dy, d2y)
