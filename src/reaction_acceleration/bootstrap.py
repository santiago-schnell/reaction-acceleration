"""Residual-bootstrap uncertainty quantification for kinetic landmarks."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .derivatives import ArrayLike, Method, _as_1d_float, estimate_derivatives


def residual_bootstrap_landmark_ci(
    t: ArrayLike,
    y: ArrayLike,
    *,
    landmark_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    method: Method = "spline",
    s: Optional[float] = None,
    k: int = 3,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a landmark time.

    This uses a *residual bootstrap*:

    1. Fit a smooth model to the data, producing `yhat(t)`.
    2. Compute residuals `r = y - yhat`.
    3. Resample residuals with replacement to create `y* = yhat + r*`.
    4. Refit and re-extract the landmark for each bootstrap replicate.

    Parameters
    ----------
    t, y:
        Time points and observed signal.
    landmark_fn:
        Callable with signature `(t, yhat, dy, d2y) -> float`, returning the
        landmark time (NaN if not detected).
    n_boot:
        Number of bootstrap replicates.
    alpha:
        Significance level (default 0.05 gives a 95% CI).

    Returns
    -------
    estimate, lo, hi
        - `estimate` is the landmark extracted from the base fit.
        - `lo, hi` are percentile-bootstrap confidence bounds.

    Notes
    -----
    - If >20% of bootstrap replicates fail to detect the landmark, this
      function returns `(estimate, NaN, NaN)` to indicate instability.
    """

    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    rng = np.random.default_rng(seed)

    t_arr = _as_1d_float(t)
    y_arr = _as_1d_float(y)

    if t_arr.size != y_arr.size:
        raise ValueError("t and y must have the same length.")

    # IMPORTANT: sort once here to keep residuals aligned with fitted values.
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    y_arr = y_arr[order]

    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("t must be strictly increasing (remove or average duplicates).")

    # Base fit
    yhat, dy, d2y, _model = estimate_derivatives(t_arr, y_arr, method=method, s=s, k=k)
    estimate = float(landmark_fn(t_arr, yhat, dy, d2y))

    if not np.isfinite(estimate):
        return float("nan"), float("nan"), float("nan")

    resid = y_arr - yhat
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        r_star = rng.choice(resid, size=resid.size, replace=True)
        y_star = yhat + r_star

        yhat_b, dy_b, d2y_b, _ = estimate_derivatives(t_arr, y_star, method=method, s=s, k=k)
        boots[b] = float(landmark_fn(t_arr, yhat_b, dy_b, d2y_b))

    valid = boots[np.isfinite(boots)]

    if valid.size < 0.8 * n_boot:
        return estimate, float("nan"), float("nan")

    lo = float(np.quantile(valid, alpha / 2))
    hi = float(np.quantile(valid, 1 - alpha / 2))
    return estimate, lo, hi
