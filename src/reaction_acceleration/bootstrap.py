"""Residual-bootstrap uncertainty quantification for kinetic landmarks."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .derivatives import ArrayLike, Method, _as_1d_float, estimate_derivatives

logger = logging.getLogger(__name__)


def residual_bootstrap_landmark_ci(
    t: ArrayLike,
    y: ArrayLike,
    *,
    landmark_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    method: Method = "spline",
    s: float | None = None,
    k: int | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, dict]:
    """Compute a bootstrap confidence interval for a landmark time.

    This uses a *residual bootstrap*:

    1. Fit a smooth model to the data, producing ``yhat(t)``.
    2. Compute residuals ``r = y - yhat``.
    3. Resample residuals with replacement to create ``y* = yhat + r*``.
    4. Refit and re-extract the landmark for each bootstrap replicate.

    Parameters
    ----------
    t, y
        Time points and observed signal.
    landmark_fn
        Callable with signature ``(t, yhat, dy, d2y) -> float``, returning
        the landmark time (NaN if not detected).
    method
        Smoothing method forwarded to :func:`estimate_derivatives`.
    s, k
        Smoothing and degree parameters forwarded to
        :func:`estimate_derivatives`. If ``k`` is omitted, the estimator's
        method-specific default is used: cubic for ``method="spline"`` and
        quartic for ``method="gcv"``. The same degree is therefore used for
        the base fit and every bootstrap refit.
    n_boot
        Number of bootstrap replicates.
    alpha
        Significance level (default 0.05 gives a 95% CI).
    seed
        Seed for the bootstrap RNG (deterministic).
    return_diagnostics
        If True, also return a ``dict`` with bootstrap diagnostics: number
        of successful detections, number of failures, failure fraction,
        and the bootstrap sample mean and standard deviation.

    Returns
    -------
    estimate, lo, hi : float
        Landmark from the base fit and the lower/upper percentile-bootstrap
        confidence bounds.
    diagnostics : dict, optional
        Returned only if ``return_diagnostics=True``.

    Notes
    -----
    If more than 20% of bootstrap replicates fail to detect the landmark,
    the function returns ``(estimate, NaN, NaN)`` to indicate instability
    and emits a WARNING-level log record. Callers that want to inspect the
    failure fraction without log plumbing should pass
    ``return_diagnostics=True``.
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
        if return_diagnostics:
            diag = {
                "n_success": 0,
                "n_fail": 0,
                "fail_fraction": float("nan"),
                "bootstrap_mean": float("nan"),
                "bootstrap_std": float("nan"),
            }
            return float("nan"), float("nan"), float("nan"), diag
        return float("nan"), float("nan"), float("nan")

    resid = y_arr - yhat
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        r_star = rng.choice(resid, size=resid.size, replace=True)
        y_star = yhat + r_star
        try:
            yhat_b, dy_b, d2y_b, _ = estimate_derivatives(t_arr, y_star, method=method, s=s, k=k)
            boots[b] = float(landmark_fn(t_arr, yhat_b, dy_b, d2y_b))
        except Exception:
            boots[b] = float("nan")

    valid = boots[np.isfinite(boots)]
    n_success = int(valid.size)
    n_fail = int(n_boot - n_success)
    fail_fraction = float(n_fail / n_boot)

    unstable = valid.size < 0.8 * n_boot
    if unstable:
        logger.warning(
            "Landmark detection failed in %d / %d bootstrap replicates "
            "(%.1f%%); returning NaN CI bounds.",
            n_fail,
            n_boot,
            100.0 * fail_fraction,
        )
        lo = float("nan")
        hi = float("nan")
    else:
        lo = float(np.quantile(valid, alpha / 2))
        hi = float(np.quantile(valid, 1 - alpha / 2))

    if return_diagnostics:
        diag = {
            "n_success": n_success,
            "n_fail": n_fail,
            "fail_fraction": fail_fraction,
            "bootstrap_mean": float(np.mean(valid)) if valid.size else float("nan"),
            "bootstrap_std": float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan"),
        }
        return estimate, lo, hi, diag

    return estimate, lo, hi
