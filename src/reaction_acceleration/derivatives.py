"""Derivative estimation for progress-curve analysis.

This module implements two approaches:

1) **Smoothing spline** (recommended): fits `scipy.interpolate.UnivariateSpline`
   and differentiates analytically. Suitable for irregular sampling.

2) **Savitzky–Golay**: local-polynomial filtering (`scipy.signal.savgol_filter`).
   Requires approximately uniform sampling.

Notes
-----
- Differentiation amplifies high-frequency noise; for acceleration analysis, avoid
  finite-difference derivatives on raw data.
- For splines, a practical heuristic is `s ~= N * sigma^2`, where `sigma` is the
  measurement-noise standard deviation.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import savgol_filter
except Exception as e:  # pragma: no cover
    raise ImportError("reaction_acceleration requires SciPy (interpolate, signal).") from e


Method = Literal["spline", "savgol"]
ArrayLike = Union[np.ndarray, list, tuple]


def _as_1d_float(x: ArrayLike) -> np.ndarray:
    """Convert input to a 1D float array, checking finiteness."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or infinite values.")
    return arr


def estimate_derivatives(
    t: ArrayLike,
    y: ArrayLike,
    *,
    method: Method = "spline",
    # Spline parameters
    s: Optional[float] = None,
    k: int = 3,
    w: Optional[ArrayLike] = None,
    # Savitzky-Golay parameters
    window_length: int = 21,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Estimate y(t), dy/dt, and d2y/dt2 from noisy progress-curve data.

    Parameters
    ----------
    t, y:
        Arrays of time points and observed signal.
    method:
        "spline" (default) or "savgol".
    s:
        Smoothing factor for the spline. Heuristic: `s ~= N * sigma^2`.
        - `s = 0` gives an interpolating spline.
    k:
        Degree of spline (default: cubic `k=3`).
    w:
        Optional weights for spline fitting (e.g., inverse standard deviation).
    window_length, polyorder:
        Savitzky–Golay parameters (uniform sampling only). `window_length` must
        be odd and > polyorder.

    Returns
    -------
    yhat, dy, d2y, model
        Smoothed function, first derivative, second derivative, and the fitted
        model object.

    Raises
    ------
    ValueError
        If input arrays are invalid, or if sampling assumptions for the chosen
        method are violated.
    """

    t_arr = _as_1d_float(t)
    y_arr = _as_1d_float(y)

    if t_arr.size != y_arr.size:
        raise ValueError(f"Length mismatch: t ({t_arr.size}) vs y ({y_arr.size}).")

    if t_arr.size < k + 1:
        raise ValueError(f"Too few data points ({t_arr.size}) for spline degree k={k}.")

    # Sort by time and enforce strict monotonicity
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    y_arr = y_arr[order]

    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("t must be strictly increasing (remove or average duplicates).")

    if method == "spline":
        if w is not None:
            w_arr = _as_1d_float(w)
            if w_arr.size != t_arr.size:
                raise ValueError("Weights w must match the length of t.")
        else:
            w_arr = None

        spl = UnivariateSpline(t_arr, y_arr, w=w_arr, s=s, k=k)
        yhat = spl(t_arr)
        dy = spl.derivative(1)(t_arr)
        d2y = spl.derivative(2)(t_arr)
        return yhat, dy, d2y, spl

    if method == "savgol":
        # Check approximate uniform sampling
        dt = np.diff(t_arr)
        dt_mean = float(np.mean(dt))
        if dt_mean <= 0:
            raise ValueError("Invalid time grid.")
        if float(np.std(dt)) > 0.05 * dt_mean:
            raise ValueError(
                "Savitzky–Golay requires (approximately) uniform sampling; "
                "use method='spline' for irregular sampling."
            )

        n = t_arr.size
        # Ensure an odd window length <= n
        max_odd = n if (n % 2 == 1) else (n - 1)
        wl = min(window_length, max_odd)
        if wl % 2 == 0:
            wl -= 1
        if wl < polyorder + 2:
            raise ValueError(
                f"window_length ({wl}) too small for polyorder ({polyorder}). "
                "Increase data density or reduce polyorder."
            )

        yhat = savgol_filter(y_arr, wl, polyorder, deriv=0)
        dy = savgol_filter(y_arr, wl, polyorder, deriv=1, delta=dt_mean)
        d2y = savgol_filter(y_arr, wl, polyorder, deriv=2, delta=dt_mean)

        model = {"window_length": wl, "polyorder": polyorder, "delta": dt_mean}
        return yhat, dy, d2y, model

    raise ValueError(f"Unknown method: {method}")
