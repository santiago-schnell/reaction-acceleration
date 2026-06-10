"""Derivative estimation for progress-curve analysis.

This module implements two approaches:

1) **Smoothing spline** (recommended): fits `scipy.interpolate.UnivariateSpline`
   and differentiates analytically. Suitable for irregular sampling.

2) **Savitzky-Golay**: local-polynomial filtering (`scipy.signal.savgol_filter`).
   Requires approximately uniform sampling.

Notes
-----
- Differentiation amplifies high-frequency noise; for acceleration analysis, avoid
  finite-difference derivatives on raw data.
- For splines, a practical heuristic is `s ~= N * sigma^2`, where `sigma` is the
  measurement-noise standard deviation.
"""

from __future__ import annotations

import sys
from typing import Literal, Union

import numpy as np

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

try:
    from scipy.interpolate import BSpline, UnivariateSpline
    from scipy.linalg import eigh
    from scipy.signal import savgol_filter
except Exception as e:  # pragma: no cover
    raise ImportError("reaction_acceleration requires SciPy (interpolate, signal, linalg).") from e


Method = Literal["spline", "gcv", "savgol"]
ArrayLike: TypeAlias = Union[np.ndarray, list, tuple]


def _as_1d_float(x: ArrayLike) -> np.ndarray:
    """Convert input to a 1D float array, checking finiteness."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN or infinite values.")
    return arr


def _pspline_gcv(
    t: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 4,
    n_inner_knots: int = 40,
    n_lambda: int = 50,
    lam_grid: tuple[float, float] = (1e-8, 1e4),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Penalized B-spline (P-spline) smoother with a GCV-selected penalty.

    Implements the Eilers-Marx penalized B-spline with a second-order
    difference penalty, selecting the smoothing penalty ``lambda`` by
    generalized cross-validation (GCV). Unlike the fixed ``s = c n sigma**2``
    rule used with ``method="spline"``, the penalty is chosen from the data,
    which removes the systematic landmark bias documented in the SI.

    The basis depends only on the (sorted) time grid, so a single
    Demmler-Reinsch generalized-eigendecomposition diagonalizes the problem
    once; GCV selection and every fit are then a handful of vector
    operations, which keeps residual-bootstrap loops inexpensive.

    Parameters
    ----------
    t, y:
        Sorted time points and observations.
    k:
        B-spline degree (default quartic, ``k=4``). A quartic basis yields a
        smoother, more reliable second derivative than a cubic for landmark
        work, because the second derivative is then piecewise quadratic
        rather than piecewise linear.
    n_inner_knots:
        Number of equally spaced interior knots (default 40).
    n_lambda, lam_grid:
        Size and (log-spaced) span of the GCV penalty search grid.

    Returns
    -------
    yhat, dy, d2y, model
        Smoothed signal, first and second derivatives evaluated on ``t``,
        and a ``dict`` model containing the fitted :class:`BSpline` and the
        GCV-selected ``lam``.

    References
    ----------
    P. H. C. Eilers, B. D. Marx, "Flexible smoothing with B-splines and
    penalties," Statistical Science 11 (1996) 89-121.
    """

    n = t.size
    p = n_inner_knots + k + 1
    if n < p + 2:
        # Too few points for the requested basis; fall back to fewer knots.
        n_inner_knots = max(k, n - k - 3)
        p = n_inner_knots + k + 1

    a, b = float(t[0]), float(t[-1])
    inner = np.linspace(a, b, n_inner_knots + 2)[1:-1]
    knots = np.concatenate(([a] * (k + 1), inner, [b] * (k + 1)))

    bmat = BSpline.design_matrix(t, knots, k).toarray()  # (n, p)
    p = bmat.shape[1]

    # Second-order difference penalty on the B-spline coefficients.
    diff2 = np.diff(np.eye(p), n=2, axis=0)
    penalty = diff2.T @ diff2
    btb = bmat.T @ bmat

    # Demmler-Reinsch: solve P u = gamma (B^T B) u with B^T B-orthonormal U.
    gamma, u_mat = eigh(penalty, btb)
    bty = bmat.T @ y
    z = u_mat.T @ bty
    yty = float(y @ y)

    lams = np.logspace(np.log10(lam_grid[0]), np.log10(lam_grid[1]), n_lambda)
    shrink = 1.0 / (1.0 + np.outer(lams, gamma))  # (n_lambda, p)
    coeff_btb = np.sum((shrink * z) ** 2, axis=1)
    coeff_bty = (shrink * z) @ z
    rss = yty - 2.0 * coeff_bty + coeff_btb
    trace_s = shrink.sum(axis=1)
    denom = n - trace_s
    gcv = np.where(denom > 0, n * rss / np.square(denom), np.inf)

    j = int(np.argmin(gcv))
    coeff = u_mat @ (shrink[j] * z)
    spl = BSpline(knots, coeff, k)
    yhat = spl(t)
    dy = spl(t, nu=1)
    d2y = spl(t, nu=2)
    model = {"spline": spl, "lam": float(lams[j]), "k": k, "n_inner_knots": n_inner_knots}
    return yhat, dy, d2y, model


def estimate_derivatives(
    t: ArrayLike,
    y: ArrayLike,
    *,
    method: Method = "spline",
    # Spline parameters
    s: float | None = None,
    k: int | None = None,
    w: ArrayLike | None = None,
    # Savitzky-Golay parameters
    window_length: int = 21,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Estimate y(t), dy/dt, and d2y/dt2 from noisy progress-curve data.

    Parameters
    ----------
    t, y:
        Arrays of time points and observed signal.
    method:
        "spline" (default; fixed-penalty cubic smoothing spline, requires
        ``s``), "gcv" (penalized B-spline with a GCV-selected penalty;
        recommended for quantitative landmark work because it removes the
        smoothing bias of the fixed-``s`` rule), or "savgol".
    s:
        Smoothing factor for ``method="spline"``. Heuristic: `s ~= N * sigma^2`
        (``s ~= 1.5-2 N sigma^2`` for second-derivative stability). Ignored by
        ``method="gcv"``, which selects the penalty from the data.
    k:
        Spline degree. Defaults to cubic (``k=3``) for ``method="spline"`` and
        quartic (``k=4``) for ``method="gcv"``.
    w:
        Optional weights for spline fitting (e.g., inverse standard deviation).
    window_length, polyorder:
        Savitzky-Golay parameters (uniform sampling only). `window_length` must
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

    # Spline degree defaults by method: cubic for the fixed-penalty spline,
    # quartic for the GCV P-spline (a smoother second derivative for landmarks).
    if k is None:
        k = 4 if method == "gcv" else 3

    if t_arr.size < k + 1:
        raise ValueError(f"Too few data points ({t_arr.size}) for spline degree k={k}.")

    # Sort by time and enforce strict monotonicity
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    y_arr = y_arr[order]

    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("t must be strictly increasing (remove or average duplicates).")

    if method == "gcv":
        return _pspline_gcv(t_arr, y_arr, k=k)

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
                "Savitzky-Golay requires (approximately) uniform sampling; "
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
