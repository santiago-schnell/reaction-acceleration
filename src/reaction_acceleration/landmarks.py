"""Landmark extraction utilities.

Currently includes a zero-crossing detector with sub-sample precision and
correct handling of samples that land exactly on zero.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .derivatives import _as_1d_float


def primary_zero_crossing_time(
    t: np.ndarray,
    z: np.ndarray,
    *,
    direction: Literal["pos_to_neg", "neg_to_pos"] = "pos_to_neg",
) -> float:
    """Return the first time `z(t)` crosses zero in the specified direction.

    Parameters
    ----------
    t, z:
        Time points and signal values.
    direction:
        - ``"pos_to_neg"``: find the first positive-to-negative crossing
        - ``"neg_to_pos"``: find the first negative-to-positive crossing

    Returns
    -------
    float
        Interpolated crossing time (linear interpolation), or NaN if no
        crossing is found.

    Notes
    -----
    Exact-zero samples are handled by looking one sample in each direction:
    a transition `z[i-1] > 0` then `z[i] == 0` (or vice versa) is treated as
    a crossing at time ``t[i]``. This fixes a subtle defect where a sample
    landing exactly on zero could be silently missed if the immediately
    surrounding signs did not satisfy the strict inequality test.
    """

    t_arr = _as_1d_float(t)
    z_arr = _as_1d_float(z)

    if t_arr.size != z_arr.size:
        raise ValueError("t and z must have the same length.")

    if t_arr.size < 2:
        return float("nan")

    want_pos_to_neg = direction == "pos_to_neg"

    # Pairwise sweep
    for i in range(1, t_arr.size):
        z0 = float(z_arr[i - 1])
        z1 = float(z_arr[i])

        # --- Exact zero at sample i is treated as a crossing if z0 has the
        #     required sign (or, for a flat region, if the next non-zero
        #     sample is in the required direction).
        if z1 == 0.0:
            # Look backward for the last non-zero sample
            z_before = z0
            j = i - 1
            while z_before == 0.0 and j > 0:
                j -= 1
                z_before = float(z_arr[j])
            if want_pos_to_neg and z_before > 0.0:
                # If the very next non-zero sample is negative, the true
                # crossing is at t[i]; if it's positive, the signal merely
                # touched zero and returned, which is not a crossing.
                k = i + 1
                z_after = 0.0
                while k < t_arr.size and z_after == 0.0:
                    z_after = float(z_arr[k])
                    k += 1
                if z_after < 0.0 or (z_after == 0.0 and k >= t_arr.size):
                    return float(t_arr[i])
            if (not want_pos_to_neg) and z_before < 0.0:
                k = i + 1
                z_after = 0.0
                while k < t_arr.size and z_after == 0.0:
                    z_after = float(z_arr[k])
                    k += 1
                if z_after > 0.0 or (z_after == 0.0 and k >= t_arr.size):
                    return float(t_arr[i])
            # else: not a crossing; continue
            continue

        # --- Strict sign change between consecutive non-zero samples.
        if want_pos_to_neg:
            crosses = (z0 > 0.0) and (z1 < 0.0)
        else:
            crosses = (z0 < 0.0) and (z1 > 0.0)

        if crosses:
            t0 = float(t_arr[i - 1])
            t1 = float(t_arr[i])
            # Linear interpolation for sub-sample accuracy.
            if z1 == z0:
                return t0
            return float(t0 - z0 * (t1 - t0) / (z1 - z0))

    return float("nan")


def acceleration_zero_crossing_time(
    t: np.ndarray,
    dy: np.ndarray,
    d2y: np.ndarray,
    *,
    direction: Literal["pos_to_neg", "neg_to_pos"] = "pos_to_neg",
    edge_fraction: float = 0.05,
) -> float:
    """Return the acceleration zero crossing nearest the rate extremum.

    This is the recommended inflection-time landmark for sigmoidal
    (autocatalytic) and single-peaked progress curves. For such curves the
    true inflection point coincides with an extremum of the rate ``dy``:
    a ``pos_to_neg`` crossing of the acceleration ``d2y`` sits at the rate
    *maximum*, and a ``neg_to_pos`` crossing sits at the rate *minimum*.

    Unlike :func:`primary_zero_crossing_time`, which returns the *first*
    crossing in the requested direction, this detector selects, among all
    crossings in that direction, the one closest in time to the rate
    extremum. On noisy second-derivative traces the naive "first crossing"
    is frequently captured by a spurious early sign flip and is dragged
    toward ``t = 0``; anchoring on the rate extremum removes that failure
    mode and yields stable bootstrap confidence intervals.

    Parameters
    ----------
    t, dy, d2y:
        Time points, first derivative (rate), and second derivative
        (acceleration). All three must have the same length.
    direction:
        - ``"pos_to_neg"``: crossing nearest the rate maximum (default;
          appropriate for autocatalytic/sigmoidal product curves).
        - ``"neg_to_pos"``: crossing nearest the rate minimum
          (appropriate for the intermediate of a consecutive reaction).
    edge_fraction:
        Fraction of the samples at each end of the record that is excluded
        when locating the rate-extremum anchor and when accepting crossings
        (default 0.05, i.e. the first and last 5 % of points). Second
        derivatives are least reliable near the boundaries, and on bootstrap
        resamples the rate extremum can otherwise migrate to an endpoint and
        drag the landmark to a spurious tail crossing. The guard is disabled
        (``edge_fraction <= 0``) for very short records, where it would empty
        the interior; in that case the unguarded behaviour is recovered.

    Returns
    -------
    float
        Interpolated crossing time (linear interpolation), or NaN if no
        crossing in the requested direction is found.

    Notes
    -----
    The rate extremum is used purely as an operational anchor to
    disambiguate multiple candidate crossings; it is not a mechanistic
    discriminator. Detection uses strict sign changes between consecutive
    samples with linear sub-grid interpolation; because it operates on a
    smoothed second derivative, exact zeros at sample points do not arise
    in practice.
    """

    t_arr = _as_1d_float(t)
    dy_arr = _as_1d_float(dy)
    d2y_arr = _as_1d_float(d2y)

    if not (t_arr.size == dy_arr.size == d2y_arr.size):
        raise ValueError("t, dy, and d2y must have the same length.")

    if t_arr.size < 2:
        return float("nan")

    want_pos_to_neg = direction == "pos_to_neg"

    # Interior guard: ignore the first/last ``edge`` samples when locating the
    # rate-extremum anchor and when accepting crossings. The boundary regions
    # are where the smoothed second derivative is least reliable (cf. the
    # boundary-derivative caveat in the SI). On bootstrap resamples the rate
    # extremum can otherwise migrate to an endpoint, after which the
    # nearest-crossing rule returns a spurious tail landmark.
    n = t_arr.size
    edge = int(np.floor(max(0.0, edge_fraction) * n))
    edge = min(edge, (n - 1) // 2)  # never empty the interior
    lo_i, hi_i = edge, n - edge  # interior index range [lo_i, hi_i)

    # Rate extremum (anchor) restricted to the interior.
    interior = slice(lo_i, hi_i)
    if want_pos_to_neg:
        t_anchor = float(t_arr[lo_i + int(np.argmax(dy_arr[interior]))])
    else:
        t_anchor = float(t_arr[lo_i + int(np.argmin(dy_arr[interior]))])

    # Collect every crossing in the requested direction (sub-grid interpolated),
    # restricted to interior sample pairs.
    crossings = []
    for i in range(max(1, lo_i + 1), hi_i):
        z0 = float(d2y_arr[i - 1])
        z1 = float(d2y_arr[i])
        if want_pos_to_neg:
            crosses = (z0 > 0.0) and (z1 < 0.0)
        else:
            crosses = (z0 < 0.0) and (z1 > 0.0)
        if crosses:
            t0 = float(t_arr[i - 1])
            t1 = float(t_arr[i])
            tc = t0 if z1 == z0 else t0 - z0 * (t1 - t0) / (z1 - z0)
            crossings.append(float(tc))

    if not crossings:
        return float("nan")

    cand = np.asarray(crossings, dtype=float)
    return float(cand[int(np.argmin(np.abs(cand - t_anchor)))])
