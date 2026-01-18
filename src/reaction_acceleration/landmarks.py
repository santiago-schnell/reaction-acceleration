"""Landmark extraction utilities.

Currently includes a minimal zero-crossing detector with sub-sample precision.
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
        - "pos_to_neg": find the first positive-to-negative crossing
        - "neg_to_pos": find the first negative-to-positive crossing

    Returns
    -------
    float
        Interpolated crossing time (linear interpolation), or NaN if no crossing
        is found.

    Notes
    -----
    - If a sample hits *exactly* zero, this is treated as a crossing time.
    - This function assumes `t` is increasing and `z` is aligned with `t`.
    """

    t_arr = _as_1d_float(t)
    z_arr = _as_1d_float(z)

    if t_arr.size != z_arr.size:
        raise ValueError("t and z must have the same length.")

    # Iterate pairwise for robust handling of exact zeros
    for i in range(1, t_arr.size):
        z0 = float(z_arr[i - 1])
        z1 = float(z_arr[i])

        # Exact zeros
        if z1 == 0.0:
            # If z0 had the required sign, treat as a crossing
            if direction == "pos_to_neg" and z0 > 0:
                return float(t_arr[i])
            if direction == "neg_to_pos" and z0 < 0:
                return float(t_arr[i])
            # Otherwise continue; could be a flat region

        if direction == "pos_to_neg":
            crosses = (z0 > 0.0) and (z1 < 0.0)
        else:
            crosses = (z0 < 0.0) and (z1 > 0.0)

        if crosses:
            t0 = float(t_arr[i - 1])
            t1 = float(t_arr[i])
            # Linear interpolation
            if z1 == z0:
                return t0
            return float(t0 - z0 * (t1 - t0) / (z1 - z0))

    return float("nan")
