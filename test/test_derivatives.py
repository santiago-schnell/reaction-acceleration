"""Tests for reaction_acceleration.derivatives and .landmarks."""

from __future__ import annotations

import numpy as np
import pytest

from reaction_acceleration import (
    estimate_derivatives,
    primary_zero_crossing_time,
)

# ---------------------------------------------------------------------------
# Spline branch
# ---------------------------------------------------------------------------


def test_spline_second_derivative_exponential_decay():
    """Spline derivatives should be accurate on noiseless smooth data."""
    t = np.linspace(0.0, 3.0, 200)
    k = 1.7
    y = np.exp(-k * t)

    _yhat, _dy, d2y, _ = estimate_derivatives(t, y, method="spline", s=0.0)

    d2y_true = (k**2) * np.exp(-k * t)
    rmse = float(np.sqrt(np.mean((d2y - d2y_true) ** 2)))
    assert rmse < 1e-2

    # No sign change expected: d2y is always positive.
    t0 = primary_zero_crossing_time(t, d2y, direction="pos_to_neg")
    assert np.isnan(t0)


def test_spline_handles_unordered_input():
    """Spline fit should sort time silently."""
    rng = np.random.default_rng(7)
    t_sorted = np.linspace(0.0, 3.0, 200)
    y_sorted = np.exp(-1.7 * t_sorted)

    perm = rng.permutation(t_sorted.size)
    t_perm = t_sorted[perm]
    y_perm = y_sorted[perm]

    yhat, _dy, _d2y, _ = estimate_derivatives(t_perm, y_perm, method="spline", s=0.0)

    # yhat is returned at the sorted grid; check it matches the clean signal.
    assert np.allclose(yhat, y_sorted, atol=1e-6)


# ---------------------------------------------------------------------------
# Savitzky-Golay branch
# ---------------------------------------------------------------------------


def test_savgol_second_derivative_exponential_decay():
    """Savitzky-Golay should give reasonable second derivatives on uniform grids."""
    t = np.linspace(0.0, 3.0, 200)
    k = 1.7
    y = np.exp(-k * t)

    _yhat, _dy, d2y, model = estimate_derivatives(
        t,
        y,
        method="savgol",
        window_length=21,
        polyorder=3,
    )

    d2y_true = (k**2) * np.exp(-k * t)
    # SG boundary artifacts degrade the RMSE near the edges; check the interior.
    interior = slice(20, -20)
    rmse_interior = float(np.sqrt(np.mean((d2y[interior] - d2y_true[interior]) ** 2)))
    assert rmse_interior < 5e-2

    assert isinstance(model, dict)
    assert model["window_length"] == 21
    assert model["polyorder"] == 3


def test_savgol_rejects_nonuniform_sampling():
    """Savitzky-Golay is only defined for uniform grids; must raise otherwise."""
    t = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.35,
            0.5,
            0.75,
            1.0,
            1.1,
            1.3,
            1.45,
            1.6,
            1.9,
            2.1,
            2.25,
            2.4,
            2.5,
            2.7,
            2.8,
            2.9,
            3.0,
        ]
    )
    y = np.exp(-1.7 * t)
    with pytest.raises(ValueError, match="uniform"):
        estimate_derivatives(t, y, method="savgol", window_length=9, polyorder=3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_mismatched_lengths():
    t = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 9)  # wrong length
    with pytest.raises(ValueError, match="Length mismatch"):
        estimate_derivatives(t, y, method="spline", s=0.0)


def test_rejects_duplicate_time_points():
    t = np.array([0.0, 0.1, 0.1, 0.2, 0.3])  # duplicate at 0.1
    y = np.array([1.0, 0.9, 0.91, 0.81, 0.73])
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_derivatives(t, y, method="spline", s=0.0)


def test_rejects_non_finite_input():
    t = np.array([0.0, 0.1, 0.2, np.nan, 0.4])
    y = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    with pytest.raises(ValueError, match=r"NaN|finite"):
        estimate_derivatives(t, y, method="spline", s=0.0)


def test_rejects_unknown_method():
    t = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_derivatives(t, y, method="not-a-method", s=0.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Zero-crossing detector
# ---------------------------------------------------------------------------


def test_primary_zero_crossing_time_linear_interpolation():
    """Canonical pos->neg case with sub-sample precision."""
    t = np.array([0.0, 1.0, 2.0])
    z = np.array([+1.0, -1.0, -2.0])
    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isclose(tc, 0.5)


def test_primary_zero_crossing_time_neg_to_pos():
    t = np.array([0.0, 1.0, 2.0])
    z = np.array([-2.0, -1.0, +3.0])
    tc = primary_zero_crossing_time(t, z, direction="neg_to_pos")
    # Crossing in (1, 2) with slope 4; zero at t = 1 - (-1)*(1/4) = 1.25
    assert np.isclose(tc, 1.25)


def test_primary_zero_crossing_exact_zero_sample_pos_to_neg():
    """A sample landing exactly on zero with a flanking sign change must be
    reported, not silently skipped."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([+2.0, 0.0, -1.0, -3.0])
    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isclose(tc, 1.0)


def test_primary_zero_crossing_exact_zero_touch_is_not_a_crossing():
    """A signal that touches zero and returns to the same sign must not be
    reported as a crossing."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([+2.0, 0.0, +1.0, +3.0])  # touches zero then goes positive again
    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isnan(tc)


def test_primary_zero_crossing_no_crossing_returns_nan():
    t = np.array([0.0, 1.0, 2.0])
    z = np.array([+1.0, +2.0, +3.0])
    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isnan(tc)


def test_primary_zero_crossing_rejects_mismatched_lengths():
    t = np.array([0.0, 1.0, 2.0])
    z = np.array([+1.0, -1.0])
    with pytest.raises(ValueError, match="same length"):
        primary_zero_crossing_time(t, z, direction="pos_to_neg")


def test_primary_zero_crossing_single_point_returns_nan():
    t = np.array([0.0])
    z = np.array([+1.0])
    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isnan(tc)
