"""Tests for reaction_acceleration.landmarks.acceleration_zero_crossing_time."""

from __future__ import annotations

import numpy as np
import pytest

from reaction_acceleration import (
    acceleration_zero_crossing_time,
    estimate_derivatives,
    primary_zero_crossing_time,
)


def _logistic_B(t, k=1.5, A_tot=1.0, B0=0.02):
    denom = 1.0 + ((A_tot / B0) - 1.0) * np.exp(-k * A_tot * t)
    return A_tot / denom


def test_clean_logistic_recovers_true_inflection():
    """On the noise-free logistic curve the detector recovers t* = ln(49)/1.5."""
    t = np.linspace(0.0, 6.0, 400)
    B = _logistic_B(t)
    s = 1e-6  # near-interpolating: derivatives essentially analytic
    _, dB, d2B, _ = estimate_derivatives(t, B, method="spline", s=s)
    t_star = acceleration_zero_crossing_time(t, dB, d2B, direction="pos_to_neg")
    t_true = np.log(49.0) / 1.5
    assert abs(t_star - t_true) < 0.05


def test_robust_detector_ignores_spurious_early_crossing():
    """A spurious early pos->neg blip must not capture the landmark.

    The genuine crossing sits at the rate maximum (t = 6); an injected early
    blip near t ~ 0.85 produces a pos->neg crossing that the naive *first*
    crossing detector returns, while the rate-anchored detector ignores it.
    """
    t = np.linspace(0.0, 10.0, 100)  # no sample lands exactly on 6.0
    dy = np.exp(-0.5 * ((t - 6.0) / 1.0) ** 2)  # rate peaks at t = 6
    d2y = -(t - 6.0) * dy  # pos for t<6, neg for t>6

    # Inject a spurious early pos->neg wiggle around index 8-9 (t ~ 0.8-0.9).
    d2y = d2y.copy()
    d2y[8] += 0.5
    d2y[9] -= 0.5

    t_first = primary_zero_crossing_time(t, d2y, direction="pos_to_neg")
    t_robust = acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")

    assert t_first < 1.5  # naive detector trapped by the blip
    assert abs(t_robust - 6.0) < 0.2  # robust detector finds the real one


def test_neg_to_pos_anchors_on_rate_minimum():
    """For a neg->pos search the anchor is the rate minimum."""
    t = np.linspace(0.0, 10.0, 100)
    dy = -np.exp(-0.5 * ((t - 7.0) / 1.0) ** 2)  # rate has a minimum at t = 7
    d2y = (t - 7.0) * np.exp(-0.5 * ((t - 7.0) / 1.0) ** 2)  # neg then pos at t=7
    t_robust = acceleration_zero_crossing_time(t, dy, d2y, direction="neg_to_pos")
    assert abs(t_robust - 7.0) < 0.2


def test_no_crossing_returns_nan():
    t = np.linspace(0.0, 5.0, 50)
    dy = np.ones_like(t)
    d2y = np.ones_like(t)  # strictly positive: no pos->neg crossing
    assert np.isnan(acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg"))


def test_length_mismatch_raises():
    t = np.linspace(0.0, 5.0, 50)
    dy = np.ones(50)
    d2y = np.ones(49)
    with pytest.raises(ValueError, match="same length"):
        acceleration_zero_crossing_time(t, dy, d2y)


def test_interior_guard_rejects_boundary_anchor():
    """The interior guard must stop a boundary rate-extremum from capturing
    the landmark.

    The rate ``dy`` has a genuine interior peak at t = 3 but a slightly larger
    spike at the final sample; the acceleration ``d2y`` has an interior
    pos->neg crossing at t ~ 3 and a spurious one near the end. With the guard
    disabled the anchor migrates to the endpoint and the detector returns the
    tail crossing; with the default guard the anchor is pinned to the interior
    and the true crossing is recovered.
    """
    t = np.linspace(0.0, 10.0, 100)
    dy = np.exp(-0.5 * ((t - 3.0) / 0.8) ** 2)
    dy[-1] = 1.05  # boundary spike -> argmax at the endpoint
    d2y = -(t - 3.0) * np.exp(-0.5 * ((t - 3.0) / 0.8) ** 2)
    d2y[-3:] = [0.4, -0.2, -0.5]  # inject a boundary pos->neg crossing

    t_unguarded = acceleration_zero_crossing_time(
        t, dy, d2y, direction="pos_to_neg", edge_fraction=0.0
    )
    t_guarded = acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")

    assert t_unguarded > 8.0  # endpoint anchor -> spurious tail crossing
    assert abs(t_guarded - 3.0) < 0.25  # guard recovers the interior crossing


def test_guard_disabled_on_short_records():
    """For very short records the guard must collapse to the unguarded result
    rather than emptying the interior."""
    t = np.array([0.0, 1.0, 2.0])
    dy = np.array([0.0, 1.0, 0.0])  # peak at the middle sample
    d2y = np.array([1.0, 0.5, -1.0])  # single pos->neg crossing between i=1,2
    tc = acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")
    assert np.isfinite(tc) and 1.0 < tc <= 2.0
