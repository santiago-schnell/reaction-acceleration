"""Tests for reaction_acceleration.bootstrap."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from reaction_acceleration import (
    primary_zero_crossing_time,
    residual_bootstrap_landmark_ci,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logistic_B(t, k=1.5, A_tot=1.0, B0=0.02):
    """Analytical solution for A + B -> 2B with mass conservation."""
    denom = 1.0 + ((A_tot / B0) - 1.0) * np.exp(-k * A_tot * t)
    return A_tot / denom


def _landmark_first_pos_to_neg_zero(t, yhat, dy, d2y):
    """Canonical autocatalysis landmark: first pos->neg zero of d2y."""
    return primary_zero_crossing_time(t, d2y, direction="pos_to_neg")


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_residual_bootstrap_sorts_time_and_returns_ci():
    """Bootstrap must tolerate unordered input and produce a finite, containing CI."""
    rng = np.random.default_rng(0)

    t_sorted = np.linspace(0.0, 6.0, 180)
    y_true = _logistic_B(t_sorted)

    # Small noise ensures non-zero residuals but keeps the landmark stable.
    sigma = 1e-4
    y_obs = y_true + rng.normal(0.0, sigma, size=y_true.shape)

    # Permute time to validate internal sorting/realignment.
    perm = rng.permutation(len(t_sorted))
    t = t_sorted[perm]
    y = y_obs[perm]

    s = 2.0 * len(t_sorted) * (sigma**2)

    est, lo, hi = residual_bootstrap_landmark_ci(
        t,
        y,
        landmark_fn=_landmark_first_pos_to_neg_zero,
        method="spline",
        s=s,
        n_boot=50,
        alpha=0.05,
        seed=1,
    )

    assert np.isfinite(est)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= est <= hi

    # Theoretical inflection time for logistic model at B = A_tot/2.
    t_theory = np.log((1.0 / 0.02) - 1.0) / (1.5 * 1.0)
    assert abs(est - t_theory) < 0.2


# ---------------------------------------------------------------------------
# Regression: SI-canonical autocatalysis
# ---------------------------------------------------------------------------


def test_canonical_si_autocatalysis_landmark_matches_theory():
    """The SI claims the canonical autocatalysis landmark is t* ~ 2.59 s at
    k=1.5, A_tot=1, B0=0.02, sigma=0.01. This regression test locks in that
    behaviour."""
    rng = np.random.default_rng(42)
    k, A_tot, B0 = 1.5, 1.0, 0.02
    n, sigma = 100, 0.01
    t = np.linspace(0.0, 6.0, n)
    B_clean = _logistic_B(t, k=k, A_tot=A_tot, B0=B0)
    y = B_clean + sigma * rng.standard_normal(n)

    s = 2.0 * n * sigma**2
    est, lo, hi = residual_bootstrap_landmark_ci(
        t,
        y,
        landmark_fn=_landmark_first_pos_to_neg_zero,
        method="spline",
        s=s,
        n_boot=200,
        alpha=0.05,
        seed=101,
    )

    t_theory = np.log((A_tot - B0) / B0) / (k * A_tot)  # ~2.59 s

    # Regression: the base-fit estimate should land in a plausible window.
    # Note: 95% CI coverage is a statistical property averaged over many
    # noise realisations and is not guaranteed for a single fixed seed,
    # so we do not assert lo <= t_theory <= hi here. (See SI section 8.3
    # for a coverage study across noise levels and realisations.)
    assert np.isfinite(est) and np.isfinite(lo) and np.isfinite(hi)
    assert 2.2 < est < 3.0
    assert abs(est - t_theory) < 0.3
    assert lo < hi, "Bootstrap CI must be ordered."


# ---------------------------------------------------------------------------
# Diagnostics return path
# ---------------------------------------------------------------------------


def test_return_diagnostics_reports_fail_count():
    """With return_diagnostics=True, callers should get a success/failure
    breakdown and the bootstrap sample mean/std."""
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 6.0, 180)
    y_true = _logistic_B(t)
    y = y_true + rng.normal(0.0, 1e-4, size=t.shape)
    s = 2.0 * len(t) * (1e-4**2)

    _est, _lo, _hi, diag = residual_bootstrap_landmark_ci(
        t,
        y,
        landmark_fn=_landmark_first_pos_to_neg_zero,
        method="spline",
        s=s,
        n_boot=40,
        seed=3,
        return_diagnostics=True,
    )

    assert set(diag.keys()) == {
        "n_success",
        "n_fail",
        "fail_fraction",
        "bootstrap_mean",
        "bootstrap_std",
    }
    assert diag["n_success"] + diag["n_fail"] == 40
    assert 0.0 <= diag["fail_fraction"] <= 1.0
    assert np.isfinite(diag["bootstrap_mean"])
    assert np.isfinite(diag["bootstrap_std"])


def test_unstable_landmark_warns_and_returns_nan_ci(caplog):
    """When >20% of bootstrap replicates fail, the CI must be NaN and a
    WARNING must be emitted. We force this by using a landmark function
    that returns NaN for most bootstrap replicates."""
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 6.0, 120)
    y = _logistic_B(t) + rng.normal(0.0, 1e-4, size=t.shape)
    s = 2.0 * len(t) * (1e-4**2)

    call_count = {"n": 0}

    def flaky_landmark(t, yhat, dy, d2y):
        """Return a valid landmark only on the very first call (base fit);
        return NaN for subsequent (bootstrap) calls to force the unstable
        path."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            return primary_zero_crossing_time(t, d2y, direction="pos_to_neg")
        return float("nan")

    with caplog.at_level(logging.WARNING, logger="reaction_acceleration.bootstrap"):
        est, lo, hi = residual_bootstrap_landmark_ci(
            t,
            y,
            landmark_fn=flaky_landmark,
            method="spline",
            s=s,
            n_boot=20,
            seed=7,
        )

    assert np.isfinite(est)  # base fit succeeded
    assert np.isnan(lo) and np.isnan(hi)  # all bootstrap replicates failed
    assert any("bootstrap replicates" in r.message for r in caplog.records)


def test_gcv_bootstrap_default_matches_explicit_quartic():
    """The bootstrap must inherit the GCV estimator's quartic default."""
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 6.0, 100)
    y = _logistic_B(t) + rng.normal(0.0, 0.01, size=t.shape)

    common = dict(
        landmark_fn=_landmark_first_pos_to_neg_zero,
        method="gcv",
        n_boot=20,
        alpha=0.05,
        seed=17,
    )

    implicit = residual_bootstrap_landmark_ci(t, y, **common)
    explicit = residual_bootstrap_landmark_ci(t, y, k=4, **common)

    assert np.allclose(
        implicit,
        explicit,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_bootstrap_rejects_bad_n_boot():
    t = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="n_boot"):
        residual_bootstrap_landmark_ci(
            t,
            y,
            landmark_fn=_landmark_first_pos_to_neg_zero,
            n_boot=0,
        )


def test_bootstrap_rejects_bad_alpha():
    t = np.linspace(0.0, 1.0, 10)
    y = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="alpha"):
        residual_bootstrap_landmark_ci(
            t,
            y,
            landmark_fn=_landmark_first_pos_to_neg_zero,
            n_boot=10,
            alpha=1.5,
        )
