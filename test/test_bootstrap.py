import numpy as np

from reaction_acceleration import estimate_derivatives, primary_zero_crossing_time, residual_bootstrap_landmark_ci


def _logistic_B(t, k=1.5, A_tot=1.0, B0=0.02):
    denom = 1.0 + ((A_tot / B0) - 1.0) * np.exp(-k * A_tot * t)
    return A_tot / denom


def _landmark_first_pos_to_neg_zero(t, yhat, dy, d2y):
    return primary_zero_crossing_time(t, d2y, direction="pos_to_neg")


def test_residual_bootstrap_sorts_time_and_returns_ci():
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

    # Base fit should succeed
    yhat, dy, d2y, _ = estimate_derivatives(t, y, method="spline", s=s)
    t_star = _landmark_first_pos_to_neg_zero(np.sort(t), yhat, dy, d2y)

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

    # Theoretical inflection time for logistic model at B = A_tot/2
    # t_inf = ln((A_tot/B0)-1) / (k*A_tot)
    t_theory = np.log((1.0 / 0.02) - 1.0) / (1.5 * 1.0)
    assert abs(est - t_theory) < 0.2
