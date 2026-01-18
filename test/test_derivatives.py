import numpy as np

from reaction_acceleration import estimate_derivatives, primary_zero_crossing_time


def test_spline_second_derivative_exponential_decay():
    """Spline derivatives should be accurate on noiseless smooth data."""
    t = np.linspace(0.0, 3.0, 200)
    k = 1.7
    y = np.exp(-k * t)

    yhat, dy, d2y, _ = estimate_derivatives(t, y, method="spline", s=0.0)

    d2y_true = (k**2) * np.exp(-k * t)
    rmse = float(np.sqrt(np.mean((d2y - d2y_true) ** 2)))

    assert rmse < 1e-2

    # No sign change expected: always positive
    t0 = primary_zero_crossing_time(t, d2y, direction="pos_to_neg")
    assert np.isnan(t0)


def test_primary_zero_crossing_time_linear_interpolation():
    t = np.array([0.0, 1.0, 2.0])
    z = np.array([+1.0, -1.0, -2.0])

    tc = primary_zero_crossing_time(t, z, direction="pos_to_neg")
    assert np.isclose(tc, 0.5)
