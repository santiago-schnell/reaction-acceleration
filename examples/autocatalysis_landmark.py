"""Worked example: autocatalysis inflection-time landmark.

This reproduces the Supporting-Information style workflow:

- simulate A + B -> 2B (mass-action)
- add homoscedastic Gaussian noise to B(t)
- estimate derivatives two ways: the fixed-`s` smoothing spline (cautionary,
  for continuity with the SI tables) and the recommended GCV P-spline
- extract an inflection-time landmark from the acceleration signal
- compute a residual-bootstrap confidence interval for each

Run (from repo root):

```bash
python examples/autocatalysis_landmark.py
```

The diagnostic plot is saved to `outputs/examples/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from reaction_acceleration import (
    acceleration_zero_crossing_time,
    estimate_derivatives,
    residual_bootstrap_landmark_ci,
)


def autocatalysis_rhs(y, t, k):
    """RHS for A + B -> 2B under mass-action kinetics."""
    A, B = y
    return [-k * A * B, +k * A * B]


def landmark_inflection(t, yhat, dy, d2y) -> float:
    """Robust autocatalytic landmark.

    Thin wrapper around the public :func:`acceleration_zero_crossing_time`:
    the pos->neg acceleration zero-crossing nearest the rate maximum. This
    is the recommended detector for sigmoidal progress curves; it is robust
    to the spurious early zero-crossings that a naive "first crossing"
    picks up in noisy second-derivative traces.
    """
    return acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")


def main() -> None:
    # Data seed fixed for reproducibility; matches SI Sec. 8.3 and the
    # Sec. 10 listing so the example, the listing, and the table all agree.
    rng = np.random.default_rng(42)

    # ----------------------------------------------------------------------
    # 1) Synthetic data (canonical SI parameters)
    # ----------------------------------------------------------------------
    k = 1.5
    y0 = [0.98, 0.02]  # [A]0, [B]0  ->  A_tot = A0 + B0 = 1.0
    t = np.linspace(0.0, 6.0, 100)

    sol = odeint(autocatalysis_rhs, y0, t, args=(k,))
    B_true = sol[:, 1]

    sigma = 0.01
    B_obs = B_true + rng.normal(0.0, sigma, size=B_true.shape)

    # ----------------------------------------------------------------------
    # 2) Derivative estimation (smoothing spline)
    # ----------------------------------------------------------------------
    # Heuristic: s ~= N * sigma^2 ; for second-derivative stability we use
    # a slightly larger factor (s = 2 N sigma^2).
    s = 2.0 * len(t) * (sigma**2)

    B_hat, dB_hat, d2B_hat, _model = estimate_derivatives(t, B_obs, method="spline", s=s)

    t_star = landmark_inflection(t, B_hat, dB_hat, d2B_hat)

    # ----------------------------------------------------------------------
    # 3) Bootstrap CI
    # ----------------------------------------------------------------------
    est, lo, hi = residual_bootstrap_landmark_ci(
        t,
        B_obs,
        landmark_fn=landmark_inflection,
        method="spline",
        s=s,
        n_boot=500,
        alpha=0.05,
        seed=1,
    )

    # ----------------------------------------------------------------------
    # 4) Report
    # ----------------------------------------------------------------------
    # For A_tot = 1.0 and B0 = 0.02, the true inflection is:
    #   t*_true = ln((A_tot - B0) / B0) / (k * A_tot) = ln(49) / 1.5 ~ 2.595 s
    A_tot = sum(y0)
    t_true = float(np.log((A_tot - y0[1]) / y0[1]) / (k * A_tot))

    print("\n" + "=" * 68)
    print("Autocatalysis landmark: acceleration zero-crossing near v_max")
    print("=" * 68)
    print(f"  True inflection (theory)  : {t_true:.4f} s")
    print("  --- Cautionary: fixed rule s = 2 N sigma^2 ---")
    print(f"  Smoothing factor (s)      : {s:.4e}")
    print(f"  Base-fit estimate (t*)    : {t_star:.4f} s")
    print(f"  Bootstrap base estimate   : {est:.4f} s")
    print(f"  95% CI (percentile)       : [{lo:.4f}, {hi:.4f}] s")
    print(
        f"  CI contains truth         : " f"{'yes' if lo <= t_true <= hi else 'no (single-seed)'}"
    )

    # Recommended pipeline: GCV-selected penalty (lower-bias point estimate).
    _Bg, dBg, d2Bg, model_gcv = estimate_derivatives(t, B_obs, method="gcv")
    t_star_gcv = landmark_inflection(t, _Bg, dBg, d2Bg)
    _est_g, lo_g, hi_g = residual_bootstrap_landmark_ci(
        t,
        B_obs,
        landmark_fn=landmark_inflection,
        method="gcv",
        n_boot=500,
        alpha=0.05,
        seed=1,
    )
    print("  --- Recommended: GCV P-spline (data-driven penalty) ---")
    print(f"  GCV-selected lambda       : {model_gcv['lam']:.4e}")
    print(f"  Base-fit estimate (t*)    : {t_star_gcv:.4f} s")
    print(f"  95% CI (percentile)       : [{lo_g:.4f}, {hi_g:.4f}] s")
    print(
        f"  CI contains truth         : "
        f"{'yes' if lo_g <= t_true <= hi_g else 'no (single-seed)'}"
    )
    print("=" * 68 + "\n")

    # ----------------------------------------------------------------------
    # 5) Diagnostic plot
    # ----------------------------------------------------------------------
    outdir = Path(__file__).resolve().parents[1] / "outputs" / "examples"
    outdir.mkdir(parents=True, exist_ok=True)

    _fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax0 = axes[0]
    ax0.scatter(t, B_obs, s=10, color="gray", alpha=0.5, label="Noisy data")
    ax0.plot(t, B_hat, linewidth=2, label="Spline fit")
    ax0.set_ylabel("[B]")
    ax0.legend(loc="upper left", frameon=False)
    ax0.set_title(r"Step 1: Smoothing ($A+B \to 2B$)", fontsize=10, fontweight="bold")

    ax1 = axes[1]
    ax1.plot(t, d2B_hat, linewidth=1.5, label=r"Acceleration $d^2B/dt^2$")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if np.isfinite(t_star):
        ax1.axvline(t_star, color="black", linestyle=":", label=f"t* = {t_star:.2f} s")
        ax1.scatter([t_star], [0], color="black", zorder=10, s=40)
    ax1.set_ylabel(r"$d^2B/dt^2$")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title("Step 2: Identifying the landmark", fontsize=10, fontweight="bold")

    plt.tight_layout()

    out_png = outdir / "autocatalysis_landmark.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved diagnostic plot: {out_png}")


if __name__ == "__main__":
    main()
