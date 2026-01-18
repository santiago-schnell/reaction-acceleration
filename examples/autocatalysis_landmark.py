"""Worked example: autocatalysis inflection-time landmark.

This reproduces the Supporting-Information style workflow:

- simulate A + B -> 2B (mass-action)
- add homoscedastic Gaussian noise to B(t)
- estimate derivatives using a smoothing spline
- extract an inflection-time landmark from the acceleration signal
- compute a residual-bootstrap confidence interval

Run (from repo root):

```bash
python examples/autocatalysis_landmark.py
```

The diagnostic plot is saved to `outputs/examples/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from reaction_acceleration import (
    estimate_derivatives,
    residual_bootstrap_landmark_ci,
)


def autocatalysis_rhs(y, t, k):
    """RHS for A + B -> 2B under mass-action kinetics."""
    A, B = y
    return [-k * A * B, +k * A * B]


def landmark_zero_crossing_near_vmax(t, yhat, dy, d2y) -> float:
    """Robust autocatalytic landmark: pos->neg zero crossing nearest vmax.

    For ideal autocatalysis, the inflection point satisfies:
      1) d^2B/dt^2 crosses zero (pos -> neg)
      2) dB/dt is maximal at the same time

    In noisy data, d^2B/dt^2 may have multiple zero-crossings. This detector
    chooses the pos->neg crossing nearest the time of maximal estimated rate.
    """
    t = np.asarray(t, dtype=float)
    dy = np.asarray(dy, dtype=float)
    d2y = np.asarray(d2y, dtype=float)

    t_vmax = float(t[int(np.argmax(dy))])

    candidates: list[float] = []
    for i in range(1, len(d2y)):
        z0 = float(d2y[i - 1])
        z1 = float(d2y[i])
        if z0 > 0.0 and z1 < 0.0:
            t0 = float(t[i - 1])
            t1 = float(t[i])
            if z1 != z0:
                tc = t0 - z0 * (t1 - t0) / (z1 - z0)
                candidates.append(float(tc))

    if not candidates:
        return float("nan")

    cand = np.asarray(candidates, dtype=float)
    return float(cand[int(np.argmin(np.abs(cand - t_vmax)))])


def main() -> None:
    rng = np.random.default_rng(0)

    # ----------------------------------------------------------------------
    # 1) Synthetic data
    # ----------------------------------------------------------------------
    k = 1.5
    y0 = [0.98, 0.02]  # [A]0, [B]0
    t = np.linspace(0.0, 6.0, 180)

    sol = odeint(autocatalysis_rhs, y0, t, args=(k,))
    B_true = sol[:, 1]

    sigma = 0.008
    B_obs = B_true + rng.normal(0.0, sigma, size=B_true.shape)

    # ----------------------------------------------------------------------
    # 2) Derivative estimation (smoothing spline)
    # ----------------------------------------------------------------------
    # Heuristic: s ~= N * sigma^2 ; for second-derivative stability we use
    # a slightly larger factor.
    s = 2.0 * len(t) * (sigma**2)

    B_hat, dB_hat, d2B_hat, _model = estimate_derivatives(t, B_obs, method="spline", s=s)

    t_star = landmark_zero_crossing_near_vmax(t, B_hat, dB_hat, d2B_hat)

    # ----------------------------------------------------------------------
    # 3) Bootstrap CI
    # ----------------------------------------------------------------------
    est, lo, hi = residual_bootstrap_landmark_ci(
        t,
        B_obs,
        landmark_fn=landmark_zero_crossing_near_vmax,
        method="spline",
        s=s,
        n_boot=400,
        alpha=0.05,
        seed=1,
    )

    # ----------------------------------------------------------------------
    # 4) Report
    # ----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Autocatalysis landmark: acceleration zero-crossing near v_max")
    print("=" * 68)
    print(f"  Smoothing factor (s)     : {s:.4e}")
    print(f"  Base-fit estimate (t*)   : {t_star:.4f} s")
    print(f"  Bootstrap base estimate  : {est:.4f} s")
    print(f"  95% CI (percentile)      : [{lo:.4f}, {hi:.4f}] s")
    print("=" * 68 + "\n")

    # ----------------------------------------------------------------------
    # 5) Diagnostic plot
    # ----------------------------------------------------------------------
    outdir = Path(__file__).resolve().parents[1] / "outputs" / "examples"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

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
    ax1.set_title("Step 2: Identifying the fingerprint", fontsize=10, fontweight="bold")

    plt.tight_layout()

    out_png = outdir / "autocatalysis_landmark.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved diagnostic plot: {out_png}")


if __name__ == "__main__":
    main()
