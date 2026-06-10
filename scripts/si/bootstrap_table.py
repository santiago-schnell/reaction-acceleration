"""Regenerate the SI bootstrap table (Section 8.3).

Single canonical noise realization: fit, extract the inflection landmark,
and compute a residual-bootstrap 95 % confidence interval. The fixed-rule
result is the cautionary case (its point estimate carries the smoothing bias
of Section 7.6); the GCV result is the recommended pipeline, whose point
estimate is de-biased. The data seed and bootstrap seed are fixed so the
fixed-rule table is reproducible and identical to the runnable Section 10
listing (same seeds, same detector).

Run from the repository root:
    python scripts/si/bootstrap_table.py
"""

from __future__ import annotations

import numpy as np
from _si_common import (
    T_STAR_TRUE,
    clean_signal,
    estimate_inflection,
    estimate_inflection_gcv,
    grid,
    landmark_fn,
    smoothing_factor,
)

from reaction_acceleration import residual_bootstrap_landmark_ci

SIGMA = 0.01
N = 100
N_BOOT = 500
DATA_SEED = 42  # matches the Section 10 listing
BOOT_SEED = 1  # matches the Section 10 listing


def main() -> None:
    t = grid(N)
    b_obs = clean_signal(t) + np.random.default_rng(DATA_SEED).normal(0, SIGMA, N)
    s = smoothing_factor(N, SIGMA, factor=2.0)  # = 0.02

    point = estimate_inflection(t, b_obs, s)
    _est, lo, hi, diag = residual_bootstrap_landmark_ci(
        t,
        b_obs,
        landmark_fn=landmark_fn,
        method="spline",
        s=s,
        n_boot=N_BOOT,
        alpha=0.05,
        seed=BOOT_SEED,
        return_diagnostics=True,
    )

    print("# SI Sec. 8.3 - bootstrap CI for the autocatalysis inflection")
    print(
        f"# n={N}, sigma={SIGMA}, s={s:.3f}, B={N_BOOT}, "
        f"data_seed={DATA_SEED}, boot_seed={BOOT_SEED}"
    )
    print(f"{'Statistic':32s} Value")
    print(f"{'True inflection t*_true':32s} {T_STAR_TRUE:.2f} s")
    print(f"{'Point estimate t*_hat':32s} {point:.2f} s")
    print(f"{'Bootstrap mean':32s} {diag['bootstrap_mean']:.2f} s")
    print(f"{'Bootstrap std. dev.':32s} {diag['bootstrap_std']:.2f} s")
    print(f"{'95% CI (percentile)':32s} [{lo:.2f}, {hi:.2f}] s")
    print(
        f"{'Failed detections':32s} {diag['n_fail']} / {N_BOOT} "
        f"({100*diag['fail_fraction']:.1f}%)"
    )
    print(f"{'CI contains truth':32s} {'yes' if lo <= T_STAR_TRUE <= hi else 'no'}")

    # Recommended pipeline: GCV-selected penalty (de-biased point estimate).
    point_gcv = estimate_inflection_gcv(t, b_obs)
    _eg, lo_g, hi_g, diag_g = residual_bootstrap_landmark_ci(
        t,
        b_obs,
        landmark_fn=landmark_fn,
        method="gcv",
        n_boot=N_BOOT,
        alpha=0.05,
        seed=BOOT_SEED,
        return_diagnostics=True,
    )
    print("\n# Recommended pipeline (GCV P-spline, same data and seeds)")
    print(f"{'Point estimate t*_hat (GCV)':32s} {point_gcv:.2f} s")
    print(f"{'Bootstrap mean (GCV)':32s} {diag_g['bootstrap_mean']:.2f} s")
    print(f"{'95% CI (percentile, GCV)':32s} [{lo_g:.2f}, {hi_g:.2f}] s")
    print(f"{'CI contains truth (GCV)':32s} {'yes' if lo_g <= T_STAR_TRUE <= hi_g else 'no'}")


if __name__ == "__main__":
    main()
