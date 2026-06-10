"""Regenerate the SI smoothing-parameter selection table (Section 6.6).

For the canonical autocatalytic system with sigma = 0.01 M and n = 100
points, scan the spline smoothing factor s and report the mean and standard
deviation of the estimated inflection time over a fixed set of noise
realizations. Demonstrates the bias-variance trade-off: too little smoothing
is high-variance, too much is biased.

Run from the repository root:
    python scripts/si/smoothing_table.py
"""

from __future__ import annotations

import numpy as np
from _si_common import T_STAR_TRUE, clean_signal, estimate_inflection, grid

SIGMA = 0.01
N = 100
N_REALISATIONS = 30
MASTER_SEED = 20240517
S_OVER_NSIG2 = [0.5, 1.0, 2.0, 5.0]


def main() -> None:
    t = grid(N)
    b_clean = clean_signal(t)
    rng = np.random.default_rng(MASTER_SEED)

    # Pre-draw a fixed noise matrix so every column sees the same realisations.
    noise = rng.standard_normal((N_REALISATIONS, N)) * SIGMA

    print("# SI Sec. 7.6 - smoothing-parameter selection")
    print(
        f"# canonical autocatalysis, n={N}, sigma={SIGMA}, "
        f"{N_REALISATIONS} realisations, true t* = {T_STAR_TRUE:.3f} s"
    )
    print(f"{'s':>7} {'s/(n*sig^2)':>12} {'t* mean':>9} {'t* std':>8}  comment")
    base = N * SIGMA**2
    for ratio in S_OVER_NSIG2:
        s = ratio * base
        ests = []
        for r in range(N_REALISATIONS):
            est = estimate_inflection(t, b_clean + noise[r], s)
            if np.isfinite(est):
                ests.append(est)
        ests = np.asarray(ests)
        mean, std = float(np.mean(ests)), float(np.std(ests, ddof=1))
        comment = {
            0.5: "under-smoothed; noisy d2[B]",
            1.0: "reasonable",
            2.0: "good balance",
            5.0: "over-smoothed; biased",
        }[ratio]
        print(f"{s:>7.3f} {ratio:>12.1f} {mean:>9.3f} {std:>8.3f}  {comment}")


if __name__ == "__main__":
    main()
