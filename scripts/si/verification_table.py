"""Regenerate the SI verification table (Section 8.3).

Coverage study for the autocatalysis inflection landmark across relative
noise levels. For each noise level, draw a fixed number of independent
realisations, estimate the landmark with the fixed smoothing rule
s = 2 n sigma**2, and report:

  * bias      = mean(t*_hat) - t*_true
  * RMSE      = sqrt(mean((t*_hat - t*_true)**2))
  * coverage  = fraction of 95 % bootstrap CIs that contain t*_true
  * detection = fraction of realisations in which a landmark was found

The script also writes the canonical benchmark source data to
``data/benchmarks/autocatalysis_operating_characteristics.csv``. Figure 5
reads that CSV, so the table and the figure have a single numerical source.

Run from the repository root:
    python scripts/si/verification_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from _si_common import (
    T_STAR_TRUE,
    clean_signal,
    estimate_inflection,
    grid,
    landmark_fn,
    smoothing_factor,
)

from reaction_acceleration import residual_bootstrap_landmark_ci

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_CSV = ROOT / "data" / "benchmarks" / "autocatalysis_operating_characteristics.csv"

N = 100
N_REALISATIONS = 50
N_BOOT = 300
MASTER_SEED = 7
BOOT_SEED = 1
SIGNAL_RANGE = 1.0
REL_NOISE = [0.005, 0.010, 0.020, 0.050]

CSV_FIELDS = [
    "rel_noise",
    "noise_percent",
    "n",
    "n_realisations",
    "n_boot",
    "n_detected",
    "n_covered",
    "smoothing_factor",
    "bias_s",
    "rmse_s",
    "coverage_percent",
    "coverage_low_percent",
    "coverage_high_percent",
    "detection_percent",
    "detection_low_percent",
    "detection_high_percent",
]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        return float("nan"), float("nan")

    phat = successes / total
    denom = 1.0 + z**2 / total
    centre = (phat + z**2 / (2.0 * total)) / denom
    half_width = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * total)) / total) / denom
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def compute_metrics() -> list[dict[str, float | int]]:
    """Compute the canonical operating-characteristic grid."""

    t = grid(N)
    b_clean = clean_signal(t)
    rng = np.random.default_rng(MASTER_SEED)
    rows: list[dict[str, float | int]] = []

    for rel in REL_NOISE:
        sigma = rel * SIGNAL_RANGE
        s = smoothing_factor(N, sigma, factor=2.0)
        ests: list[float] = []
        covered = 0
        detected = 0

        for _ in range(N_REALISATIONS):
            y = b_clean + rng.normal(0.0, sigma, N)
            point = estimate_inflection(t, y, s)
            if not np.isfinite(point):
                continue
            detected += 1
            ests.append(float(point))
            _, lo, hi = residual_bootstrap_landmark_ci(
                t,
                y,
                landmark_fn=landmark_fn,
                method="spline",
                s=s,
                n_boot=N_BOOT,
                alpha=0.05,
                seed=BOOT_SEED,
            )
            if np.isfinite(lo) and lo <= T_STAR_TRUE <= hi:
                covered += 1

        ests_arr = np.asarray(ests, dtype=float)
        bias = float(np.mean(ests_arr) - T_STAR_TRUE) if ests_arr.size else float("nan")
        rmse = (
            float(np.sqrt(np.mean((ests_arr - T_STAR_TRUE) ** 2)))
            if ests_arr.size
            else float("nan")
        )
        cov = 100.0 * covered / detected if detected else float("nan")
        det = 100.0 * detected / N_REALISATIONS
        cov_low, cov_high = wilson_interval(covered, detected)
        det_low, det_high = wilson_interval(detected, N_REALISATIONS)

        rows.append(
            {
                "rel_noise": float(rel),
                "noise_percent": float(100.0 * rel),
                "n": N,
                "n_realisations": N_REALISATIONS,
                "n_boot": N_BOOT,
                "n_detected": detected,
                "n_covered": covered,
                "smoothing_factor": float(s),
                "bias_s": bias,
                "rmse_s": rmse,
                "coverage_percent": float(cov),
                "coverage_low_percent": float(100.0 * cov_low),
                "coverage_high_percent": float(100.0 * cov_high),
                "detection_percent": float(det),
                "detection_low_percent": float(100.0 * det_low),
                "detection_high_percent": float(100.0 * det_high),
            }
        )

    return rows


def write_metrics_csv(rows: list[dict[str, float | int]], path: Path = OUTPUT_CSV) -> Path:
    """Write operating-characteristic metrics to CSV and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(rows: list[dict[str, float | int]]) -> None:
    """Print the SI table in human-readable form."""

    print("# SI Sec. 8.3 - verification grid")
    print(
        f"# n={N}, {N_REALISATIONS} realisations/level, B={N_BOOT}, "
        f"s=2*n*sigma^2, true t* = {T_STAR_TRUE:.3f} s"
    )
    print(
        f"{'sigma/range':>11} {'Bias (s)':>10} {'RMSE (s)':>9} "
        f"{'CI coverage':>12} {'Detection':>10}"
    )

    for row in rows:
        print(
            f"{row['noise_percent']:>10.1f}% "
            f"{row['bias_s']:>+10.3f} "
            f"{row['rmse_s']:>9.3f} "
            f"{row['coverage_percent']:>11.0f}% "
            f"{row['detection_percent']:>9.0f}%"
        )


def main() -> None:
    rows = compute_metrics()
    csv_path = write_metrics_csv(rows)
    print_table(rows)
    print(f"# wrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
