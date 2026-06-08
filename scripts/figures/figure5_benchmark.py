"""Figure 5: operating characteristics of acceleration-landmark recovery.

The figure is rendered from the canonical benchmark CSV stored in
``data/benchmarks/`` and written by ``scripts/si/verification_table.py``.
If the CSV is absent, this script computes it first by calling the same
benchmark routine. That design keeps the SI table and the figure tied to a
single numerical source of truth.

Outputs
-------
- ``outputs/figures/Figure5_benchmark.pdf``
- ``outputs/figures/Figure5_benchmark.png``
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parents[2]
METRICS_CSV = ROOT / "data" / "benchmarks" / "autocatalysis_operating_characteristics.csv"

apply_style()

COLORS = {
    "metric": "#2166AC",
    "coverage": "#B2182B",
    "target": "#666666",
}


def ensure_metrics_csv() -> Path:
    """Return the benchmark CSV, generating it from the SI routine if needed."""

    if METRICS_CSV.exists():
        return METRICS_CSV

    si_dir = ROOT / "scripts" / "si"
    sys.path.insert(0, str(si_dir))
    from verification_table import compute_metrics, write_metrics_csv

    return write_metrics_csv(compute_metrics(), METRICS_CSV)


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    """Load benchmark metrics from CSV as NumPy arrays."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    keys = [
        "noise_percent",
        "bias_s",
        "rmse_s",
        "coverage_percent",
        "coverage_low_percent",
        "coverage_high_percent",
        "detection_percent",
        "detection_low_percent",
        "detection_high_percent",
    ]
    return {key: np.asarray([float(row[key]) for row in rows], dtype=float) for key in keys}


def style_axis(axis, ylabel: str, label: str, *, add_zero: bool = False) -> None:
    """Apply the shared formatting for each panel."""

    axis.set_xlabel(r"Noise, $\sigma$/range (%)")
    axis.set_ylabel(ylabel)
    axis.text(-0.25, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", direction="in", top=True, right=True)
    if add_zero:
        axis.axhline(0, color=COLORS["target"], linewidth=0.8, linestyle=":")


def main() -> None:
    metrics = load_metrics(ensure_metrics_csv())
    noise = metrics["noise_percent"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    fig.subplots_adjust(
        wspace=0.35,
        hspace=0.45,
        left=0.10,
        right=0.98,
        top=0.94,
        bottom=0.13,
    )

    axes[0, 0].plot(noise, metrics["bias_s"], marker="o", color=COLORS["metric"])
    style_axis(axes[0, 0], "Bias in $t^*$ (s)", "(a)", add_zero=True)

    axes[0, 1].plot(noise, metrics["rmse_s"], marker="o", color=COLORS["metric"])
    style_axis(axes[0, 1], "RMSE in $t^*$ (s)", "(b)")

    coverage_yerr = np.vstack(
        (
            metrics["coverage_percent"] - metrics["coverage_low_percent"],
            metrics["coverage_high_percent"] - metrics["coverage_percent"],
        )
    )
    axes[1, 0].errorbar(
        noise,
        metrics["coverage_percent"],
        yerr=coverage_yerr,
        marker="o",
        capsize=2,
        color=COLORS["metric"],
    )
    style_axis(axes[1, 0], "95% CI coverage (%)", "(c)")
    axes[1, 0].axhline(95, color=COLORS["coverage"], linewidth=0.8, linestyle="--")
    axes[1, 0].text(noise[-1], 96.5, "nominal 95%", ha="right", va="bottom", fontsize=8)
    axes[1, 0].set_ylim(45, 105)

    detection_yerr = np.vstack(
        (
            metrics["detection_percent"] - metrics["detection_low_percent"],
            metrics["detection_high_percent"] - metrics["detection_percent"],
        )
    )
    axes[1, 1].errorbar(
        noise,
        metrics["detection_percent"],
        yerr=detection_yerr,
        marker="o",
        capsize=2,
        color=COLORS["metric"],
    )
    style_axis(axes[1, 1], "Detection (%)", "(d)")
    axes[1, 1].set_ylim(85, 103)

    outdir = ROOT / "outputs" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "Figure5_benchmark.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(outdir / "Figure5_benchmark.png", format="png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
