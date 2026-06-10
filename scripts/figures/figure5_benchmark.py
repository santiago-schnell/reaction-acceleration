"""Figure 5: operating characteristics of acceleration-landmark recovery.

The figure is rendered from the canonical benchmark CSV stored in
``data/benchmarks/`` and written by ``scripts/si/verification_table.py``.
If the CSV is absent, this script computes it first by calling the same
benchmark routine. That design keeps the SI table and the figure tied to a
single numerical source of truth.

Two pipelines are overlaid in every panel:
  * the recommended GCV P-spline (filled markers, solid lines), and
  * the cautionary fixed rule s = 2 n sigma**2 (open markers, dashed lines).

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
    "gcv": "#2166AC",      # recommended
    "fixed": "#B2182B",    # cautionary
    "target": "#666666",
}
STYLE = {
    "gcv": dict(marker="o", linestyle="-", markerfacecolor=COLORS["gcv"]),
    "fixed": dict(marker="s", linestyle="--", markerfacecolor="white"),
}
LABEL = {"gcv": "GCV (recommended)", "fixed": r"fixed $s=2n\sigma^2$"}


def ensure_metrics_csv() -> Path:
    """Return the benchmark CSV, generating it from the SI routine if needed."""

    if METRICS_CSV.exists():
        return METRICS_CSV

    si_dir = ROOT / "scripts" / "si"
    sys.path.insert(0, str(si_dir))
    from verification_table import compute_metrics, write_metrics_csv

    return write_metrics_csv(compute_metrics(), METRICS_CSV)


def load_metrics(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load benchmark metrics from CSV, split by pipeline."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    keys = [
        "noise_percent", "bias_s", "rmse_s", "half_width_s",
        "coverage_percent", "coverage_low_percent", "coverage_high_percent",
        "detection_percent", "detection_low_percent", "detection_high_percent",
    ]
    out: dict[str, dict[str, np.ndarray]] = {}
    for pipeline in ("gcv", "fixed"):
        sub = [r for r in rows if r.get("pipeline", "fixed") == pipeline]
        if not sub:
            continue
        out[pipeline] = {
            k: np.asarray([float(r[k]) for r in sub], dtype=float) for k in keys
        }
    return out


def style_axis(axis, ylabel: str, label: str, *, add_zero: bool = False) -> None:
    axis.set_xlabel(r"Noise, $\sigma$/range (%)")
    axis.set_ylabel(ylabel)
    axis.text(-0.25, 1.05, label, transform=axis.transAxes, fontweight="bold", fontsize=10)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", direction="in", top=True, right=True)
    if add_zero:
        axis.axhline(0, color=COLORS["target"], linewidth=0.8, linestyle=":")


def _plot_series(axis, data, ykey, *, yerr_keys=None):
    for pipeline in ("gcv", "fixed"):
        if pipeline not in data:
            continue
        d = data[pipeline]
        kw = dict(STYLE[pipeline], color=COLORS[pipeline], label=LABEL[pipeline],
                  capsize=2, markersize=5)
        if yerr_keys is not None:
            lo, hi = yerr_keys
            yerr = np.vstack((d[ykey] - d[lo], d[hi] - d[ykey]))
            axis.errorbar(d["noise_percent"], d[ykey], yerr=yerr, **kw)
        else:
            axis.errorbar(d["noise_percent"], d[ykey], **kw)


def main() -> None:
    data = load_metrics(ensure_metrics_csv())

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    fig.subplots_adjust(wspace=0.35, hspace=0.45, left=0.10, right=0.98, top=0.95, bottom=0.12)

    _plot_series(axes[0, 0], data, "bias_s")
    style_axis(axes[0, 0], "Bias in $t^*$ (s)", "(a)", add_zero=True)

    _plot_series(axes[0, 1], data, "rmse_s")
    style_axis(axes[0, 1], "RMSE in $t^*$ (s)", "(b)")

    _plot_series(
        axes[1, 0], data, "coverage_percent",
        yerr_keys=("coverage_low_percent", "coverage_high_percent"),
    )
    style_axis(axes[1, 0], "95% CI coverage (%)", "(c)")
    axes[1, 0].axhline(95, color=COLORS["target"], linewidth=0.8, linestyle="--")
    axes[1, 0].text(axes[1, 0].get_xlim()[1], 93, "nominal 95%", ha="right",
                    va="top", fontsize=8, color=COLORS["target"])
    axes[1, 0].set_ylim(45, 107)

    _plot_series(
        axes[1, 1], data, "detection_percent",
        yerr_keys=("detection_low_percent", "detection_high_percent"),
    )
    style_axis(axes[1, 1], "Detection (%)", "(d)")
    axes[1, 1].set_ylim(85, 103)

    outdir = ROOT / "outputs" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "Figure5_benchmark.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(outdir / "Figure5_benchmark.png", format="png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
