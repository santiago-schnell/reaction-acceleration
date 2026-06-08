"""Shared Matplotlib style for manuscript figures.

All figure scripts in ``scripts/figures/`` import ``apply_style`` from
this module rather than re-declaring their own ``plt.rcParams.update``
block. This keeps typography, tick direction, and line width consistent
across every panel of every figure in the paper.

Usage
-----
    from _style import apply_style, COLORS
    apply_style()
    # ... build figure ...
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Shared palette used across figures. Adding colours here rather than
# per-script keeps brand consistency automatic.
COLORS = {
    "auto": "#B2182B",  # autocatalytic (deep red)
    "first": "#2166AC",  # first-order  (blue)
    "accel_pos": "#4DAF4A",  # positive acceleration shading (green)
    "accel_neg": "#E66101",  # negative acceleration shading (orange)
    "gray": "#666666",  # guide lines
    "ref_gray": "#888888",  # reference lines
}


def apply_style() -> None:
    """Apply the shared manuscript rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "lines.linewidth": 1.5,
        }
    )
