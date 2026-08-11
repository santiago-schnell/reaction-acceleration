"""graphical_abstract.py

Graphical abstract for the ChemSystemsChem research article:
"Reaction Acceleration: Reviving the Second Derivative in Chemical Kinetics".[cite: 1]

The figure emphasizes that the sign pattern of the acceleration 
(d^2[B]/dt^2 for a monitored progress variable B) distinguishes:[cite: 1]

- single-step relaxation (no sign change; typically negative for product 
  formation when the rate decreases),[cite: 1]
- intermediacy in consecutive reactions (negative-to-positive),[cite: 1]
- feedback in autocatalysis (positive-to-negative).[cite: 1]

Units are omitted for compactness; the goal is conceptual rather than 
quantitative.[cite: 1]

Scientific clarification
------------------------
The monitored quantity is explicitly identified as `x(t) = [B](t)` in all
three panels. Thus `B` is the product in the first-order and autocatalytic
panels and the intermediate in the consecutive-reaction panel. This removes
the possible implication that the acceleration of every progress variable in
a first-order relaxation is necessarily negative.

The figure is deterministic: no random numbers are used.

Dependencies: NumPy, Matplotlib.[cite: 1]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
# 1. Style & Configuration[cite: 1]
# ------------------------------------------------------------------------------

try:
    # Repository execution: scripts/figures/_style.py is available.
    from _style import apply_style
except ImportError:
    # Stand-alone fallback so that this single file remains runnable.
    def apply_style() -> None:
        """Apply the manuscript's plotting style."""
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

apply_style()

COLORS = {
    "line": "#444444",      # dark gray[cite: 1]
    "pos": "#4DAF4A",       # positive acceleration: green[cite: 1]
    "neg": "#E66101",       # negative acceleration: orange[cite: 1]
    "zero_dot": "#D00000",  # zero crossing: red[cite: 1]
}


# ------------------------------------------------------------------------------
# 2. Analytical Acceleration Functions[cite: 1]
# ------------------------------------------------------------------------------

def acc_first_order(t: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Product acceleration for A -> B with x(t) = [B](t)."""
    return -(k**2) * np.exp(-k * t)


def acc_consecutive(
    t: np.ndarray,
    k1: float = 1.0,
    k2: float = 0.5,
) -> np.ndarray:
    """Intermediate acceleration for A -> B -> C with x(t) = [B](t)."""
    if np.isclose(k1, k2):
        # Equal-rate limit: [B] = A0*k*t*exp(-k*t), with A0 normalized to 1.
        k = 0.5 * (k1 + k2)
        return (k**2) * np.exp(-k * t) * (k * t - 2.0)

    term1 = (k1**2) * np.exp(-k1 * t)
    term2 = (k2**2) * np.exp(-k2 * t)
    prefactor = k1 / (k2 - k1)
    return prefactor * (term1 - term2)


def acc_autocatalytic(
    t: np.ndarray,
    k: float = 1.5,
    a_tot: float = 1.0,
    b0: float = 0.02,
) -> np.ndarray:
    """Product acceleration for A + B -> 2B with x(t) = [B](t)."""
    if not (0.0 < b0 < a_tot):
        raise ValueError("b0 must satisfy 0 < b0 < a_tot.")

    denominator = 1.0 + ((a_tot / b0) - 1.0) * np.exp(-k * a_tot * t)
    b = a_tot / denominator
    return (k**2) * b * (a_tot - b) * (a_tot - 2.0 * b)


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a curve to unit maximum absolute magnitude."""
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Acceleration curve has no finite, nonzero scale.")
    return values / scale


def _zero_crossing_times(t: np.ndarray, y: np.ndarray) -> list[float]:
    """Return linearly interpolated zero-crossing times."""
    indices = np.flatnonzero(np.diff(np.signbit(y)))
    crossings: list[float] = []
    for index in indices:
        t0, t1 = float(t[index]), float(t[index + 1])
        y0, y1 = float(y[index]), float(y[index + 1])
        crossings.append(t0 - y0 * (t1 - t0) / (y1 - y0))
    return crossings


# ------------------------------------------------------------------------------
# 3. Plotting with Custom Positioning[cite: 1]
# ------------------------------------------------------------------------------

def plot_landmark(
    ax: plt.Axes,
    t: np.ndarray,
    acceleration: np.ndarray,
    title: str,
    reaction: str,
    landmark_text: str,
    text_position: tuple[float, float],
) -> None:
    """
    Draws a single panel with the acceleration curve, shading, and landmark.[cite: 1]
    text_position: tuple (x, y) in axes coordinates for the landmark label.[cite: 1]
    """
    
    # Draw curve[cite: 1]
    ax.plot(t, acceleration, color=COLORS["line"], zorder=5)
    
    # Zero line[cite: 1]
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # Shading[cite: 1]
    ax.fill_between(
        t,
        0.0,
        acceleration,
        where=acceleration > 0.0,
        color=COLORS["pos"],
        alpha=0.25,
        interpolate=True,
    )
    ax.fill_between(
        t,
        0.0,
        acceleration,
        where=acceleration < 0.0,
        color=COLORS["neg"],
        alpha=0.25,
        interpolate=True,
    )

    # Landmark Dot (Zero Crossing)[cite: 1]
    for crossing in _zero_crossing_times(t, acceleration):
        ax.scatter(
            [crossing],
            [0.0],
            color=COLORS["zero_dot"],
            s=50,
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
        )

    # Annotations[cite: 1]
    ax.set_title(f"{title}\n{reaction}", fontsize=11, fontweight="bold", pad=10)

    # Text Label with Custom Position[cite: 1]
    horizontal_alignment = "right" if text_position[0] > 0.5 else "left"
    vertical_alignment = "top" if text_position[1] > 0.5 else "bottom"
    
    ax.text(
        text_position[0],
        text_position[1],
        landmark_text,
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontsize=9,
        fontstyle="italic",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
    )

    # Clean axes[cite: 1]
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)


def _output_directory(script_path: Path) -> Path:
    """Choose the repository output directory when the expected tree exists."""
    parents = script_path.resolve().parents
    if len(parents) >= 3 and script_path.parent.name == "figures" and script_path.parent.parent.name == "scripts":
        return parents[2] / "outputs" / "figures"
    return script_path.resolve().parent / "outputs" / "figures"


def main() -> tuple[Path, Path]:
    """Generate vector PDF and 300-dpi PNG outputs."""
    t = np.linspace(0.0, 6.0, 500)

    # Normalized Data[cite: 1]
    first_order = _normalize(acc_first_order(t))
    consecutive = _normalize(acc_consecutive(t))
    autocatalytic = _normalize(acc_autocatalytic(t))

    # Sanity checks for the intended taxonomy.
    assert np.all(first_order < 0.0), "First-order product acceleration must remain negative."
    consecutive_crossings = _zero_crossing_times(t, consecutive)
    autocatalytic_crossings = _zero_crossing_times(t, autocatalytic)
    assert len(consecutive_crossings) == 1, "Consecutive intermediate should have one zero crossing."
    assert len(autocatalytic_crossings) == 1, "Autocatalytic product should have one zero crossing."
    assert consecutive[0] < 0.0 < consecutive[-1], "Expected a negative-to-positive transition."
    assert autocatalytic[0] > 0.0 > autocatalytic[-1], "Expected a positive-to-negative transition."

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25), constrained_layout=True)

    # Panel 1: First Order[cite: 1]
    # Curve is always negative (bottom). Place text at Top-Right.[cite: 1]
    plot_landmark(
        axes[0],
        t,
        first_order,
        title="Relaxation",
        reaction=r"($A \rightarrow B$)",
        landmark_text="Always Negative",
        text_position=(0.95, 0.90),
    ) 

    # Panel 2: Consecutive[cite: 1]
    # Curve ends positive (top). Place text at Bottom-Right.[cite: 1]
    plot_landmark(
        axes[1],
        t,
        consecutive,
        title="Intermediate",
        reaction=r"($A \rightarrow B \rightarrow C$)",
        landmark_text=r"Sign Change: $(-)\rightarrow(+)$",
        text_position=(0.95, 0.05),
    )

    # Panel 3: Autocatalysis[cite: 1]
    # Curve ends negative (bottom). Place text at Top-Right.[cite: 1]
    plot_landmark(
        axes[2],
        t,
        autocatalytic,
        title="Autocatalysis",
        reaction=r"($A + B \rightarrow 2B$)",
        landmark_text=r"Sign Change: $(+)\rightarrow(-)$",
        text_position=(0.95, 0.90),
    )

    fig.supylabel(
        r"Progress-variable acceleration ($\ddot{x}$)",
        fontsize=12,
        x=-0.045,
    )
    fig.suptitle(
        r"Monitored quantity in every panel: $x(t)=[B](t)$",
        fontsize=10.5,
        fontweight="semibold",
    )

    output_dir = _output_directory(Path(__file__))
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "Graphical_Abstract.pdf"
    png_path = output_dir / "Graphical_Abstract.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    main()
