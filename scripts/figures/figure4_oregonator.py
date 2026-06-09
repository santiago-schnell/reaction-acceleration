r"""
Figure 4: Oregonator Acceleration Landmarks
-------------------------------------------------------
This script generates a three-panel figure illustrating acceleration landmarks
in the Oregonator representation of the Belousov-Zhabotinsky reaction.

The rate `dZ/dt` and acceleration `d2Z/dt2` are computed directly from the
mathematical identity of the model equations, precluding numerical differentiation
artifacts.

Designed for absolute reproducibility, this script meticulously details
the kinetic parameters, numerical solutions to the differential equations,
and the explicit rendering logic utilized to generate the publication-quality
schematics.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Biomedical Data Sciences, Geisel School of Medicine.

Outputs:
    - Figure4_oregonator.pdf
    - Figure4_oregonator.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import odeint

# ------------------------------------------------------------------------------
# 1. Configuration & Style Integration
# ------------------------------------------------------------------------------
apply_style()

COLORS = {
    "Z": "#2166AC",  # Blue (Oxidized species / Cerium)
    "Rate": "#B2182B",  # Red (Kinetic Rate)
    "accel_pos": "#4DAF4A",  # Green (Positive Acceleration phase)
    "accel_neg": "#E66101",  # Orange (Negative Acceleration phase)
    "gray": "#666666",  # Guide lines
    "dot": "#D00000",  # Red (Zero-crossing markers)
}

# ------------------------------------------------------------------------------
# 2. Mathematical Definition & Data Generation
# ------------------------------------------------------------------------------
# Classical dimensionless parameters for the Oregonator model
EPSILON = 0.04
EPSILON_PRIME = 0.0004
F_STOICH = 1.0
Q_PARAM = 0.0008


def oregonator_rates(state: np.ndarray | list[float]) -> np.ndarray:
    """Compute the Oregonator rate vector for state = (X, Y, Z)."""
    x, y, z = state
    # Prevent variables from approaching zero to maintain numerical stability
    x, y, z = max(float(x), 1e-12), max(float(y), 1e-12), max(float(z), 1e-12)

    # Differential equations describing the non-linear kinetic topology
    dxdt = (Q_PARAM * y - x * y + x * (1.0 - x)) / EPSILON
    dydt = (-Q_PARAM * y - x * y + F_STOICH * z) / EPSILON_PRIME
    dzdt = x - z

    return np.array([dxdt, dydt, dzdt], dtype=float)


def oregonator_rhs(state, _time):
    return oregonator_rates(state)


def zero_crossing_indices(values, min_distance=20):
    """Isolate de-duplicated sign-change indices to identify kinetic shifts."""
    raw = [i for i in range(1, len(values)) if values[i - 1] * values[i] < 0]
    clean = []
    for idx in raw:
        if not clean or idx - clean[-1] > min_distance:
            clean.append(idx)
    return clean


# High-resolution temporal integration to resolve rapid relaxation oscillations
time = np.linspace(0.0, 40.0, 8000)
solution = odeint(oregonator_rhs, [0.1, 0.1, 0.1], time)

# Discard initial transient state to focus entirely on the stabilized limit cycle
skip = 2000
time_plot = time[skip:] - time[skip]
solution_plot = solution[skip:, :]
X_plot = solution_plot[:, 0]
Z_plot = solution_plot[:, 2]

# Derive the precise rates and acceleration from the analytical identities
rates_plot = np.array([oregonator_rates(row) for row in solution_plot])
dZdt = rates_plot[:, 2]
d2Zdt2 = rates_plot[:, 0] - rates_plot[:, 2]

# Truncate the time series to approximately three relaxation-oscillation cycles
end_idx = 3600
time_win = time_plot[:end_idx]
X_win = X_plot[:end_idx]
Z_win = Z_plot[:end_idx]
rate_win = dZdt[:end_idx]
accel_win = d2Zdt2[:end_idx]

# Map the exact zero crossings representing the bounds of the autocatalytic phase
inflection_idx = [idx for idx in zero_crossing_indices(d2Zdt2) if idx < end_idx]
time_inf, X_inf, Z_inf, rate_inf = (
    time_plot[inflection_idx],
    X_plot[inflection_idx],
    Z_plot[inflection_idx],
    dZdt[inflection_idx],
)

# ------------------------------------------------------------------------------
# 3. Graphical Rendering
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(8.5, 2.3))
width, height, bottom = 0.23, 0.72, 0.20

# Manual axis geometry assignment for pristine multi-panel alignment
ax1 = fig.add_axes([0.08, bottom, width, height])
ax2 = fig.add_axes([0.39, bottom, width, height])
ax3 = fig.add_axes([0.75, bottom, width, height])

# --- Panel (a): Phase-space Projection ---
ax1.plot(X_win, Z_win, color=COLORS["gray"], linewidth=1.2)
ax1.scatter(X_inf, Z_inf, color=COLORS["dot"], s=25, zorder=10)

# Render directional flow vectors along the limit cycle
last_arrow_pos = np.array([-999.0, -999.0])
for i in range(0, len(X_win) - 5, 50):
    current = np.array([X_win[i], Z_win[i]])
    if np.linalg.norm(current - last_arrow_pos) > 0.15:
        ax1.annotate(
            "",
            xy=(X_win[i + 5], Z_win[i + 5]),
            xytext=(X_win[i], Z_win[i]),
            arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1},
        )
        last_arrow_pos = current

ax1.set_xlabel(r"[HBrO$_2$] ($X$)")
ax1.set_ylabel(r"[Ce$^{4+}$] ($Z$)")
ax1.text(-0.25, 1.05, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=10)

# --- Panel (b): Concentration and Velocity ---
ax2.plot(time_win, Z_win, color=COLORS["Z"], linewidth=1.5)
ax2.set_xlabel("Time (dimensionless)")
ax2.set_ylabel(r"[Ce$^{4+}$] ($Z$)", color=COLORS["Z"])
ax2.tick_params(axis="y", labelcolor=COLORS["Z"])
ax2.set_ylim(0, np.max(Z_win) * 1.1)

# Generate a secondary axis to superimpose the kinetic rate
ax2b = ax2.twinx()
ax2b.plot(time_win, rate_win, color=COLORS["Rate"], linewidth=1.0, alpha=0.8, linestyle="--")
ax2b.set_ylabel(r"Model rate $dZ/dt$", color=COLORS["Rate"])
ax2b.tick_params(axis="y", labelcolor=COLORS["Rate"])
ax2b.axhline(0, color=COLORS["gray"], linewidth=0.5, linestyle=":")
ax2b.scatter(time_inf, rate_inf, color=COLORS["dot"], s=20, zorder=10)
ax2.text(-0.25, 1.05, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=10)

# --- Panel (c): Second Derivative (Acceleration) ---
ax3.plot(time_win, accel_win, color=COLORS["gray"], linewidth=1.0)
ax3.axhline(0, color=COLORS["gray"], linewidth=0.8)

# Shading dictates periods of positive acceleration vs. negative acceleration
ax3.fill_between(time_win, 0, accel_win, where=accel_win > 0, color=COLORS["accel_pos"], alpha=0.3)
ax3.fill_between(time_win, 0, accel_win, where=accel_win < 0, color=COLORS["accel_neg"], alpha=0.3)
ax3.scatter(time_inf, np.zeros_like(time_inf), color=COLORS["dot"], s=25, zorder=10)

ax3.set_xlabel("Time (dimensionless)")
ax3.set_ylabel(r"Model accel. $d^2Z/dt^2$")
ax3.text(-0.25, 1.05, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=10)

# --- Exterior Label Positioning & Formatting ---
for axis in [ax1, ax2, ax3]:
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", direction="in", top=True, right=True)

# ------------------------------------------------------------------------------
# 4. File Output Synthesis
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "Figure4_oregonator.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(outdir / "Figure4_oregonator.png", format="png", bbox_inches="tight", dpi=300)
