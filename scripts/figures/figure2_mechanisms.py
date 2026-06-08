r"""
Figure 2: Kinetic Landmarks of Elementary Mechanisms
-------------------------------------------------------
This script generates a comparative matrix of First-order, Consecutive, and
Autocatalytic kinetics. It highlights the characteristic acceleration landmark
of each mechanism.

Designed for absolute reproducibility, this script meticulously details
the kinetic parameters, analytical and numerical solutions to the differential
equations, and the explicit rendering logic utilized to generate the
publication-quality schematics.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Department of Biomedical Data Sciences, Geisel School
    of Medicine at Dartmouth, Hanover, New Hampshire, USA

Outputs:
    - Figure2_mechanisms.pdf
    - Figure2_mechanisms.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import odeint

# ------------------------------------------------------------------------------
# 1. Configuration & Style Integration
# ------------------------------------------------------------------------------
# Apply global typographic and geometric parameters for publication standards
apply_style()

# Harmonized color palette for structural consistency across kinetic classes
colors = {
    "A": "#2166AC",  # Blue (Reactant consumption)
    "B": "#B2182B",  # Red (Primary monitored Product/Intermediate)
    "C": "#999999",  # Gray (Terminal Product)
    "accel_pos": "#4DAF4A",  # Green (Positive acceleration / speeding up)
    "accel_neg": "#E66101",  # Orange (Negative acceleration / slowing down)
    "gray": "#666666",  # Gray (Axes and guidelines)
}

# ------------------------------------------------------------------------------
# 2. Mathematical Definition & Data Generation
# ------------------------------------------------------------------------------
# Define the temporal integration domain
t = np.linspace(0, 5, 1000)
dt = t[1] - t[0]

# --- Column 1: First-order Kinetics (A -> B) ---
k1 = 1.0
A_first = np.exp(-k1 * t)
B_first = 1 - np.exp(-k1 * t)
# Velocity is strictly positive; acceleration is strictly negative
v_first = k1 * A_first
accel_first = -(k1**2) * A_first

# --- Column 2: Consecutive Kinetics (A -> B -> C) ---
k1_cons = 1.0
k2_cons = 0.6


def consecutive(y, t, k1, k2):
    """System of ODEs for consecutive first-order reactions."""
    A, B, _C = y
    return [-k1 * A, k1 * A - k2 * B, k2 * B]


sol_cons = odeint(consecutive, [1.0, 0.0, 0.0], t, args=(k1_cons, k2_cons))
A_cons, B_cons, C_cons = sol_cons[:, 0], sol_cons[:, 1], sol_cons[:, 2]

# Rate of intermediate formation
v_cons = k1_cons * A_cons - k2_cons * B_cons

# Acceleration of intermediate B using the exact Jacobian identity:
# d^2B/dt^2 = -k1*(k1+k2)*A + k2^2*B
accel_cons = -k1_cons * (k1_cons + k2_cons) * A_cons + (k2_cons**2) * B_cons

# Isolate the zero-crossing landmark (inflection point) for the consecutive mechanism
idx_inf_cons = np.where(np.diff(np.sign(accel_cons)))[0]
t_inf_cons = t[idx_inf_cons[0]] if len(idx_inf_cons) > 0 else None

# --- Column 3: Minimal Autocatalytic Kinetics (A + B -> 2B) ---
k_auto = 1.5


def autocatalytic(y, t, k):
    """System of ODEs for minimal autocatalysis."""
    A, B = y
    return [-k * A * B, k * A * B]


sol_auto = odeint(autocatalytic, [0.98, 0.02], t, args=(k_auto,))
A_auto, B_auto = sol_auto[:, 0], sol_auto[:, 1]

# Rate of autocatalytic product formation
v_auto = k_auto * A_auto * B_auto

# Acceleration for autocatalysis derived exactly from the chain rule:
# d^2B/dt^2 = k^2 * A * B * (A - B)
accel_auto = (k_auto**2) * A_auto * B_auto * (A_auto - B_auto)

# Isolate the inflection point (maximum velocity) landmark
idx_max_auto = np.argmax(v_auto)
t_inf_auto = t[idx_max_auto]
vmax_auto = v_auto[idx_max_auto]

# ------------------------------------------------------------------------------
# 3. Graphical Rendering
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0))

# Adjusted horizontal spacing (wspace) to provide breathing room
# and slightly expanded left margin to accommodate the panel letters
fig.subplots_adjust(wspace=0.35, hspace=0.35, top=0.90, bottom=0.10, left=0.12, right=0.98)

# Render column headers describing the kinetic mechanisms
cols = [
    "First-order\n(A $\\rightarrow$ B)",
    "Consecutive\n(A $\\rightarrow$ B $\\rightarrow$ C)",
    "Autocatalytic\n(A + B $\\rightarrow$ 2B)",
]
for ax, col_title in zip(axes[0], cols):
    ax.set_title(col_title, fontsize=10, fontweight="bold", pad=15)

# --- Row 1: Concentration Profiles ---
axes[0, 0].plot(t, A_first, color=colors["A"])
axes[0, 0].plot(t, B_first, color=colors["B"])

axes[0, 1].plot(t, A_cons, color=colors["A"])
axes[0, 1].plot(t, B_cons, color=colors["B"])
axes[0, 1].plot(t, C_cons, color=colors["C"], linestyle="--")
if t_inf_cons:
    axes[0, 1].axvline(t_inf_cons, color=colors["gray"], linestyle=":", lw=0.8)
    axes[0, 1].scatter(
        [t_inf_cons], [B_cons[int(idx_inf_cons[0])]], color=colors["B"], s=15, zorder=5
    )

axes[0, 2].plot(t, A_auto, color=colors["A"])
axes[0, 2].plot(t, B_auto, color=colors["B"])
axes[0, 2].axvline(t_inf_auto, color=colors["gray"], linestyle=":", lw=0.8)
axes[0, 2].scatter([t_inf_auto], [B_auto[idx_max_auto]], color=colors["B"], s=15, zorder=5)

for ax in axes[0]:
    ax.set_ylim(-0.05, 1.1)

# Apply Y-axis title only to the leftmost column to prevent text clashing
axes[0, 0].set_ylabel("Conc. (M)", labelpad=10)

# --- Row 2: Reaction Rate Profiles ---
axes[1, 0].plot(t, v_first, color=colors["B"])

axes[1, 1].plot(t, v_cons, color=colors["B"])
axes[1, 1].axhline(0, color=colors["gray"], linestyle="--", lw=0.8)
if t_inf_cons:
    axes[1, 1].axvline(t_inf_cons, color=colors["gray"], linestyle=":", lw=0.8)

axes[1, 2].plot(t, v_auto, color=colors["B"])
axes[1, 2].axvline(t_inf_auto, color=colors["gray"], linestyle=":", lw=0.8)
axes[1, 2].scatter([t_inf_auto], [vmax_auto], color=colors["B"], s=25, zorder=5)
axes[1, 2].text(
    t_inf_auto + 0.2, vmax_auto, r"$v_{\max}$", color=colors["B"], va="center", fontsize=9
)

# Apply Y-axis title only to the leftmost column
axes[1, 0].set_ylabel(r"Rate, $d[B]/dt$ (M/s)", labelpad=10)

# --- Row 3: Acceleration Profiles ---
axes[2, 0].plot(t, accel_first, color=colors["gray"], lw=1)
axes[2, 0].fill_between(t, 0, accel_first, color=colors["accel_neg"], alpha=0.2)
axes[2, 0].axhline(0, color=colors["gray"], linestyle="-", lw=0.5)

axes[2, 1].plot(t, accel_cons, color=colors["gray"], lw=1)
axes[2, 1].fill_between(
    t, 0, accel_cons, where=(accel_cons < 0), color=colors["accel_neg"], alpha=0.2
)
axes[2, 1].fill_between(
    t, 0, accel_cons, where=(accel_cons > 0), color=colors["accel_pos"], alpha=0.2
)
axes[2, 1].axhline(0, color=colors["gray"], linestyle="-", lw=0.5)
if t_inf_cons:
    axes[2, 1].axvline(t_inf_cons, color=colors["gray"], linestyle=":", lw=0.8)
    axes[2, 1].scatter([t_inf_cons], [0], color=colors["B"], s=25, zorder=5)

axes[2, 2].plot(t, accel_auto, color=colors["gray"], lw=1)
axes[2, 2].fill_between(
    t, 0, accel_auto, where=(accel_auto > 0), color=colors["accel_pos"], alpha=0.2
)
axes[2, 2].fill_between(
    t, 0, accel_auto, where=(accel_auto < 0), color=colors["accel_neg"], alpha=0.2
)
axes[2, 2].axhline(0, color=colors["gray"], linestyle="-", lw=0.5)
axes[2, 2].axvline(t_inf_auto, color=colors["gray"], linestyle=":", lw=0.8)
axes[2, 2].scatter([t_inf_auto], [0], color=colors["B"], s=25, zorder=5)

for ax in axes[2]:
    ax.set_xlabel("Time (s)")

# Apply Y-axis title only to the leftmost column
axes[2, 0].set_ylabel(r"Accel., $d^2[B]/dt^2$ (M/s$^2$)", labelpad=10)

# --- Exterior Label Positioning & Formatting ---
letters = "abcdefghi"
# Position panel letters cleanly outside the boxes
label_x = -0.22
label_y = 1.05

for i, ax in enumerate(axes.flatten()):
    ax.text(
        label_x,
        label_y,
        f"({letters[i]})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlim(0, 5)

# ------------------------------------------------------------------------------
# 4. File Output Synthesis
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "Figure2_mechanisms.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(outdir / "Figure2_mechanisms.png", format="png", bbox_inches="tight", dpi=300)
