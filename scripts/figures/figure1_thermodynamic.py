r"""
Figure 1: Thermodynamic Relationships in Chemical Kinetics
-----------------------------------------------------------
This script generates a three-panel figure illustrating the fundamental
relationships between affinity, velocity, and acceleration as a reaction
approaches equilibrium.

Designed for absolute reproducibility, this script meticulously details
the thermodynamic parameters, the analytical solutions to the kinetic
differential equations, and the explicit rendering logic utilized to
generate the publication-quality schematics.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Department of Biomedical Data Sciences, Geisel School
    of Medicine at Dartmouth, Hanover, New Hampshire, USA

Panel (a): Affinity (A) vs. Extent of reaction (ξ)
Panel (b): Velocity (v) vs. Time (t)
Panel (c): Reaction acceleration (\dot{v} = dv/dt) vs. Time (t)

The system modeled is a reversible first-order reaction: A ⇌ B.

Outputs:
    - Figure1_thermodynamic.pdf
    - Figure1_thermodynamic.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import apply_style
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------------------
# 1. Configuration & Style Integration
# ------------------------------------------------------------------------------
# Apply global typographic and geometric parameters for publication standards
apply_style()

# Harmonized color palette for thermodynamic consistency
colors = {
    "affinity": "#2166AC",  # Blue (Concentration/Potential analog)
    "velocity": "#B2182B",  # Red (Kinetic Rate analog)
    "accel_neg": "#E66101",  # Orange (Negative Acceleration / Deceleration)
    "gray": "#666666",  # Gray (Structural guidelines)
}

# ------------------------------------------------------------------------------
# 2. Mathematical Definition & Data Generation
# ------------------------------------------------------------------------------
# Explicit declaration of physical parameters to guarantee reproducible states.
# The system models a reversible first-order transition A ⇌ B.
R = 8.314  # Universal Gas Constant (J/(mol*K))
T = 298  # Standard Temperature (K)
K = 10  # Equilibrium constant (Dimensionless, K > 1 favors B)
A0 = 1.0  # Initial concentration of species [A] (M)

# --- Panel (a): Affinity Computation ---
# Equilibrium extent derived from mass conservation: xi_eq = [A]0 * K / (1 + K)
xi_eq = (K / (1 + K)) * A0

# Generate highly resolved extent values (xi) approaching equilibrium
xi = np.linspace(0.001, xi_eq * 0.999, 500)

# Thermodynamic Affinity calculation: A = -ΔG = RT * ln(K/Q)
# The reaction quotient Q represents [B]/[A] = xi / (A0 - xi)
A_energy = R * T * np.log(K * (A0 - xi) / xi) / 1000  # Conversion to kJ/mol

# --- Panels (b) & (c): Kinetic Evolution Computation ---
# Define rate constants maintaining strict thermodynamic fidelity (K = k1 / k-1)
k1 = 1.0
km1 = k1 / K

# Temporal array for integrating the kinetic profile
t = np.linspace(0, 5, 500)

# Analytical solution for [A](t) describing exponential relaxation to equilibrium
Aeq = A0 / (1 + K)
A_conc = Aeq + (A0 - Aeq) * np.exp(-(k1 + km1) * t)

# Velocity (v) computed as the net forward reaction rate = k1[A] - k-1[B]
v = k1 * A_conc - km1 * (A0 - A_conc)

# Acceleration (\dot{v}) computed as the temporal derivative of velocity (dv/dt).
# As the system relaxes, velocity monotonically decreases, yielding a strictly
# negative acceleration (deceleration) profile.
accel = -(k1 + km1) * v

# ------------------------------------------------------------------------------
# 3. Graphical Rendering
# ------------------------------------------------------------------------------
# Instantiate a 1x3 coordinate geometry
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

# Spatial adjustments to prevent label occlusion
fig.subplots_adjust(wspace=0.45, top=0.85, bottom=0.20, left=0.08, right=0.98)

# --- Panel (a): Affinity Geometry ---
ax1 = axes[0]
ax1.plot(xi, A_energy, color=colors["gray"], linewidth=1.5)
ax1.fill_between(xi, 0, A_energy, color=colors["affinity"], alpha=0.1)
ax1.axhline(y=0, color=colors["gray"], linestyle="--", linewidth=0.8)
ax1.axvline(x=xi_eq, color=colors["gray"], linestyle=":", linewidth=0.8)

ax1.set_xlabel(r"Extent of reaction, $\xi$")
ax1.set_ylabel(r"Affinity, $A$ (kJ/mol)")
ax1.set_xlim(0, 1)
ax1.set_ylim(-2, 8)

# Exterior panel designation for consistent formatting
ax1.text(-0.25, 1.05, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=10)

ax1.annotate(
    r"$\xi_{\mathrm{eq}}$",
    xy=(xi_eq, 0),
    xycoords="data",
    xytext=(-5, 5),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=9,
    color=colors["gray"],
)

# --- Panel (b): Velocity Geometry ---
ax2 = axes[1]
ax2.plot(t, v, color=colors["gray"], linewidth=1.5)
ax2.axhline(y=0, color=colors["gray"], linestyle="--", linewidth=0.8)
ax2.fill_between(t, 0, v, where=(v > 0), alpha=0.15, color=colors["velocity"])
ax2.text(
    0.5, 0.6, r"$v > 0$", fontsize=9, color=colors["velocity"], transform=ax2.transAxes, ha="center"
)

ax2.set_xlabel(r"Time, $t$ (s)")
ax2.set_ylabel(r"Velocity, $v$ (M/s)")
ax2.set_xlim(0, 5)
ax2.text(-0.25, 1.05, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=10)

# --- Panel (c): Acceleration Geometry ---
ax3 = axes[2]
ax3.plot(t, accel, color=colors["gray"], linewidth=1.5)
ax3.axhline(y=0, color=colors["gray"], linestyle="--", linewidth=0.8)
ax3.fill_between(t, 0, accel, where=(accel < 0), alpha=0.2, color=colors["accel_neg"])
ax3.text(
    0.5,
    0.3,
    r"$\dot{v} < 0$",
    fontsize=9,
    color=colors["accel_neg"],
    transform=ax3.transAxes,
    ha="center",
)

ax3.set_xlabel(r"Time, $t$ (s)")
ax3.set_ylabel(r"Acceleration, $\dot{v}$ (M/s$^2$)")
ax3.set_xlim(0, 5)
ax3.text(-0.25, 1.05, "(c)", transform=ax3.transAxes, fontweight="bold", fontsize=10)

# --- Minor Typographic Formatting ---
for ax in axes:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

# ------------------------------------------------------------------------------
# 4. File Output Synthesis
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "Figure1_thermodynamic.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(outdir / "Figure1_thermodynamic.png", format="png", bbox_inches="tight", dpi=300)
print("Figure 1 generated and archived successfully.")
