"""
Figure 1: Thermodynamic Relationships in Chemical Kinetics
-----------------------------------------------------------
This script generates a three-panel figure illustrating the fundamental
relationships between affinity, velocity, and acceleration as a reaction
approaches equilibrium.

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Configuration & Style
# ------------------------------------------------------------------------------
# Set global plot parameters to match academic publication standards
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
})

# Define the harmonized color palette
colors = {
    'affinity': '#2166AC',      # Blue (matches Concentration/Potential)
    'velocity': '#B2182B',      # Red (matches Rate)
    'accel_neg': '#E66101',     # Orange (Negative Acceleration/Deceleration)
    'gray': '#666666',          # Gray for lines/axes
}

# ------------------------------------------------------------------------------
# 2. Data Generation
# ------------------------------------------------------------------------------
# Physical parameters for a reversible first-order reaction A <=> B
R = 8.314   # Gas constant (J/(mol*K))
T = 298     # Temperature (K)
K = 10      # Equilibrium constant (K > 1 favors products)
A0 = 1.0    # Initial concentration of [A] (M)

# --- Panel (a): Affinity vs. Extent of Reaction ---
# Equilibrium extent: xi_eq = [A]0 * K / (1 + K)
xi_eq = (K / (1 + K)) * A0

# Generate extent values (xi) from near-zero to near-equilibrium
xi = np.linspace(0.001, xi_eq * 0.999, 500)

# Thermodynamic Affinity: A = -deltaG = RT * ln(K/Q)
# where Q = [B]/[A] = xi / (A0 - xi)
A_energy = R * T * np.log(K * (A0 - xi) / xi) / 1000  # Convert to kJ/mol

# --- Panels (b) & (c): Kinetics vs. Time ---
# Rate constants consistent with K = k1 / k-1
k1 = 1.0        
km1 = k1 / K    

t = np.linspace(0, 5, 500)

# Analytical solution for [A](t)
# [A](t) approaches equilibrium exponentially
Aeq = A0 / (1 + K)
A_conc = Aeq + (A0 - Aeq) * np.exp(-(k1 + km1) * t)

# Velocity (v) = rate of forward net reaction = k1[A] - k-1[B]
v = k1 * A_conc - km1 * (A0 - A_conc)

# Acceleration (a) = dv/dt
# Since the system relaxes to equilibrium, velocity decreases monotonically.
# Therefore, acceleration is strictly negative (deceleration).
accel = -(k1 + km1) * v

# ------------------------------------------------------------------------------
# 3. Plotting
# ------------------------------------------------------------------------------
# Create figure with 1 row x 3 columns
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))

# Adjust spacing to ensure labels don't overlap
fig.subplots_adjust(wspace=0.45, top=0.85, bottom=0.20, left=0.08, right=0.98)

# --- Panel (a): Affinity ---
ax1 = axes[0]
ax1.plot(xi, A_energy, color=colors['gray'], linewidth=1.5)
# Fill to visualize the thermodynamic potential
ax1.fill_between(xi, 0, A_energy, color=colors['affinity'], alpha=0.1)

# Reference lines
ax1.axhline(y=0, color=colors['gray'], linestyle='--', linewidth=0.8)
ax1.axvline(x=xi_eq, color=colors['gray'], linestyle=':', linewidth=0.8)

# Labels & Annotations
ax1.set_xlabel(r'Extent of reaction, $\xi$')
ax1.set_ylabel(r'Affinity, $A$ (kJ/mol)')
ax1.set_xlim(0, 1)
ax1.set_ylim(-2, 8)
# Panel label outside the plot area
ax1.text(-0.20, 1.05, '(a)', transform=ax1.transAxes, fontweight='bold', fontsize=10)

ax1.annotate(r'$\xi_{\mathrm{eq}}$', xy=(xi_eq, 0), xycoords='data',
             xytext=(-5, 5), textcoords='offset points',
             ha='right', va='bottom', fontsize=9, color=colors['gray'])

# --- Panel (b): Velocity ---
ax2 = axes[1]
ax2.plot(t, v, color=colors['gray'], linewidth=1.5)
ax2.axhline(y=0, color=colors['gray'], linestyle='--', linewidth=0.8)

# Shading: Red for Rate (Positive Velocity)
ax2.fill_between(t, 0, v, where=(v > 0), alpha=0.15, color=colors['velocity'])
ax2.text(0.5, 0.6, r'$v > 0$', fontsize=9, color=colors['velocity'], 
         transform=ax2.transAxes, ha='center')

# Labels
ax2.set_xlabel(r'Time, $t$ (s)')
ax2.set_ylabel(r'Velocity, $v$ (M/s)')
ax2.set_xlim(0, 5)
ax2.text(-0.20, 1.05, '(b)', transform=ax2.transAxes, fontweight='bold', fontsize=10)

# --- Panel (c): Reaction acceleration ---
ax3 = axes[2]
ax3.plot(t, accel, color=colors['gray'], linewidth=1.5)
ax3.axhline(y=0, color=colors['gray'], linestyle='--', linewidth=0.8)

# Shading: Orange for Negative Acceleration (Deceleration)
# This aligns with the "Traffic Light" scheme of Figs 2-4.
ax3.fill_between(t, 0, accel, where=(accel < 0), alpha=0.2, color=colors['accel_neg'])
ax3.text(0.5, 0.3, r'$\dot{v} < 0$', fontsize=9, color=colors['accel_neg'], 
         transform=ax3.transAxes, ha='center')

# Labels
ax3.set_xlabel(r'Time, $t$ (s)')
ax3.set_ylabel(r'Acceleration, $\dot{v}$ (M/s$^2$)')
ax3.set_xlim(0, 5)
ax3.text(-0.20, 1.05, '(c)', transform=ax3.transAxes, fontweight='bold', fontsize=10)

# --- Formatting ---
for ax in axes:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

# ------------------------------------------------------------------------------
# 4. Save Output
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / 'outputs' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / 'Figure1_thermodynamic.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(outdir / 'Figure1_thermodynamic.png', format='png', bbox_inches='tight', dpi=300)
print("Figure 1 saved successfully.")