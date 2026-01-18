"""
Figure 2: Kinetic Fingerprints of Elementary Mechanisms
-------------------------------------------------------
This script generates a comparative matrix of First-order, Consecutive, and 
Autocatalytic kinetics. It highlights the unique acceleration "fingerprint" 
of each mechanism.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Department of Biomedical Data Sciences, Geisel School
    of Medicine at Dartmouth, Hanover, New Hampshire, USA

Outputs:
    - Figure2_mechanisms.pdf
    - Figure2_mechanisms.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import odeint
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Configuration & Style
# ------------------------------------------------------------------------------
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

colors = {
    'A': '#2166AC',          # Blue (Reactant)
    'B': '#B2182B',          # Red (Product/Intermediate)
    'C': '#999999',          # Gray (Final Product)
    'accel_pos': '#4DAF4A',  # Green (Positive acceleration)
    'accel_neg': '#E66101',  # Orange (Negative acceleration)
    'gray': '#666666',       # Gray (Guidelines/Text)
}

# ------------------------------------------------------------------------------
# 2. Data Generation
# ------------------------------------------------------------------------------
t = np.linspace(0, 5, 1000)
dt = t[1] - t[0]

# --- Column 1: First-order (A -> B) ---
k1 = 1.0
A_first = np.exp(-k1 * t)
B_first = 1 - np.exp(-k1 * t)
v_first = k1 * A_first  # rate of B formation: d[B]/dt
accel_first = -k1**2 * A_first  # acceleration: d^2[B]/dt^2

# --- Column 2: Consecutive (A -> B -> C) ---
k1_cons = 1.0
k2_cons = 0.6
def consecutive(y, t, k1, k2):
    A, B, C = y
    dAdt = -k1 * A
    dBdt = k1 * A - k2 * B
    dCdt = k2 * B
    return [dAdt, dBdt, dCdt]

sol_cons = odeint(consecutive, [1.0, 0.0, 0.0], t, args=(k1_cons, k2_cons))
A_cons, B_cons, C_cons = sol_cons[:, 0], sol_cons[:, 1], sol_cons[:, 2]
v_cons = k1_cons * A_cons - k2_cons * B_cons  # rate of B: d[B]/dt

# Acceleration of B (second derivative): d^2[B]/dt^2
# Using the exact ODE identity avoids numerical differentiation artifacts.
# From dB/dt = k1*A - k2*B and dA/dt = -k1*A:
#   d2B/dt2 = k1*dA/dt - k2*dB/dt = -k1*(k1+k2)*A + k2^2*B
accel_cons = -k1_cons * (k1_cons + k2_cons) * A_cons + (k2_cons**2) * B_cons

# Find zero crossing for consecutive acceleration (Inflection in B)
idx_inf_cons = np.where(np.diff(np.sign(accel_cons)))[0]
t_inf_cons = t[idx_inf_cons[0]] if len(idx_inf_cons) > 0 else None

# --- Column 3: Autocatalytic (A + B -> 2B) ---
k_auto = 2.0
def autocatalytic(y, t, k):
    A, B = y
    dAdt = -k * A * B
    dBdt = k * A * B
    return [dAdt, dBdt]

sol_auto = odeint(autocatalytic, [1.0, 0.01], t, args=(k_auto,))
A_auto, B_auto = sol_auto[:, 0], sol_auto[:, 1]
v_auto = k_auto * A_auto * B_auto  # rate of B: d[B]/dt

# Acceleration for autocatalysis (exact): d2B/dt2 = k^2 * A * B * (A - B)
# since dB/dt = kAB and dA/dt = -kAB.
accel_auto = (k_auto**2) * A_auto * B_auto * (A_auto - B_auto)

# Find max rate (Inflection in B)
idx_max_auto = np.argmax(v_auto)
t_inf_auto = t[idx_max_auto]
vmax_auto = v_auto[idx_max_auto]

# ------------------------------------------------------------------------------
# 3. Plotting
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0))

# Adjusted Margins: 
# Increased top slightly to prevent label/title clash
fig.subplots_adjust(wspace=0.45, hspace=0.35, top=0.90, bottom=0.10, left=0.10, right=0.98)

# Column Titles
cols = ['First-order\n(A $\\rightarrow$ B)', 'Consecutive\n(A $\\rightarrow$ B $\\rightarrow$ C)', 'Autocatalytic\n(A + B $\\rightarrow$ 2B)']
for ax, col_title in zip(axes[0], cols):
    ax.set_title(col_title, fontsize=10, fontweight='bold', pad=15)

# --- Row 1: Concentration ---
# Col 1
axes[0,0].plot(t, A_first, color=colors['A'], label='[A]')
axes[0,0].plot(t, B_first, color=colors['B'], label='[B]')
# Col 2
axes[0,1].plot(t, A_cons, color=colors['A'])
axes[0,1].plot(t, B_cons, color=colors['B'])
axes[0,1].plot(t, C_cons, color=colors['C'], linestyle='--', label='[C]')
if t_inf_cons:
    axes[0,1].axvline(t_inf_cons, color=colors['gray'], linestyle=':', lw=0.8)
    axes[0,1].scatter([t_inf_cons], [B_cons[int(idx_inf_cons[0])]], color=colors['B'], s=15, zorder=5)
# Col 3
axes[0,2].plot(t, A_auto, color=colors['A'])
axes[0,2].plot(t, B_auto, color=colors['B'])
axes[0,2].axvline(t_inf_auto, color=colors['gray'], linestyle=':', lw=0.8)
axes[0,2].scatter([t_inf_auto], [B_auto[idx_max_auto]], color=colors['B'], s=15, zorder=5)

for ax in axes[0]:
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('Conc. (M)')

# --- Row 2: Rate ---
# Col 1
axes[1,0].plot(t, v_first, color=colors['B'])
# Col 2
axes[1,1].plot(t, v_cons, color=colors['B'])
axes[1,1].axhline(0, color=colors['gray'], linestyle='--', lw=0.8)
if t_inf_cons:
    axes[1,1].axvline(t_inf_cons, color=colors['gray'], linestyle=':', lw=0.8)
# Col 3
axes[1,2].plot(t, v_auto, color=colors['B'])
axes[1,2].axvline(t_inf_auto, color=colors['gray'], linestyle=':', lw=0.8)
axes[1,2].scatter([t_inf_auto], [vmax_auto], color=colors['B'], s=25, zorder=5)
axes[1,2].text(t_inf_auto + 0.2, vmax_auto, r'$v_{\max}$', color=colors['B'], va='center', fontsize=9)

for ax in axes[1]:
    ax.set_ylabel(r'Rate, $d[B]/dt$ (M/s)')

# --- Row 3: Acceleration ---
# Col 1 (Always Neg)
axes[2,0].plot(t, accel_first, color=colors['gray'], lw=1)
axes[2,0].fill_between(t, 0, accel_first, color=colors['accel_neg'], alpha=0.2)
axes[2,0].axhline(0, color=colors['gray'], linestyle='-', lw=0.5)

# Col 2 (Neg -> Pos)
axes[2,1].plot(t, accel_cons, color=colors['gray'], lw=1)
axes[2,1].fill_between(t, 0, accel_cons, where=(accel_cons < 0), color=colors['accel_neg'], alpha=0.2)
axes[2,1].fill_between(t, 0, accel_cons, where=(accel_cons > 0), color=colors['accel_pos'], alpha=0.2)
axes[2,1].axhline(0, color=colors['gray'], linestyle='-', lw=0.5)
if t_inf_cons:
    axes[2,1].axvline(t_inf_cons, color=colors['gray'], linestyle=':', lw=0.8)
    axes[2,1].scatter([t_inf_cons], [0], color=colors['B'], s=25, zorder=5)

# Col 3 (Pos -> Neg)
axes[2,2].plot(t, accel_auto, color=colors['gray'], lw=1)
axes[2,2].fill_between(t, 0, accel_auto, where=(accel_auto > 0), color=colors['accel_pos'], alpha=0.2)
axes[2,2].fill_between(t, 0, accel_auto, where=(accel_auto < 0), color=colors['accel_neg'], alpha=0.2)
axes[2,2].axhline(0, color=colors['gray'], linestyle='-', lw=0.5)
axes[2,2].axvline(t_inf_auto, color=colors['gray'], linestyle=':', lw=0.8)
axes[2,2].scatter([t_inf_auto], [0], color=colors['B'], s=25, zorder=5)

for ax in axes[2]:
    ax.set_ylabel(r'Accel., $d^2[B]/dt^2$ (M/s$^2$)')
    ax.set_xlabel('Time (s)')

# --- Labels & Formatting ---
letters = 'abcdefghi'
# Standardized label position (outside top-left)
# Using -0.25 offset to clear the y-axis labels
label_x = -0.25
label_y = 1.10

for i, ax in enumerate(axes.flatten()):
    ax.text(label_x, label_y, f'({letters[i]})', transform=ax.transAxes, 
            fontweight='bold', fontsize=10, va='bottom')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlim(0, 5)

# ------------------------------------------------------------------------------
# 4. Save Output
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / 'outputs' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / 'Figure2_mechanisms.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(outdir / 'Figure2_mechanisms.png', format='png', bbox_inches='tight', dpi=300)
print("Figure 2 saved successfully.")
