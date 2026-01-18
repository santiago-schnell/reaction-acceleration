"""
Figure 3: Autocatalysis Diagnostic
----------------------------------
This script generates a three-panel comparison of autocatalytic vs. first-order
kinetics.
- Legend removed (info moved to caption).
- Top margin adjusted to utilize space.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Department of Biomedical Data Sciences, Geisel School
    of Medicine at Dartmouth, Hanover, New Hampshire, USA

Outputs:
    - Figure3_autocatalysis.pdf
    - Figure3_autocatalysis.png
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
    'axes.titlesize': 11, # Standardized to 11 (was 10)
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
})

colors = {
    'auto': '#B2182B',       # Red (Autocatalytic)
    'first': '#2166AC',      # Blue (First-order)
    'accel_pos': '#4DAF4A',  # Green (Positive accel)
    'accel_neg': '#E66101',  # Orange (Negative accel)
    'gray': '#666666',       # Gray (Guidelines)
}

# ------------------------------------------------------------------------------
# 2. Data Generation
# ------------------------------------------------------------------------------
t = np.linspace(0, 6, 1000)
dt = t[1] - t[0]

# --- Autocatalytic (A + B -> 2B) ---
k_auto = 1.5
def autocatalytic(y, tt, k):
    A, B = y
    dAdt = -k * A * B
    dBdt = k * A * B
    return [dAdt, dBdt]

y0_auto = [0.98, 0.02]
sol_auto = odeint(autocatalytic, y0_auto, t, args=(k_auto,))
A_auto, B_auto = sol_auto[:, 0], sol_auto[:, 1]
v_auto = k_auto * A_auto * B_auto
accel_auto = np.gradient(v_auto, dt)

# --- First-Order (A -> B) ---
k_first = 0.7
A_first = np.exp(-k_first * t)
B_first = 1 - np.exp(-k_first * t)
v_first = k_first * A_first
accel_first = -k_first**2 * A_first

# --- Inflection Point Analysis ---
inflection_idx = int(np.argmax(v_auto))
t_inflection = t[inflection_idx]
B_inflection = B_auto[inflection_idx]
vmax = v_auto[inflection_idx]

# ------------------------------------------------------------------------------
# 3. Plotting
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

# Adjusted Margins:
# Consistent with Figure 1/4 spacing
fig.subplots_adjust(wspace=0.50, top=0.85, bottom=0.20, left=0.08, right=0.98)

# Label position configuration (Standardized)
label_x = -0.20
label_y = 1.05

# --- Panel (a): Concentration ---
ax1 = axes[0]
ax1.plot(t, B_auto, color=colors['auto'], linewidth=2)
ax1.plot(t, B_first, color=colors['first'], linewidth=1.5, linestyle='--')

ax1.axvline(x=t_inflection, color=colors['gray'], linestyle=':', linewidth=0.8, alpha=0.6)
ax1.scatter([t_inflection], [B_inflection], color=colors['auto'], s=30, zorder=5)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Product, [B]')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 1.1)
# Moved Label Outside
ax1.text(label_x, label_y, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='bottom')

# --- Panel (b): Rate ---
ax2 = axes[1]
ax2.plot(t, v_auto, color=colors['auto'], linewidth=2)
ax2.plot(t, v_first, color=colors['first'], linewidth=1.5, linestyle='--')

ax2.axvline(x=t_inflection, color=colors['gray'], linestyle=':', linewidth=0.8, alpha=0.6)
ax2.scatter([t_inflection], [vmax], color=colors['auto'], s=30, zorder=5)
ax2.text(t_inflection + 0.2, vmax + 0.05, r'$v_{\max}$', color=colors['auto'], va='center', fontsize=9)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel(r'Rate, $d[B]/dt$ (M/s)')
ax2.set_xlim(0, 6)
# Moved Label Outside
ax2.text(label_x, label_y, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='bottom')

# --- Panel (c): Acceleration ---
ax3 = axes[2]
ax3.plot(t, accel_auto, color=colors['auto'], linewidth=2)
ax3.plot(t, accel_first, color=colors['first'], linewidth=1.5, linestyle='--')

ax3.axhline(y=0, color=colors['gray'], linestyle='-', linewidth=0.8, alpha=0.6)
ax3.axvline(x=t_inflection, color=colors['gray'], linestyle=':', linewidth=0.8, alpha=0.6)
ax3.scatter([t_inflection], [0], color=colors['auto'], s=30, zorder=5)

ax3.fill_between(t, 0, accel_auto, where=(accel_auto > 0), alpha=0.2, color=colors['accel_pos'])
ax3.fill_between(t, 0, accel_auto, where=(accel_auto < 0), alpha=0.2, color=colors['accel_neg'])

ax3.set_xlabel('Time (s)')
ax3.set_ylabel(r'Accel., $d^2[B]/dt^2$ (M/s$^2$)')
ax3.set_xlim(0, 6)
# Moved Label Outside
ax3.text(label_x, label_y, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='bottom')

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
plt.savefig(outdir / 'Figure3_autocatalysis.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(outdir / 'Figure3_autocatalysis.png', format='png', bbox_inches='tight', dpi=300)

print("Figure 3 saved successfully.")
