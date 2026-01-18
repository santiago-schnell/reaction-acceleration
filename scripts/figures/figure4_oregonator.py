"""
Figure 4: Kinetic Fingerprint of the Oregonator Model
-----------------------------------------------------
This script generates a high-quality 3-panel figure illustrating the
limit cycle dynamics and the "acceleration fingerprint" of the
Belousov-Zhabotinsky reaction (Oregonator model).

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Department of Biomedical Data Sciences, Geisel School
    of Medicine at Dartmouth, Hanover, New Hampshire, USA

Key Features:
1.  **Layout**: 1 row x 3 columns with custom asymmetric spacing to
    accommodate the secondary Y-axis in the middle panel.
2.  **Panels**:
    - (a) Phase Space: Limit cycle ($[HBrO_2]$ vs. $[Ce^{4+}]$) with
          direction arrows and red dots marking inflection points.
    - (b) Time Series: Concentration ($[Z]$) and Rate ($dZ/dt$) on dual axes.
    - (c) Acceleration: $d^2Z/dt^2$ with sign-based shading (Green/Orange).
3.  **Style**: Minimalist, publication-quality (Arial/Helvetica fonts).
    - No internal legends or chart junk.
    - Panel labels (a, b, c) placed outside the axes.

Outputs:
    - Figure4_oregonator.pdf
    - Figure4_oregonator.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import odeint

# ------------------------------------------------------------------------------
# 1. Configuration & Style
# ------------------------------------------------------------------------------
# Update matplotlib rcParams for publication-quality rendering
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

# Define a consistent color palette matching Figs 1-3
colors = {
    'Z': '#2166AC',          # Blue (Concentration)
    'Rate': '#B2182B',       # Red (Rate)
    'accel_pos': '#4DAF4A',  # Green (Positive Acceleration)
    'accel_neg': '#E66101',  # Orange (Negative Acceleration)
    'gray': '#666666',       # Gray for axes/grid
    'dot': '#D00000',        # Red dot for inflection points
}

# ------------------------------------------------------------------------------
# 2. Data Generation (Oregonator Model)
# ------------------------------------------------------------------------------
# Dimensionless parameters for the Oregonator model
epsilon = 0.04
epsilon_prime = 0.0004
f = 1.0
q = 0.0008

def oregonator(y, t, eps, eps_p, f, q):
    """
    Defines the Oregonator differential equations.
    x: [HBrO2], Y: [Br-], z: [Ce4+]
    """
    x, Y, z = y
    # Clamp values to avoid numerical instability near zero
    x = max(x, 1e-12)
    Y = max(Y, 1e-12)
    z = max(z, 1e-12)
    
    dxdt = (q * Y - x * Y + x * (1 - x)) / eps
    dYdt = (-q * Y - x * Y + f * z) / eps_p
    dzdt = x - z
    return [dxdt, dYdt, dzdt]

# Integration Setup
# Use high resolution (8000 points) to capture stiff relaxation oscillations accurately
t = np.linspace(0, 40, 8000)
dt = t[1] - t[0]
y0 = [0.1, 0.1, 0.1] # Initial conditions
sol = odeint(oregonator, y0, t, args=(epsilon, epsilon_prime, f, q))

# Transient Removal
# Skip the initial portion to ensure the system has settled onto the limit cycle
skip = 2000
t_plot = t[skip:] - t[skip]
X_plot = sol[skip:, 0]
Z_plot = sol[skip:, 2]

# Calculate Derivatives
# Rate (v) and Acceleration (a) for variable Z
dZdt = np.gradient(Z_plot, dt)
d2Zdt2 = np.gradient(dZdt, dt)

# Identify Inflection Points (Zero Crossings of Acceleration)
zero_crossings_idx = []
for i in range(1, len(d2Zdt2)):
    if d2Zdt2[i-1] * d2Zdt2[i] < 0: # Check for sign change
        zero_crossings_idx.append(i)

# Filter Inflection Points
# Remove duplicate detections caused by numerical noise (points too close together)
clean_inf_idx = []
if len(zero_crossings_idx) > 0:
    clean_inf_idx.append(zero_crossings_idx[0])
    for idx in zero_crossings_idx[1:]:
        if idx - clean_inf_idx[-1] > 20: # Minimum distance threshold
            clean_inf_idx.append(idx)

# Window Selection
# Select a time window of ~2 cycles (approx t=0 to 18) for clear visualization
# This "zoom" makes the individual phases of the oscillation distinct.
end_idx = 3600
t_win = t_plot[:end_idx]
X_win = X_plot[:end_idx]
Z_win = Z_plot[:end_idx]
rate_win = dZdt[:end_idx]
accel_win = d2Zdt2[:end_idx]

# Extract inflection points within the selected window
inf_idx_win = [i for i in clean_inf_idx if i < end_idx]
t_inf = t_plot[inf_idx_win]
Z_inf = Z_plot[inf_idx_win]
X_inf = X_plot[inf_idx_win]
rate_inf = dZdt[inf_idx_win]

# ------------------------------------------------------------------------------
# 3. Plotting with Explicit Layout
# ------------------------------------------------------------------------------
# Use specific figsize as requested for optimal spacing
fig = plt.figure(figsize=(8.5, 2.3))

# Define geometry [left, bottom, width, height]
# Width of each panel (fraction of figure width)
w = 0.23
h = 0.72
b = 0.20

# Horizontal positions (customized for spacing)
l1 = 0.08  # Panel (a) start
l2 = 0.39  # Panel (b) start (Gap (a)-(b) ~ 0.08)
l3 = 0.75  # Panel (c) start (Gap (b)-(c) ~ 0.13 to fit secondary Y-axis)

ax1 = fig.add_axes([l1, b, w, h])
ax2 = fig.add_axes([l2, b, w, h])
ax3 = fig.add_axes([l3, b, w, h])

# --- Panel (a): Phase Space ---
ax1.plot(X_win, Z_win, color=colors['gray'], linewidth=1.2)
ax1.scatter(X_inf, Z_inf, color=colors['dot'], s=25, zorder=10)

# Add Direction Arrows
# Place arrows based on Euclidean distance to avoid clustering near origin
arrow_dist_threshold = 0.15
last_arrow_pos = np.array([-999, -999])
for i in range(0, len(X_win)-5, 50):
    curr_pos = np.array([X_win[i], Z_win[i]])
    dist = np.linalg.norm(curr_pos - last_arrow_pos)
    if dist > arrow_dist_threshold:
        ax1.annotate('', xy=(X_win[i+5], Z_win[i+5]), xytext=(X_win[i], Z_win[i]),
                     arrowprops=dict(arrowstyle='->', color=colors['gray'], lw=1))
        last_arrow_pos = curr_pos

ax1.set_xlabel(r'[HBrO$_2$] ($X$)')
ax1.set_ylabel(r'[Ce$^{4+}$] ($Z$)')
# Panel label placed outside plot area
ax1.text(-0.20, 1.05, '(a)', transform=ax1.transAxes, fontweight='bold', fontsize=10)

# --- Panel (b): Concentration & Rate ---
ax2.plot(t_win, Z_win, color=colors['Z'], linewidth=1.5)
ax2.set_xlabel('Time (dimensionless)')
ax2.set_ylabel(r'[Ce$^{4+}$] ($Z$)', color=colors['Z'])
ax2.tick_params(axis='y', labelcolor=colors['Z'])
ax2.set_ylim(0, np.max(Z_win)*1.1)

# Secondary Axis for Rate
ax2b = ax2.twinx()
ax2b.plot(t_win, rate_win, color=colors['Rate'], linewidth=1.0, alpha=0.8, linestyle='--')
ax2b.set_ylabel('Rate $dZ/dt$', color=colors['Rate'])
ax2b.tick_params(axis='y', labelcolor=colors['Rate'])
ax2b.axhline(0, color=colors['gray'], lw=0.5, linestyle=':')
ax2b.scatter(t_inf, rate_inf, color=colors['dot'], s=20, zorder=10)

ax2.text(-0.20, 1.05, '(b)', transform=ax2.transAxes, fontweight='bold', fontsize=10)

# --- Panel (c): Acceleration ---
ax3.plot(t_win, accel_win, color=colors['gray'], linewidth=1.0)
ax3.axhline(0, color=colors['gray'], lw=0.8)

# Acceleration Shading
ax3.fill_between(t_win, 0, accel_win, where=(accel_win>0), color=colors['accel_pos'], alpha=0.3)
ax3.fill_between(t_win, 0, accel_win, where=(accel_win<0), color=colors['accel_neg'], alpha=0.3)
ax3.scatter(t_inf, np.zeros_like(t_inf), color=colors['dot'], s=25, zorder=10)

ax3.set_xlabel('Time (dimensionless)')
ax3.set_ylabel(r'Accel. $d^2Z/dt^2$')
ax3.text(-0.20, 1.05, '(c)', transform=ax3.transAxes, fontweight='bold', fontsize=10)

# --- Formatting ---
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

# --- Save Output ---
outdir = Path(__file__).resolve().parents[2] / 'outputs' / 'figures'
outdir.mkdir(parents=True, exist_ok=True)
out_pdf = outdir / 'Figure4_oregonator.pdf'
out_png = outdir / 'Figure4_oregonator.png'
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(out_png, format='png', bbox_inches='tight', dpi=300)
print(f"Figures saved: {out_pdf}, {out_png}")
