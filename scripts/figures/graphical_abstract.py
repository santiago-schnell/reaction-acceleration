"""graphical_abstract.py

Graphical abstract for the ChemSystemsChem Concept article:
"Reaction Acceleration: Reviving the Second Derivative in Chemical Kinetics".

The figure emphasizes that the sign pattern of the acceleration
(d^2[B]/dt^2 for a monitored progress variable B) distinguishes:

- single-step relaxation (no sign change; typically negative for product
  formation when the rate decreases),
- intermediacy in consecutive reactions (negative-to-positive),
- feedback in autocatalysis (positive-to-negative).

Units are omitted for compactness; the goal is conceptual rather than
quantitative.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Style & Configuration
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
})

colors = {
    'line': '#444444',       # Dark Gray
    'pos': '#4DAF4A',        # Green
    'neg': '#E66101',        # Orange
    'zero_dot': '#D00000',   # Red
}

# ------------------------------------------------------------------------------
# 2. Analytical Acceleration Functions
# ------------------------------------------------------------------------------

def acc_first_order(t, k=1.0):
    return - (k**2) * np.exp(-k * t)

def acc_consecutive(t, k1=1.0, k2=0.5):
    term1 = (k1**2) * np.exp(-k1 * t)
    term2 = (k2**2) * np.exp(-k2 * t)
    prefactor = k1 / (k2 - k1)
    return prefactor * (term1 - term2)

def acc_autocatalytic(t, k=1.5, A_tot=1.0, B0=0.02):
    denom = 1.0 + ((A_tot/B0) - 1.0) * np.exp(-k * A_tot * t)
    B = A_tot / denom
    return (k**2) * B * (A_tot - B) * (A_tot - 2.0 * B)

# ------------------------------------------------------------------------------
# 3. Plotting with Custom Positioning
# ------------------------------------------------------------------------------

def plot_fingerprint(ax, t, acc_data, title, reaction, fingerprint_text, text_pos):
    """
    Draws a single panel with the acceleration curve, shading, and landmark.
    text_pos: tuple (x, y) in axes coordinates for the fingerprint label.
    """
    
    # Draw curve
    ax.plot(t, acc_data, color=colors['line'], zorder=5)
    
    # Zero line
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    # Shading
    ax.fill_between(t, 0, acc_data, where=(acc_data > 0), 
                    color=colors['pos'], alpha=0.25, interpolate=True)
    ax.fill_between(t, 0, acc_data, where=(acc_data < 0), 
                    color=colors['neg'], alpha=0.25, interpolate=True)

    # Landmark Dot (Zero Crossing)
    sign_change = np.where(np.diff(np.sign(acc_data)))[0]
    if len(sign_change) > 0:
        for idx in sign_change:
            t0, t1 = t[idx], t[idx+1]
            y0, y1 = acc_data[idx], acc_data[idx+1]
            t_cross = t0 - y0 * (t1 - t0) / (y1 - y0)
            ax.scatter([t_cross], [0], color=colors['zero_dot'], s=50, 
                       edgecolor='white', linewidth=1.5, zorder=10)

    # Annotations
    ax.set_title(f"{title}\n{reaction}", fontsize=11, fontweight='bold', pad=10)
    
    # Text Label with Custom Position
    # Using alignment 'right' or 'left' based on x-position could be dynamic,
    # but fixed 'right' alignment with careful coordinates works well.
    ha = 'right' if text_pos[0] > 0.5 else 'left'
    va = 'top' if text_pos[1] > 0.5 else 'bottom'
    
    ax.text(text_pos[0], text_pos[1], fingerprint_text, transform=ax.transAxes, 
            ha=ha, va=va, fontsize=9, fontstyle='italic',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    # Clean axes
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)

def main():
    t = np.linspace(0, 6, 500)
    
    # Normalized Data
    a1 = acc_first_order(t)
    a1 /= np.max(np.abs(a1))
    
    a2 = acc_consecutive(t)
    a2 /= np.max(np.abs(a2))
    
    a3 = acc_autocatalytic(t)
    a3 /= np.max(np.abs(a3))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0), constrained_layout=True)
    
    # Panel 1: First Order
    # Curve is always negative (bottom). Place text at Top-Right.
    plot_fingerprint(axes[0], t, a1, 
                     title="Relaxation", 
                     reaction=r"($A \rightarrow B$)", 
                     fingerprint_text="Always Negative",
                     text_pos=(0.95, 0.90)) # Top Right
    
    # Panel 2: Consecutive
    # Curve ends positive (top). Place text at Bottom-Right.
    plot_fingerprint(axes[1], t, a2, 
                     title="Intermediate", 
                     reaction=r"($A \rightarrow B \rightarrow C$)", 
                     fingerprint_text=r"Sign Change: $(-)\rightarrow(+)$",
                     text_pos=(0.95, 0.05)) # Bottom Right

    # Panel 3: Autocatalysis
    # Curve ends negative (bottom). Place text at Top-Right.
    plot_fingerprint(axes[2], t, a3, 
                     title="Autocatalysis", 
                     reaction=r"($A + B \rightarrow 2B$)", 
                     fingerprint_text=r"Sign Change: $(+)\rightarrow(-)$",
                     text_pos=(0.95, 0.90)) # Top Right (Corrected)

    fig.supylabel(r"Reaction Acceleration ($\ddot{\xi}$)", fontsize=12, x=-0.01)

    outdir = Path(__file__).resolve().parents[2] / 'outputs' / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'Graphical_Abstract.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(outdir / 'Graphical_Abstract.png', format='png', bbox_inches='tight', dpi=300)
    print("Graphical Abstract saved successfully.")

if __name__ == "__main__":
    main()
