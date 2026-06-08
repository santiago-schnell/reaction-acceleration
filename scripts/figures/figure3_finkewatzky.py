r"""
Figure 3: Finke-Watzky Two-Step Autocatalysis
---------------------------------------------
Illustrates how the acceleration landmark position in the Finke-Watzky two-step
model (slow nucleation A -> B at rate k1; fast autocatalytic growth A + B -> 2B
at rate k2) encodes the dimensionless mechanistic ratio k1/(k2 A_tot).

Designed for absolute reproducibility, this script meticulously details
the kinetic parameters, numerical solutions to the differential equations,
and the explicit rendering logic utilized to generate the publication-quality
schematics.

Author: Santiago Schnell
Contact: santiago.schnell@dartmouth.edu
Affiliation: Department of Mathematics, Dartmouth; Department of Biochemistry
    & Cell Biology, and Biomedical Data Sciences, Geisel School of Medicine.

Outputs:
    - Figure3_finkewatzky.pdf
    - Figure3_finkewatzky.png
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
apply_style()

# Sequential gradient palette denoting increasing ratios of nucleation-to-growth
case_colors = ["#FCAE91", "#DE2D26", "#67000D"]
accel_pos = "#4DAF4A"
accel_neg = "#E66101"
ref_gray = "#888888"

# ------------------------------------------------------------------------------
# 2. Mathematical Definition & Data Generation
# ------------------------------------------------------------------------------
A_tot = 1.0  # Total conserved quantity [A]0 + [B]0
k2 = 1.5  # Autocatalytic growth rate constant (M^-1 s^-1)

# Dimensionless mechanistic ratios to evaluate: k1 / (k2 * A_tot)
ratios = [1e-3, 1e-2, 1e-1]

t = np.linspace(0, 10, 2000)


def finke_watzky_rhs(B, tt, k1, k2_, At):
    """ODE system for the combined continuous nucleation and autocatalytic surface growth."""
    return (k1 + k2_ * B) * (At - B)


cases = []
# Iterate through the mechanistic ratios, solving the ODE and determining the acceleration landmark
for r in ratios:
    k1 = r * k2 * A_tot
    B = odeint(finke_watzky_rhs, 0.0, t, args=(k1, k2, A_tot)).ravel()

    # Explicit computation of derivatives for precision
    dBdt = (k1 + k2 * B) * (A_tot - B)
    d2Bdt2 = dBdt * (k2 * A_tot - 2 * k2 * B - k1)

    # Analytical zero-crossing constraint representing the kinetic shift
    B_star = A_tot / 2.0 - k1 / (2.0 * k2)
    t_star_idx = int(np.argmin(np.abs(B - B_star)))

    cases.append(
        {
            "k1": k1,
            "ratio": r,
            "B": B,
            "dBdt": dBdt,
            "d2Bdt2": d2Bdt2,
            "B_star": B_star,
            "t_star": t[t_star_idx],
        }
    )

# ------------------------------------------------------------------------------
# 3. Graphical Rendering
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
fig.subplots_adjust(wspace=0.38, top=0.93, bottom=0.17, left=0.08, right=0.985)

# --- Panel (a): Product concentration ---
ax = axes[0]
for c, col in zip(cases, case_colors):
    ax.plot(t, c["B"], color=col)
    # Mark the precise mathematical inflection point
    ax.plot(
        c["t_star"],
        c["B_star"],
        "o",
        color=col,
        markersize=6,
        markeredgecolor="k",
        markeredgewidth=0.7,
        zorder=5,
    )

ax.axhline(A_tot / 2, color=ref_gray, linestyle=":", linewidth=0.8, zorder=0)
ax.text(9.7, A_tot / 2 + 0.025, r"$A_\mathrm{tot}/2$", color=ref_gray, fontsize=8, ha="right")

# Annotate the B* offset dimension for the deepest-red curve to illustrate the shift
deepest = cases[-1]
x_bracket = 6.5
y_top = A_tot / 2.0
y_bot = deepest["B_star"]
ax.annotate(
    "",
    xy=(x_bracket, y_bot),
    xytext=(x_bracket, y_top),
    arrowprops=dict(
        arrowstyle="|-|,widthA=0.3,widthB=0.3", color="k", lw=0.8, shrinkA=0, shrinkB=0
    ),
)
ax.text(
    x_bracket + 0.25, (y_top + y_bot) / 2.0, r"$k_1/(2 k_2)$", fontsize=6, ha="left", va="center"
)
ax.plot(
    [deepest["t_star"] + 0.15, x_bracket],
    [y_bot, y_bot],
    color=ref_gray,
    lw=0.6,
    linestyle="-",
    zorder=1,
)

ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel(r"$[B]$ (M)")
ax.set_ylim(-0.03, 1.10)

# --- Panel (b): Reaction rate ---
ax = axes[1]
for c, col in zip(cases, case_colors):
    ax.plot(t, c["dBdt"], color=col)
    v_max_idx = int(np.argmax(c["dBdt"]))
    ax.plot(
        t[v_max_idx],
        c["dBdt"][v_max_idx],
        "o",
        color=col,
        markersize=6,
        markeredgecolor="k",
        markeredgewidth=0.7,
        zorder=5,
    )

ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel(r"$d[B]/dt$ (M s$^{-1}$)")

deepest = cases[-1]
v_max_idx = int(np.argmax(deepest["dBdt"]))
ax.annotate(
    r"$v_\mathrm{max}$",
    xy=(t[v_max_idx], deepest["dBdt"][v_max_idx]),
    xytext=(t[v_max_idx] + 0.9, deepest["dBdt"][v_max_idx] + 0.00005),
    fontsize=9,
    color="k",
)

# --- Panel (c): Acceleration ---
ax = axes[2]
for c, col in zip(cases, case_colors):
    ax.plot(t, c["d2Bdt2"], color=col)

deepest = cases[-1]
ax.fill_between(
    t, 0, deepest["d2Bdt2"], where=deepest["d2Bdt2"] >= 0, color=accel_pos, alpha=0.18, linewidth=0
)
ax.fill_between(
    t, 0, deepest["d2Bdt2"], where=deepest["d2Bdt2"] < 0, color=accel_neg, alpha=0.18, linewidth=0
)

for c, col in zip(cases, case_colors):
    ax.plot(
        c["t_star"],
        0,
        "o",
        color=col,
        markersize=6,
        markeredgecolor="k",
        markeredgewidth=0.7,
        zorder=5,
    )

ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Time, $t$ (s)")
ax.set_ylabel(r"$d^{2}[B]/dt^{2}$ (M s$^{-2}$)")

# --- Exterior Label Positioning & Formatting ---
letters = "abc"
label_x = -0.25
label_y = 1.05

for i, ax in enumerate(axes):
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
    ax.set_xlim(0, 10)

# ------------------------------------------------------------------------------
# 4. File Output Synthesis
# ------------------------------------------------------------------------------
outdir = Path(__file__).resolve().parents[2] / "outputs" / "figures"
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "Figure3_finkewatzky.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.savefig(outdir / "Figure3_finkewatzky.png", format="png", bbox_inches="tight", dpi=300)
