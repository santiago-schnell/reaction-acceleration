# Manuscript figure scripts

These scripts regenerate every figure in the paper and the graphical
abstract. They are deterministic (no randomness) and produce both PDF
(for the journal) and PNG (for quick inspection).

## Files

| Script | Figure | Content |
|---|---|---|
| `figure1_thermodynamic.py` | Figure 1 | Affinity → velocity → acceleration cascade for a first-order approach to equilibrium |
| `figure2_mechanisms.py` | Figure 2 | Side-by-side concentration, rate, and acceleration for four canonical mechanisms |
| `figure3_finkewatzky.py` | Figure 3 | Finke–Watzky two-step autocatalysis: landmark shift as a readout of nucleation-to-growth ratio |
| `figure4_oregonator.py` | Figure 4 | Oregonator relaxation oscillations with model-derived acceleration |
| `figure5_benchmark.py` | Figure 5 | Operating characteristics for acceleration-landmark recovery |
| `graphical_abstract.py` | Graphical Abstract | One-panel summary |

## Shared style

`_style.py` defines a shared Matplotlib style (fonts, sizes, dpi, tick
behaviour) used by every figure script. Changing a font or size in one
place propagates to every panel in the paper.

## Running

From the repository root (so that outputs land in the repository's
`outputs/figures/` directory):

```bash
python scripts/figures/figure1_thermodynamic.py
python scripts/figures/figure2_mechanisms.py
python scripts/figures/figure3_finkewatzky.py
python scripts/figures/figure4_oregonator.py
python scripts/figures/figure5_benchmark.py
python scripts/figures/graphical_abstract.py
```

Each script takes a few seconds on a modern laptop; the Oregonator is
the slowest because it integrates a stiff ODE. Figure 5 reads the seeded
verification metrics from `data/benchmarks/autocatalysis_operating_characteristics.csv`.
If that CSV is absent, `figure5_benchmark.py` regenerates it by calling the
same benchmark routine used by `scripts/si/verification_table.py`, so the
main-text figure and Supporting-Information table have a single numerical
source of truth.

## Requirements

```bash
python -m pip install -e ".[viz]"
```

This installs Matplotlib in addition to the core library. If you see an
`ImportError` for Matplotlib, this extras install was missed.

## Output location

```
outputs/figures/
├─ Figure1_thermodynamic.pdf
├─ Figure1_thermodynamic.png
├─ Figure2_mechanisms.pdf
├─ Figure2_mechanisms.png
├─ Figure3_finkewatzky.pdf
├─ Figure3_finkewatzky.png
├─ Figure4_oregonator.pdf
├─ Figure4_oregonator.png
├─ Figure5_benchmark.pdf
├─ Figure5_benchmark.png
├─ Graphical_Abstract.pdf
└─ Graphical_Abstract.png
```

The `outputs/` directory is not version controlled (see `.gitignore`);
the scripts create it automatically. The benchmark CSV used by Figure 5 is
version controlled under `data/benchmarks/` because it is the small, seeded
source-data table for both Figure 5 and the Supporting-Information table.
