# Reproducibility

This repository includes scripts to regenerate the manuscript figures.

## Installation

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e ".[viz]"
```

## Regenerate figures

Run (from the repository root):

```bash
python scripts/figures/figure1_thermodynamic.py
python scripts/figures/figure2_mechanisms.py
python scripts/figures/figure3_autocatalysis.py
python scripts/figures/figure4_oregonator.py
python scripts/figures/graphical_abstract.py
```

Outputs are written to:

- `outputs/figures/`

## Notes

- The figure scripts are deterministic (no RNG) except where explicitly stated.
- The Oregonator script uses a dense time grid to resolve stiff relaxation oscillations; runtime depends on your machine.
