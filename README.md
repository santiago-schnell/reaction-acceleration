# Reaction Acceleration (Kinetic Curvature) - Python Tools

This repository provides lightweight Python utilities for **reaction acceleration analysis** from kinetic progress curves.

Reaction acceleration here means the **time derivative of the rate** (equivalently, the **second time derivative** of a calibrated progress variable). Practically, the workflow is:

1. **Smooth** a noisy progress curve (smoothing spline or Savitzky-Golay).
2. **Differentiate analytically** to obtain dy/dt (rate) and d2y/dt2 (acceleration).
3. **Extract curvature landmarks** (e.g., inflection times via zero crossings of d2y/dt2).
4. **Quantify uncertainty** in landmark times via a residual bootstrap.

The code is designed to be:
- Reviewer friendly: small dependency footprint (NumPy/SciPy; Matplotlib only for plots).
- Reproducible: deterministic examples, unit tests, and CI.
- Easy to read: short modules and explicit function names.


## Start here

### Install

Recommended (installs the core library plus Matplotlib for examples and figure scripts):

```bash
python -m pip install -U pip
python -m pip install -e ".[viz]"
```

If you only want the core analysis utilities (no plotting dependencies):

```bash
python -m pip install -e .
```


### Run the worked example (autocatalysis inflection time)

```bash
python examples/autocatalysis_landmark.py
```

This prints (i) a base-fit landmark estimate and (ii) a residual-bootstrap confidence interval, and writes a diagnostic plot to:

- `outputs/examples/`


### Regenerate the manuscript figures

```bash
python scripts/figures/figure1_thermodynamic.py
python scripts/figures/figure2_mechanisms.py
python scripts/figures/figure3_autocatalysis.py
python scripts/figures/figure4_oregonator.py
python scripts/figures/graphical_abstract.py
```

Figures are written to:

- `outputs/figures/`


### Run the unit tests

```bash
python -m pip install -e ".[dev]"
pytest
```


## Repository layout (quick map)

```text
reaction-acceleration/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
├─ .gitignore
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ src/
│  └─ reaction_acceleration/
│     ├─ __init__.py
│     ├─ derivatives.py          # smooth + differentiate
│     ├─ landmarks.py            # zero-crossing landmarks
│     ├─ bootstrap.py            # residual bootstrap CI
│     └─ cli.py                  # optional: ra-sanity entry point
├─ examples/
│  ├─ autocatalysis_landmark.py  # end-to-end demo + diagnostic plot
│  └─ README.md
├─ scripts/
│  └─ figures/
│     ├─ figure1_thermodynamic.py
│     ├─ figure2_mechanisms.py
│     ├─ figure3_autocatalysis.py
│     ├─ figure4_oregonator.py
│     ├─ graphical_abstract.py
│     └─ README.md
├─ tests/
│  ├─ test_derivatives.py
│  └─ test_bootstrap.py
└─ docs/
   ├─ reproducibility.md
   └─ methodology-notes.md
```

Notes:
- Generated outputs are written to `outputs/...` and are intentionally **not** version controlled (see `.gitignore`).
- The example and figure scripts create the `outputs/` subdirectories automatically.


## API in one minute

```python
import numpy as np

from reaction_acceleration import (
    estimate_derivatives,
    primary_zero_crossing_time,
    residual_bootstrap_landmark_ci,
)

# Example: inflection time from a generic progress curve y(t)

t = np.linspace(0.0, 6.0, 200)
y = ...  # concentration or calibrated signal (1D array)

# 1) Smooth and differentiate
# For splines, a practical starting point is s ~ N * sigma^2.
yhat, dy, d2y, model = estimate_derivatives(t, y, method="spline", s=0.02)

# 2) Landmark: first pos->neg zero crossing of acceleration
# (common operational definition for autocatalytic-like sigmoidal growth)
t_star = primary_zero_crossing_time(t, d2y, direction="pos_to_neg")

# 3) Bootstrap CI for the same landmark

def landmark_fn(t, yhat, dy, d2y):
    return primary_zero_crossing_time(t, d2y, direction="pos_to_neg")

est, lo, hi = residual_bootstrap_landmark_ci(
    t,
    y,
    landmark_fn=landmark_fn,
    method="spline",
    s=0.02,
    n_boot=500,
    seed=0,
)

print(f"t*: {est:.3f} s (95% CI [{lo:.3f}, {hi:.3f}])")
```


## Numerical stability and best practice

- Differentiation strongly amplifies noise; avoid finite differences on raw experimental data.
- For splines, choose the smoothing factor based on the expected noise level. A common starting point is:
  - `s ~ N * sigma^2`
  - for second-derivative landmarks, slightly higher smoothing (e.g., `1.2-2.0` times this value) is often more stable.
- Check landmark robustness:
  - vary smoothing strength by plus/minus 50% and confirm the landmark time is stable;
  - inspect residuals for structure (over-smoothing) or excessive high-frequency noise (under-smoothing);
  - if many bootstrap replicates fail to detect the landmark, treat the feature as unstable at current data quality.

See `docs/methodology-notes.md` for a short, practical discussion.


## License

GNU General Public License v3.0 (see `LICENSE`).


## Citation

If you use this software, please cite the accompanying Concept article and/or this repository (see `CITATION.cff`).
