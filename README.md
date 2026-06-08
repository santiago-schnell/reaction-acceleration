# Reaction Acceleration — Python Tools

Companion code for the research article
**"Reaction Acceleration: Reviving the Second Derivative in Chemical
Kinetics"** (Schnell, *ChemSystemsChem*, submitted 2026).

This repository lets you take a noisy concentration-vs-time curve from a
chemical experiment and recover the **acceleration** of the reaction —
the second derivative of the progress variable — together with uncertainty estimates
conditional on the chosen smoothing model. The acceleration is the hidden diagnostic that
tells you *how a reaction is changing*: when it speeds up, when it slows
down, and where the transitions sit.

---

## Who is this for?

- **You have kinetic time-course data** (e.g., spectroscopy, pH,
  fluorescence) and want to locate inflection points and other curvature
  features rigorously, with confidence intervals rather than
  eyeballed estimates.
- **You are reviewing or extending the paper** and want to reproduce
  the manuscript figures, the test suite, and the canonical worked example.
- **You teach or learn kinetics** and want worked Python examples of the
  methods in the Supporting Information.

You do **not** need to be a Python expert. The instructions below assume
you can open a terminal and type commands; they explain what each
command does so you are never typing something blindly.

---

## What you will need

One of:

- **macOS or Linux** with a recent Python installation (3.9 or later).
- **Windows 10 / 11** with Python installed from
  [python.org](https://www.python.org/downloads/) or via the Windows
  Store.

To check that Python is available, open a terminal (Terminal on macOS,
PowerShell or Command Prompt on Windows, any terminal on Linux) and type:

```bash
python --version
```

You should see something like `Python 3.11.4`. If you see
`Python 2.x` or the command is not found, install Python 3.9+ before
continuing.

---

## Step 1 — Get the code

### Option A: download a ZIP (simplest)

1. On the repository page on GitHub, click the green **Code** button.
2. Choose **Download ZIP**.
3. Unzip it to a folder you can find again. The folder will be called
   something like `reaction-acceleration-main`.

### Option B: clone with git (if you already use git)

```bash
git clone https://github.com/santiago-schnell/reaction-acceleration.git
cd reaction-acceleration
```

---

## Step 2 — Install the package

Change into the repository folder, then install:

```bash
cd reaction-acceleration-main      # or whatever you called the folder
python -m pip install -U pip
python -m pip install -e ".[viz]"
```

**What each command does:**

- `cd reaction-acceleration-main` — moves your terminal *into* the
  repository folder. Every command below assumes you are here.
- `python -m pip install -U pip` — updates `pip` (Python's package
  manager) to the latest version. Not strictly required, but it
  prevents many confusing error messages.
- `python -m pip install -e ".[viz]"` — installs this package along with
  Matplotlib, which you need for the example and figure scripts. The
  `-e` flag means "editable": changes you make to the source will take
  effect without reinstalling.

If you only want the core library without plotting, use
`pip install -e .` instead. The tests and the example both require the
`[viz]` extras.

### Expected output

The last command prints a few dozen lines ending with something like:

```
Successfully installed reaction-acceleration-0.5.0 ...
```

---

## Step 3 — Run the worked example

The simplest way to see everything working end-to-end is the autocatalysis
worked example:

```bash
python examples/autocatalysis_landmark.py
```

### What this does

1. Simulates a noisy autocatalytic progress curve
   (A + B → 2 B, the canonical case from the Supporting Information).
2. Smooths it with a spline and analytically differentiates twice to
   obtain the reaction acceleration.
3. Locates the inflection point (where the acceleration crosses zero).
4. Quantifies uncertainty with a residual bootstrap.
5. Writes a diagnostic plot to `outputs/examples/`.

### Expected output

```
====================================================================
Autocatalysis landmark: acceleration zero-crossing near v_max
====================================================================
  True inflection (theory)  : 2.5945 s
  Smoothing factor (s)      : 2.0000e-02
  Base-fit estimate (t*)    : 2.4324 s
  Bootstrap base estimate   : 2.4324 s
  95% CI (percentile)       : [2.2302, 2.6537] s
  CI contains truth         : yes
====================================================================

Saved diagnostic plot: /path/to/outputs/examples/autocatalysis_landmark.png
```

The "True inflection" is the analytical value from theory (2.5945 s for
these parameters). The base-fit and bootstrap estimates should be close
to that, and the 95% confidence interval should usually (though not
always — this is a statistical property) contain the truth.

Open the PNG in `outputs/examples/` to see the smoothed curve, the
estimated acceleration, and the inflection point marked.

---

## Step 4 — Regenerate the manuscript figures

```bash
python scripts/figures/figure1_thermodynamic.py
python scripts/figures/figure2_mechanisms.py
python scripts/figures/figure3_finkewatzky.py
python scripts/figures/figure4_oregonator.py
python scripts/figures/figure5_benchmark.py
python scripts/figures/graphical_abstract.py
```

Each script creates PDF and PNG versions in `outputs/figures/`. Figure 5 reads the benchmark CSV in `data/benchmarks/`, which is written by `scripts/si/verification_table.py`; if the CSV is absent, it regenerates the benchmark first. Running all six scripts takes about a minute on a modern laptop when the benchmark CSV is already present.

---

## Step 5 — Run the test suite

```bash
python -m pip install -e ".[dev]"
pytest
```

The first command installs testing dependencies; you only need to do it
once. The second runs every test.

### Expected output

```
.....................                                                    [100%]
26 passed in 4.32s
```

If any test fails, it is likely because your NumPy or SciPy version
differs from the one used to develop the code. The canonical
autocatalysis regression test has been written to tolerate minor
numerical drift between versions; a hard failure elsewhere is worth
investigating.

---

## Using the library in your own code

After `pip install -e .`, the public API is importable from the top-level
package. The package exports four functions — `estimate_derivatives`,
`primary_zero_crossing_time`, `acceleration_zero_crossing_time`, and
`residual_bootstrap_landmark_ci`; the example below uses three of them:

```python
import numpy as np
from reaction_acceleration import (
    estimate_derivatives,
    acceleration_zero_crossing_time,
    residual_bootstrap_landmark_ci,
)

# Your experimental data
t = np.linspace(0.0, 6.0, 200)    # time in seconds
y = ...                           # concentration, 1-D array of length 200

# 1) Smooth and differentiate
#    A practical starting point for the smoothing parameter s is
#    2 * n * sigma^2, where sigma is the noise standard deviation.
n = len(t)
sigma = 0.01                      # noise estimate for your instrument
s = 2.0 * n * sigma**2
yhat, dy, d2y, _model = estimate_derivatives(t, y, method="spline", s=s)

# 2) Locate the inflection point. For sigmoidal (autocatalytic) curves
#    use acceleration_zero_crossing_time: it returns the positive-to-negative
#    acceleration crossing nearest the rate maximum, which is robust to the
#    spurious early zero-crossings that a naive "first crossing" picks up in
#    noisy second-derivative traces.
t_star = acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")

# 3) Bootstrap confidence interval
def landmark_fn(t, yhat, dy, d2y):
    return acceleration_zero_crossing_time(t, dy, d2y, direction="pos_to_neg")

est, lo, hi = residual_bootstrap_landmark_ci(
    t, y,
    landmark_fn=landmark_fn,
    method="spline",
    s=s,
    n_boot=500,
    seed=0,           # fixing the seed makes the CI reproducible
)

print(f"t* = {est:.3f} s (95% CI [{lo:.3f}, {hi:.3f}])")
```

### Diagnostic mode

If you want to see how many bootstrap replicates actually succeeded:

```python
est, lo, hi, diag = residual_bootstrap_landmark_ci(
    t, y,
    landmark_fn=landmark_fn,
    s=s,
    n_boot=500,
    return_diagnostics=True,
)

print(f"Successful replicates: {diag['n_success']} / 500")
print(f"Failure fraction: {diag['fail_fraction']:.1%}")
print(f"Bootstrap mean / std: {diag['bootstrap_mean']:.4f} / {diag['bootstrap_std']:.4f}")
```

If more than 20% of replicates fail, the library returns `NaN` for the
CI bounds and emits a logging warning — the landmark is not statistically
stable at that data quality.

---

## Common errors and what to do about them

| Error message | What it means | Fix |
|---|---|---|
| `python: command not found` | Python is not installed, or not on your PATH | Install Python 3.9+ from python.org; on Windows, tick "Add Python to PATH" during install |
| `ModuleNotFoundError: No module named 'reaction_acceleration'` | You are outside the repo folder, or `pip install` was skipped | `cd` into the repo folder and rerun `pip install -e ".[viz]"` |
| `ValueError: t must be strictly increasing` | Your time array has duplicates or is unsorted | Sort your data and remove duplicates; in a pinch, `numpy.unique(t, return_index=True)` helps |
| `ValueError: Length mismatch: t (...) vs y (...)` | Your `t` and `y` arrays are different sizes | Trim to the shorter, or align from the raw instrument file |
| `ValueError: Input contains NaN or infinite values` | Missing or corrupt data | Remove or interpolate the bad samples before calling |
| `ValueError: ... requires (approximately) uniform sampling` | You asked for Savitzky–Golay on irregular data | Use `method="spline"` instead |
| Bootstrap returns `NaN` bounds plus a warning | More than 20% of replicates failed; landmark is unstable | Reduce noise, reduce smoothing, or accept that the feature is not reliably detectable |

---

## Repository layout

```
reaction-acceleration/
├─ README.md             <-- you are here
├─ LICENSE
├─ CITATION.cff
├─ pyproject.toml        <-- package metadata
├─ requirements.txt
├─ requirements-dev.txt
├─ .gitignore
├─ .github/
│  └─ workflows/
│     └─ ci.yml          <-- continuous-integration config
├─ src/
│  └─ reaction_acceleration/
│     ├─ __init__.py
│     ├─ derivatives.py          <-- smooth + differentiate
│     ├─ landmarks.py            <-- zero-crossing landmarks
│     ├─ bootstrap.py            <-- residual bootstrap CI
│     └─ cli.py                  <-- optional: ra-sanity entry point
├─ data/
│  └─ benchmarks/
│     └─ autocatalysis_operating_characteristics.csv  <-- source data for Figure 5 / SI Table 1
├─ examples/
│  ├─ autocatalysis_landmark.py  <-- end-to-end demo + diagnostic plot
│  └─ README.md
├─ scripts/
│  ├─ figures/
│  │  ├─ _style.py               <-- shared matplotlib style
│  │  ├─ figure1_thermodynamic.py
│  │  ├─ figure2_mechanisms.py
│  │  ├─ figure3_finkewatzky.py
│  │  ├─ figure4_oregonator.py
│  │  ├─ figure5_benchmark.py
│  │  ├─ graphical_abstract.py
│  │  └─ README.md
│  └─ si/
│     ├─ _si_common.py            <-- shared canonical simulation setup
│     ├─ smoothing_table.py       <-- SI Sec. 6.6
│     ├─ bootstrap_table.py       <-- SI Sec. 7.3
│     └─ verification_table.py    <-- SI Sec. 8.3 / Figure 5 data
├─ test/
│  ├─ test_derivatives.py
│  ├─ test_landmarks.py
│  └─ test_bootstrap.py
└─ docs/
   ├─ reproducibility.md
   └─ methodology-notes.md
```

Generated outputs are written to `outputs/…` and are intentionally not
version controlled. A fresh checkout therefore will not contain an
`outputs/` directory. The example and figure scripts create the following
ignored directories automatically when needed:

```
outputs/
├─ examples/      # diagnostic plots from examples/
└─ figures/       # regenerated manuscript figures and graphical abstract
```

The small seeded benchmark CSV used by Figure 5 and SI Table 1 is treated
as source data and is version controlled under `data/benchmarks/` rather
than under `outputs/`.

---

## Glossary

- **Progress curve.** The measured concentration (or a calibrated
  observable) as a function of time, typically sigmoidal or
  monotonically approaching a plateau.
- **Rate.** The first time derivative of the progress curve,
  `dy/dt` (units of concentration per unit time).
- **Acceleration.** The second time derivative, `d²y/dt²`. The sign
  tells you whether the rate is rising (+) or falling (−); a
  zero-crossing is the inflection point of the progress curve.
- **Landmark.** A geometric feature of the curve — an inflection time,
  a rate maximum, the midpoint of a logistic rise — that encodes
  mechanism-specific information.
- **Smoothing spline.** A piecewise-cubic function fitted to noisy data
  with a tunable trade-off between data fidelity and smoothness. Here
  controlled by the parameter `s`; a useful starting point is
  `s ≈ 2·n·σ²`.
- **Residual bootstrap.** A non-parametric uncertainty-quantification
  method: fit the data, resample the fit's residuals to create synthetic
  datasets, refit each one, and read off percentile confidence intervals
  from the distribution of refitted landmarks. These intervals quantify
  variance conditional on the chosen smoother; smoothing-induced bias must
  be assessed by sensitivity analysis or simulation calibration.
- **Inflection point.** The time at which the curve's concavity changes
  sign, equivalently where the acceleration is zero. For autocatalytic
  sigmoidal curves this often coincides with the maximum rate.

For the full mathematical context, see `docs/methodology-notes.md` and
the Supporting Information of the paper.

---

## Citation

If you use this software in published work, please cite the accompanying
research article and/or this repository. A machine-readable citation
is in `CITATION.cff`.

---

## License

GNU General Public License v3.0 — see `LICENSE`.


## Archival release

For publication, tag the repository as `v0.5.0` and archive that exact tag
through Zenodo or an equivalent service. Replace the mutable GitHub-only
citation in the manuscript with the archived release DOI once it is minted;
until then, the `CITATION.cff` file records the submitted version and the
canonical repository URL without inventing a DOI.
