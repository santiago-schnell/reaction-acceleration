# Methodology notes: derivative estimation and numerical stability

Reaction acceleration analysis is fundamentally a **second-derivative** problem. Numerical differentiation amplifies noise strongly, so the recommended workflow is:

1. **Smooth first** (fit a spline or local polynomial)
2. **Differentiate analytically** (differentiate the fitted smooth representation)
3. **Extract landmarks** (zero-crossings, extrema)
4. **Quantify uncertainty** (bootstrap)

## Smoothing splines

For irregular sampling, cubic smoothing splines are recommended. In SciPy this is implemented via `scipy.interpolate.UnivariateSpline`.

A practical heuristic is to set the smoothing factor

- `s ≈ N * σ^2`

where `N` is the number of observations and `σ` is the (estimated) measurement noise standard deviation.

For derivative-heavy analysis, slightly more smoothing (e.g., `s = 1.2–2.0 * N * σ^2`) can improve second-derivative stability.

## Savitzky–Golay

For **uniformly sampled** data, Savitzky–Golay filtering can be convenient, but it is more sensitive to sampling irregularities and boundary effects.

## Landmark robustness

Recommended checks before reporting an inflection time:

- **Sensitivity sweep:** vary the smoothing strength by ±50% and confirm that the landmark time remains stable.
- **Residual inspection:** residuals should look like noise (no clear temporal structure).
- **Bootstrap stability:** if many bootstrap replicates fail to detect the landmark, the feature is not robust at the current data quality.

## Conditioning and stiffness

- Derivative estimation becomes ill-conditioned when the sampling interval is very small relative to the noise scale.
- For stiff kinetic models (e.g., relaxation oscillators), simulated time series can have sharp features; ensure dense sampling in the fast phases.

