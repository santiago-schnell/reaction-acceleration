# Methodology notes: derivative estimation and numerical stability

This note is a practitioner's summary of the methods implemented in
`reaction_acceleration`. For the full mathematical treatment, derivations,
worked numerical examples, and verification output, see the Supporting
Information (SI) of the accompanying paper; specific pointers are given
below.

---

## The core problem

Reaction acceleration analysis is fundamentally a **second-derivative**
problem applied to noisy experimental data. The naive approach, finite
differences of raw measurements, fails dramatically because differentiation
amplifies noise. If the noise standard deviation on the signal is $\sigma$
and the sampling interval is $\Delta t$, then the naive second-difference
estimator has variance

$$
\mathrm{Var}(\hat{x}'') = \frac{6\sigma^2}{\Delta t^4}.
$$

The $\Delta t^{-4}$ dependence means that denser sampling makes the naive
estimator *worse*, not better. See SI §7.1 for the derivation.

The recommended workflow is therefore:

1. **Smooth first**: fit a spline or Savitzky–Golay filter to the data.
2. **Differentiate analytically**: obtain $\mathrm{d}\hat{y}/\mathrm{d}t$
   and $\mathrm{d}^2\hat{y}/\mathrm{d}t^2$ from the fitted representation.
3. **Extract landmarks**: zero crossings, extrema, inflection points.
4. **Quantify uncertainty**: residual bootstrap over the fit.

This is exactly what `estimate_derivatives`, `primary_zero_crossing_time`
/ `acceleration_zero_crossing_time` (the latter recommended for sigmoidal
curves, as it anchors the crossing on the rate maximum), and
`residual_bootstrap_landmark_ci` implement.

---

### Data-driven penalty selection (recommended): `method="gcv"`

Rather than fixing `s` by a rule of thumb, the package can select the
smoothing penalty directly from the data by **generalised cross-validation
(GCV)**, exposed as `estimate_derivatives(..., method="gcv")`. This fits a
penalized B-spline (P-spline; Eilers & Marx 1996) with a second-difference
penalty and chooses the penalty $\lambda$ that minimises

$$
\mathrm{GCV}(\lambda) = \frac{n\,\lVert y - \hat{x}_\lambda\rVert^2}{(n - \mathrm{tr}\,S_\lambda)^2},
$$

where $S_\lambda$ is the smoother (hat) matrix. Because the basis depends
only on the time grid, a single Demmler–Reinsch diagonalisation makes both
GCV selection and bootstrap refits inexpensive. A quartic basis ($k=4$) is
the default, giving a piecewise-quadratic — rather than piecewise-linear —
second derivative.

On the canonical autocatalysis benchmark, GCV removes essentially all of
the fixed rule's landmark bias (point-estimate bias falls from roughly
$-0.11$ to $-0.15$ s down to within $\pm 0.03$ s across $\sigma/\text{range}
\in [0.5\%, 5\%]$) and restores 95% bootstrap-CI coverage to its nominal
level (SI §9.3). It is the recommended choice for quantitative landmark
work. The fixed-`s` rule below is retained as a transparent, easily audited
fallback and as the cautionary example in SI §7.6.

## Smoothing splines (fixed penalty): `method="spline"`

For irregularly or regularly sampled data, cubic smoothing splines
implemented via `scipy.interpolate.UnivariateSpline` minimise

$$
\sum_{i=1}^n w_i\,(y_i - \hat{x}(t_i))^2 + \lambda \int (\hat{x}''(t))^2\,\mathrm{d}t.
$$

The SciPy parameter `s` controls the fidelity-smoothness trade-off (it
is related to $\lambda$ via the data weights).

### Choosing `s`

The practical heuristic is
$$
s \approx n\,\sigma^2,
$$
where $n$ is the number of observations and $\sigma$ is the estimated
measurement-noise standard deviation. This corresponds to expecting the
sum of squared residuals to equal the expected total noise variance.

For **second-derivative landmark analysis**, slightly more smoothing
improves stability. The recommended starting point is
$$
s = 2\,n\,\sigma^2,
$$
which is the starting point used in SI §7.6 and in the bootstrap example of §8.3. A sensitivity sweep
with $s \in [n\sigma^2,\ 3\,n\sigma^2]$ should show the landmark time stable
to within its bootstrap CI.

### When to vary `s`

- **Landmark drifts substantially** as `s` is changed → landmark is not
  robust; collect more data, reduce noise, or reconsider the mechanism.
- **Residuals show systematic structure** → over-smoothing; reduce `s`.
- **Second derivative has many spurious sign changes** → under-smoothing;
  increase `s`.

### Known bias at high noise

The fixed rule $s = 2n\sigma^2$ starts to over-smooth at relative noise
levels of $\sigma/\text{range} \geq 2\%$, introducing a bias of up to
0.2 s in the autocatalysis inflection time (SI §8.3). In this regime,
either use a data-driven criterion such as generalised cross-validation
to select `s`, or widen the sensitivity sweep and report the landmark
range rather than a single value. The residual bootstrap quantifies
variance but not smoothing-induced bias, so a narrow CI at high noise
is not a guarantee of accuracy.

---

## Savitzky–Golay (uniform grids only)

For strictly uniformly sampled data, Savitzky–Golay local-polynomial
filtering (`scipy.signal.savgol_filter`) is a lightweight alternative:

- window length: 11–21 points for moderate-noise kinetic data
- polynomial order: 3 (cubic) for second derivatives

The method is less tolerant of sampling irregularities and boundary
effects than smoothing splines. `estimate_derivatives(method="savgol")`
will raise `ValueError` if the time grid is non-uniform; use the spline
method in that case.

---

## Residual bootstrap for landmark uncertainty

Once a landmark time $\hat{t}^\ast$ has been extracted from the smoothed
fit, the residual bootstrap quantifies its uncertainty without needing
analytical derivatives of the landmark with respect to the data:

1. Fit and record residuals $r_i = y_i - \hat{x}(t_i)$.
2. For $b = 1,\ldots,B$: resample residuals with replacement, form
   $y_i^{(b)} = \hat{x}(t_i) + r_i^\ast$, refit, re-extract
   $\hat{t}^{\ast(b)}$.
3. Report the 95% percentile confidence interval
   $[\hat{t}^\ast_{2.5\%},\ \hat{t}^\ast_{97.5\%}]$.

$B = 500$ replicates is typically sufficient. The
`residual_bootstrap_landmark_ci` implementation returns NaN bounds and
emits a warning if more than 20% of bootstrap replicates fail to detect
the landmark (feature unstable at current data quality). Pass
`return_diagnostics=True` to inspect the success/failure breakdown.

See SI §8 for the full bootstrap algorithm and §8.3 for a worked numerical example on the canonical autocatalytic data set.

---

## Verification: before trusting a landmark

Recommended checks before reporting an acceleration-derived landmark:

- **Sensitivity sweep.** Vary `s` by ±50% and confirm that the landmark
  time remains within the bootstrap CI.
- **Residual inspection.** Plot residuals versus time and versus fitted
  value; both should look like random noise.
- **Bootstrap failure rate.** A high failure rate (≥20%) indicates an
  unstable landmark; do not report a CI.
- **Multi-species cross-check.** If multiple species are measured,
  their inflection times should be consistent with the mechanism's
  stoichiometry (all species share the same reaction-acceleration
  zero-crossing time).

SI §8 specifies an extended verification grid across mechanisms, noise
levels, sampling densities, and sampling patterns. SI §9.3 reports
sample bias, RMSE, coverage, and detection-rate numbers that can be
used as benchmarks.

---

## Conditioning and stiffness

- Derivative estimation becomes ill-conditioned when $\Delta t$ is very
  small relative to the noise scale. If the noise is near the bit-level
  of the instrument, smoothing parameters should be chosen from the
  upper end of the recommended range.
- For stiff kinetic models (e.g., the relaxation oscillator of the main
  text, Figure 4), simulated time series contain sharp transitions. Sample
  densely across the fast phases and verify that the spline tracks the
  transitions without introducing Gibbs-like oscillations.

---

## Pointers into the Supporting Information

| Topic | SI section |
|---|---|
| Noise-amplification derivation | §7.1 |
| Spline vs Savitzky–Golay | §7.3 and §7.4 |
| Smoothing-parameter heuristic | §7.5 |
| Smoothing-parameter worked example | §7.6 |
| Residual bootstrap algorithm | §8.1 |
| Bootstrap worked example | §8.3 |
| Verification framework | §8 |
| Sample verification output (bias/RMSE/coverage) | §9.3 |
| Full Python implementation details | §9 |
