# Worked examples

This folder contains runnable end-to-end examples that demonstrate the
workflow from noisy synthetic data to a landmark time with a confidence
interval. They are intended to be read alongside the Supporting
Information of the paper.

## Files

- **`autocatalysis_landmark.py`** — canonical autocatalytic progress
  curve (A + B → 2 B). Simulates noisy data, smooths, differentiates,
  locates the inflection-point landmark, and quantifies its uncertainty
  with a residual bootstrap. Implements the canonical workflow discussed
  in Supporting Information §4.4, §7.6, and §8.3.

## Running the examples

From the repository root:

```bash
python examples/autocatalysis_landmark.py
```

Expected terminal output:

```
====================================================================
Autocatalysis landmark: acceleration zero-crossing near v_max
====================================================================
  True inflection (theory)  : 2.5945 s
  --- Cautionary: fixed rule s = 2 N sigma^2 ---
  Smoothing factor (s)      : 2.0000e-02
  Base-fit estimate (t*)    : 2.4324 s
  Bootstrap base estimate   : 2.4324 s
  95% CI (percentile)       : [2.2302, 2.6145] s
  CI contains truth         : yes
  --- Recommended: GCV P-spline (data-driven penalty) ---
  GCV-selected lambda       : 6.5513e+00
  Base-fit estimate (t*)    : 2.5677 s
  95% CI (percentile)       : [2.4533, 2.6816] s
  CI contains truth         : yes
====================================================================
```

A diagnostic plot is also written to `outputs/examples/`. The plot shows
the noisy data, the smoothed fit, the estimated acceleration, and the
located inflection time.

## Adapting these examples to your own data

The example is written deliberately as a short, linear script so that
each step is easy to replace:

1. **Replace the data-generation step** (`sol = odeint(...)`) with code
   that loads your own `t` and `y` arrays.
2. **Use the recommended GCV estimator** (`method="gcv"`) when the aim is
   quantitative landmark recovery. The fixed rule
   `s = 2 * len(t) * sigma**2` is retained as a transparent sensitivity and
   cautionary comparison.
3. **Choose a landmark function** appropriate to your system. The
   supplied `landmark_inflection` wraps `acceleration_zero_crossing_time`
   and is tailored to autocatalytic sigmoids; for the intermediate of a
   consecutive reaction you would instead call
   `acceleration_zero_crossing_time(t, dy, d2y, direction="neg_to_pos")`,
   which anchors the crossing on the rate minimum.

See the main `README.md` and `docs/methodology-notes.md` for further
guidance.
