"""Small command-line entry points.

Currently provides a single command:

- `ra-sanity`: runs a minimal deterministic sanity check.

The full test suite should be run via `pytest`.
"""

from __future__ import annotations

import sys

import numpy as np

from .derivatives import estimate_derivatives
from .landmarks import primary_zero_crossing_time


def main(argv: list[str] | None = None) -> int:
    """Run a minimal deterministic sanity test.

    Returns an exit code suitable for shell use.
    """
    _ = argv  # reserved for future options

    try:
        t = np.linspace(0.0, 3.0, 200)
        k = 1.7
        y = np.exp(-k * t)

        # Exact (noise-free) data: use interpolating spline
        yhat, dy, d2y, _ = estimate_derivatives(t, y, method="spline", s=0.0)

        d2y_true = (k**2) * np.exp(-k * t)
        rmse = float(np.sqrt(np.mean((d2y - d2y_true) ** 2)))

        # No zero crossing expected (all positive)
        t0 = primary_zero_crossing_time(t, d2y, direction="pos_to_neg")

        print("Sanity check:")
        print(f"  d2y RMSE: {rmse:.3e}")
        print(f"  first pos->neg zero crossing: {t0}")

        if rmse > 1e-2:
            print("FAIL: derivative RMSE too large.")
            return 1
        if np.isfinite(t0):
            print("FAIL: detected a spurious zero crossing.")
            return 1

        print("PASS")
        return 0

    except Exception as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
