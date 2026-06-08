"""reaction_acceleration

Lightweight tools for estimating reaction acceleration (second derivatives) from
kinetic progress curves.

Public API
----------
- estimate_derivatives
- primary_zero_crossing_time
- acceleration_zero_crossing_time
- residual_bootstrap_landmark_ci

The package is intentionally small and focuses on reproducible workflows.
"""

from .bootstrap import residual_bootstrap_landmark_ci
from .derivatives import estimate_derivatives
from .landmarks import acceleration_zero_crossing_time, primary_zero_crossing_time

__all__ = [
    "acceleration_zero_crossing_time",
    "estimate_derivatives",
    "primary_zero_crossing_time",
    "residual_bootstrap_landmark_ci",
]

__version__ = "0.5.0"
