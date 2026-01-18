"""reaction_acceleration

Lightweight tools for estimating reaction acceleration (second derivatives) from
kinetic progress curves.

Public API
----------
- estimate_derivatives
- primary_zero_crossing_time
- residual_bootstrap_landmark_ci

The package is intentionally small and focuses on reproducible workflows.
"""

from .derivatives import estimate_derivatives
from .landmarks import primary_zero_crossing_time
from .bootstrap import residual_bootstrap_landmark_ci

__all__ = [
    "estimate_derivatives",
    "primary_zero_crossing_time",
    "residual_bootstrap_landmark_ci",
]

__version__ = "0.1.0"
