"""Public finite-sample and asymptotic null-law API.

Finite-m inversions, numerical safeguards, special functions, and the Landau
approximation live in dedicated implementation modules behind this public API.
"""

from ._constants import EULER_MASCHERONI
from ._finite_null_laws import HalfCauchyMean, HarmonicMean
from ._landau import Landau
from ._numerics import (
  NumericalCalibrationError,
  NumericalIntegrationError,
  NumericalRootError,
)
from ._special_functions import scaled_exponential_integral

euler_gamma = EULER_MASCHERONI

__all__ = [
  "HalfCauchyMean",
  "HarmonicMean",
  "Landau",
  "NumericalCalibrationError",
  "NumericalIntegrationError",
  "NumericalRootError",
  "euler_gamma",
  "scaled_exponential_integral",
]
