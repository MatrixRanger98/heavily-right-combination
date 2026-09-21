"""Compatibility imports for the historical distribution module.

New code should import from :mod:`heavily_right.null_laws`. The implementations now live
in focused private modules; these names remain to avoid breaking older scripts.
"""

from __future__ import annotations

import numpy as np

from ._constants import EULER_MASCHERONI
from ._finite_null_laws import HalfCauchyMean, HarmonicMean
from ._landau import Landau
from ._special_functions import scaled_exponential_integral

euler_gamma = EULER_MASCHERONI


def expi_over_exp(
  x: float | np.ndarray,
  n: int = 150,
) -> float | np.ndarray:
  """Compatibility name for the scaled exponential integral.

  ``n`` is retained solely for callers of the former fixed-term series.
  """
  del n
  return scaled_exponential_integral(x)


# Historical lowercase class names remain exact aliases.
hcauchy_mean = HalfCauchyMean
harmonic_mean = HarmonicMean
landau = Landau


__all__ = [
  "HalfCauchyMean",
  "HarmonicMean",
  "Landau",
  "euler_gamma",
  "expi_over_exp",
  "harmonic_mean",
  "hcauchy_mean",
  "landau",
]
