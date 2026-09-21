"""Correlation-matrix constructors shared by simulation scripts."""

from __future__ import annotations

import numpy as np


def ar1_correlation(dimension: int, rho: float) -> np.ndarray:
  """Return the AR(1) correlation matrix with entries ``rho**|i-j|``."""
  power = np.abs(np.arange(dimension).reshape(-1, 1) - np.arange(dimension))
  return rho**power


def equicorrelation(dimension: int, rho: float) -> np.ndarray:
  """Return a constant-off-diagonal (equicorrelation) matrix."""
  return rho * np.ones((dimension, dimension)) + (1 - rho) * np.eye(dimension)


# Existing scripts historically used these names.  Keep aliases while new code
# uses names that describe the returned correlation structure.
cor_ar_1 = ar1_correlation
cor_fixed = equicorrelation


__all__ = ["ar1_correlation", "cor_ar_1", "cor_fixed", "equicorrelation"]
