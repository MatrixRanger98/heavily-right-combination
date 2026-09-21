"""Overflow-safe special functions used by finite-m null laws."""

from __future__ import annotations

import numpy as np
from scipy.special import expi


def scaled_exponential_integral(x: float | np.ndarray) -> float | np.ndarray:
  """Return ``exp(-x) * Ei(x)`` without large-argument overflow.

  Moderate positive arguments use SciPy's implementation directly. Arguments
  at least 50 use the optimally truncated asymptotic expansion
  ``sum(k! / x**(k+1))``.
  """
  values = np.asarray(x, dtype=float)
  if np.any(values <= 0.0):
    raise ValueError("scaled exponential integral requires positive x")
  result = np.empty_like(values)
  regular = values < 50.0
  result[regular] = np.exp(-values[regular]) * expi(values[regular])
  for index in np.ndindex(values.shape):
    if regular[index]:
      continue
    value = float(values[index])
    total = term = 1.0 / value
    previous = abs(term)
    for order in range(1, 80):
      term *= order / value
      magnitude = abs(term)
      if magnitude > previous:
        break
      total += term
      if magnitude <= 2.0e-16 * abs(total):
        break
      previous = magnitude
    result[index] = total
  return float(result) if values.ndim == 0 else result


__all__ = ["scaled_exponential_integral"]
