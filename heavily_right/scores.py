"""Reusable transforms from p-values to heavy-right-tailed scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PValueInput = float | Sequence[float] | np.ndarray
ScoreResult = float | FloatArray


def _validated_pvalues(p_values: PValueInput) -> FloatArray:
  values = np.asarray(p_values, dtype=float)
  if not np.isfinite(values).all():
    raise ValueError("p-values must be finite")
  if np.any((values < 0.0) | (values > 1.0)):
    raise ValueError("p-values must lie in [0, 1]")
  return values


def _scalar_or_array(values: FloatArray) -> ScoreResult:
  return float(values) if values.ndim == 0 else values


def half_cauchy_score(p_values: PValueInput) -> ScoreResult:
  """Return ``cot(pi * p / 2)`` elementwise.

  This positive score maps ``p=0`` to positive infinity and ``p=1`` to zero.
  """
  values = _validated_pvalues(p_values)
  with np.errstate(divide="ignore"):
    scores = 1.0 / np.tan(np.pi * values / 2.0)
  return _scalar_or_array(scores)


def reciprocal_score(p_values: PValueInput) -> ScoreResult:
  """Return the reciprocal score ``1 / p`` elementwise."""
  values = _validated_pvalues(p_values)
  with np.errstate(divide="ignore"):
    scores = 1.0 / values
  return _scalar_or_array(scores)


__all__ = ["PValueInput", "ScoreResult", "half_cauchy_score", "reciprocal_score"]
