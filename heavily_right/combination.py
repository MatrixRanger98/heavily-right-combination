"""P-value combination rules used by the meta-analysis procedures.

This module contains only the combination layer: it converts vectors of
study-level p-values to global scores, p-values, and decisions.  Confidence
sets and study-level calibration remain in :mod:`heavily_right.meta_analysis`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray
from scipy.stats import cauchy, gamma, halfnorm, norm

from .null_laws import HalfCauchyMean, HarmonicMean, Landau, euler_gamma
from .scores import half_cauchy_score, reciprocal_score

CombinationMethod = Literal[
  "HCauchy",
  "EHMP",
  "HMP",
  "Cauchy",
  "Levy",
  "Fisher",
  "Stouffer",
  "Bonferroni",
  "Simes",
]
SUPPORTED_METHODS: frozenset[str] = frozenset(get_args(CombinationMethod))
FloatArray = NDArray[np.float64]
ScoreResult = float | FloatArray


def cot(x: Any) -> Any:
  """Return the cotangent without relying on ``tan(pi / 2)`` as infinity."""
  return 1 / np.tan(x)


class CombinationTest:
  """Combine study-level p-values using one of the supported rules.

  ``arg`` is either an integer number of tests, which selects equal weights,
  or a one-dimensional vector of finite, positive weights summing to one.

  ``HCauchy`` and ``EHMP`` use exact finite-m independence calibration through
  1,000 tests and the Landau approximation above that size. ``HMP`` always
  uses the asymptotic Landau calibration.

  ``level`` is the significance level. P-values have shape ``(..., m)``;
  outputs have shape ``(...)``. Dependence assumptions are not inferred
  from the input. Invalid shapes, weights, or probabilities raise ValueError;
  failed finite-law computations raise NumericalCalibrationError subclasses.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import CombinationTest
  >>> test = CombinationTest([0.4, 0.6], method="HCauchy", level=0.05)
  >>> pvalues = np.array([[0.01, 0.2], [0.3, 0.4]])
  >>> combined = test.get_global_p(pvalues)
  >>> assert combined.shape == (2,) and np.all((0 <= combined) & (combined <= 1))
  >>> assert test.make_decision(pvalues).shape == (2,)
  >>> assert np.isclose(CombinationTest(1).get_global_p(0.2), 0.2)
  """

  def __init__(
    self,
    arg: np.integer | float | np.floating | Iterable[float] | np.ndarray,
    method: CombinationMethod = "HCauchy",
    level: float = 0.05,
    **kwargs: Any,
  ) -> None:
    del kwargs
    if isinstance(arg, (int, float, np.integer, np.floating)):
      numeric_arg = float(arg)
      if (
        isinstance(arg, bool)
        or not np.isfinite(numeric_arg)
        or numeric_arg < 1.0
        or not numeric_arg.is_integer()
      ):
        raise ValueError("number of tests must be a positive integer")
      self.__num_test = int(numeric_arg)
      self.__weights = np.ones(self.__num_test) / self.__num_test
    elif isinstance(arg, Iterable):
      values = arg if isinstance(arg, np.ndarray) else list(arg)
      weights = np.asarray(values, dtype=float)
      if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional vector")
      if not np.isfinite(weights).all():
        raise ValueError("weights must be finite")
      if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive")
      if not np.isclose(weights.sum(), 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("weights must sum to one")
      self.__num_test = weights.size
      self.__weights = weights.copy()
    else:
      raise TypeError(
        "provide an integer number of tests or a one-dimensional weight vector"
      )
    self.__weights.setflags(write=False)

    self.__method = self._validated_method(method)
    if self.__method == "Fisher" and self.__num_test != 1:
      warnings.warn("Equal weights are used for Fisher's test.", stacklevel=2)
    self.__level = self._validated_level(level)
    self.__init_threshold()

  @staticmethod
  def _validated_method(method: str) -> CombinationMethod:
    if method not in SUPPORTED_METHODS:
      choices = ", ".join(sorted(SUPPORTED_METHODS))
      raise ValueError(f"unknown combination method {method!r}; choose {choices}")
    return cast("CombinationMethod", method)

  @staticmethod
  def _validated_level(level: float) -> float:
    level = float(level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
      raise ValueError("level must lie strictly between zero and one")
    return level

  def _validated_pvalues(
    self,
    p_vector: float | Sequence[float] | np.ndarray,
  ) -> FloatArray:
    pvalues = np.asarray(p_vector, dtype=float)
    if pvalues.ndim == 0:
      if self.__num_test != 1:
        raise ValueError("scalar p-values are valid only for one test")
      pvalues = pvalues.reshape(1)
    if pvalues.shape[-1] != self.__num_test:
      raise ValueError(
        "the final p-value dimension must match the number of tests"
      )
    if not np.isfinite(pvalues).all():
      raise ValueError("p-values must be finite")
    if np.any((pvalues < 0.0) | (pvalues > 1.0)):
      raise ValueError("p-values must lie in [0, 1]")
    return pvalues

  @staticmethod
  def _scalar_or_array(value: Any) -> ScoreResult:
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array

  @staticmethod
  def _evaluate_scalar_null(
    function: Callable[[float], float],
    score: np.ndarray,
  ) -> ScoreResult:
    if score.ndim == 0:
      value = float(score)
      return 0.0 if np.isposinf(value) else function(value)
    return np.array(
      [
        0.0 if np.isposinf(value) else function(float(value))
        for value in score.flat
      ]
    ).reshape(score.shape)

  @property
  def num_test(self) -> int:
    return self.__num_test

  @property
  def weights(self) -> FloatArray:
    return self.__weights

  @property
  def method(self) -> CombinationMethod:
    return self.__method

  @property
  def level(self) -> float:
    return self.__level

  @property
  def threshold(self) -> float:
    return self.__threshold

  def get_threshold_from_p(self, level: float) -> float:
    """Return the global-score threshold for a significance level."""
    level = self._validated_level(level)
    if self.__num_test == 1:
      return -level
    elif self.__method == "HCauchy":
      if self.__num_test <= 1000:
        return HalfCauchyMean.ppf(1 - level, self.__weights)
      location = (
        (-sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma)
        * 2
        / np.pi
      )
      return Landau.ppf(1 - level, location, 1)
    elif self.__method == "EHMP":
      if self.__num_test <= 1000:
        return HarmonicMean.ppf(1 - level, self.__weights)
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return Landau.ppf(1 - level, location, np.pi / 2)
    elif self.__method == "HMP":
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return Landau.ppf(1 - level, location, np.pi / 2)
    elif self.__method == "Cauchy":
      return cauchy.ppf(1 - level)
    elif self.__method == "Levy":
      return 1 / norm.ppf((1 + level) / 2) ** 2
    elif self.__method == "Fisher":
      return gamma.ppf(1 - level, self.__num_test)
    elif self.__method == "Stouffer":
      return norm.ppf(1 - level)
    elif self.__method == "Bonferroni" or self.__method == "Simes":
      return -level / self.__num_test
    raise NotImplementedError("Method not implemented.")

  def __init_threshold(self) -> None:
    self.__threshold = self.get_threshold_from_p(self.__level)

  def change_method(self, new_method: CombinationMethod) -> None:
    self.__method = self._validated_method(new_method)
    if self.__method == "Fisher" and self.__num_test != 1:
      warnings.warn("Equal weights are used for Fisher's test.", stacklevel=2)
    self.__init_threshold()

  def change_level(self, new_level: float) -> None:
    self.__level = self._validated_level(new_level)
    self.__init_threshold()

  def get_p_from_score(
    self, score: float | Sequence[float] | np.ndarray
  ) -> ScoreResult:
    """Convert a global score to its global p-value."""
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
      raise ValueError("scores must not contain NaN")
    if self.__num_test == 1:
      return self._scalar_or_array(-score)
    elif self.__method == "HCauchy":
      if self.__num_test <= 1000:
        return self._evaluate_scalar_null(
          lambda value: float(HalfCauchyMean.sf(value, self.__weights)),
          score,
        )
      location = (
        (-sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma)
        * 2
        / np.pi
      )
      return Landau.sf(score, location, 1)
    elif self.__method == "EHMP":
      if self.__num_test <= 1000:
        return self._evaluate_scalar_null(
          lambda value: float(HarmonicMean.sf(value, self.__weights)),
          score,
        )
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return Landau.sf(score, location, np.pi / 2)
    elif self.__method == "HMP":
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return Landau.sf(score, location, np.pi / 2)
    elif self.__method == "Cauchy":
      return cauchy.sf(score)
    elif self.__method == "Levy":
      return halfnorm.cdf(1 / np.sqrt(score))
    elif self.__method == "Fisher":
      return gamma.sf(score, self.__num_test)
    elif self.__method == "Stouffer":
      return norm.sf(score)
    elif self.__method == "Bonferroni" or self.__method == "Simes":
      return np.minimum(1, -score * self.__num_test)
    raise NotImplementedError("Method not implemented.")

  def get_global_score(
    self, p_vector: float | Sequence[float] | np.ndarray
  ) -> ScoreResult:
    """Combine individual p-values into the method-specific global score."""
    p_vector = self._validated_pvalues(p_vector)
    if self.__num_test == 1:
      result = -np.squeeze(p_vector, axis=-1)
    elif self.__method == "HCauchy":
      result = np.asarray(half_cauchy_score(p_vector)) @ self.__weights
    elif self.__method == "EHMP" or self.__method == "HMP":
      result = np.asarray(reciprocal_score(p_vector)) @ self.__weights
    elif self.__method == "Cauchy":
      result = cot(p_vector * np.pi) @ self.__weights
    elif self.__method == "Levy":
      result = (
        1
        / norm.ppf((1 + p_vector) / 2) ** 2
        @ self.__weights
        / (np.sum(np.sqrt(self.__weights))) ** 2
      )
    elif self.__method == "Fisher":
      result = -np.sum(np.log(p_vector), axis=-1)
    elif self.__method == "Stouffer":
      result = (
        -norm.ppf(p_vector) @ self.__weights
        / np.sqrt(np.sum(self.__weights**2))
      )
    elif self.__method == "Bonferroni":
      result = -p_vector.min(axis=-1)
    elif self.__method == "Simes":
      adjusted = np.sort(p_vector, axis=-1) / np.arange(1, self.__num_test + 1)
      result = -adjusted.min(axis=-1)
    else:
      raise NotImplementedError("Method not implemented.")
    return self._scalar_or_array(result)

  def get_global_p(
    self, p_vector: float | Sequence[float] | np.ndarray
  ) -> ScoreResult:
    """Combine individual p-values and return the corresponding global p-value."""
    return self.get_p_from_score(self.get_global_score(p_vector))

  def make_decision(
    self,
    p_vector: float | Sequence[float] | np.ndarray,
  ) -> bool | NDArray[np.bool_]:
    """Return whether the combined test rejects at the configured level."""
    decision = np.asarray(self.get_global_score(p_vector)) >= self.__threshold
    return bool(decision) if decision.ndim == 0 else decision


# Compatibility with the class name used by the existing scripts and paper code.
combination_test = CombinationTest


__all__ = [
  "CombinationMethod",
  "CombinationTest",
  "combination_test",
  "cot",
]
