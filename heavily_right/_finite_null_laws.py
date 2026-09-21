"""Exact finite-m null laws for heavy-right score averages."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import sici

from ._constants import EULER_MASCHERONI
from ._half_cauchy_lower_tail import lower_tail
from ._harmonic_mean_lower_tail import lower_tail as reciprocal_lower_tail
from ._numerics import (
  IntegrationResult,
  WeightInput,
  integrate_checked,
  normalize_weights,
  quantile_from_cdf,
)
from ._special_functions import scaled_exponential_integral


class HalfCauchyMean:
  """Distribution of a weighted sum of independent standard half-Cauchy variables.

  Public import: ``from heavily_right import HalfCauchyMean``. Methods take a
  scalar argument and ``w`` as an equal-weight count or nonnegative normalized
  weights. ``ppf`` takes a CDF probability, not a survival probability.
  ``precision=True`` on sf/cdf/pdf adds an estimated absolute numerical error, not a
  rigorous total-error bound. Numerical failures raise typed exceptions.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import HalfCauchyMean
  >>> cutoff = HalfCauchyMean.ppf(0.95, 2)
  >>> probability, quadrature_error = HalfCauchyMean.sf(cutoff, 2, precision=True)
  >>> assert np.isclose(probability, 0.05, atol=1e-7)
  >>> assert quadrature_error >= 0
  >>> assert np.isclose(HalfCauchyMean.sf(1.0, 1), 0.5)
  """

  @staticmethod
  def _lower_tail_limit(weights: np.ndarray) -> float:
    """Select the stable contour; this is not a zero-probability shortcut."""
    location = (-sum(weights * np.log(weights)) + 1 - EULER_MASCHERONI) * 2 / np.pi
    return max(0.5, float(location) - 1.0)

  @staticmethod
  def __pre_cdf(z: float | np.float64, x: float | np.float64, w: np.ndarray) -> float | np.float64:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - EULER_MASCHERONI) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    f_star = (
      -2
      / np.pi
      * (
        np.cos(n * z * w) * (sici(n * z * w)[0] - np.pi / 2)
        - np.sin(n * z * w) * sici(n * z * w)[1]
      )
    )
    f_comp = -f_star + 2 * np.cos(n * z * w) + 2j * np.sin(n * z * w)
    f_sum = sum(np.log(f_comp))
    f_int = 2 * np.exp(f_sum - x * n * z).imag / (2 * np.pi * z)
    return f_int

  @staticmethod
  def __pre_pdf(z: float | np.float64, x: float | np.float64, w: np.ndarray) -> float | np.float64:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - EULER_MASCHERONI) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    f_star = (
      -2
      / np.pi
      * (
        np.cos(n * z * w) * (sici(n * z * w)[0] - np.pi / 2)
        - np.sin(n * z * w) * sici(n * z * w)[1]
      )
    )
    f_comp = -f_star + 2 * np.cos(n * z * w) + 2j * np.sin(n * z * w)
    f_sum = sum(np.log(f_comp))
    f_int = 2 * np.exp(f_sum - x * n * z).imag * n / (2 * np.pi)
    return f_int

  @classmethod
  def sf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x <= 0.0:
      return (1.0, 0.0) if precision else 1.0
    if x == np.inf:
      return (0.0, 0.0) if precision else 0.0
    if active_weight.size == 1:
      value = float(2.0 / np.pi * np.arctan(active_weight[0] / x))
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = lower_tail(x, active_weight, cdf=True)
      result = float(1.0 - value)
      return (result, error + np.finfo(float).eps) if precision else result
    return integrate_checked(
      cls.__pre_cdf,
      0.0,
      x,
      weight,
      precision=precision,
      value_bounds=(0.0, 1.0),
    )

  @classmethod
  def cdf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x <= 0.0:
      return (0.0, 0.0) if precision else 0.0
    if x == np.inf:
      return (1.0, 0.0) if precision else 1.0
    if active_weight.size == 1:
      value = float(2.0 / np.pi * np.arctan(x / active_weight[0]))
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = lower_tail(x, active_weight, cdf=True)
      return (value, error) if precision else value
    comp = cls.sf(x, weight, precision, check_w=False)
    if precision:
      return 1 - comp[0], comp[1]
    return 1 - comp

  @classmethod
  def pdf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x <= 0.0 or x == np.inf:
      return (0.0, 0.0) if precision else 0.0
    if active_weight.size == 1:
      value = float(
        2.0
        / np.pi
        * active_weight[0]
        / (x * x + active_weight[0] * active_weight[0])
      )
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = lower_tail(x, active_weight)
      return (value, error) if precision else value
    return integrate_checked(
      cls.__pre_pdf,
      0.0,
      x,
      weight,
      precision=precision,
      value_bounds=(0.0, None),
    )

  @classmethod
  def ppf(
    cls,
    alpha: float,
    w: WeightInput,
    check_w: bool = True,
  ) -> float:
    weight = normalize_weights(w, checked=check_w)
    return quantile_from_cdf(
      alpha,
      lambda value: float(cls.cdf(value, weight, check_w=False)),
      lower_bound=0.0,
    )


class HarmonicMean:
  """Null law of the weighted reciprocal score, not its reciprocal.

  For independent uniform p-values this is the distribution of ``sum(w/p)``.
  Public methods and weight conventions match HalfCauchyMean. An upper-tail
  cutoff uses ``ppf(1 - level, w)``; precision=True returns an estimated
  absolute numerical error for sf/cdf/pdf. Lower tails use a separately
  validated shifted Laplace contour. This is finite-sample numerical inversion,
  distinct from the asymptotic HMP combination method.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import HarmonicMean
  >>> cutoff = HarmonicMean.ppf(0.95, 1)
  >>> assert np.isclose(cutoff, 20.0)
  >>> assert np.isclose(HarmonicMean.sf(cutoff, 1), 0.05)
  """

  @staticmethod
  def _support_residual(x: float, original: WeightInput, weights: np.ndarray) -> float:
    """Keep count means' exact support; compensate supplied-vector cancellation."""
    if isinstance(original, (int, float, np.integer, np.floating)):
      return x - 1.0
    return math.fsum([x, *(-weights)])

  @staticmethod
  def _lower_tail_limit(weights: np.ndarray) -> float:
    """Contour-selection threshold, never a probability-zero shortcut."""
    location = -math.fsum(weights * np.log(weights)) + 1 - EULER_MASCHERONI
    return max(math.fsum(weights) + 0.5, location - 1.0)

  @staticmethod
  def __pre_cdf(z: float | np.float64, x: float | np.float64, w: np.ndarray) -> float | np.float64:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - EULER_MASCHERONI) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    ei_2_over_exp = (n * z * w) * scaled_exponential_integral(n * z * w) - 1
    f_comp = -ei_2_over_exp + 1j * np.pi * np.exp(np.log(n * z * w) - n * z * w)
    f_sum = sum(np.log(f_comp) + n * z * w)
    f_int = 2 * np.exp(f_sum - x * n * z).imag / (2 * np.pi * z)
    return f_int

  @staticmethod
  def __pre_pdf(z: float | np.float64, x: float | np.float64, w: np.ndarray) -> float | np.float64:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - EULER_MASCHERONI) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    ei_2_over_exp = (n * z * w) * scaled_exponential_integral(n * z * w) - 1
    f_comp = -ei_2_over_exp + 1j * np.pi * np.exp(np.log(n * z * w) - n * z * w)
    f_sum = sum(np.log(f_comp) + n * z * w)
    f_int = 2 * np.exp(f_sum - x * n * z).imag * n / (2 * np.pi)
    return f_int

  @classmethod
  def sf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x == np.inf:
      return (0.0, 0.0) if precision else 0.0
    residual = cls._support_residual(x, w, active_weight)
    if residual <= 0:
      return (1.0, 0.0) if precision else 1.0
    if active_weight.size == 1:
      support = float(active_weight[0])
      value = 1.0 if x <= support else support / x
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = reciprocal_lower_tail(x, active_weight, cdf=True, shifted_x=residual)
      result = float(1.0 - value)
      return (result, error + np.finfo(float).eps) if precision else result
    return integrate_checked(
      cls.__pre_cdf,
      1.0e-150,
      x,
      weight,
      precision=precision,
      value_bounds=(0.0, 1.0),
    )

  @classmethod
  def cdf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x == np.inf:
      return (1.0, 0.0) if precision else 1.0
    residual = cls._support_residual(x, w, active_weight)
    if residual <= 0:
      return (0.0, 0.0) if precision else 0.0
    if active_weight.size == 1:
      value = float(residual / x)
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = reciprocal_lower_tail(x, active_weight, cdf=True, shifted_x=residual)
      return (value, error) if precision else value
    comp = cls.sf(x, weight, precision, check_w=False)
    if precision:
      return 1 - comp[0], comp[1]
    return 1 - comp

  @classmethod
  def pdf(
    cls,
    x: float,
    w: WeightInput,
    precision: bool = False,
    check_w: bool = True,
  ) -> IntegrationResult:
    weight = normalize_weights(w, checked=check_w)
    active_weight = weight[weight != 0.0]
    if np.isnan(x):
      raise ValueError("x must not be NaN")
    if x == np.inf:
      return (0.0, 0.0) if precision else 0.0
    residual = cls._support_residual(x, w, active_weight)
    if residual <= 0:
      return (0.0, 0.0) if precision else 0.0
    if active_weight.size == 1:
      support = float(active_weight[0])
      value = (support / x) / x
      return (value, 0.0) if precision else value
    if x <= cls._lower_tail_limit(active_weight):
      value, error = reciprocal_lower_tail(x, active_weight, shifted_x=residual)
      return (value, error) if precision else value
    return integrate_checked(
      cls.__pre_pdf,
      1.0e-150,
      x,
      weight,
      precision=precision,
      value_bounds=(0.0, None),
    )

  @classmethod
  def ppf(
    cls,
    alpha: float,
    w: WeightInput,
    check_w: bool = True,
  ) -> float:
    weight = normalize_weights(w, checked=check_w)
    scalar_count = isinstance(w, (int, float, np.integer, np.floating))
    lower = 1.0 if scalar_count else float(np.nextafter(math.fsum(weight), -np.inf))
    return quantile_from_cdf(
      alpha,
      lambda value: float(cls.cdf(value, w, check_w=check_w)),
      lower_bound=lower,
    )

__all__ = ["HalfCauchyMean", "HarmonicMean"]
