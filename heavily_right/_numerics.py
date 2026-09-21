"""Private numerical adapters shared by heavy-tailed distributions."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

FloatArray = NDArray[np.float64]
WeightInput = float | np.integer | Sequence[float] | FloatArray
IntegrationResult = float | tuple[float, float]
Integrand = Callable[[float, float, FloatArray], float | np.floating]
ProbabilityFunction = Callable[[float], float]

QUADRATURE_ABSOLUTE_TOLERANCE = 2.0e-13
QUADRATURE_RELATIVE_TOLERANCE = 2.0e-11
QUADRATURE_SUBDIVISION_LIMIT = 600
ROOT_ABSOLUTE_TOLERANCE = 2.0e-10
ROOT_RELATIVE_TOLERANCE = 2.0e-11


class NumericalCalibrationError(RuntimeError):
  """Base error for a failed null-law calibration calculation."""


class NumericalIntegrationError(NumericalCalibrationError):
  """Raised when characteristic-function inversion is unreliable."""


class NumericalRootError(NumericalCalibrationError):
  """Raised when a requested probability cannot be bracketed or inverted."""


def normalize_weights(value: WeightInput, *, checked: bool) -> FloatArray:
  """Validate a positive integer count or normalized nonnegative weights."""
  if not checked:
    return np.asarray(value, dtype=float)
  if isinstance(value, (int, float, np.integer, np.floating)):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 1 or int(value) != value:
      raise ValueError("a scalar study count must be a positive integer")
    count = int(value)
    return np.full(count, 1.0 / count)
  weight = np.asarray(value, dtype=float)
  if weight.ndim != 1 or weight.size == 0:
    raise ValueError("weights must be a nonempty one-dimensional vector")
  if not np.isfinite(weight).all() or np.any(weight < 0.0):
    raise ValueError("weights must be finite and nonnegative")
  if not np.isclose(weight.sum(), 1.0, rtol=1.0e-10, atol=1.0e-12):
    raise ValueError("weights must sum to one")
  return weight


def integrate_checked(
  integrand: Integrand,
  lower_bound: float,
  x: float,
  weights: FloatArray,
  *,
  precision: bool,
  value_bounds: tuple[float | None, float | None],
) -> IntegrationResult:
  """Evaluate and validate an improper characteristic-function integral."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", IntegrationWarning)
    try:
      value, error = quad(
        integrand,
        lower_bound,
        np.inf,
        args=(x, weights),
        epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
        epsrel=QUADRATURE_RELATIVE_TOLERANCE,
        limit=QUADRATURE_SUBDIVISION_LIMIT,
      )
    except (ArithmeticError, RuntimeWarning, ValueError) as numerical_error:
      raise NumericalIntegrationError(
        "characteristic-function inversion raised a numerical error"
      ) from numerical_error
  tolerance = max(
    QUADRATURE_ABSOLUTE_TOLERANCE,
    QUADRATURE_RELATIVE_TOLERANCE * abs(value),
  )
  integration_warnings = [
    warning
    for warning in caught
    if issubclass(warning.category, IntegrationWarning)
  ]
  if integration_warnings and error > 25.0 * tolerance:
    messages = "; ".join(str(warning.message) for warning in integration_warnings)
    raise NumericalIntegrationError(
      "characteristic-function inversion did not converge: "
      f"{messages}; value={value}, estimated error={error}, "
      f"tolerance={tolerance}"
    )
  if not np.isfinite(value) or not np.isfinite(error) or error < 0 or error > 100.0 * tolerance:
    raise NumericalIntegrationError(
      "characteristic-function inversion failed its error check: "
      f"value={value}, estimated error={error}, tolerance={tolerance}"
    )
  lower, upper = value_bounds
  range_tolerance = max(5.0e-9, 25.0 * tolerance)
  if lower is not None and value < lower - range_tolerance:
    raise NumericalIntegrationError(
      f"inversion result {value} is below its lower bound {lower}"
    )
  if upper is not None and value > upper + range_tolerance:
    raise NumericalIntegrationError(
      f"inversion result {value} is above its upper bound {upper}"
    )
  if lower is not None:
    value = max(lower, value)
  if upper is not None:
    value = min(upper, value)
  return (value, error) if precision else value


def quantile_from_cdf(
  alpha: float,
  cdf: ProbabilityFunction,
  *,
  lower_bound: float,
  initial_upper_bound: float = 100.0,
) -> float:
  """Invert a checked CDF using deterministic geometric bracketing."""
  if not 0.0 < alpha < 1.0:
    raise ValueError("alpha must lie strictly between zero and one")
  lower = float(lower_bound)
  upper = max(float(initial_upper_bound), lower + 1.0)
  for _ in range(200):
    if cdf(upper) >= alpha:
      break
    lower = upper
    upper *= 2.0
  else:
    raise NumericalRootError(
      f"failed to bracket the quantile for alpha={alpha}"
    )
  try:
    return float(
      brentq(
        lambda value: cdf(value) - alpha,
        lower,
        upper,
        xtol=ROOT_ABSOLUTE_TOLERANCE,
        rtol=ROOT_RELATIVE_TOLERANCE,
      )
    )
  except (RuntimeError, ValueError) as numerical_error:
    raise NumericalRootError(
      f"failed to invert the CDF for alpha={alpha}"
    ) from numerical_error


__all__ = [
  "IntegrationResult",
  "NumericalCalibrationError",
  "NumericalIntegrationError",
  "NumericalRootError",
  "WeightInput",
]
