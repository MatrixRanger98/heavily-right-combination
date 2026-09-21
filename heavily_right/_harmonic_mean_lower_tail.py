"""Shifted positive-real Laplace inversion for reciprocal-score lower tails.

For Y with density y^-2 on y>=1, U=Y-1 has transform
L(s)=1-s*exp(s)*E1(s)=integral_0^infinity t*exp(-t)/(s+t) dt.
Factoring out the exact support shift avoids exponential cancellation in the
rotated contour. Error estimates include floating arithmetic allowances but
are not interval-arithmetic certificates.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import exp1

from ._half_cauchy_lower_tail import _checked_integral
from ._numerics import (
  QUADRATURE_ABSOLUTE_TOLERANCE,
  QUADRATURE_RELATIVE_TOLERANCE,
  FloatArray,
  NumericalIntegrationError,
)

_EPSILON = np.finfo(float).eps
_SMALLEST_FLOAT = np.nextafter(0.0, 1.0)
_ASYMPTOTIC_THRESHOLD = 50.0


def _asymptotic(s: NDArray[np.complex128]) -> tuple[NDArray[np.complex128],
                                                NDArray[np.complex128]]:
  """Return L(s) and -s*L'(s) from 32 inverse-power terms.

  The Stieltjes representation bounds the transform remainder by
  33!/|s|^33 throughout Re(s)>=0: |s+t|>=|s| there. The bound is below
  8e-20 at |s|=50. Computing L directly avoids subtracting two near-unit terms.
  """
  term = 1.0 / s
  value = term.copy()
  derivative = term.copy()
  for k in range(1, 32):
    term *= -(k + 1) / s
    value += term
    derivative += (k + 1) * term
  return value, derivative


def _laplace(s: NDArray[np.complex128]) -> NDArray[np.complex128]:
  large = np.abs(s) >= _ASYMPTOTIC_THRESHOLD
  value = np.empty_like(s)
  z = s[~large]
  value[~large] = 1.0 - z * np.exp(z) * exp1(z)
  value[large] = _asymptotic(s[large])[0]
  if not np.isfinite(value).all() or np.any(value == 0):
    raise NumericalIntegrationError("nonfinite/zero shifted reciprocal Laplace transform")
  return value


def _tilted_mean(s: FloatArray) -> FloatArray:
  """-L'(s)/L(s) for real positive s; used only to choose the contour."""
  large = s >= _ASYMPTOTIC_THRESHOLD
  result = np.empty_like(s)
  z = s[~large]
  scaled_e1 = np.exp(z) * exp1(z)
  result[~large] = ((z + 1.0) * scaled_e1 - 1.0) / (1.0 - z * scaled_e1)
  z = s[large]
  value, derivative = _asymptotic(z.astype(complex))
  result[large] = (derivative / value).real / z
  if not np.isfinite(result).all() or np.any(result <= 0):
    raise NumericalIntegrationError("invalid shifted reciprocal tilted mean")
  return result


def _boundary_expansion(
  x: float, weights: FloatArray, *, cdf: bool,
) -> tuple[float, float] | None:
  """Positive-convolution simplex bound and two near-boundary terms.

  The product prod(1+x*T_j/w_j)^-2, T~Dirichlet(1,...,1), is bounded
  below by 1-2*sum(x*T_j/w_j). Its Taylor remainder is nonnegative and
  at most 2*(sum(x*T_j/w_j))^2+sum((x*T_j/w_j)^2).
  """
  m = weights.size
  log_upper = ((m - 1) * math.log(x) - math.lgamma(m)
               - math.fsum(np.log(weights)))
  if cdf:
    log_upper += math.log(x) - math.log(m)
  log_allowance = 16 * _EPSILON * (1 + m + abs(log_upper))
  if log_upper + log_allowance < math.log(_SMALLEST_FLOAT):
    return 0.0, _SMALLEST_FLOAT
  if x > 1e-6 * float(weights.min()):
    return None
  ratios = x / weights
  total = float(ratios.sum())
  first = 2.0 * total / m
  second = (2.0 * total**2 + 4.0 * float(np.dot(ratios, ratios))) / (m * (m + 1))
  if cdf:
    first *= m / (m + 1)
    second *= m / (m + 2)
  upper = math.exp(log_upper)
  roundoff = 16 * _EPSILON * (1 + m + abs(log_upper)) * upper
  return upper * (1.0 - first), max(upper * second + roundoff, _SMALLEST_FLOAT)


def lower_tail(
  x: float, weights: FloatArray, *, cdf: bool = False, shifted_x: float | None = None,
) -> tuple[float, float]:
  """Invert at positive distance from support without rounding that support.

  Vector weights represent their supplied binary floats. Scalar-count callers
  pass x-1 explicitly to retain the mathematical mean's exact support at one.
  """
  if shifted_x is None:
    shifted_x = math.fsum([x, *(-weights)])
  if shifted_x <= 0 or not math.isfinite(shifted_x):
    raise NumericalIntegrationError("reciprocal inversion requires positive finite support distance")
  boundary = _boundary_expansion(shifted_x, weights, cdf=cdf)
  if boundary is not None:
    return boundary
  unique, counts = np.unique(weights, return_counts=True)
  m = weights.size

  def mean_at(a: float) -> float:
    arguments = a * unique
    if not np.isfinite(arguments).all() or np.any(arguments <= 0):
      raise NumericalIntegrationError("reciprocal tilt is outside floating-point range")
    return float(np.dot(counts * unique, _tilted_mean(arguments)))

  # Each shifted tilted density is decreasing, hence its mean is <=1/a;
  # the total tilted mean is <=m/a. This gives a deterministic strict bracket.
  high = 2.0 * m / shifted_x
  if not math.isfinite(high):
    raise NumericalIntegrationError("reciprocal tilt bracket overflowed")
  low = high
  for _ in range(1100):
    if mean_at(low) >= shifted_x:
      break
    low /= 2.0
  else:
    raise NumericalIntegrationError("could not bracket the reciprocal tilt")
  try:
    tilt = brentq(lambda a: mean_at(a) - shifted_x, low, high,
                  xtol=np.finfo(float).tiny, rtol=8 * _EPSILON)
  except (RuntimeError, ValueError) as cause:
    raise NumericalIntegrationError("could not solve the reciprocal tilt") from cause
  factors = _laplace((tilt * unique).astype(complex)).real
  if np.any(factors <= 0):
    raise NumericalIntegrationError("nonpositive real reciprocal Laplace transform")
  log_factors = np.log(factors)
  log_transform = float(np.dot(counts, log_factors))
  log_prefactor = tilt * shifted_x + log_transform
  # Chernoff bounds the CDF. The PDF additionally uses the smallest supremum
  # of the decreasing tilted component densities: 1/(w_j*L(tilt*w_j)).
  log_bound = log_prefactor
  if not cdf:
    log_bound -= float(np.max(np.log(unique) + log_factors))
  log_allowance = 64 * _EPSILON * (1 + m + abs(log_transform) + abs(log_bound))
  if log_bound + log_allowance < math.log(_SMALLEST_FLOAT):
    return 0.0, _SMALLEST_FLOAT
  frequency_scale = tilt / math.sqrt(m)
  log_scale = log_prefactor + math.log(frequency_scale / math.pi)
  if cdf:
    log_scale -= math.log(tilt)

  def transform(u: float) -> complex:
    s = complex(tilt, u * frequency_scale)
    value = complex(np.exp(np.dot(counts, np.log(_laplace(s * unique))) - log_transform))
    return value * tilt / s if cdf else value

  frequency = frequency_scale * shifted_x
  normalized = None
  if m >= 8 and float(unique.max()) <= 4 * float(unique.min()):
    try:
      normalized, normalized_error = _checked_integral(
        lambda u: (transform(u) * np.exp(1j * frequency * u)).real,
      )
      accumulation = abs(normalized)
    except NumericalIntegrationError:
      pass  # Only the separately checked oscillatory quadrature may replace it.
  if normalized is None:
    real, real_error = _checked_integral(lambda u: transform(u).real, frequency)
    imag, imag_error = _checked_integral(lambda u: transform(u).imag, frequency, sine=True)
    normalized = real - imag
    normalized_error = real_error + imag_error
    accumulation = abs(real) + abs(imag)
  if normalized <= 0:
    raise NumericalIntegrationError("reciprocal lower-tail integral was not positive")
  value = math.exp(log_scale + math.log(normalized))
  roundoff = 32 * _EPSILON * (1 + m + abs(log_transform) + abs(log_scale))
  error = max(math.exp(log_scale + math.log(normalized_error + roundoff * accumulation)),
              _SMALLEST_FLOAT)
  tolerance = max(QUADRATURE_ABSOLUTE_TOLERANCE,
                  QUADRATURE_RELATIVE_TOLERANCE * abs(value))
  if (not math.isfinite(value) or not math.isfinite(error) or value < 0
      or (cdf and value > 1) or error > 100 * tolerance):
    raise NumericalIntegrationError(
      f"reciprocal lower-tail result failed validation: value={value}, estimated error={error}"
    )
  return value, error
