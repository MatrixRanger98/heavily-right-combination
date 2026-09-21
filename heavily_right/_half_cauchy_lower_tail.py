"""Stable lower-tail PDF/CDF of weighted positive Half-Cauchy sums.

The inverse Laplace contour stays in the positive half-plane and is tilted to
the requested mean. This avoids the exponential cancellation of the rotated
contour in the lower tail. Near zero, a positive-convolution expansion has an
analytic remainder bound. Reported errors include quadrature and a floating
arithmetic allowance; they are estimates, not interval-arithmetic certificates.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import exp1

from ._numerics import (
  QUADRATURE_ABSOLUTE_TOLERANCE,
  QUADRATURE_RELATIVE_TOLERANCE,
  QUADRATURE_SUBDIVISION_LIMIT,
  FloatArray,
  NumericalIntegrationError,
)

_EPSILON = np.finfo(float).eps
_SMALLEST_FLOAT = np.nextafter(0.0, 1.0)
_ASYMPTOTIC_THRESHOLD = 60.0


def _laplace(s: NDArray[np.complex128]) -> NDArray[np.complex128]:
  """L(s)=E exp(-s X), Re(s)>0, without overflowing exp(±is).

  For |s|>=60 use 26 terms of (2/pi) sum (-1)^k (2k)!/s^(2k+1).
  Rotating the defining integral by at most pi/4 bounds the remainder by
  (2/pi)*52!/(|s|/sqrt(2))^53, below 2.8e-19 at the switch. This is uniform
  throughout Re(s)>0, including large imaginary arguments; no negative-real
  E1 branch or unrecorded tail clipping is used in that regime.
  """
  large = np.abs(s) >= _ASYMPTOTIC_THRESHOLD
  value = np.empty_like(s)
  z = s[~large]
  value[~large] = (
    np.exp(-1j * z) * exp1(-1j * z) - np.exp(1j * z) * exp1(1j * z)
  ) / (1j * np.pi)
  z = s[large]
  if z.size:
    inverse_square = (1.0 / z) ** 2
    term = np.ones_like(z)
    total = term.copy()
    for k in range(1, 26):
      term *= -(2 * k) * (2 * k - 1) * inverse_square
      total += term
    value[large] = (2.0 / np.pi) * total / z
  if not np.isfinite(value).all() or np.any(value == 0):
    raise NumericalIntegrationError("nonfinite/zero Half-Cauchy Laplace transform")
  return value


def _tilted_mean(s: FloatArray) -> FloatArray:
  """-L'(s)/L(s) for real s>0; only selects the inversion contour."""
  value = _laplace(s.astype(complex)).real
  large = s >= _ASYMPTOTIC_THRESHOLD
  result = np.empty_like(s)
  z = s[~large]
  derivative = -(
    np.exp(-1j * z) * exp1(-1j * z) + np.exp(1j * z) * exp1(1j * z)
  ).real / np.pi
  result[~large] = -derivative / value[~large]
  z = s[large]
  if z.size:
    inverse_square = (1.0 / z) ** 2
    term = np.ones_like(z)
    total = term.copy()
    for k in range(1, 26):
      term *= -(2 * k) * (2 * k - 1) * inverse_square
      total += (2 * k + 1) * term
    # Divide only after forming the O(1) ratio; 1/z**2 may underflow even
    # when the leading tilted mean 1/z remains representable.
    result[large] = ((2.0 / np.pi) * total / (z * value[large])) / z
  if not np.isfinite(result).all() or np.any(result <= 0):
    raise NumericalIntegrationError("invalid Half-Cauchy tilted mean")
  return result


def _boundary_expansion(x: float, weights: FloatArray, *, cdf: bool) -> tuple[float, float] | None:
  """Two simplex-moment terms, accepted only with a small relative remainder.

  f(x)=U(x) E prod[1+(x T_j/w_j)^2]^-1, T~Dirichlet(1,...,1).
  The expansion 1-a1 has signed remainder in [0,a2]. Integrating each
  monomial gives the corresponding CDF factors m/(m+2k).
  """
  m = weights.size
  log_upper = (m * math.log(2.0 / math.pi) + (m - 1) * math.log(x)
               - math.lgamma(m) - math.fsum(np.log(weights)))
  if cdf:
    log_upper += math.log(x) - math.log(m)
  if log_upper < math.log(_SMALLEST_FLOAT):
    # This global simplex bound is valid even outside the expansion regime.
    return 0.0, _SMALLEST_FLOAT
  if x > 1e-4 * float(weights.min()):
    return None
  ratios_squared = (x / weights) ** 2
  sum_squares = float(ratios_squared.sum())
  first = 2.0 * sum_squares / (m * (m + 1))
  second = (22.0 * float(np.dot(ratios_squared, ratios_squared))
            + 2.0 * sum_squares**2) / (m * (m + 1) * (m + 2) * (m + 3))
  if cdf:
    first *= m / (m + 2)
    second *= m / (m + 4)
  upper = math.exp(log_upper)
  value = upper * (1.0 - first)
  roundoff = 16 * _EPSILON * (1 + m + abs(log_upper)) * upper
  return value, max(upper * second + roundoff, _SMALLEST_FLOAT)


def _checked_integral(
  integrand: Callable[[float], float], frequency: float | None = None, *, sine: bool = False,
) -> tuple[float, float]:
  """Checked improper quadrature, optionally using sine/cosine tail cycles."""
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", IntegrationWarning)
    try:
      if frequency is None:
        value, error = quad(
          integrand, 0.0, np.inf,
          epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
          epsrel=QUADRATURE_RELATIVE_TOLERANCE,
          limit=QUADRATURE_SUBDIVISION_LIMIT,
        )
      else:
        value, error = quad(
          integrand, 0.0, np.inf, weight="sin" if sine else "cos", wvar=frequency,
          epsabs=QUADRATURE_ABSOLUTE_TOLERANCE,
          limit=QUADRATURE_SUBDIVISION_LIMIT, limlst=100,
        )
    except (ArithmeticError, ValueError, RuntimeWarning) as cause:
      raise NumericalIntegrationError("lower-tail Laplace quadrature failed") from cause
  tolerance = max(QUADRATURE_ABSOLUTE_TOLERANCE,
                  QUADRATURE_RELATIVE_TOLERANCE * abs(value))
  if (not math.isfinite(value) or abs(value) == np.finfo(float).max
      or not math.isfinite(error) or error < 0
      or error > (25.0 if caught else 100.0) * tolerance):
    raise NumericalIntegrationError(
      f"lower-tail Laplace quadrature did not converge: value={value}, "
      f"estimated error={error}; " + "; ".join(str(item.message) for item in caught)
    )
  return value, error


def lower_tail(x: float, weights: FloatArray, *, cdf: bool = False) -> tuple[float, float]:
  """Evaluate PDF or CDF at finite x>0 with positive active weights."""
  boundary = _boundary_expansion(x, weights, cdf=cdf)
  if boundary is not None:
    return boundary
  unique, counts = np.unique(weights, return_counts=True)
  m = weights.size

  def mean_at(a: float) -> float:
    arguments = a * unique
    if not np.isfinite(arguments).all() or np.any(arguments <= 0):
      raise NumericalIntegrationError("Half-Cauchy tilt is outside floating-point range")
    return float(np.dot(counts * unique, _tilted_mean(arguments)))

  # Tilted means decrease continuously from infinity to zero. The bound
  # mean_at(a)<=m/a gives a strict bracket with a factor-two rounding margin.
  high = 2.0 * m / x
  if not math.isfinite(high):
    raise NumericalIntegrationError("Half-Cauchy tilt bracket overflowed")
  low = high
  for _ in range(1100):
    if mean_at(low) >= x:
      break
    low /= 2.0
  else:
    raise NumericalIntegrationError("could not bracket the Half-Cauchy tilt")
  try:
    tilt = brentq(lambda a: mean_at(a) - x, low, high,
                  xtol=np.finfo(float).tiny, rtol=8 * _EPSILON)
  except (RuntimeError, ValueError) as cause:
    raise NumericalIntegrationError("could not solve the Half-Cauchy tilt") from cause
  log_factors = np.log(_laplace((tilt * unique).astype(complex)).real)
  log_transform = float(np.dot(counts, log_factors))
  log_prefactor = tilt * x + log_transform
  # Chernoff gives F(x)<=exp(log_prefactor). For the PDF, each tilted
  # component density is decreasing, with supremum 2/(pi*w*L(tilt*w));
  # the convolution supremum is bounded by the smallest component supremum.
  log_bound = log_prefactor
  if not cdf:
    log_bound += math.log(2.0 / math.pi) - float(np.max(np.log(unique) + log_factors))
  log_allowance = 64 * _EPSILON * (1 + m + abs(log_transform) + abs(log_bound))
  if log_bound + log_allowance < math.log(_SMALLEST_FLOAT):
    return 0.0, _SMALLEST_FLOAT
  frequency_scale = tilt / math.sqrt(m)
  log_scale = log_prefactor + math.log(frequency_scale / math.pi)
  if cdf:
    log_scale -= math.log(tilt)

  def transform(u: float) -> complex:
    s = complex(tilt, u * frequency_scale)
    result = complex(np.exp(np.dot(counts, np.log(_laplace(s * unique))) - log_transform))
    return result * tilt / s if cdf else result

  frequency = frequency_scale * x
  # With many comparable weights, combining phases before quadrature avoids
  # resolving two rapidly oscillating factors that cancel at the saddle.
  # Small/skewed systems have longer algebraic tails: use Fourier tail cycles.
  combined_phase = m >= 8 and float(unique.max()) <= 4 * float(unique.min())
  normalized = None
  if combined_phase:
    try:
      normalized, normalized_error = _checked_integral(
        lambda u: (transform(u) * np.exp(1j * frequency * u)).real,
      )
      accumulation = abs(normalized)
    except NumericalIntegrationError:
      # This is an alternate quadrature, never an unchecked value/zero fallback.
      pass
  if normalized is None:
    real, real_error = _checked_integral(lambda u: transform(u).real, frequency)
    imag, imag_error = _checked_integral(lambda u: transform(u).imag, frequency, sine=True)
    normalized = real - imag
    normalized_error = real_error + imag_error
    accumulation = abs(real) + abs(imag)
  if normalized <= 0:
    raise NumericalIntegrationError("lower-tail Laplace integral was not positive")
  # Combine logs before exponentiation to retain representable subnormal tails.
  value = math.exp(log_scale + math.log(normalized))
  # Include transform/normalization accumulation, not only QUADPACK's estimate.
  roundoff = 32 * _EPSILON * (1 + m + abs(log_transform) + abs(log_scale))
  error = max(math.exp(log_scale + math.log(normalized_error + roundoff * accumulation)),
              _SMALLEST_FLOAT)
  tolerance = max(QUADRATURE_ABSOLUTE_TOLERANCE,
                  QUADRATURE_RELATIVE_TOLERANCE * abs(value))
  if (not math.isfinite(value) or not math.isfinite(error) or value < 0
      or (cdf and value > 1) or error > 100 * tolerance):
    raise NumericalIntegrationError(
      f"lower-tail Laplace result failed validation: value={value}, estimated error={error}"
    )
  return value, error
