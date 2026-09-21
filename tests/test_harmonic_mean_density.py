"""Independent reciprocal-score lower-tail and integration-failure regressions."""

from __future__ import annotations

import math
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from heavily_right import HarmonicMean, NumericalIntegrationError
from heavily_right._harmonic_mean_lower_tail import _laplace


def _two_score_convolution(x: float, weights: list[float], *, cdf: bool) -> float:
  a, b = sorted(weights)
  # v=1-a/y absorbs the entire density a/y^2 of the narrower component.
  width = x - math.fsum(weights)
  upper = width / (a + width)

  def integrand(v: float) -> float:
    remaining = x - a / (1 - v)
    return (remaining - b) / remaining if cdf else b / remaining**2

  value, error = quad(integrand, 0, upper, epsabs=2e-14, epsrel=2e-12, limit=300)
  if error > 1e-11:
    raise AssertionError("positive bivariate convolution oracle failed")
  return float(value)


class HarmonicMeanDensityTests(unittest.TestCase):
  def test_compensated_support_residual_matches_exact_input_float_oracles(self) -> None:
    cases = (
      ([1e-12, 1-1e-12], np.nextafter(1.0, 2.0)),
      ([0.5, np.nextafter(0.5, 1.0)], np.nextafter(1.0, 2.0)),
      ([0.5, np.nextafter(0.5, 1.0)], 1.0),
      ([1e-12, np.nextafter(1-1e-12, 0.0)], 1.0),
    )
    for weights, x in cases:
      with localcontext() as context:
        context.prec = 90
        a, b = (Decimal.from_float(float(w)) for w in weights)
        point = Decimal.from_float(float(x))
        if point <= a+b:
          expected_pdf = expected_cdf = 0.0
        else:
          logarithm = ((point-a)*(point-b)/(a*b)).ln()
          expected_pdf = float(a*b*((1/a+1/b-1/(point-a)-1/(point-b))/point**2
                                    + 2*logarithm/point**3))
          expected_cdf = float(1-(a+b)/point-a*b/point**2*logarithm)
      for quantity, expected in (("pdf", expected_pdf), ("cdf", expected_cdf)):
        value, error = getattr(HarmonicMean, quantity)(x, weights, precision=True)
        with self.subTest(weights=weights, x=x, quantity=quantity):
          self.assertAlmostEqual(value, expected, delta=max(2e-11*expected, 1e-320))
          self.assertLessEqual(abs(value-expected), error + 32*np.finfo(float).eps*expected)
          if expected:
            self.assertGreater(value, 0)

  def test_scalar_count_has_exact_mean_support_without_renormalizing_vectors(self) -> None:
    for quantity in ("pdf", "cdf"):
      function = getattr(HarmonicMean, quantity)
      self.assertEqual(function(1.0, 3), 0.0)
      # Three supplied binary approximations of 1/3 sum to slightly less than
      # one as exact real inputs. They are not silently renormalized.
      self.assertGreater(function(1.0, [1/3, 1/3, 1/3]), 0.0)

  def test_single_coordinate_keeps_representable_subnormal_density(self) -> None:
    for x in (1e155, 1e160):
      with localcontext() as context:
        context.prec = 90
        expected = float(1 / Decimal.from_float(x)**2)
      value = HarmonicMean.pdf(x, 1)
      self.assertGreater(value, 0.0)
      self.assertAlmostEqual(value, expected, delta=np.nextafter(0.0, 1.0))

  def test_failed_precision_table_points_match_high_precision_inverse_laplace(self) -> None:
    # Independent 70-digit mpmath deHoog inversions, degrees 80 and 120;
    # recipe: Note/reciprocal_lower_tail_oracle_2026-09-21.py.
    for m, x, pdf, cdf in (
      (100, 2.0, 3.87326949695958998489883916623205e-7,
       1.45701416916221742627928387779698e-8),
      (1000, 4.0, 9.35136133693843362075990181561913e-6,
       6.70333929539934515919109941847425e-7),
    ):
      for quantity, expected in (("pdf", pdf), ("cdf", cdf)):
        value, error = getattr(HarmonicMean, quantity)(x, m, precision=True)
        with self.subTest(m=m, x=x, quantity=quantity):
          self.assertGreater(value, 0)
          self.assertLess(abs(value / expected - 1), 1e-10)
          self.assertLessEqual(abs(value - expected), error + 64*np.finfo(float).eps*expected)
          self.assertLess(error / value, 1e-8)
      self.assertAlmostEqual(HarmonicMean.sf(x, m) + HarmonicMean.cdf(x, m), 1, places=15)

  def test_shifted_laplace_matches_positive_stieltjes_integral(self) -> None:
    for s in (0.1+1j, 10+20j, 49+1j, 50+0j, 1e-4+50j, 100+200j, 1e6+1e6j):
      real = quad(lambda t, s=s: (t*math.exp(-t)/(s+t)).real, 0, 50,
                  epsabs=2e-15, epsrel=2e-13, limit=300)[0]
      imag = quad(lambda t, s=s: (t*math.exp(-t)/(s+t)).imag, 0, 50,
                  epsabs=2e-15, epsrel=2e-13, limit=300)[0]
      value = complex(_laplace(np.array([s], dtype=complex))[0])
      with self.subTest(s=s):
        self.assertLess(abs(value - complex(real, imag)), 2e-13*abs(value) + 2e-15)

  def test_weighted_two_coordinate_pdf_and_cdf_match_positive_convolution(self) -> None:
    for weights in ([0.5, 0.5], [0.2, 0.8], [1e-6, 1-1e-6]):
      for x in (1.0001, 1.01, 1.1, 1.5, 2.0):
        for quantity in ("pdf", "cdf"):
          expected = _two_score_convolution(x, weights, cdf=quantity == "cdf")
          actual = getattr(HarmonicMean, quantity)(x, weights)
          with self.subTest(weights=weights, x=x, quantity=quantity):
            self.assertAlmostEqual(actual, expected, delta=2e-10*abs(expected) + 2e-13)

  def test_equal_two_coordinate_survival_matches_exact_identity(self) -> None:
    for x in (1.01, 1.1, 1.5, 2.0, 5.0, 20.0):
      expected = 1/x + math.log(2*x - 1)/(2*x*x)
      self.assertAlmostEqual(HarmonicMean.sf(x, 2), expected, delta=2e-13)

  def test_support_single_coordinate_nonfinite_inputs_and_zero_weights(self) -> None:
    for count in (1, 2, 100):
      for x in (-math.inf, 0, 1):
        self.assertEqual(HarmonicMean.pdf(x, count, precision=True), (0, 0))
        self.assertEqual(HarmonicMean.cdf(x, count, precision=True), (0, 0))
        self.assertEqual(HarmonicMean.sf(x, count, precision=True), (1, 0))
      self.assertEqual(HarmonicMean.pdf(math.inf, count, precision=True), (0, 0))
      self.assertEqual(HarmonicMean.cdf(math.inf, count, precision=True), (1, 0))
      self.assertEqual(HarmonicMean.sf(math.inf, count, precision=True), (0, 0))
      for quantity in ("pdf", "cdf", "sf"):
        with self.assertRaises(ValueError):
          getattr(HarmonicMean, quantity)(math.nan, count)
    for x in (1.01, 2.0, 20.0):
      self.assertEqual(HarmonicMean.pdf(x, 1), 1/(x*x))
      self.assertEqual(HarmonicMean.sf(x, [0, 1, 0]), 1/x)
      self.assertAlmostEqual(HarmonicMean.cdf(x, 1), 1 - 1/x, places=15)
      for quantity in ("pdf", "cdf", "sf"):
        function = getattr(HarmonicMean, quantity)
        self.assertAlmostEqual(function(x, [0.2, 0, 0.3, 0.5]),
                               function(x, [0.5, 0.3, 0.2]), places=13)

  def test_representable_near_support_densities_are_not_clipped(self) -> None:
    for m, x in ((10, np.nextafter(1.0, 2.0)), (10, 1.0001),
                 (100, 1.5), (1000, 1.5), (1000, 2.0), (1000, 3.0)):
      for quantity in ("pdf", "cdf"):
        value, error = getattr(HarmonicMean, quantity)(x, m, precision=True)
        with self.subTest(m=m, x=x, quantity=quantity):
          self.assertTrue(math.isfinite(value) and math.isfinite(error))
          self.assertGreater(value, 0)
          self.assertGreater(error, 0)
          self.assertLess(error / value, 1e-8)

  def test_underflow_has_positive_bound_not_a_guessed_zero(self) -> None:
    for x in (1.0001, 1.01, 1.1):
      for quantity in ("pdf", "cdf"):
        value, error = getattr(HarmonicMean, quantity)(x, 1000, precision=True)
        with self.subTest(x=x, quantity=quantity):
          self.assertEqual(value, 0)
          self.assertTrue(math.isfinite(error))
          self.assertGreaterEqual(error, np.nextafter(0.0, 1.0))

  def test_extreme_weights_reduce_continuously_to_single_reciprocal_score(self) -> None:
    for tiny in (1e-100, 1e-200):
      self.assertAlmostEqual(HarmonicMean.pdf(1.1, [tiny, 1.0]), 1/1.1**2, delta=2e-12)
      self.assertAlmostEqual(HarmonicMean.cdf(1.1, [tiny, 1.0]), 1-1/1.1, delta=2e-12)

  def test_contours_meet_continuously_and_cdf_derivative_matches_pdf(self) -> None:
    for m in (2, 10, 100, 1000):
      threshold = HarmonicMean._lower_tail_limit(np.full(m, 1/m))
      h = 1e-5 * threshold
      for quantity in ("pdf", "cdf"):
        function = getattr(HarmonicMean, quantity)
        left = 2*function(threshold-h, m) - function(threshold-2*h, m)
        right = 2*function(threshold+h, m) - function(threshold+2*h, m)
        with self.subTest(m=m, quantity=quantity):
          self.assertAlmostEqual(left, right, delta=2e-8)
      derivative = (HarmonicMean.cdf(threshold+h, m)-HarmonicMean.cdf(threshold-h, m))/(2*h)
      self.assertAlmostEqual(derivative, HarmonicMean.pdf(threshold, m), delta=2e-8)

  def test_lower_and_upper_quantile_round_trips(self) -> None:
    for m, alpha in ((2, 1e-4), (10, 0.01), (100, 1e-5), (2, 0.95), (100, 0.95)):
      value = HarmonicMean.ppf(alpha, m)
      with self.subTest(m=m, alpha=alpha):
        self.assertAlmostEqual(HarmonicMean.cdf(value, m), alpha, delta=2e-9)

  def test_bad_quadrature_and_laplace_values_raise_typed_errors(self) -> None:
    # The reciprocal route deliberately reuses the checked quadrature adapter;
    # neither warnings nor sentinels are accepted as an alternate zero result.
    for result in ((math.nan, math.nan), (math.inf, 0), (1, math.inf),
                   (1, -1), (1, 1), (np.finfo(float).max, 0)):
      with self.subTest(result=result), \
           patch("heavily_right._half_cauchy_lower_tail.quad", return_value=result), \
           self.assertRaises(NumericalIntegrationError):
        HarmonicMean.pdf(2.0, 100)
    with patch("heavily_right._harmonic_mean_lower_tail.exp1", return_value=math.nan), \
         self.assertRaises(NumericalIntegrationError):
      HarmonicMean.pdf(2.0, 100)


if __name__ == "__main__":
  unittest.main()
