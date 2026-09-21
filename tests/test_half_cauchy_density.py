"""Independent convolution and support-boundary checks for the finite-m PDF.

The two-coordinate oracle integrates positive densities over a finite interval;
it does not share the production characteristic-function/Laplace inversion.
The tiny-density checks use elementary simplex bounds, not an absolute tolerance
that would accidentally accept zero for a representable positive density.
"""

from __future__ import annotations

import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from heavily_right import HalfCauchyMean, NumericalIntegrationError


def _two_half_cauchy_convolution(x: float, a: float, b: float) -> float:
  """Density of a*H1+b*H2, via t=a*tan(theta) in its convolution.

  Integrate the narrower scale first. The substitution absorbs its sharp
  density peak analytically, including for the 1e-6 weight below. No normalized
  weight constraint is needed for this independent oracle.
  """
  a, b = min(a, b), max(a, b)
  with warnings.catch_warnings():
    warnings.simplefilter("error", IntegrationWarning)
    value, error = quad(
      lambda theta: (
        4.0 * b
        / (math.pi**2 * ((x - a * math.tan(theta))**2 + b * b))
      ),
      0.0,
      math.atan(x / a),
      epsabs=2.0e-14,
      epsrel=2.0e-13,
      limit=200,
    )
  if not math.isfinite(value) or error > max(2.0e-14, 2.0e-13 * value):
    raise AssertionError("The independent finite convolution did not converge")
  return float(value)


class HalfCauchyDensityTests(unittest.TestCase):
  def assert_finite_density(self, value: float, error: float) -> None:
    self.assertTrue(math.isfinite(value))
    self.assertGreater(value, 0.0)
    self.assertTrue(math.isfinite(error))
    self.assertGreaterEqual(error, 0.0)

  def assert_error_covers_reference(
    self, value: float, error: float, expected: float,
  ) -> None:
    # These are cross-validation checks, not claims that the estimate is a
    # universally rigorous bound. Allow ordinary rounding of the reference.
    allowance = 64 * np.finfo(float).eps * abs(expected) + np.nextafter(0.0, 1.0)
    self.assertLessEqual(abs(value - expected), error + allowance)

  def test_two_coordinate_density_matches_positive_convolution(self) -> None:
    for weights in ((0.5, 0.5), (0.2, 0.8), (1.0e-6, 1.0 - 1.0e-6)):
      for x in (1.0e-6, 0.01, 0.1, 0.5, 2.0):
        with self.subTest(weights=weights, x=x):
          expected = _two_half_cauchy_convolution(x, *weights)
          value, error = HalfCauchyMean.pdf(x, weights, precision=True)
          self.assert_finite_density(value, error)
          self.assertAlmostEqual(
            value, expected, delta=1.0e-14 + 1.0e-10 * expected
          )

  def test_mean_versus_sum_density_has_the_required_jacobian(self) -> None:
    # If M=(H1+H2)/2, f_M(x)=2*f_(H1+H2)(2*x), not f_sum(2*x).
    for x in (0.01, 0.5, 2.0):
      with self.subTest(x=x):
        expected = 2.0 * _two_half_cauchy_convolution(2.0 * x, 1.0, 1.0)
        self.assertAlmostEqual(
          HalfCauchyMean.pdf(x, 2), expected,
          delta=1.0e-14 + 1.0e-10 * expected,
        )

  def test_tiny_m10_density_obeys_elementary_convolution_bounds(self) -> None:
    # Each simplex coordinate lies in [0, m*x]. Bounding each factor
    # 1/(1+t_j**2) gives U/(1+(m*x)**2)**m <= f(x) <= U.
    # The grid value is the actual former Landau plotting failure, not zero.
    grid_x = float(np.arange(-1.0, 9.0, 0.01)[100])
    self.assertGreater(grid_x, 0.0)
    m = 10
    for x in (8.88e-16, grid_x, 0.001, 0.01):
      with self.subTest(x=x):
        log_upper = (
          m * math.log(2.0 / math.pi)
          + m * math.log(m)
          + (m - 1) * math.log(x)
          - math.lgamma(m)
        )
        upper = math.exp(log_upper)
        lower = math.exp(log_upper - m * math.log1p((m * x)**2))
        value, error = HalfCauchyMean.pdf(x, m, precision=True)
        self.assert_finite_density(value, error)
        self.assertGreaterEqual(value, lower * (1.0 - 2.0e-12))
        self.assertLessEqual(value, upper * (1.0 + 2.0e-12))

  def test_m10_small_to_moderate_and_tail_results_have_error_estimates(self) -> None:
    # Exercise multiple numerical regimes without coupling the public contract
    # to a private, tunable dispatch threshold.
    for x in (0.1, 0.2, 0.4, 0.5, 1.0, 4.0, 8.0, 10.0):
      with self.subTest(x=x):
        result = HalfCauchyMean.pdf(x, 10, precision=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assert_finite_density(*result)

  def test_m10_density_agrees_with_independent_high_precision_references(self) -> None:
    # Independent 50/80-digit de Hoog and Cohen inversions; see the project
    # note HALF_CAUCHY_DENSITY_ORACLE_2026-09-21.md for the reproducible oracle.
    references = (
      (0.01, 3.0077997241492881812495813810650164e-16),
      (0.1, 2.5302244695915063994790903221141080e-7),
      (0.2, 8.1541631828556601331656170693661547e-5),
      (0.5, 0.034037379736545367572470973723265807),
    )
    for x, expected in references:
      with self.subTest(x=x):
        value, error = HalfCauchyMean.pdf(x, 10, precision=True)
        self.assert_finite_density(value, error)
        # Relative comparison is intentional: a fixed 1e-14 absolute tolerance
        # would accept the erroneous zero answer for the first reference.
        self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-9)
        self.assert_error_covers_reference(value, error, expected)

  def test_direct_small_cdf_matches_high_precision_inversion(self) -> None:
    # Direct 65-digit inversion of L(s/m)**m/s, independent of 1-sf(x).
    references = (
      (0.01, 3.0087107333065109274232392175898600e-19),
      (0.1, 2.6037233244268005771237709828850620e-9),
      (0.2, 1.8051385443885068623919975651059092e-6),
      (0.5, 0.0026247015692195846242226436063277600),
    )
    for x, expected in references:
      with self.subTest(x=x):
        value, error = HalfCauchyMean.cdf(x, 10, precision=True)
        self.assert_finite_density(value, error)
        self.assertLess(value, 1.0)
        self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-9)
        self.assert_error_covers_reference(value, error, expected)

  def test_pdf_and_cdf_join_smoothly_across_the_contour_switch(self) -> None:
    for m in (2, 10, 100, 1000):
      limit = HalfCauchyMean._lower_tail_limit(np.full(m, 1.0 / m))
      h = 1.0e-4 * max(1.0, limit)
      pdf = np.array([
        HalfCauchyMean.pdf(limit + offset * h, m, precision=True)
        for offset in (-2, -1, 0, 1, 2)
      ])
      cdf = np.array([
        HalfCauchyMean.cdf(limit + offset * h, m, precision=True)
        for offset in (-2, -1, 0, 1, 2)
      ])
      with self.subTest(m=m):
        for value, error in pdf:
          self.assert_finite_density(value, error)
        for value, error in cdf:
          self.assert_finite_density(value, error)
          self.assertLess(value, 1.0)
        for quantities in (pdf, cdf):
          # Extrapolate each side to the same x. Distinct x values must not
          # simply be equal: a real slope is expected across the switch.
          left = 2.0 * quantities[1, 0] - quantities[0, 0]
          right = 2.0 * quantities[3, 0] - quantities[4, 0]
          numerical_error = (
            2.0 * quantities[1, 1] + quantities[0, 1]
            + 2.0 * quantities[3, 1] + quantities[4, 1]
          )
          self.assertAlmostEqual(
            left, right,
            delta=numerical_error + 2.0e-6 * quantities[2, 0],
          )
        # On each side separately, the CDF difference/h must approximate the
        # mean endpoint PDF (trapezoid error is second order in h).
        for lo, hi in ((0, 1), (3, 4)):
          derivative = (cdf[hi, 0] - cdf[lo, 0]) / h
          expected = 0.5 * (pdf[hi, 0] + pdf[lo, 0])
          numerical_error = (cdf[hi, 1] + cdf[lo, 1]) / h
          self.assertAlmostEqual(
            derivative, expected,
            delta=numerical_error + 2.0e-6 * expected,
          )

  def test_failed_fourier_quadrature_raises_instead_of_substituting_zero(self) -> None:
    for bad_result in ((math.nan, math.nan), (0.1, math.inf), (0.1, -1.0), (0.1, 1.0)):
      for quantity in ("pdf", "cdf", "sf"):
        with self.subTest(result=bad_result, quantity=quantity):
          with patch(
            "heavily_right._half_cauchy_lower_tail.quad", return_value=bad_result,
          ) as integration, self.assertRaises(NumericalIntegrationError):
            getattr(HalfCauchyMean, quantity)(0.1, 10)
          self.assertTrue(integration.called)

  def test_many_coordinate_lower_tail_is_not_replaced_by_zero(self) -> None:
    for m, x in ((100, 1.0), (1000, 2.5)):
      with self.subTest(m=m, x=x):
        self.assert_finite_density(*HalfCauchyMean.pdf(x, m, precision=True))

  def test_extremely_small_representable_density_keeps_relative_accuracy(self) -> None:
    # Independent 320/250-digit de Hoog and Cohen inversions. Both values are
    # representable: treating them as numerical zero would lose information.
    references = (
      (0.4, 2.84394213347286742500945130530454868e-249),
      (0.5, 5.17741175302779504717576382707483726e-187),
    )
    for x, expected in references:
      with self.subTest(x=x):
        value, error = HalfCauchyMean.pdf(x, 1000, precision=True)
        self.assert_finite_density(value, error)
        self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-8)
        self.assert_error_covers_reference(value, error, expected)

  def test_true_underflow_retains_a_positive_absolute_error(self) -> None:
    smallest = float(np.nextafter(0.0, 1.0))
    for x in (0.0001, 0.1, 0.3):
      for quantity in ("pdf", "cdf"):
        with self.subTest(x=x, quantity=quantity):
          value, error = getattr(HalfCauchyMean, quantity)(x, 1000, precision=True)
          self.assertEqual(value, 0.0)
          self.assertTrue(math.isfinite(error))
          self.assertGreaterEqual(error, smallest)

  def test_extreme_weight_scale_keeps_density_and_direct_cdf(self) -> None:
    # At x=a, for scales (a,1), the second density lies between
    # 2/[pi*(1+a**2)] and 2/pi throughout the convolution. Thus the PDF
    # differs from 1/pi by relative O(a**2), and integrating atan(t/a)
    # gives the CDF leading term below with the same relative bound.
    cdf_coefficient = 1.0 / math.pi - 2.0 * math.log(2.0) / math.pi**2
    for a in (1.0e-100, 1.0e-200):
      for quantity, expected in (
        ("pdf", 1.0 / math.pi), ("cdf", a * cdf_coefficient),
      ):
        with self.subTest(weight=a, quantity=quantity):
          value, error = getattr(HalfCauchyMean, quantity)(a, [a, 1.0], precision=True)
          self.assert_finite_density(value, error)
          self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-9)
          self.assert_error_covers_reference(value, error, expected)

  def test_quadpack_finite_failure_sentinel_cannot_masquerade_as_success(self) -> None:
    # QUADPACK can return DBL_MAX on failure. It is finite; two equal sentinels
    # must not subtract to a spurious zero density in the split Fourier route.
    sentinel = float(np.finfo(float).max)
    for weights in (10, (0.2, 0.8)):
      for quantity in ("pdf", "cdf", "sf"):
        with self.subTest(weights=weights, quantity=quantity):
          with patch(
            "heavily_right._half_cauchy_lower_tail.quad",
            return_value=(sentinel, 0.0),
          ) as integration, self.assertRaises(NumericalIntegrationError):
            getattr(HalfCauchyMean, quantity)(0.1, weights)
          self.assertTrue(integration.called)

  def test_former_m1000_lower_tail_shortcut_is_not_an_error_certificate(self) -> None:
    # This is the former location-minus-2.5 cutoff. Independent inversion puts
    # its CDF above the shortcut's claimed 1e-8 absolute error.
    x = 2.1667664604475318268
    expected_pdf = 2.5857730958474028568e-6
    expected_cdf = 1.040069978848716663e-7
    pdf, pdf_error = HalfCauchyMean.pdf(x, 1000, precision=True)
    cdf, cdf_error = HalfCauchyMean.cdf(x, 1000, precision=True)
    sf, sf_error = HalfCauchyMean.sf(x, 1000, precision=True)
    self.assert_finite_density(pdf, pdf_error)
    self.assertAlmostEqual(pdf / expected_pdf, 1.0, delta=2.0e-8)
    self.assert_error_covers_reference(pdf, pdf_error, expected_pdf)
    self.assertGreater(cdf, 0.0)
    self.assertAlmostEqual(cdf / expected_cdf, 1.0, delta=2.0e-8)
    self.assert_error_covers_reference(cdf, cdf_error, expected_cdf)
    self.assertAlmostEqual(sf, 1.0 - expected_cdf, delta=2.0e-15)
    self.assertAlmostEqual(cdf + sf, 1.0, delta=2.0e-15)
    for error in (cdf_error, sf_error):
      self.assertTrue(math.isfinite(error))
      self.assertGreaterEqual(error, 0.0)

  def test_precision_switch_preserves_the_scalar_value(self) -> None:
    for x, weights in ((0.01, 10), (0.5, 2), (2.0, [0.2, 0.3, 0.5])):
      with self.subTest(x=x, weights=weights):
        scalar = HalfCauchyMean.pdf(x, weights)
        value, error = HalfCauchyMean.pdf(x, weights, precision=True)
        self.assertIsInstance(scalar, float)
        self.assert_finite_density(value, error)
        self.assertEqual(scalar, value)

  def test_zero_weights_and_permutations_do_not_change_the_density(self) -> None:
    weights = np.array([0.0, 0.2, 0.3, 0.5])
    for x in (0.01, 0.5, 2.0):
      with self.subTest(x=x):
        expected = HalfCauchyMean.pdf(x, weights[1:])
        for reordered in (weights, weights[::-1], weights[[2, 0, 3, 1]]):
          value = HalfCauchyMean.pdf(x, reordered)
          self.assertAlmostEqual(
            value, expected, delta=1.0e-14 + 1.0e-10 * expected
          )

  def test_single_active_coordinate_uses_the_exact_density(self) -> None:
    for x in (1.0e-100, 1.0e-6, 0.5, 2.0, 1.0e100):
      expected = 2.0 / math.pi / (1.0 + x * x)
      for weights in (1, [0.0, 1.0, 0.0]):
        with self.subTest(x=x, weights=weights):
          value, error = HalfCauchyMean.pdf(x, weights, precision=True)
          self.assertGreater(value, 0.0)
          self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-14)
          self.assertEqual(error, 0.0)

  def test_support_boundary_and_infinite_arguments(self) -> None:
    # Preserve the existing implementation's isolated-point convention
    # f(0)=0, even though the one-sided m=1 limit is 2/pi.
    for weights in (1, 2, [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]):
      for x in (-math.inf, -1.0, 0.0, math.inf):
        with self.subTest(x=x, weights=weights):
          self.assertEqual(HalfCauchyMean.pdf(x, weights), 0.0)
          self.assertEqual(
            HalfCauchyMean.pdf(x, weights, precision=True), (0.0, 0.0)
          )

  def test_nan_argument_is_rejected_consistently(self) -> None:
    for weights in (1, 2, [0.0, 1.0, 0.0]):
      with self.subTest(weights=weights), self.assertRaises(ValueError):
        HalfCauchyMean.pdf(math.nan, weights)

  def test_invalid_weights_are_rejected_before_boundary_shortcuts(self) -> None:
    invalid_weights = (
      0, -1, 1.5, True, math.nan, [], [0.0, 0.0], [-0.2, 1.2],
      [0.2, 0.3], [math.nan, 1.0], [math.inf, 0.0], [[0.5, 0.5]],
    )
    for weights in invalid_weights:
      for x in (0.0, 1.0):
        with self.subTest(weights=weights, x=x), self.assertRaises(ValueError):
          HalfCauchyMean.pdf(x, weights)

  def test_survival_and_quantile_calibration_are_independent_of_the_pdf(self) -> None:
    # Pre-fix SF/PPF snapshots. A density repair must not silently change the
    # upper-tail calibration or start routing its inversion through the PDF.
    cases = (
      (2, 0.1440317144517751, 13.68751142121654),
      (10, 0.19573871447209878, 15.403475649506728),
      ([0.2, 0.3, 0.5], 0.15356066843681823, 14.086189994681837),
    )
    with patch.object(HalfCauchyMean, "pdf", side_effect=AssertionError("PDF called")):
      for weights, expected_sf, expected_ppf in cases:
        with self.subTest(weights=weights):
          self.assertAlmostEqual(
            HalfCauchyMean.sf(5.0, weights), expected_sf, delta=2.0e-11
          )
          self.assertAlmostEqual(
            HalfCauchyMean.ppf(0.95, weights), expected_ppf, delta=2.0e-8
          )


if __name__ == "__main__":
  unittest.main()
