"""Numerical and identity checks for the canonical finite-m null laws."""

import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expi

from heavily_right._numerics import NumericalIntegrationError, integrate_checked
from heavily_right.null_laws import (
  HalfCauchyMean,
  HarmonicMean,
  Landau,
  scaled_exponential_integral,
)


class ScaledExponentialIntegralTests(unittest.TestCase):
  def test_direct_branch_matches_scipy_before_overflow(self) -> None:
    values = np.array([1.0e-6, 0.1, 1.0, 10.0, 49.0])
    expected = np.exp(-values) * expi(values)
    assert_allclose(scaled_exponential_integral(values), expected, rtol=2.0e-14)

  def test_asymptotic_branch_is_accurate_at_large_arguments(self) -> None:
    for value in (50.0, 100.0):
      with self.subTest(value=value):
        expected = math.exp(-value) * expi(value)
        self.assertAlmostEqual(
          scaled_exponential_integral(value), expected, places=14
        )
    self.assertAlmostEqual(
      scaled_exponential_integral(1000.0),
      0.0010010020060241204,
      places=17,
    )


class FiniteNullLawTests(unittest.TestCase):
  def test_invalid_null_law_weights_are_rejected(self) -> None:
    for law in (HalfCauchyMean, HarmonicMean):
      for weights in (0, -1, 1.5, True, np.nan, [], [0., 0.], [-.2, 1.2], [.2, .3], [np.nan, 1.]):
        with self.subTest(law=law.__name__, weights=weights), self.assertRaises(ValueError):
          law.sf(5.0, weights)

  def test_zero_weights_with_multiple_active_coordinates(self) -> None:
    for law in (HalfCauchyMean, HarmonicMean):
      self.assertAlmostEqual(law.sf(5.0, [0.5, 0., 0.5]), law.sf(5.0, 2), places=13)

  def test_nonfinite_quadrature_error_is_rejected(self) -> None:
    with patch("heavily_right._numerics.quad", return_value=(0.5, math.nan)), self.assertRaises(NumericalIntegrationError):
      integrate_checked(lambda z, x, weights: z, 0.0, 1.0, np.ones(1), precision=False, value_bounds=(0., 1.))

  def test_exact_single_coordinate_identities(self) -> None:
    for value in (0.5, 2.0, 20.0):
      with self.subTest(score="half-cauchy", value=value):
        expected = 2.0 / math.pi * math.atan(1.0 / value)
        self.assertAlmostEqual(HalfCauchyMean.sf(value, 1), expected, places=14)
      with self.subTest(score="reciprocal", value=value):
        expected = 1.0 if value <= 1.0 else 1.0 / value
        self.assertAlmostEqual(HarmonicMean.sf(value, 1), expected, places=14)

  def test_zero_weight_coordinates_reduce_to_the_active_coordinate(self) -> None:
    weights = np.array([0.0, 1.0, 0.0])
    self.assertAlmostEqual(
      HalfCauchyMean.sf(5.0, weights),
      HalfCauchyMean.sf(5.0, 1),
      places=14,
    )
    self.assertAlmostEqual(
      HarmonicMean.sf(5.0, weights),
      HarmonicMean.sf(5.0, 1),
      places=14,
    )

  def test_bivariate_reciprocal_tail_matches_exact_identity(self) -> None:
    for value in (2.0, 5.0, 20.0):
      expected = 1.0 / value + math.log(
        4.0 * (value - 0.5) ** 2
      ) / (4.0 * value * value)
      self.assertAlmostEqual(
        HarmonicMean.sf(value, np.array([0.5, 0.5])),
        expected,
        places=13,
      )

  def test_survival_functions_are_bounded_monotone_and_permutation_invariant(
    self,
  ) -> None:
    weights = np.array([0.2, 0.3, 0.5])
    reversed_weights = weights[::-1]
    for distribution in (HalfCauchyMean, HarmonicMean):
      values = np.array(
        [distribution.sf(point, weights) for point in (2.0, 5.0, 20.0)]
      )
      with self.subTest(distribution=distribution.__name__):
        self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))
        self.assertTrue(np.all(np.diff(values) <= 0.0))
        self.assertAlmostEqual(
          distribution.sf(5.0, weights),
          distribution.sf(5.0, reversed_weights),
          places=13,
        )

  def test_quantile_round_trip_and_probability_validation(self) -> None:
    for distribution in (HalfCauchyMean, HarmonicMean):
      quantile = distribution.ppf(0.95, 2)
      self.assertAlmostEqual(distribution.sf(quantile, 2), 0.05, places=9)
      for invalid in (0.0, 1.0, -0.1, 1.1):
        with self.subTest(
          distribution=distribution.__name__, alpha=invalid
        ), self.assertRaises(ValueError):
          distribution.ppf(invalid, 2)

  def test_failed_quadrature_raises_typed_error(self) -> None:
    with patch(
      "heavily_right._numerics.quad", return_value=(math.nan, math.nan)
    ), self.assertRaises(NumericalIntegrationError):
      integrate_checked(
        lambda z, x, weights: z + x + weights[0],
        0.0,
        1.0,
        np.ones(1),
        precision=False,
        value_bounds=(0.0, 1.0),
      )


class LandauRegressionTests(unittest.TestCase):
  def test_piecewise_density_and_cdf_snapshots(self) -> None:
    points = np.array([-10.0, -5.5, -1.0, 0.0, 1.0, 5.0, 50.0, 300.0])
    assert_allclose(
      Landau.pdf(points),
      [
        0.0,
        0.0,
        0.221762209,
        0.2622401260878838,
        0.163531241,
        0.0265588931,
        0.000274514184,
        7.21869609e-6,
      ],
      rtol=8.0e-9,
      atol=1.0e-15,
    )
    assert_allclose(
      Landau.cdf(points),
      [
        0.0,
        0.0,
        0.09616096,
        0.3652387234877862,
        0.57786676,
        0.85880423,
        0.98668964,
        0.99785396,
      ],
      rtol=8.0e-8,
      atol=1.0e-15,
    )

  def test_quantile_snapshot_and_scalar_return(self) -> None:
    probabilities = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])
    assert_allclose(
      Landau.ppf(probabilities),
      [-1.96126544, -1.62750616, -0.982837609, 0.575629876, 7.12867869, 66.0201227],
      rtol=8.0e-8,
    )
    self.assertIsInstance(Landau.pdf(0.0), float)
    self.assertIsInstance(Landau.cdf(0.0), float)


if __name__ == "__main__":
  unittest.main()
