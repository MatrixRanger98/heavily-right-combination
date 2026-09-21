"""Independent checks for validated convex score minima and empty-set bounds."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult, brentq

from heavily_right import EmptySetNumericalError, MetaAnalysisMD
from heavily_right.minimum import minimize_score_region
from heavily_right.projected_region import ProjectedScoreRegion


def smooth_region(separation):
  return ProjectedScoreRegion(
    dimension=2, weights=np.array([0.5, 0.5]), method="HCauchy",
    xi_hat=np.array([[-separation, 0.0], [separation, 0.0]]),
    sigma=np.repeat(np.eye(2)[None], 2, axis=0),
    sub_dim=np.array([[0, 1], [0, 1]]), degrees_freedom=None,
    projections=np.eye(2), equal_sub_dim=True,
  )


class ConvexMinimumTests(unittest.TestCase):
  def test_extreme_weight_clipped_domain_cannot_certify_false_emptiness(self):
    weights = np.array([1.0, 1.0e-151])
    estimates = np.array([[0.0, 0.0], [3.0, 0.0]])
    covariance = np.array([np.eye(2), 1.0e-10 * np.eye(2)])
    subdimensions = np.array([[0, 1], [0, 1]])
    analysis = MetaAnalysisMD(weights, dim=2)
    region = ProjectedScoreRegion(
      dimension=2, weights=weights, method="HCauchy", xi_hat=estimates,
      sigma=covariance, sub_dim=subdimensions, degrees_freedom=None,
      projections=np.eye(2), equal_sub_dim=True,
    )
    stationary = brentq(
      lambda x: region._score_gradient(np.array([x, 0.0]))[1][0],
      2.99972, 2.99976, xtol=1.0e-15,
    )
    self.assertTrue(analysis.check_cover(
      np.zeros(2), estimates, covariance, subdimensions, None, np.eye(2),
    ))
    minimum = minimize_score_region(region, np.array([stationary, 0.0]), cutoff=analysis.threshold)
    self.assertFalse(minimum.success)
    self.assertFalse(minimum.diagnostics.empty_certified)
    self.assertIn("probability floor", minimum.diagnostics.message)
    with self.assertRaises(EmptySetNumericalError) as context:
      analysis.support_interval(
        [1.0, 0.0], estimates, covariance, subdimensions, None, np.eye(2),
        x0=[stationary, 0.0], find_min=False,
      )
    self.assertEqual(context.exception.diagnostics.removals, ())

  def test_extreme_weight_domain_guard_also_applies_to_symmetric_center(self):
    analysis = MetaAnalysisMD([1.0, 1.0e-151], dim=2)
    with self.assertRaises(EmptySetNumericalError) as context:
      analysis.support_interval(
        [1.0, 0.0], np.zeros((2, 2)), np.array([np.eye(2), 1.0e-10 * np.eye(2)]),
        np.array([[0, 1], [0, 1]]), None, np.eye(2),
      )
    self.assertIn("probability floor", str(context.exception))
    self.assertEqual(context.exception.diagnostics.removals, ())

  def test_disconnected_weak_component_cannot_consume_support_budget(self):
    mass = 1.0e-12
    analysis = MetaAnalysisMD([0.9 - mass, 0.1, mass], dim=2)
    result = analysis.support_interval(
      [1.0, 0.0], np.array([[0.0], [0.7], [0.0]]), np.ones((3, 1, 1)),
      [np.array([0]), np.array([0]), np.array([1])], np.ones(3), np.eye(2),
      x0=[0.0, 5.0e10], find_min=False,
    )
    self.assertTrue(result.success)
    assert_allclose(result.minimum_point, [0.0, 0.0], atol=1.0e-8)
    assert_allclose(
      [result.lower, result.upper],
      [-(analysis.threshold - 0.07) / (1.0 - mass), (analysis.threshold + 0.07) / (1.0 - mass)],
      atol=1.0e-7,
    )

  def test_connected_weak_direction_recovers_or_fails_explicitly(self):
    mass = 1.0e-12
    analysis = MetaAnalysisMD([0.9 - mass, 0.1, mass], dim=2)
    try:
      result = analysis.support_interval(
        [1.0, 0.0], np.array([[0.0], [0.7], [0.0]]), np.ones((3, 1, 1)),
        [np.array([0]), np.array([1]), np.array([2])], np.ones(3),
        np.array([[1.0, 0.0], [1.0, 1.0e-12], [0.0, 1.0]]),
        x0=[0.0, 5.0e10], find_min=False,
      )
    except EmptySetNumericalError as error:
      self.assertIn("minimum", str(error))
      self.assertEqual(error.diagnostics.removals, ())
      return
    self.assertTrue(result.success)
    assert_allclose(
      [result.lower, result.upper],
      [-(analysis.threshold - 0.07) / (1.0 - mass), (analysis.threshold + 0.07) / (1.0 - mass)],
      atol=1.0e-6,
    )

  def test_weak_multistudy_component_requires_a_minimum_value_gap(self):
    mass = 1.0e-12
    analysis = MetaAnalysisMD([0.5 * (1.0 - mass)] * 2 + [0.9 * mass, 0.1 * mass], dim=2)
    try:
      result = analysis.support_interval(
        [1.0, 0.0], np.array([[0.0], [0.0], [0.0], [1.0e12]]),
        np.ones((4, 1, 1)), [np.array([0]), np.array([0]), np.array([1]), np.array([1])],
        np.ones(4), np.eye(2), find_min=False,
      )
    except EmptySetNumericalError as error:
      self.assertIn("gap", str(error))
      self.assertEqual(error.diagnostics.removals, ())
      return
    self.assertTrue(result.success)
    half_width = (analysis.threshold - 0.1) / (1.0 - mass)
    assert_allclose([result.lower, result.upper], [-half_width, half_width], atol=1.0e-6)

  def test_verified_initial_point_avoids_repeated_optimization(self):
    region = smooth_region(0.2)
    with patch("heavily_right.minimum.minimize", side_effect=AssertionError("unnecessary optimization")):
      result = minimize_score_region(region, np.zeros(2), cutoff=14.0)
    self.assertTrue(result.success)
    self.assertEqual(result.diagnostics.solver, "validated-start")
    self.assertEqual(result.diagnostics.iterations, 0)

  def test_smooth_compatible_score_finds_known_minimum(self):
    region = smooth_region(0.2)
    result = minimize_score_region(region, np.array([5.0, 2.0]), cutoff=14.0)
    self.assertTrue(result.success, result.diagnostics.message)
    assert_allclose(region.point_from_standardized(result.point), [0.0, 0.0], atol=1.0e-8)
    self.assertFalse(result.diagnostics.empty_certified)

  def test_unequal_scalar_weights_recover_exact_cusp_minimum(self):
    region = ProjectedScoreRegion(
      dimension=2, weights=np.array([0.4, 0.1, 0.5]), method="HCauchy",
      xi_hat=np.array([[0.0], [0.2], [0.0]]), sigma=np.ones((3, 1, 1)),
      sub_dim=np.array([[0], [0], [1]]), degrees_freedom=None,
      projections=np.eye(2), equal_sub_dim=True,
    )
    result = minimize_score_region(region, cutoff=15.0)
    self.assertTrue(result.success, result.diagnostics.message)
    assert_allclose(region.point_from_standardized(result.point), [0.0, 0.0], atol=1.0e-8)
    self.assertLess(result.diagnostics.stationarity_residual, 1.0e-8)

  def test_empty_decision_requires_a_cutoff_lower_bound(self):
    analysis = MetaAnalysisMD(2, dim=2)
    region = smooth_region(3.0)
    result = minimize_score_region(region, cutoff=analysis.threshold)
    self.assertTrue(result.success, result.diagnostics.message)
    self.assertTrue(result.diagnostics.empty_certified)
    self.assertGreater(result.diagnostics.feasible_set_score_lower_bound, analysis.threshold)
    self.assertLessEqual(result.diagnostics.feasible_set_score_lower_bound, result.score)

  def test_successful_optimizer_flag_is_not_a_stationarity_certificate(self):
    region = smooth_region(0.2)
    bogus = OptimizeResult(x=np.array([8.0, 0.0]), success=True, nit=1, message="fake success")
    with patch("heavily_right.minimum.minimize", return_value=bogus):
      result = minimize_score_region(region, cutoff=14.0)
    # The independent known GLS candidate protects this case; the distant
    # flagged point must never be used to establish empty-set evidence.
    assert_allclose(result.point, [0.0, 0.0], atol=1.0e-8)
    self.assertFalse(result.diagnostics.empty_certified)

  def test_nonstationary_optimizer_success_is_rejected(self):
    region = ProjectedScoreRegion(
      dimension=2, weights=np.array([0.4, 0.1, 0.5]), method="HCauchy",
      xi_hat=np.array([[0.0], [0.2], [0.0]]), sigma=np.ones((3, 1, 1)),
      sub_dim=np.array([[0], [0], [1]]), degrees_freedom=None,
      projections=np.eye(2), equal_sub_dim=True,
    )
    bogus = OptimizeResult(x=np.zeros(5), success=True, nit=1, message="fake success at GLS")
    with patch("heavily_right.minimum.minimize", return_value=bogus):
      result = minimize_score_region(region, cutoff=0.001)
    self.assertFalse(result.success)
    self.assertFalse(result.diagnostics.empty_certified)
    self.assertGreater(result.diagnostics.stationarity_residual, 1.0e-4)

  def test_clipped_tails_never_certify_a_minimum_or_empty_set(self):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      result = minimize_score_region(smooth_region(100.0), cutoff=14.0, maxiter=2)
    self.assertFalse(result.success)
    self.assertFalse(result.diagnostics.empty_certified)
    self.assertIsNone(result.diagnostics.feasible_set_score_lower_bound)
    self.assertIn("tail", result.diagnostics.message)

  def test_public_probability_floor_invalidates_an_otherwise_representable_tail(self):
    # chi2.sf(27**2, 2) is about 5e-159: representable in float64, but below
    # the public MD p-value floor of 1e-150. It must still fail closed.
    result = minimize_score_region(smooth_region(27.0), cutoff=14.0, maxiter=2)
    self.assertFalse(result.success)
    self.assertFalse(result.diagnostics.empty_certified)
    self.assertIsNone(result.diagnostics.feasible_set_score_lower_bound)
    self.assertIn("clipped", result.diagnostics.message)

  def test_iteration_budget_and_invalid_inputs(self):
    for value in (True, 0, 1.5):
      with self.subTest(value=value), self.assertRaises(ValueError):
        minimize_score_region(smooth_region(0.2), maxiter=value)
    with self.assertRaises(ValueError):
      minimize_score_region(smooth_region(0.2), initial_point=np.array([np.nan, 0.0]))


if __name__ == "__main__":
  unittest.main()
