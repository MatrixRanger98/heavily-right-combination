"""Bounded adversarial cases with independent analytic optimizer targets."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import chi2, norm

from heavily_right import EmptySetNumericalError, EmptySetPolicy, MetaAnalysis1D, MetaAnalysisMD
from heavily_right.projected_region import ProjectedScoreRegion
from heavily_right.support import ray_boundary, solve_support_point


def sphere_support(*, radius=1.0, scale=1.0, direction=None, **options):
  """The exact feasible set is a sphere, independent of positive score scale."""
  if direction is None:
    direction = np.array([1.0, 0.3])

  def score(point):
    return float(scale * (point @ point))

  return solve_support_point(
    score=score,
    score_gradient=lambda point: (score(point), 2.0 * scale * point),
    score_gradient_hessian=lambda point: (
      score(point), 2.0 * scale * point, 2.0 * scale * np.eye(2)
    ),
    feasible_point=np.zeros(2),
    direction=direction,
    cutoff=scale * radius**2,
    **options,
  )


class SupportAdversarialTests(unittest.TestCase):
  def test_direction_magnitude_does_not_change_support_point(self):
    direction = np.array([1.0, 0.3])
    exact_point = direction / np.linalg.norm(direction)
    for multiplier in (1.0e-150, 1.0e-12, 1.0, 1.0e12, 1.0e150):
      with self.subTest(multiplier=multiplier):
        result = sphere_support(
          direction=multiplier * direction, maxiter=30, newton_maxiter=20,
        )
        self.assertTrue(result.diagnostics.success, result.diagnostics.message)
        assert_allclose(result.point, exact_point, rtol=2.0e-7, atol=1.0e-9)
        self.assertAlmostEqual(result.value / multiplier, np.linalg.norm(direction), places=6)

  def test_small_positive_radius_retains_relative_accuracy(self):
    for radius in (1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12):
      with self.subTest(radius=radius):
        result = sphere_support(radius=radius, maxiter=30, newton_maxiter=20)
        self.assertTrue(result.diagnostics.success, result.diagnostics.message)
        assert_allclose(np.linalg.norm(result.point) / radius, 1.0, rtol=2.0e-6)

  def test_equivalent_score_scales_have_same_support(self):
    for scale in (1.0e-20, 1.0e-10, 1.0, 1.0e10, 1.0e20):
      with self.subTest(scale=scale):
        result = sphere_support(
          scale=scale, direction=np.array([1.0, 0.0]),
          newton_maxiter=0, maxiter=30,
        )
        self.assertTrue(result.diagnostics.success, result.diagnostics.message)
        assert_allclose(result.point, [1.0, 0.0], atol=1.0e-7)

  def test_tiny_score_units_do_not_certify_a_point_outside_the_set(self):
    bad_success = OptimizeResult(
      x=np.array([2.0, 0.0]), success=True, nit=1,
      message="injected optimizer success outside the unit disk",
    )
    with patch("heavily_right.support.minimize", return_value=bad_success):
      result = sphere_support(
        scale=1.0e-20, direction=np.array([1.0, 0.0]),
        newton_maxiter=0, maxiter=3,
      )
    # Recovery is allowed, but claiming success at the injected point is not.
    if result.diagnostics.success:
      self.assertLessEqual(np.linalg.norm(result.point), 1.0 + 1.0e-6)
      assert_allclose(result.value, 1.0, atol=1.0e-6)

  def test_slsqp_numerical_exceptions_recover_or_fail_explicitly(self):
    for exception in (FloatingPointError, OverflowError, np.linalg.LinAlgError):
      calls = []

      def fail_slsqp(*args, captured_calls=calls, captured_exception=exception, **kwargs):
        captured_calls.append(kwargs["method"])
        if kwargs["method"] == "SLSQP":
          raise captured_exception("forced SLSQP arithmetic failure")
        return minimize(*args, **kwargs)

      with self.subTest(exception=exception.__name__):
        with patch("heavily_right.support.minimize", side_effect=fail_slsqp):
          result = sphere_support(
            direction=np.array([1.0, 0.0]), newton_maxiter=0, maxiter=30,
          )
        self.assertTrue(result.diagnostics.success, result.diagnostics.message)
        assert_allclose(result.point, [1.0, 0.0], atol=1.0e-7)
        self.assertIn("quadratic", result.diagnostics.solver)
        self.assertLessEqual(len(calls), 10)

  def test_all_fallback_exceptions_do_not_become_success(self):
    with patch(
      "heavily_right.support.minimize",
      side_effect=FloatingPointError("forced all-fallback failure"),
    ) as optimizer:
      result = sphere_support(newton_maxiter=0, maxiter=3)
    self.assertFalse(result.diagnostics.success)
    self.assertLessEqual(optimizer.call_count, 10)
    self.assertIn("forced all-fallback failure", result.diagnostics.message)

  def test_wrong_sign_optimizer_success_is_never_certified(self):
    wrong_endpoint = OptimizeResult(
      x=np.array([-1.0, 0.0]), success=True, nit=3,
      message="injected negative-multiplier endpoint",
    )
    with patch("heavily_right.support.minimize", return_value=wrong_endpoint) as optimizer:
      result = sphere_support(
        direction=np.array([1.0, 0.0]), newton_maxiter=0, maxiter=3,
      )
    self.assertFalse(result.diagnostics.success)
    self.assertLessEqual(optimizer.call_count, 10)
    self.assertLessEqual(result.diagnostics.iterations, 30)

  def test_unbounded_ray_and_nan_trials_fail_with_bounded_work(self):
    calls = 0

    def flat_score(point):
      nonlocal calls
      calls += 1
      return 0.0

    with self.assertRaisesRegex(RuntimeError, "bracket"):
      ray_boundary(flat_score, np.zeros(2), np.array([1.0, 0.0]), 1.0, maximum_distance=4.0)
    self.assertLessEqual(calls, 8)
    with self.assertRaises((ValueError, FloatingPointError, RuntimeError)):
      ray_boundary(
        lambda point: float(point @ point) if np.linalg.norm(point) < 0.5 else np.nan,
        np.zeros(2), np.array([1.0, 0.0]), 1.0,
      )

  def test_singleton_does_not_return_a_large_uncertified_interval(self):
    result = sphere_support(radius=0.0, maxiter=20, newton_maxiter=10)
    if result.diagnostics.success:
      self.assertLess(np.linalg.norm(result.point), 1.0e-7)
    self.assertLessEqual(result.diagnostics.iterations, 3 * 10 + 10 * 20)

  def test_redundant_active_cusps_respect_disabled_dense_hessian(self):
    region = ProjectedScoreRegion(
      dimension=2, weights=np.array([0.5, 0.25, 0.25]), method="HCauchy",
      xi_hat=np.zeros((3, 1)), sigma=np.ones((3, 1, 1)),
      sub_dim=np.array([[0], [1], [1]]), degrees_freedom=None,
      projections=np.eye(2), equal_sub_dim=True,
    )
    direction = np.array([1.0, 0.2]) * region.scale
    component = region.component(direction, base_z=np.zeros(2))
    actual_evaluate = region._evaluate

    def no_dense_hessian(*args, **kwargs):
      self.assertFalse(kwargs.get("with_hessian", True))
      return actual_evaluate(*args, **kwargs)

    with patch.object(region, "_evaluate", side_effect=no_dense_hessian):
      result = solve_support_point(
        score=component.score, score_gradient=component._score_gradient,
        score_gradient_hessian=component._score_gradient_hessian,
        feasible_point=np.zeros(2), direction=direction, cutoff=0.2,
        newton_maxiter=0, maxiter=100,
        candidate_refiner=component.certify_support_candidate,
      )
    self.assertTrue(result.diagnostics.success, result.diagnostics.message)
    expected = norm.isf((2.0 / np.pi * np.arctan(1.0 / 0.4)) / 2.0)
    assert_allclose(region.point_from_standardized(result.point), [expected, 0.0], atol=1.0e-7)
    self.assertEqual(result.diagnostics.certificate_kind, "subgradient-kkt")


class InferenceAdversarialTests(unittest.TestCase):
  def test_optimizer_success_alone_cannot_authorize_a_study_deletion(self):
    wrong_minimum = OptimizeResult(
      x=np.array([8.0, 0.0]), success=True,
      message="injected successful stop away from the global minimum",
    )
    with patch("heavily_right.multivariate.minimize", return_value=wrong_minimum):
      try:
        result = MetaAnalysisMD(2, dim=2).support_interval(
          [1.0, 0.0], [[0.0, 0.0], [0.2, 0.0]],
          np.repeat(np.eye(2)[None], 2, axis=0), sub_dim=2,
          empty_set_policy=EmptySetPolicy(warn=False),
        )
      except EmptySetNumericalError as error:
        diagnostics = error.diagnostics
      else:
        diagnostics = result.empty_set_diagnostics
    self.assertIsNotNone(diagnostics)
    assert diagnostics is not None
    self.assertEqual(diagnostics.removals, ())

  def test_translated_inconsistent_regions_do_not_become_symmetric(self):
    analysis = MetaAnalysisMD([0.2, 0.8], dim=2)
    estimates = np.array([[0.0, 0.0], [0.04, 0.0]])
    covariance = np.array([np.eye(2) * 0.02, np.eye(2) * 0.04])
    direction = np.array([1.0, 0.3])
    original = analysis.support_interval(direction, estimates, covariance, sub_dim=2)
    shifted = analysis.support_interval(direction, estimates + 1.0e9, covariance, sub_dim=2)
    self.assertTrue(original.success)
    self.assertFalse(original.central_symmetry_used)
    self.assertFalse(shifted.central_symmetry_used)
    if shifted.success:
      assert_allclose(
        np.array([shifted.lower, shifted.upper]) - 1.0e9 * direction.sum(),
        [original.lower, original.upper], atol=2.0e-6, rtol=0.0,
      )
    else:
      self.assertRegex(
        shifted.lower_diagnostics.message + shifted.upper_diagnostics.message,
        "precision|recenter|reconstruction|represent",
      )

  def test_unrepresentable_offset_endpoint_is_not_reported_certified(self):
    result = MetaAnalysisMD(1, dim=2).support_interval(
      [1.0, 0.0], [[1.0e16, 1.0e16]], np.eye(2) * 1.0e-4, sub_dim=2,
    )
    self.assertFalse(result.success)
    self.assertRegex(
      result.lower_diagnostics.message + result.upper_diagnostics.message,
      "precision|recenter|reconstruction|represent",
    )

  def test_rotated_ill_conditioned_single_block_matches_ellipsoid(self):
    rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    covariance = rotation @ np.diag([1.0, 1.0e-6]) @ rotation.T
    direction = np.array([1.0, -0.9])
    result = MetaAnalysisMD(1, dim=2).support_interval(
      direction, [[0.0, 0.0]], covariance, sub_dim=2,
      maxiter=100, newton_maxiter=50,
    )
    exact = np.sqrt(chi2.isf(0.05, 2) * float(direction @ covariance @ direction))
    self.assertTrue(result.success, result.upper_diagnostics.message)
    assert_allclose([result.lower, result.upper], [-exact, exact], rtol=2.0e-6)

  def test_scalar_translation_preserves_interval_and_estimate(self):
    analysis = MetaAnalysis1D(3)
    estimates = np.array([0.1, 0.2, 0.3])
    original = analysis.confidence_interval_result(estimates, np.full(3, 0.1))
    shifted = analysis.confidence_interval_result(1.0e6 + estimates, np.full(3, 0.1))
    self.assertFalse(shifted.empty_set_diagnostics.encountered)
    assert_allclose(
      np.array([shifted.lower, shifted.upper, shifted.estimate]) - 1.0e6,
      [original.lower, original.upper, original.estimate], atol=2.0e-8,
    )

  def test_scalar_tiny_units_cannot_create_an_empty_set(self):
    analysis = MetaAnalysis1D(3)
    estimates = np.array([0.1, 0.2, 0.3])
    original = analysis.confidence_interval_result(estimates, np.full(3, 0.1))
    scaled = analysis.confidence_interval_result(estimates * 1.0e-12, np.full(3, 1.0e-13))
    self.assertFalse(scaled.empty_set_diagnostics.encountered)
    assert_allclose(
      np.array([scaled.lower, scaled.upper, scaled.estimate]) / 1.0e-12,
      [original.lower, original.upper, original.estimate], atol=2.0e-7,
    )


if __name__ == "__main__":
  unittest.main()
