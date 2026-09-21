"""Regression tests for smooth projected-region support calculations."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult, minimize
from scipy.special import beta as beta_function
from scipy.stats import f, norm

from heavily_right.meta_analysis import MetaAnalysisMD
from heavily_right.projected_region import ProjectedScoreRegion
from heavily_right.support import solve_support_point
from reproduction.network_meta_analysis.senn2013 import (
  DESIGN_MATRIX,
  ESTIMATES,
  STANDARD_ERRORS,
)


class ProjectedScoreDerivativeTests(unittest.TestCase):
  def test_nonconvex_multistudy_degrees_are_not_globally_certified(self):
    with self.assertRaisesRegex(ValueError, "convex-score regime"):
      ProjectedScoreRegion(
        dimension=2, weights=np.array([0.5, 0.5]), method="HCauchy",
        xi_hat=np.zeros((2, 2)), sigma=np.array([np.eye(2), np.eye(2)]),
        sub_dim=np.array([[0, 1], [0, 1]]), degrees_freedom=np.array([2.0, 2.0]),
        projections=np.eye(2), equal_sub_dim=True,
      )

  def test_invalid_covariance_is_rejected_before_optimization(self):
    for covariance in (
      np.array([[1.0, 0.5], [0.0, 1.0]]),
      np.array([[1.0, 2.0], [2.0, 1.0]]),
      np.array([[np.nan, 0.0], [0.0, 1.0]]),
    ):
      with self.subTest(covariance=covariance), self.assertRaises(ValueError):
        ProjectedScoreRegion(
          dimension=2,
          weights=np.ones(1),
          method="HCauchy",
          xi_hat=np.zeros((1, 2)),
          sigma=covariance[np.newaxis],
          sub_dim=np.array([[0, 1]]),
          degrees_freedom=None,
          projections=np.eye(2),
          equal_sub_dim=True,
        )

  def setUp(self):
    self.region = ProjectedScoreRegion(
      dimension=5,
      weights=np.array([0.5, 0.5]),
      method="HCauchy",
      xi_hat=np.zeros((2, 4)),
      sigma=np.array(
        [
          np.eye(4),
          np.diag([1.2, 0.8, 1.1, 0.9]),
        ]
      ),
      sub_dim=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
      degrees_freedom=np.array([15, 17]),
      projections=np.eye(5),
      equal_sub_dim=True,
    )

  def test_analytic_gradient_and_hessian_match_central_differences(self):
    point = np.array([0.3, -0.2, 0.25, 0.1, -0.15])
    _, gradient, hessian = self.region._score_gradient_hessian(point)
    basis = np.eye(point.size)
    step = 2.0e-6
    finite_gradient = np.array(
      [
        (
          self.region.score(point + step * direction)
          - self.region.score(point - step * direction)
        )
        / (2.0 * step)
        for direction in basis
      ]
    )
    finite_hessian = np.column_stack(
      [
        (
          self.region._score_gradient(point + step * direction)[1]
          - self.region._score_gradient(point - step * direction)[1]
        )
        / (2.0 * step)
        for direction in basis
      ]
    )

    assert_allclose(gradient, finite_gradient, rtol=2.0e-6, atol=2.0e-8)
    assert_allclose(hessian, finite_hessian, rtol=2.0e-6, atol=2.0e-8)
    assert_allclose(hessian, hessian.T, atol=1.0e-14)

  def test_q4_f_density_derivative_has_finite_zero_limit(self):
    numerator_df = 4
    denominator_df = 12
    expected = (numerator_df / denominator_df) ** 2 / beta_function(2, 6)
    actual = ProjectedScoreRegion._f_density_derivative(
      0.0,
      numerator_df,
      denominator_df,
      density=0.0,
    )
    self.assertAlmostEqual(actual, expected, places=14)

  def test_underflowed_tail_probability_is_safe_for_line_search(self):
    value, derivative, second = self.region._transform(
      pvalue=0.0,
      density=0.0,
      density_derivative=0.0,
      argument_scale=1.0,
    )
    self.assertTrue(np.isfinite(value))
    self.assertEqual(derivative, 0.0)
    self.assertEqual(second, 0.0)

  def test_q1_derivatives_remain_exact_near_the_residual_cusp(self):
    region = ProjectedScoreRegion(
      dimension=1,
      weights=np.array([0.5, 0.5]),
      method="HCauchy",
      xi_hat=np.zeros((2, 1)),
      sigma=np.ones((2, 1, 1)),
      sub_dim=np.zeros((2, 1), dtype=int),
      degrees_freedom=None,
      projections=np.eye(1),
      equal_sub_dim=True,
    )
    for sign in (-1.0, 1.0):
      _, gradient, hessian = region._score_gradient_hessian(
        np.array([sign * 1.0e-10])
      )
      assert_allclose(gradient, [sign * np.sqrt(np.pi / 2.0)], rtol=1.0e-9)
      self.assertTrue(np.isfinite(hessian).all())
    score, subgradient, hessian = region._score_gradient_hessian(np.zeros(1))
    self.assertTrue(np.isfinite(score))
    assert_allclose(subgradient, [0.0])
    self.assertFalse(np.isfinite(hessian).all())

  def test_nearby_incompatible_cusps_are_not_snapped_to_one_point(self):
    region = ProjectedScoreRegion(
      dimension=1,
      weights=np.array([0.5, 0.5]),
      method="HCauchy",
      xi_hat=np.array([[0.0], [5.0e-10]]),
      sigma=np.ones((2, 1, 1)),
      sub_dim=np.zeros((2, 1), dtype=int),
      degrees_freedom=None,
      projections=np.eye(1),
      equal_sub_dim=True,
    )
    component = region.component(np.ones(1), base_z=np.zeros(1))
    self.assertIsNone(component.certify_support_candidate(np.zeros(1), np.ones(1), 0.2))


class SupportIntervalTests(unittest.TestCase):
  def test_epigraph_fallback_handles_five_simultaneous_scalar_cusps(self):
    dimension = 6
    region = ProjectedScoreRegion(
      dimension=dimension, weights=np.full(dimension, 1.0 / dimension),
      method="HCauchy", xi_hat=np.zeros((dimension, 1)),
      sigma=np.ones((dimension, 1, 1)), sub_dim=np.arange(dimension)[:, None],
      degrees_freedom=None, projections=np.eye(dimension), equal_sub_dim=True,
    )
    original_direction = np.array([1.0, 0.05, 0.04, 0.03, 0.02, 0.01])
    direction = original_direction * region.scale
    component = region.component(direction, base_z=np.zeros(dimension))
    result = component.solve_nonsmooth_support_candidate(
      np.r_[0.1, np.zeros(dimension - 1)], direction, 0.2,
    )
    self.assertIsNotNone(result)
    self.assertEqual(result.solver, "epigraph-active-set-newton")
    self.assertEqual(len(result.attempts), 16)
    self.assertGreater(result.evaluations, 0)
    pvalue = 2.0 / np.pi * np.arctan(1.0 / (dimension * 0.2))
    expected = np.r_[norm.isf(pvalue / 2.0), np.zeros(dimension - 1)]
    assert_allclose(region.point_from_standardized(result.point), expected, atol=1.0e-8)
    self.assertLess(abs(result.score - 0.2), 1.0e-9)
    eta = float(result.gradient @ direction) / (direction @ direction)
    self.assertGreater(eta, 0.0)
    assert_allclose(result.gradient, eta * direction, atol=1.0e-8)

  def test_nonsmooth_scalar_endpoint_uses_bounded_subgradient_certificate(self):
    region = ProjectedScoreRegion(
      dimension=2,
      weights=np.array([0.5, 0.5]),
      method="HCauchy",
      xi_hat=np.zeros((2, 1)),
      sigma=np.ones((2, 1, 1)),
      sub_dim=np.array([[0], [1]]),
      degrees_freedom=None,
      projections=np.eye(2),
      equal_sub_dim=True,
    )
    direction = np.array([1.0, 0.2]) * region.scale
    component = region.component(direction, base_z=np.zeros(2))
    result = solve_support_point(
      score=component.score,
      score_gradient=component._score_gradient,
      score_gradient_hessian=component._score_gradient_hessian,
      feasible_point=np.zeros(2),
      direction=direction,
      cutoff=0.2,
      candidate_refiner=component.certify_support_candidate,
    )
    point = region.point_from_standardized(result.point)
    pvalue = 2.0 / np.pi * np.arctan(1.0 / 0.4)
    expected = norm.isf(pvalue / 2.0)
    self.assertTrue(result.diagnostics.success, result.diagnostics.message)
    self.assertEqual(result.diagnostics.certificate_kind, "subgradient-kkt")
    assert_allclose(point, [expected, 0.0], atol=1.0e-8)
    self.assertLess(result.diagnostics.kkt_residual, 1.0e-7)

  def test_disabled_newton_never_evaluates_a_hessian(self):
    def forbidden_hessian(point):
      self.fail("the dimension-cap path must not evaluate a dense Hessian")

    result = solve_support_point(
      score=lambda point: float(point @ point),
      score_gradient=lambda point: (float(point @ point), 2.0 * point),
      score_gradient_hessian=forbidden_hessian,
      feasible_point=np.zeros(2),
      direction=np.array([1.0, 0.0]),
      cutoff=1.0,
      newton_maxiter=0,
    )
    self.assertTrue(result.diagnostics.success)
    self.assertEqual(result.diagnostics.solver, "slsqp")

  def test_hessian_numerical_failure_recovers_with_slsqp(self):
    def failed_hessian(point):
      raise FloatingPointError("curvature evaluation failed")

    result = solve_support_point(
      score=lambda point: float(point @ point),
      score_gradient=lambda point: (float(point @ point), 2.0 * point),
      score_gradient_hessian=failed_hessian,
      feasible_point=np.zeros(2),
      direction=np.array([1.0, 0.0]),
      cutoff=1.0,
    )
    self.assertTrue(result.diagnostics.success)
    self.assertEqual(result.diagnostics.solver, "slsqp")

  def test_support_certificate_allows_translated_negative_support(self):
    center = np.array([-3.0, 0.0])

    def score(point):
      return float((point - center) @ (point - center))

    result = solve_support_point(
      score=score,
      score_gradient=lambda point: (score(point), 2.0 * (point - center)),
      score_gradient_hessian=lambda point: (
        score(point), 2.0 * (point - center), 2.0 * np.eye(2)
      ),
      feasible_point=center,
      direction=np.array([1.0, 0.0]),
      cutoff=1.0,
    )
    self.assertTrue(result.diagnostics.success)
    self.assertAlmostEqual(result.value, -2.0)

  def test_penalty_is_used_only_after_slsqp_failure(self):
    direction = np.array([1.0, 0.0])

    def score(point):
      return float(point @ point)

    def score_gradient(point):
      return score(point), 2.0 * point

    def score_gradient_hessian(point):
      return score(point), 2.0 * point, 2.0 * np.eye(2)

    scipy_minimize = minimize

    def staged_minimize(*args, **kwargs):
      if kwargs.get("method") == "SLSQP":
        return OptimizeResult(
          # This point has zero unsigned merit but the wrong multiplier
          # sign. A later certified penalty candidate must replace it.
          x=np.array([-1.0, 0.0]),
          success=False,
          nit=1,
          message="forced SLSQP failure",
        )
      return scipy_minimize(*args, **kwargs)

    with patch("heavily_right.support.minimize", side_effect=staged_minimize):
      result = solve_support_point(
        score=score,
        score_gradient=score_gradient,
        score_gradient_hessian=score_gradient_hessian,
        feasible_point=np.zeros(2),
        direction=direction,
        cutoff=1.0,
        newton_maxiter=0,
      )

    self.assertTrue(result.diagnostics.success)
    self.assertEqual(result.diagnostics.solver, "quadratic-penalty")
    self.assertIn("forced SLSQP failure", result.diagnostics.message)
    assert_allclose(result.point, [1.0, 0.0], atol=1.0e-10)

  def test_single_block_hotelling_interval_matches_analytic_ellipsoid(self):
    dimension = 4
    degrees_freedom = 20
    estimate = np.array([0.3, -0.2, 0.5, 0.1])
    covariance = np.array(
      [
        [1.0, 0.1, 0.0, 0.0],
        [0.1, 0.7, 0.05, 0.0],
        [0.0, 0.05, 0.5, 0.02],
        [0.0, 0.0, 0.02, 0.3],
      ]
    ) / 30
    direction = np.array([0.2, -0.4, 0.7, 0.5])
    analysis = MetaAnalysisMD(1, dim=dimension, level=0.05)

    result = analysis.support_interval(
      direction,
      estimate[np.newaxis, :],
      covariance[np.newaxis, :, :],
      sub_dim=dimension,
      df=degrees_freedom,
    )

    denominator_df = degrees_freedom + 1 - dimension
    critical_quadratic = (
      f.ppf(0.95, dimension, denominator_df)
      * dimension
      * degrees_freedom
      / denominator_df
    )
    half_width = np.sqrt(
      critical_quadratic * direction @ covariance @ direction
    )
    center = direction @ estimate
    assert_allclose(result.lower, center - half_width, atol=2.0e-12)
    assert_allclose(result.upper, center + half_width, atol=2.0e-12)
    self.assertTrue(result.success)
    self.assertTrue(result.central_symmetry_used)
    self.assertEqual(result.upper_diagnostics.solver, "analytic-ellipsoid")
    self.assertEqual(
      result.lower_diagnostics.solver,
      "analytic-ellipsoid",
    )
    self.assertGreaterEqual(result.lower_diagnostics.evaluations, 1)
    self.assertGreater(result.upper_diagnostics.eta, 0.0)
    self.assertLess(abs(result.upper_diagnostics.boundary_error), 1.0e-11)
    self.assertLess(result.upper_diagnostics.kkt_residual, 1.0e-11)

  def test_overlapping_inconsistent_system_matches_direct_constrained_solve(self):
    estimate = np.array([[0.1, -0.2], [0.4, 0.3]])
    covariance = np.array(
      [
        [[0.12, 0.02], [0.02, 0.09]],
        [[0.08, -0.01], [-0.01, 0.11]],
      ]
    )
    sub_dim = np.array([[0, 1], [1, 2]])
    degrees_freedom = np.array([18, 21])
    direction = np.array([0.7, -0.2, 0.5])
    analysis = MetaAnalysisMD(2, dim=3, method="HCauchy", level=0.05)

    result = analysis.support_interval(
      direction,
      estimate,
      covariance,
      sub_dim,
      degrees_freedom,
    )

    def score(point):
      pvalues = analysis.get_p_vector(
        point,
        estimate,
        covariance,
        sub_dim,
        degrees_freedom,
      )
      return analysis.get_global_score(pvalues)

    feasible = minimize(
      score,
      np.array([0.1, 0.1, 0.3]),
      method="Powell",
      options={"xtol": 1.0e-11, "ftol": 1.0e-11},
    ).x
    direct_values = []
    for sign in (-1.0, 1.0):
      constrained = minimize(
        lambda point, sign=sign: -sign * direction @ point,
        feasible,
        jac=lambda point, sign=sign: -sign * direction,
        constraints={
          "type": "ineq",
          "fun": lambda point: analysis.threshold - score(point),
        },
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 2000},
      )
      self.assertTrue(constrained.success, constrained.message)
      direct_values.append(float(direction @ constrained.x))

    assert_allclose(
      [result.lower, result.upper],
      direct_values,
      rtol=1.0e-8,
      atol=1.0e-8,
    )
    self.assertTrue(result.success)
    self.assertFalse(result.central_symmetry_used)
    self.assertEqual(result.lower_diagnostics.solver, "kkt-newton")
    self.assertEqual(result.upper_diagnostics.solver, "kkt-newton")
    self.assertGreater(result.lower_diagnostics.eta, 0.0)
    self.assertGreater(result.upper_diagnostics.eta, 0.0)

  def test_multistudy_support_is_invariant_to_small_measurement_units(self):
    estimate = np.array([[0.1, -0.2], [0.4, 0.3]])
    covariance = np.array([
      [[0.12, 0.02], [0.02, 0.09]],
      [[0.08, -0.01], [-0.01, 0.11]],
    ])
    analysis = MetaAnalysisMD(2, dim=2, method="HCauchy", level=0.05)
    direction = np.array([0.7, -0.2])
    reference = analysis.support_interval(direction, estimate, covariance, sub_dim=2)
    units = 1.0e-12
    scaled = analysis.support_interval(
      direction, estimate * units, covariance * units**2, sub_dim=2,
    )
    self.assertTrue(reference.success)
    self.assertTrue(scaled.success)
    assert_allclose(
      np.array([scaled.lower, scaled.upper]) / units,
      [reference.lower, reference.upper], rtol=1.0e-7, atol=1.0e-9,
    )

  def test_components_limit_dense_solve_and_dimension_cap_selects_fallback(self):
    dimension = 8
    analysis = MetaAnalysisMD(2, dim=dimension, level=0.05)
    estimate = np.zeros((2, 4))
    covariance = np.repeat(np.eye(4)[np.newaxis, :, :], 2, axis=0)
    sub_dim = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    direction = np.zeros(dimension)
    direction[0] = 1.0

    original_inverse = np.linalg.inv
    with patch(
      "heavily_right.projected_region.np.linalg.inv",
      wraps=original_inverse,
    ) as inverse:
      newton = analysis.support_interval(
        direction,
        estimate,
        covariance,
        sub_dim,
        df=np.array([20, 20]),
        dense_newton_max_dimension=4,
      )
    fallback = analysis.support_interval(
      direction,
      estimate,
      covariance,
      sub_dim,
      df=np.array([20, 20]),
      dense_newton_max_dimension=3,
    )

    self.assertEqual(newton.active_dimension, 4)
    inverted_shapes = [call.args[0].shape for call in inverse.call_args_list]
    self.assertNotIn((dimension, dimension), inverted_shapes)
    self.assertTrue(all(max(shape) <= 4 for shape in inverted_shapes))
    self.assertEqual(newton.upper_diagnostics.solver, "kkt-newton")
    self.assertEqual(fallback.active_dimension, 4)
    self.assertEqual(fallback.upper_diagnostics.solver, "slsqp")
    self.assertIn(
      "dense KKT Newton disabled",
      fallback.upper_diagnostics.message,
    )
    self.assertTrue(fallback.success)
    assert_allclose(fallback.lower, newton.lower, atol=2.0e-7)
    assert_allclose(fallback.upper, newton.upper, atol=2.0e-7)

  def test_network_meta_analysis_q1_contrast_converges(self):
    estimates = np.delete(np.delete(ESTIMATES, 0), 5)
    standard_errors = np.delete(np.delete(STANDARD_ERRORS, 0), 5)
    projections = np.delete(np.delete(DESIGN_MATRIX, 0, axis=0), 5, axis=0)
    analysis = MetaAnalysisMD(
      len(estimates),
      dim=projections.shape[1],
      method="HCauchy",
      level=0.05,
    )
    direction = np.eye(projections.shape[1])[0] - np.eye(
      projections.shape[1]
    )[1]

    result = analysis.support_interval(
      direction,
      estimates,
      standard_errors**2,
      sub_dim=1,
      df=None,
      projs=projections,
    )

    self.assertTrue(result.success)
    assert_allclose(result.lower, -0.5786995762, atol=2.0e-8)
    assert_allclose(result.upper, 0.8223210519, atol=2.0e-8)
    self.assertLess(abs(result.lower_diagnostics.boundary_error), 1.0e-9)
    self.assertLess(abs(result.upper_diagnostics.boundary_error), 1.0e-9)

  def test_network_meta_analysis_metformin_direction_is_certified(self):
    estimates = np.delete(np.delete(ESTIMATES, 0), 5)
    standard_errors = np.delete(np.delete(STANDARD_ERRORS, 0), 5)
    projections = np.delete(np.delete(DESIGN_MATRIX, 0, axis=0), 5, axis=0)
    analysis = MetaAnalysisMD(
      len(estimates),
      dim=projections.shape[1],
      method="HCauchy",
      level=0.05,
    )

    result = analysis.support_interval(
      np.eye(projections.shape[1])[2],
      estimates,
      standard_errors**2,
      sub_dim=1,
      df=None,
      projs=projections,
    )

    self.assertTrue(result.success)
    assert_allclose(result.lower, -1.1232903127, atol=2.0e-8)
    assert_allclose(result.upper, -0.8426895586, atol=2.0e-8)
    self.assertLess(abs(result.lower_diagnostics.boundary_error), 1.0e-8)
    self.assertLess(result.lower_diagnostics.kkt_residual, 1.0e-8)
    self.assertEqual(result.lower_diagnostics.certificate_kind, "subgradient-kkt")
    self.assertIn("SLSQP", result.lower_diagnostics.message)

  def test_network_meta_analysis_benfluorex_metformin_cusp_is_certified(self):
    retained = np.delete(np.arange(len(ESTIMATES)), [0, 6])
    # The reproduction policy renormalizes these inherited equal weights.
    weights = np.full(len(ESTIMATES), 1.0 / len(ESTIMATES))[retained]
    weights /= weights.sum()
    analysis = MetaAnalysisMD(weights, dim=DESIGN_MATRIX.shape[1], level=0.05)
    direction = np.eye(DESIGN_MATRIX.shape[1])[1] - np.eye(DESIGN_MATRIX.shape[1])[2]
    result = analysis.support_interval(
      direction,
      ESTIMATES[retained],
      STANDARD_ERRORS[retained] ** 2,
      sub_dim=1,
      projs=DESIGN_MATRIX[retained],
    )
    self.assertTrue(result.success, result.upper_diagnostics.message)
    self.assertEqual(result.upper_diagnostics.certificate_kind, "subgradient-kkt")
    self.assertLess(abs(result.upper_diagnostics.boundary_error), 1.0e-8)
    self.assertLess(result.upper_diagnostics.kkt_residual, 1.0e-8)

  def test_network_meta_analysis_metformin_miglitol_cusp_is_certified(self):
    retained = np.delete(np.arange(len(ESTIMATES)), [0, 6])
    weights = np.full(len(ESTIMATES), 1.0 / len(ESTIMATES))[retained]
    weights /= weights.sum()
    analysis = MetaAnalysisMD(weights, dim=DESIGN_MATRIX.shape[1], level=0.05)
    direction = np.eye(DESIGN_MATRIX.shape[1])[2] - np.eye(DESIGN_MATRIX.shape[1])[3]
    result = analysis.support_interval(
      direction, ESTIMATES[retained], STANDARD_ERRORS[retained] ** 2,
      sub_dim=1, projs=DESIGN_MATRIX[retained],
    )
    self.assertTrue(result.success, result.lower_diagnostics.message)
    self.assertEqual(result.lower_diagnostics.certificate_kind, "subgradient-kkt")
    self.assertLess(abs(result.lower_diagnostics.boundary_error), 1.0e-8)
    self.assertLess(result.lower_diagnostics.kkt_residual, 1.0e-8)

  def test_network_meta_analysis_metformin_pioglitazone_active_set_solve(self):
    retained = np.delete(np.arange(len(ESTIMATES)), [0, 6])
    weights = np.full(len(ESTIMATES), 1.0 / len(ESTIMATES))[retained]
    weights /= weights.sum()
    analysis = MetaAnalysisMD(weights, dim=DESIGN_MATRIX.shape[1], level=0.05)
    direction = np.eye(DESIGN_MATRIX.shape[1])[2] - np.eye(DESIGN_MATRIX.shape[1])[4]
    result = analysis.support_interval(
      direction, ESTIMATES[retained], STANDARD_ERRORS[retained] ** 2,
      sub_dim=1, projs=DESIGN_MATRIX[retained],
    )
    self.assertTrue(result.success, result.lower_diagnostics.message)
    self.assertEqual(result.lower_diagnostics.certificate_kind, "subgradient-kkt")
    self.assertLess(abs(result.lower_diagnostics.boundary_error), 1.0e-8)
    self.assertLess(result.lower_diagnostics.kkt_residual, 1.0e-8)


if __name__ == "__main__":
  unittest.main()
