"""Dimensionless scalar inference and independently checked minimum regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm, t

from heavily_right.empty_sets import EmptySetNumericalError, EmptySetPolicy
from heavily_right.univariate import MetaAnalysis1D


class ScalarOptimizationTests(unittest.TestCase):
  def test_translation_and_tiny_units_preserve_interval(self):
    estimates = np.array([0.1, 0.2, 0.3])
    errors = np.array([0.1, 0.12, 0.08])
    for method in ("HCauchy", "EHMP"):
      for df in (None, 2.5):
        analysis = MetaAnalysis1D(3, method=method)
        baseline = analysis.confidence_interval_result(estimates, errors, df)
        expected = [baseline.lower, baseline.upper, baseline.estimate]
        for shift, scale in ((1000.0, 1.0), (1.0e6, 1.0), (0.0, 1.0e-12)):
          with self.subTest(method=method, df=df, shift=shift, scale=scale):
            actual = analysis.confidence_interval_result(
              shift + scale * estimates, scale * errors, df,
            )
            normalized = (np.array([actual.lower, actual.upper, actual.estimate]) - shift) / scale
            assert_allclose(normalized, expected, atol=2.0e-9, rtol=2.0e-9)
            self.assertFalse(actual.empty_set_diagnostics.encountered)
            self.assertAlmostEqual(
              analysis.find_minimizer(shift + scale * estimates, scale * errors, df),
              actual.estimate, places=12,
            )

  def test_single_study_matches_analytic_interval(self):
    for df in (None, 0.5, 2.5):
      quantile = norm.isf(0.025) if df is None else t.isf(0.025, df)
      for shift, scale in ((0.0, 1.0), (1.0e6, 1.0), (0.0, 1.0e-12)):
        with self.subTest(df=df, shift=shift, scale=scale):
          result = MetaAnalysis1D(1).confidence_interval_result([shift + 0.2 * scale], 0.1 * scale, df)
          actual = (np.array([result.lower, result.upper]) - shift) / scale
          assert_allclose(actual, 0.2 + np.array([-1, 1]) * 0.1 * quantile, atol=2.0e-9)

  def test_fractional_degrees_of_freedom_scalar_and_vector_agree(self):
    analysis = MetaAnalysis1D(3)
    estimates = np.array([0.0, 0.1, 0.2])
    errors = np.array([0.5, 0.6, 0.7])
    expected = 2 * t.sf(np.abs(0.6 - estimates) / errors, 2.5)
    for df in (2.5, [2.5] * 3, np.array([2.5] * 3)):
      with self.subTest(df=df):
        assert_allclose(analysis.get_p_vector(0.6, estimates, errors, df), expected, rtol=1.0e-14)
    scalar = analysis.confidence_interval_result(estimates, errors, 2.5)
    vector = analysis.confidence_interval_result(estimates, errors, [2.5] * 3)
    assert_allclose([scalar.lower, scalar.upper, scalar.estimate], [vector.lower, vector.upper, vector.estimate], atol=0, rtol=0)

  def test_degrees_of_freedom_are_finite_positive(self):
    analysis = MetaAnalysis1D(2)
    for df in (0, -1, np.nan, np.inf, [2.5, np.nan], [2.5, np.inf]):
      with self.subTest(df=df), self.assertRaises(ValueError):
        analysis.get_p_vector(0.0, [0.0, 0.1], 1.0, df)

  def test_invalid_convexity_regime_rejected_only_for_inference(self):
    for method in ("HCauchy", "EHMP"):
      analysis = MetaAnalysis1D(2, method=method)
      self.assertTrue(np.isfinite(analysis.get_p_vector(0.0, [0.0, 0.1], 1.0, 0.5)).all())
      with self.assertRaisesRegex(ValueError, "df >= 1"):
        analysis.confidence_interval_result([0.0, 0.1], 1.0, 0.5)

  def test_subgradient_detects_minimum_at_exact_cusp(self):
    for method in ("HCauchy", "EHMP"):
      analysis = MetaAnalysis1D([0.9, 0.1], method=method)
      problem = analysis._prepare_scalar_problem(np.array([0.0, 0.2]), np.ones(2), None)
      point, score, lower_bound = analysis._scalar_minimum(problem)
      self.assertEqual(point, 0.0)
      self.assertEqual(analysis._scalar_subgradient(point, problem)[0], 0.0)
      self.assertEqual(score, lower_bound)

  def test_analytic_scalar_derivative_matches_finite_difference(self):
    for method in ("HCauchy", "EHMP"):
      for df in (None, np.array([2.5, 3.5])):
        analysis = MetaAnalysis1D([0.4, 0.6], method=method)
        problem = analysis._prepare_scalar_problem(np.array([0.0, 0.2]), np.array([0.7, 1.1]), df)
        point = 0.6
        slope, log_scale = analysis._scalar_subgradient(point, problem)
        step = 1.0e-6
        numeric = (analysis._scalar_score(point + step, problem) - analysis._scalar_score(point - step, problem)) / (2 * step)
        self.assertAlmostEqual(slope * np.exp(log_scale), numeric, places=7)

  def test_falsely_successful_minimizer_is_independently_refined(self):
    analysis = MetaAnalysis1D(3)
    estimates = np.array([0.0, 0.1, 0.2])
    baseline = analysis.find_minimizer(estimates, np.ones(3))
    fake = SimpleNamespace(success=True, x=0.0, fun=1000.0, message="false success")
    with patch("heavily_right.univariate.minimize_scalar", return_value=fake):
      actual = analysis.confidence_interval_result(
        estimates, np.ones(3), empty_set_policy=EmptySetPolicy(warn=False),
      )
    self.assertAlmostEqual(actual.estimate, baseline, places=12)
    self.assertEqual(actual.empty_set_diagnostics.removals, ())

  def test_failed_minimizer_and_saturated_tail_never_authorize_deletion(self):
    fake = SimpleNamespace(success=False, x=0.0, fun=1.0, message="forced failure")
    with patch("heavily_right.univariate.minimize_scalar", return_value=fake):
      with self.assertRaises(EmptySetNumericalError) as caught:
        MetaAnalysis1D(2).confidence_interval_result(
          [-3.0, 3.0], 0.5, empty_set_policy=EmptySetPolicy(warn=False),
        )
    self.assertEqual(caught.exception.diagnostics.removals, ())
    with self.assertRaises(EmptySetNumericalError) as saturated:
      MetaAnalysis1D(2).confidence_interval_result(
        [-10.0, 10.0], 0.01, empty_set_policy=EmptySetPolicy(warn=False),
      )
    self.assertEqual(saturated.exception.diagnostics.removals, ())

  def test_failed_boundary_root_is_not_a_successful_interval(self):
    fake = SimpleNamespace(converged=False, root=0.0)
    with patch("heavily_right.univariate.root_scalar", return_value=fake):
      with self.assertRaises(EmptySetNumericalError):
        MetaAnalysis1D(1).confidence_interval_result([0.0], 1.0)

  def test_unrepresentable_physical_endpoints_fail_with_guidance(self):
    for count in (1, 2):
      with self.subTest(count=count), self.assertRaisesRegex(
        EmptySetNumericalError, "recenter/rescale",
      ) as caught:
        MetaAnalysis1D(count).confidence_interval_result(
          [1.0e6] * count, 1.0e-13,
          empty_set_policy=EmptySetPolicy(warn=False),
        )
      self.assertEqual(caught.exception.diagnostics.removals, ())
      self.assertEqual(caught.exception.diagnostics.final_state, "unknown")

  def test_legacy_nonconvex_methods_cannot_automatically_delete(self):
    with self.assertWarns(UserWarning):
      analysis = MetaAnalysis1D(2, method="Cauchy")
    with self.assertRaisesRegex(ValueError, "requires convex"):
      analysis.confidence_interval_result(
        [0.0, 0.2], 1.0, empty_set_policy=EmptySetPolicy(warn=False),
      )


if __name__ == "__main__":
  unittest.main()
