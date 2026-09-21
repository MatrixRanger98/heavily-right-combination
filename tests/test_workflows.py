"""Shared fitted-workflow invariants and retained-study bookkeeping."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from heavily_right.empty_sets import EmptySetPolicy, EmptySetResolutionError
from heavily_right.multivariate import MetaAnalysisMD
from heavily_right.workflows import (
  Study,
  StudyCollection,
  WorkflowFitError,
  WorkflowQueryError,
  fit_studies,
)


class StudyWorkflowTests(unittest.TestCase):
  def test_studies_copy_arrays_and_validate_covariance(self) -> None:
    estimate = np.array([.2, .3])
    study = Study(estimate, np.eye(2), np.eye(2))
    estimate[:] = 10
    assert_allclose(study.estimate, [.2, .3])
    self.assertFalse(study.estimate.flags.writeable)
    for covariance in ([[1, 2], [2, 1]], [[1, .1], [0, 1]], [[1, 0], [0, 0]]):
      with self.subTest(covariance=covariance), self.assertRaises(ValueError):
        Study([0, 0], covariance, np.eye(2))
    for value in ([np.nan], [1j], [True], ["1"]):
      with self.subTest(value=value), self.assertRaises(ValueError):
        Study(value, [[1]], [[1]])

  def test_collection_identifiability_labels_and_names(self) -> None:
    inputs = StudyCollection((Study(0, 1, [1, 0]), Study(0, 1, [0, 1])), ("a", "b"))
    self.assertEqual([study.label for study in inputs.studies], ["study-1", "study-2"])
    with self.assertRaisesRegex(ValueError, "identify"):
      StudyCollection((Study(0, 1, [1, 0]),), ("a", "b"))
    with self.assertRaisesRegex(ValueError, "unique"):
      StudyCollection(inputs.studies, ("a", "a"))
    with self.assertRaisesRegex(ValueError, "labels"):
      StudyCollection(tuple(replace(study, label="duplicate") for study in inputs.studies), ("a", "b"))

  def test_scalar_negative_projection_and_direction(self) -> None:
    fit = fit_studies(StudyCollection((Study(-4, 4, [-2]),), ("theta",)))
    assert_allclose(fit.estimate, [2])
    interval = fit.support_interval(-3)
    expected = np.array([-3 * (2 + norm.isf(.025)), -3 * (2 - norm.isf(.025))])
    assert_allclose([interval.lower, interval.upper], expected)
    self.assertTrue(interval.success)
    self.assertAlmostEqual(float(-3 * interval.lower_point[0]), interval.lower)
    self.assertTrue(fit.contains([2]))

  def test_scalar_overflow_is_not_success(self) -> None:
    fit = fit_studies(StudyCollection((Study(3, 1, [1]),), ("theta",)))
    self.assertFalse(fit.support_interval(1e308).success)

  def test_md_matches_direct_engine_with_unequal_study_dimensions(self) -> None:
    inputs = StudyCollection((
      Study([.1, .2], np.eye(2) * .2, np.eye(2), label="joint"),
      Study(.1, .2, [-1, 1], label="contrast"),
    ), ("a", "b"))
    fit = fit_studies(inputs)
    actual = fit.support_interval([-1, 1])
    expected = MetaAnalysisMD(2, dim=2).support_interval(
      [-1, 1], xi_hat=[np.array([.1, .2]), np.array([.1])],
      Sigma=[np.eye(2) * .2, np.array([[.2]])],
      sub_dim=[np.array([0, 1]), np.array([2])],
      projs=np.array([[1, 0], [0, 1], [-1, 1]]),
    )
    assert_allclose([actual.lower, actual.upper], [expected.lower, expected.upper], atol=1e-7)
    self.assertTrue(actual.success)
    self.assertTrue(fit.contains(fit.estimate))

  def test_scalar_removal_is_off_by_default_and_retained_in_subsequent_queries(self) -> None:
    inputs = StudyCollection((Study(-3, .25, [1], label="left"),
                              Study(3, .25, [1], label="right")), ("theta",))
    with self.assertRaises(EmptySetResolutionError) as error:
      fit_studies(inputs)
    self.assertEqual(error.exception.diagnostics.removals, ())
    fit = fit_studies(inputs, empty_set_policy=EmptySetPolicy(warn=False))
    self.assertEqual(len(fit.original_indices), 1)
    self.assertEqual(len(fit.active_inputs.studies), 1)
    assert_allclose(fit.weights, [1])
    self.assertTrue(fit.contains(fit.estimate))
    self.assertIs(fit.support_interval(-1).empty_set_diagnostics, fit.empty_set_diagnostics)
    self.assertEqual(len(fit.empty_set_diagnostics.removals), 1)

  def test_invalid_controls_weights_and_mixed_df_are_explicit(self) -> None:
    inputs = StudyCollection((Study(0, 1, [1]),), ("theta",))
    for kwargs in ({"weights": [.5, .5]}, {"weights": [.5]}, {"weights": [1+2j]},
                   {"weights": [True]}, {"weights": ["1"]}, {"method": "Fisher"},
                   {"maxeval": 100}, {"newton_maxiter": 1}):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        fit_studies(inputs, **kwargs)
    mixed = StudyCollection((Study(0, 1, [1]), Study(0, 1, [1], df=10)), ("theta",))
    with self.assertRaisesRegex(ValueError, "mixing"):
      fit_studies(mixed)

  def test_native_result_arrays_cannot_corrupt_cached_fit(self) -> None:
    fit = fit_studies(StudyCollection((Study([0, 0], np.eye(2), np.eye(2)),), ("a", "b")))
    first = fit.support_interval([1, 0])
    expected = first.lower_point.copy()
    with self.assertRaises(ValueError):
      first.native_result.lower_point[:] = 999
    # Even a caller deliberately reenabling writes owns a detached snapshot.
    first.native_result.lower_point.setflags(write=True)
    first.native_result.lower_point[:] = 999
    second = fit.support_interval([1, 0])
    assert_allclose(second.lower_point, expected)
    self.assertTrue(second.success)

  def test_md_removal_uses_one_retained_model_for_every_direction(self) -> None:
    inputs = StudyCollection((
      Study([-2, -2], np.eye(2) * .25, np.eye(2), label="left trial"),
      Study([2, 2], np.eye(2) * .25, np.eye(2), label="right trial"),
    ), ("a", "b"))
    fit = fit_studies(inputs, empty_set_policy=EmptySetPolicy(warn=False))
    self.assertEqual(len(fit.original_indices), 1)
    self.assertEqual(len(fit.empty_set_diagnostics.removals), 1)
    survivor = fit.active_inputs.studies[0]
    assert_allclose(fit.estimate, survivor.estimate)
    assert_allclose(fit.weights, [1])
    self.assertTrue(fit.contains(survivor.estimate))
    for direction in ([1, 0], [0, 1], [1, -1]):
      interval = fit.support_interval(direction)
      self.assertTrue(interval.success)
      self.assertIs(interval.empty_set_diagnostics, fit.empty_set_diagnostics)
      self.assertEqual(len(interval.empty_set_diagnostics.removals), 1)
    with patch.object(MetaAnalysisMD, "_support_interval_once", side_effect=RuntimeError("forced failure")):
      with self.assertRaises(WorkflowQueryError) as error:
        fit.support_interval([0, 1])
    self.assertIs(error.exception.diagnostics, fit.empty_set_diagnostics)
    self.assertEqual(len(error.exception.diagnostics.removals), 1)
    self.assertEqual(error.exception.query_diagnostics.study_labels, (survivor.label,))
    self.assertEqual(error.exception.query_diagnostics.final_state, "unknown")
    self.assertEqual(len(fit.original_indices), 1)

  def test_failed_initial_endpoint_does_not_create_successful_fit(self) -> None:
    inputs = StudyCollection((Study([0, 0], np.eye(2), np.eye(2)),), ("a", "b"))
    method = MetaAnalysisMD.support_interval

    def fail(analysis, *args, **kwargs):
      result = method(analysis, *args, **kwargs)
      return replace(result, lower_diagnostics=replace(result.lower_diagnostics, success=False))

    with patch.object(MetaAnalysisMD, "support_interval", fail):
      with self.assertRaises(WorkflowFitError) as error:
        fit_studies(inputs)
    self.assertFalse(error.exception.result.success)

    def fail_nonfinite(analysis, *args, **kwargs):
      return replace(fail(analysis, *args, **kwargs), lower_point=np.array([np.nan, np.nan]))

    with patch.object(MetaAnalysisMD, "support_interval", fail_nonfinite):
      with self.assertRaises(WorkflowFitError) as error:
        fit_studies(inputs)
    self.assertTrue(np.isnan(error.exception.result.lower_point).all())

  def test_direction_and_point_validation(self) -> None:
    fit = fit_studies(StudyCollection((Study([0, 0], np.eye(2), np.eye(2)),), ("a", "b")))
    for direction in ([0, 0], [1], [1, np.inf], [True, False]):
      with self.subTest(direction=direction), self.assertRaises(ValueError):
        fit.support_interval(direction)
    for point in ([0], [0, np.nan]):
      with self.subTest(point=point), self.assertRaises(ValueError):
        fit.score(point)


if __name__ == "__main__":
  unittest.main()
