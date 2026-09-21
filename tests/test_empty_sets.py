"""Regression tests for opt-in empty-confidence-set handling."""

import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from heavily_right.empty_sets import (
  EmptyRegionError,
  EmptySetAdjustmentWarning,
  EmptySetNumericalError,
  EmptySetPolicy,
  EmptySetResolutionError,
)
from heavily_right.meta_analysis import MetaAnalysis1D, MetaAnalysisMD


class EmptySetPolicyTests(unittest.TestCase):
  def test_policy_is_validated(self) -> None:
    with self.assertRaises(ValueError):
      EmptySetPolicy(max_removals=0)
    with self.assertRaises(ValueError):
      EmptySetPolicy(min_remaining=0)
    for value in (True, 1.5, np.nan, np.inf, "2"):
      with self.subTest(value=value), self.assertRaises(ValueError):
        EmptySetPolicy(max_removals=value)
    with self.assertRaises(ValueError):
      EmptySetPolicy(warn="no")

  def test_univariate_handling_is_off_by_default(self) -> None:
    analysis = MetaAnalysis1D(2)
    with warnings.catch_warnings(record=True) as caught:
      lower, upper, _, diagnostics = analysis.confidence_interval(
        [-3.0, 3.0],
        [0.5, 0.5],
        return_diagnostics=True,
      )
    self.assertEqual(caught, [])
    self.assertEqual(lower, upper)
    self.assertFalse(diagnostics.enabled)
    self.assertTrue(diagnostics.encountered)
    self.assertFalse(diagnostics.resolved)
    self.assertEqual(diagnostics.removals, ())

  def test_univariate_removal_is_reported_and_structured(self) -> None:
    analysis = MetaAnalysis1D(2)
    with self.assertWarns(EmptySetAdjustmentWarning):
      lower, upper, estimate, diagnostics = analysis.confidence_interval(
        [-3.0, 3.0],
        [0.5, 0.5],
        empty_set_policy=EmptySetPolicy(),
        study_labels=["left", "right"],
        return_diagnostics=True,
      )
    self.assertLess(lower, upper)
    self.assertTrue(diagnostics.resolved)
    survivor = diagnostics.final_original_indices[0]
    self.assertAlmostEqual(estimate, [-3.0, 3.0][survivor])
    removal = diagnostics.removals[0]
    self.assertEqual(removal.original_index, 1 - survivor)
    self.assertEqual(removal.study_label, ["left", "right"][1 - survivor])
    self.assertEqual(len(removal.global_minimizer), 1)
    self.assertEqual(diagnostics.last_global_minimizer, (estimate,))
    self.assertEqual(removal.remaining_original_indices, (survivor,))
    self.assertEqual(len(diagnostics.evaluations), 2)
    self.assertEqual(len(diagnostics.evaluations[0].candidates), 2)
    self.assertTrue(diagnostics.evaluations[0].empty)
    self.assertFalse(diagnostics.evaluations[1].empty)

  def test_numerical_scalar_failure_is_not_an_empty_set(self) -> None:
    # Tail clipping creates a flat objective and Brent cannot bracket it.
    with warnings.catch_warnings(record=True) as caught:
      with self.assertRaises(EmptySetNumericalError) as failure:
        MetaAnalysis1D(2).confidence_interval_result(
          [-10.0, 10.0], [0.01, 0.01], empty_set_policy=EmptySetPolicy(),
        )
    self.assertEqual(caught, [])
    record = failure.exception.diagnostics
    assert record is not None
    self.assertFalse(record.encountered)
    self.assertFalse(record.resolved)
    self.assertEqual(record.final_state, "unknown")
    self.assertIsNone(record.final_minimum_score)
    self.assertIsNone(record.last_global_minimizer)
    self.assertIsNone(record.evaluations[-1].empty)
    self.assertIsNone(json.loads(record.to_json())["final_minimum_score"])

  def test_scalar_failure_after_removal_retains_complete_trace(self) -> None:
    actual_once = MetaAnalysis1D._confidence_interval_once

    def fail_after_removal(analysis, *args, **kwargs):
      if analysis.num_test == 1:
        raise RuntimeError("forced scalar endpoint failure")
      return actual_once(analysis, *args, **kwargs)

    with patch.object(MetaAnalysis1D, "_confidence_interval_once", fail_after_removal):
      with self.assertRaises(EmptySetNumericalError) as failure:
        MetaAnalysis1D(2).confidence_interval_result(
          [-3., 3.], [.5, .5], empty_set_policy=EmptySetPolicy(warn=False),
        )
    record = failure.exception.diagnostics
    assert record is not None
    self.assertEqual(len(record.removals), 1)
    self.assertEqual(len(record.final_original_indices), 1)
    self.assertEqual(record.final_weights, (1.,))
    self.assertEqual(len(record.evaluations), 2)
    self.assertTrue(record.evaluations[0].empty)
    self.assertIsNone(record.evaluations[1].empty)
    self.assertIsNone(record.final_minimum_score)
    self.assertTrue(record.encountered)
    self.assertFalse(record.resolved)
    self.assertEqual(record.final_state, "unknown")
    self.assertIn("forced scalar endpoint failure", record.message)

  def test_threshold_failure_after_removal_marks_unknown_calibration(self) -> None:
    analysis = MetaAnalysis1D(2)
    with patch.object(MetaAnalysis1D, "get_threshold_from_p", side_effect=RuntimeError("cutoff failed")):
      with self.assertRaises(EmptySetNumericalError) as failure:
        analysis.confidence_interval_result(
          [-3., 3.], [.5, .5], empty_set_policy=EmptySetPolicy(warn=False),
        )
    record = failure.exception.diagnostics
    assert record is not None
    self.assertEqual(len(record.removals), 1)
    self.assertEqual(record.final_weights, (1.,))
    self.assertIsNone(record.final_threshold)
    self.assertEqual(record.final_state, "unknown")

  def test_weights_and_json_trace_are_retained_without_overwrite(self) -> None:
    weights = np.array([0.2, 0.3, 0.5])
    result = MetaAnalysis1D(weights).confidence_interval_result(
      [0.0, 0.0, 5.0], [1.0, 1.0, 0.5],
      empty_set_policy=EmptySetPolicy(max_removals=2, warn=False),
      study_labels=["first", "second", "outlier"],
    )
    record = result.empty_set_diagnostics
    self.assertTrue(record.encountered)
    self.assertTrue(record.resolved)
    retained = weights[list(record.final_original_indices)]
    np.testing.assert_allclose(record.final_weights, retained / retained.sum())
    self.assertEqual(record.original_weights, tuple(weights))
    for removal in record.removals:
      expected = weights[list(removal.remaining_original_indices)]
      np.testing.assert_allclose(removal.remaining_weights, expected / expected.sum())
    payload = json.loads(record.to_json())
    self.assertEqual(payload["schema_version"], 1)
    self.assertEqual(payload["policy"]["max_removals"], 2)
    self.assertEqual(payload["final_original_indices"], list(record.final_original_indices))
    with TemporaryDirectory() as directory:
      path = Path(directory) / "diagnostics.json"
      record.save_json(path)
      self.assertEqual(json.loads(path.read_text()), payload)
      with self.assertRaises(FileExistsError):
        record.save_json(path)

  def test_nonempty_policy_changes_nothing(self) -> None:
    analysis = MetaAnalysis1D([0.25, 0.75])
    default = analysis.confidence_interval_result([0.0, 0.0], [1.0, 1.0])
    enabled = analysis.confidence_interval_result(
      [0.0, 0.0], [1.0, 1.0], empty_set_policy=EmptySetPolicy(warn=False),
    )
    self.assertEqual((default.lower, default.upper), (enabled.lower, enabled.upper))
    self.assertEqual(enabled.empty_set_diagnostics.final_weights, (0.25, 0.75))
    self.assertFalse(enabled.empty_set_diagnostics.encountered)
    self.assertEqual(enabled.empty_set_diagnostics.evaluations[0].candidates, ())

  def test_multivariate_removal_preserves_default_failure_contract(self) -> None:
    analysis = MetaAnalysisMD(2, dim=2)
    estimates = np.array([[-4.0, 0.0], [4.0, 0.0]])
    covariance = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    with self.assertRaises(RuntimeError):
      analysis.support_interval(
        [1.0, 0.0],
        estimates,
        covariance,
        sub_dim=2,
      )
    with self.assertWarns(EmptySetAdjustmentWarning):
      result = analysis.support_interval(
        [1.0, 0.0],
        estimates,
        covariance,
        sub_dim=2,
        empty_set_policy=EmptySetPolicy(),
        study_labels=["left block", "right block"],
      )
    self.assertTrue(result.success)
    diagnostics = result.empty_set_diagnostics
    self.assertIsNotNone(diagnostics)
    assert diagnostics is not None
    self.assertTrue(diagnostics.resolved)
    removed = diagnostics.removals[0].original_index
    self.assertEqual(diagnostics.final_original_indices, (1 - removed,))
    self.assertEqual(diagnostics.removals[0].study_label, ["left block", "right block"][removed])
    self.assertEqual(len(diagnostics.removals[0].global_minimizer), 2)
    self.assertEqual(len(diagnostics.last_global_minimizer), 2)
    certificate = diagnostics.evaluations[0].minimum_diagnostics
    assert certificate is not None
    self.assertTrue(certificate.empty_certified)
    self.assertGreater(certificate.feasible_set_score_lower_bound, diagnostics.evaluations[0].threshold)

  def test_rank_losing_candidate_is_logged_and_skipped(self) -> None:
    actual_once = MetaAnalysisMD._support_interval_once

    def first_empty(analysis, *args, **kwargs):
      if analysis.num_test == 3:
        raise EmptyRegionError(
          point=np.zeros(2), p_values=np.array([0.001, 0.01, 0.02]),
          minimum_score=100.0, threshold=float(analysis.threshold),
        )
      return actual_once(analysis, *args, **kwargs)

    with patch.object(MetaAnalysisMD, "_support_interval_once", first_empty):
      result = MetaAnalysisMD(3, dim=2).support_interval(
        [1.0, 0.0], [0.0, -5.0, 5.0], [1.0, 1.0, 1.0],
        sub_dim=1, projs=np.array([[1., 0.], [0., 1.], [0., 1.]]),
        empty_set_policy=EmptySetPolicy(warn=False),
      )
    record = result.empty_set_diagnostics
    assert record is not None
    self.assertEqual(record.final_original_indices, (0, 2))
    candidates = record.evaluations[0].candidates
    self.assertEqual(candidates[0].original_index, 0)
    self.assertFalse(candidates[0].eligible)
    self.assertEqual(candidates[0].retained_projection_rank, 1)
    self.assertTrue(candidates[1].selected)
    self.assertEqual(len(candidates), 3)

  def test_unminimized_outside_point_cannot_trigger_deletion(self) -> None:
    with warnings.catch_warnings(record=True) as caught:
      result = MetaAnalysisMD(2, dim=2).support_interval(
        [1.0, 0.0], [[0.0, 0.0], [0.2, 0.0]],
        np.repeat(np.eye(2)[None], 2, axis=0), sub_dim=2,
        x0=np.array([8., 0.]), find_min=False,
        empty_set_policy=EmptySetPolicy(),
      )
    self.assertEqual(caught, [])
    self.assertTrue(result.success)
    self.assertEqual(result.empty_set_diagnostics.removals, ())

  def test_failed_multivariate_minimization_cannot_trigger_deletion(self) -> None:
    with patch(
      "heavily_right.multivariate.minimize_score_region",
      side_effect=RuntimeError("forced minimum iteration limit"),
    ):
      with warnings.catch_warnings(record=True) as caught:
        with self.assertRaisesRegex(EmptySetNumericalError, "minimum") as failure:
          MetaAnalysisMD(2, dim=2).support_interval(
            [1.0, 0.0], [[0.0, 0.0], [0.2, 0.0]],
            np.repeat(np.eye(2)[None], 2, axis=0), sub_dim=2,
            empty_set_policy=EmptySetPolicy(),
          )
    self.assertEqual(caught, [])
    record = failure.exception.diagnostics
    assert record is not None
    self.assertFalse(record.encountered)
    self.assertEqual(record.removals, ())
    self.assertEqual(record.final_state, "unknown")

  def test_multivariate_failure_after_removal_retains_complete_trace(self) -> None:
    actual_once = MetaAnalysisMD._support_interval_once

    def fail_after_removal(analysis, *args, **kwargs):
      if analysis.num_test == 1:
        raise RuntimeError("forced support failure")
      return actual_once(analysis, *args, **kwargs)

    with patch.object(MetaAnalysisMD, "_support_interval_once", fail_after_removal):
      with self.assertRaises(EmptySetNumericalError) as failure:
        MetaAnalysisMD(2, dim=2).support_interval(
          [1., 0.], [[-4., 0.], [4., 0.]],
          np.repeat(np.eye(2)[None], 2, axis=0), sub_dim=2,
          empty_set_policy=EmptySetPolicy(warn=False), study_labels=["left", "right"],
        )
    record = failure.exception.diagnostics
    assert record is not None
    removed = record.removals[0].original_index
    self.assertEqual(record.final_original_indices, (1 - removed,))
    self.assertEqual(record.removals[0].study_label, ["left", "right"][removed])
    self.assertEqual(record.final_weights, (1.,))
    self.assertEqual(len(record.evaluations), 2)
    self.assertTrue(record.evaluations[0].empty)
    self.assertIsNone(record.evaluations[1].empty)
    self.assertFalse(record.resolved)
    self.assertEqual(record.final_state, "unknown")
    self.assertIsNone(json.loads(record.to_json())["last_global_minimizer"])

  def test_unresolved_adjustment_raises_with_diagnostics(self) -> None:
    analysis = MetaAnalysisMD(3, dim=2)
    estimates = np.array([[-8.0, 0.0], [0.0, 0.0], [8.0, 0.0]])
    covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", EmptySetAdjustmentWarning)
      with self.assertRaises(EmptySetResolutionError) as caught:
        analysis.support_interval(
          [1.0, 0.0],
          estimates,
          covariance,
          sub_dim=2,
          empty_set_policy=EmptySetPolicy(max_removals=1),
        )
    diagnostics = caught.exception.diagnostics
    self.assertTrue(diagnostics.encountered)
    self.assertFalse(diagnostics.resolved)
    self.assertEqual(len(diagnostics.removals), 1)
    self.assertEqual(len(diagnostics.last_global_minimizer), 2)

  def test_multivariate_saturated_tails_remain_unknown_without_deletion(self) -> None:
    with self.assertRaises(EmptySetNumericalError) as failure:
      MetaAnalysisMD(2, dim=2).support_interval(
        [1., 0.], [[-10., 0.], [10., 0.]],
        np.repeat((np.eye(2) * 1.e-4)[None], 2, axis=0), sub_dim=2,
        empty_set_policy=EmptySetPolicy(warn=False),
      )
    record = failure.exception.diagnostics
    assert record is not None
    self.assertEqual(record.removals, ())
    self.assertEqual(record.final_state, "unknown")
    self.assertIsNone(record.final_minimum_score)


if __name__ == "__main__":
  unittest.main()
