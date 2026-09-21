"""Independent covariance, design, and endpoint checks for the NMA front end."""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import chi2

from heavily_right.network import NetworkData, fit_nma, network_arm_studies, network_studies


class NetworkInputTests(unittest.TestCase):
  def test_contrast_sign_and_explicit_treatment_order(self):
    data = network_studies([1, 2], [0.2, 0.3], ["b", "c"], ["a", "b"],
      reference="a", treatments=["c", "a", "b"])
    self.assertEqual(data.parameter_names, ("c", "b"))
    assert_allclose(data.studies.studies[0].projection, [[0, 1]])
    assert_allclose(data.studies.studies[1].projection, [[1, -1]])
    assert_allclose(data.contrast_direction("a", "c"), [-1, 0])

  def test_correlated_covariance_is_authoritative_and_copied(self):
    covariance = np.array([[0.06, 0.012], [0.012, 0.10]])
    data = network_studies([1, 2], [0.2, 0.3], ["b", "c"], ["a", "a"],
      study_id=["trial", "trial"], reference="a", covariance={"trial": covariance})
    covariance[0, 0] = 99
    assert_allclose(data.studies.studies[0].covariance, [[0.06, 0.012], [0.012, 0.10]])
    self.assertEqual(data.studies.studies[0].label, "trial")

  def test_covariance_only_input_does_not_require_duplicate_se(self):
    covariance = np.array([[0.06, 0.012], [0.012, 0.10]])
    for supplied in (covariance, {"trial": covariance}, [covariance]):
      with self.subTest(covariance=supplied):
        data = network_studies([1, 2], None, ["b", "c"], ["a", "a"],
          study_id=["trial", "trial"], covariance=supplied)
        assert_allclose(data.studies.studies[0].covariance, covariance)
    with self.assertRaisesRegex(ValueError, "se is required"):
      network_studies([1], None, ["b"], ["a"])

  def test_multiarm_requires_explicit_covariance_or_independence(self):
    args = ([1, 2], [0.2, 0.3], ["b", "c"], ["a", "a"])
    with self.assertRaisesRegex(ValueError, "require within-study covariance"):
      network_studies(*args, study_id=["trial", "trial"])
    data = network_studies(*args, study_id=["trial", "trial"], multiarm="independent", df=10)
    self.assertEqual(len(data.studies.studies), 2)
    self.assertTrue(all(block.df == 10 for block in data.studies.studies))
    with self.assertRaisesRegex(ValueError, "cannot discard"):
      network_studies(*args, study_id=["trial", "trial"], multiarm="independent", covariance=np.eye(2))
    with self.assertRaisesRegex(ValueError, "cannot discard"):
      network_studies(*args, study_id=["one", "two"], multiarm="independent", covariance=np.eye(2))

  def test_redundant_and_disconnected_networks_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "disconnected"):
      network_studies([1, 2], [1, 1], ["a", "c"], ["b", "d"])
    with self.assertRaisesRegex(ValueError, "redundant"):
      network_studies([1, 2, 3], [1, 1, 1], ["b", "c", "c"], ["a", "a", "b"],
        study_id=["trial"] * 3, covariance=np.eye(3))

  def test_independent_arm_covariance_retains_shared_baseline(self):
    data = network_arm_studies([0, 1, -0.5], [0.1, 0.2, 0.15],
      ["a", "b", "c"], ["trial"] * 3, reference="a")
    block = data.studies.studies[0]
    assert_allclose(block.estimate, [1, -0.5])
    assert_allclose(block.covariance, [[0.05, 0.01], [0.01, 0.0325]])
    assert_allclose(block.projection, np.eye(2))

  def test_arm_baseline_need_not_be_network_reference(self):
    data = network_arm_studies([1, 0, -0.5], [0.2, 0.1, 0.15],
      ["b", "a", "c"], ["trial"] * 3, reference="a", treatments=["a", "b", "c"])
    block = data.studies.studies[0]
    assert_allclose(block.estimate, [-1, -1.5])
    assert_allclose(block.projection, [[-1, 0], [-1, 1]])
    assert_allclose(block.covariance, [[0.05, 0.04], [0.04, 0.0625]])

  def test_full_covariance_respects_study_boundaries_and_input_order(self):
    matrix = np.array([[0.04, 0, 0.01], [0, 0.09, 0], [0.01, 0, 0.16]])
    data = network_studies([1, 2, 3], [0.2, 0.3, 0.4], ["b", "c", "c"], ["a", "b", "a"],
      study_id=["one", "two", "one"], covariance=matrix, df={"one": 12, "two": 14})
    assert_allclose(data.studies.studies[0].covariance, [[0.04, 0.01], [0.01, 0.16]])
    self.assertEqual(data.studies.studies[1].df, 14)
    matrix[0, 1] = matrix[1, 0] = 0.001
    with self.assertRaisesRegex(ValueError, "between different study"):
      network_studies([1, 2, 3], [0.2, 0.3, 0.4], ["b", "c", "c"], ["a", "b", "a"],
        study_id=["one", "two", "one"], covariance=matrix)

  def test_invalid_arm_records_and_mixed_calibrations_fail_early(self):
    for arms in (["a", "a"], ["a", "b"]):
      with self.subTest(arms=arms), self.assertRaises(ValueError):
        network_arm_studies([0, 1], [0.1, 0.2], arms, ["one", "two"])
    with self.assertRaisesRegex(ValueError, "mixing"):
      network_studies([1, 2], [0.2, 0.3], ["b", "c"], ["a", "a"], df=[None, 12])

  def test_complex_and_nonnumeric_summaries_are_not_silently_converted(self):
    for estimate in ([1 + 2j], [True], ["1.0"]):
      with self.subTest(estimate=estimate), self.assertRaisesRegex(ValueError, "real numeric"):
        network_studies(estimate, [0.2], ["b"], ["a"])
    with self.assertRaisesRegex(ValueError, "real numeric"):
      network_studies([1], [0.2], ["b"], ["a"], covariance=np.array([[0.04 + 1j]]))

  def test_network_metadata_must_match_study_parameters(self):
    data = network_studies([1, 2], [0.2, 0.3], ["b", "c"], ["a", "a"], reference="a")
    with self.assertRaisesRegex(ValueError, "order must match"):
      NetworkData(data.studies, ("a", "c", "b"), "a")


class NetworkFitTests(unittest.TestCase):
  def test_multiarm_contrast_matches_independent_ellipsoid_formula(self):
    data = network_arm_studies([0, 1, -0.5], [0.1, 0.2, 0.15],
      ["a", "b", "c"], ["trial"] * 3, reference="a")
    fit = fit_nma(data)
    interval = fit.contrast("b", "c")
    half_width = np.sqrt(chi2.isf(0.05, 2) * (0.2**2 + 0.15**2))
    self.assertTrue(interval.success)
    assert_allclose([interval.lower, interval.upper], [1.5 - half_width, 1.5 + half_width], atol=2e-9)
    assert_allclose(fit.estimate, [1, -0.5], atol=1e-12)
    self.assertTrue(fit.contains(fit.estimate))
    reversed_interval = fit.contrast("c", "b")
    assert_allclose([reversed_interval.lower, reversed_interval.upper], [-interval.upper, -interval.lower])
    self.assertEqual(len(fit.pairwise_intervals()), 3)

  def test_reference_and_within_trial_baseline_preserve_contrast_support(self):
    means, errors = np.array([0, 1, -0.5]), np.array([0.1, 0.2, 0.15])
    arms = np.array(["a", "b", "c"])
    configurations = (("a", [0, 1, 2]), ("b", [0, 1, 2]), ("a", [1, 2, 0]))
    fits = []
    for reference, order in configurations:
      data = network_arm_studies(means[order], errors[order], arms[order], ["trial"] * 3,
        reference=reference, treatments=["a", "b", "c"])
      fits.append(fit_nma(data))
    for a, b in (("a", "b"), ("b", "c"), ("c", "a")):
      expected = fits[0].contrast(a, b)
      for fit in fits[1:]:
        with self.subTest(reference=fit.data.reference, first=a, second=b):
          actual = fit.contrast(a, b)
          self.assertTrue(actual.success)
          assert_allclose([actual.lower, actual.upper], [expected.lower, expected.upper], atol=2e-9)

  def test_two_treatment_network_uses_scalar_workflow(self):
    data = network_studies([1, 1.1], [0.2, 0.3], ["b", "b"], ["a", "a"], reference="a")
    fit = fit_nma(data)
    interval = fit.contrast("b", "a")
    self.assertTrue(interval.success)
    self.assertLess(interval.lower, fit.estimate[0])
    self.assertGreater(interval.upper, fit.estimate[0])
    self.assertIs(fit.inputs, data.studies)
    self.assertEqual(fit.original_indices, (0, 1))
    self.assertEqual(len(fit.active_inputs.studies), 2)
    assert_allclose(fit.weights, [0.5, 0.5])
    self.assertEqual(fit.threshold, fit.fit.threshold)
    with self.assertRaisesRegex(ValueError, "belong"):
      fit.contrast("b", "unknown")


if __name__ == "__main__":
  unittest.main()
