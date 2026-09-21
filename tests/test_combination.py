"""Characterization tests for the extracted p-value combination layer."""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from heavily_right.combination import CombinationTest, combination_test
from heavily_right.meta_analysis import (
  MetaAnalysis1D,
  MetaAnalysisMD,
  meta_analysis_1d,
  meta_analysis_md,
)
from heavily_right.meta_analysis import combination_test as legacy_import
from heavily_right.meta_analysis import cot as legacy_cot


class CombinationTestCase(unittest.TestCase):
  def test_legacy_import_is_compatibility_alias(self):
    self.assertIs(combination_test, CombinationTest)
    self.assertIs(legacy_import, CombinationTest)
    self.assertIs(MetaAnalysis1D, meta_analysis_1d)
    self.assertIs(MetaAnalysisMD, meta_analysis_md)
    self.assertAlmostEqual(legacy_cot(np.pi / 4), 1.0)

  def test_single_study_preserves_input_p_value(self):
    analysis = CombinationTest(1, method="HCauchy", level=0.05)
    p_values = np.array([[0.01], [0.05], [0.20]])

    assert_allclose(analysis.get_global_p(p_values), p_values[:, 0])
    assert_allclose(
      analysis.make_decision(p_values),
      np.array([True, True, False]),
    )

  def test_standard_method_scores_match_legacy_formulas(self):
    p_values = np.array([0.02, 0.20, 0.70])

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      fisher = CombinationTest(3, method="Fisher")
    self.assertAlmostEqual(
      fisher.get_global_score(p_values),
      -np.log(p_values).sum(),
    )

    stouffer = CombinationTest(3, method="Stouffer")
    expected_stouffer = -norm.ppf(p_values).sum() / np.sqrt(3)
    self.assertAlmostEqual(stouffer.get_global_score(p_values), expected_stouffer)

    bonferroni = CombinationTest(3, method="Bonferroni")
    self.assertAlmostEqual(bonferroni.get_global_p(p_values), 3 * p_values.min())

  def test_threshold_round_trip_for_closed_form_methods(self):
    for method in ("Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes"):
      with self.subTest(method=method), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = CombinationTest(4, method=method)
        threshold = analysis.get_threshold_from_p(0.025)
        self.assertAlmostEqual(
          float(analysis.get_p_from_score(threshold)),
          0.025,
          places=12,
        )

  def test_all_methods_vectorize_over_replicates(self):
    p_values = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    for method in (
      "HCauchy",
      "EHMP",
      "HMP",
      "Cauchy",
      "Levy",
      "Fisher",
      "Stouffer",
      "Bonferroni",
      "Simes",
    ):
      with self.subTest(method=method), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = CombinationTest(3, method=method)
        vectorized = analysis.get_global_score(p_values)
        rowwise = np.array([analysis.get_global_score(row) for row in p_values])
        assert_allclose(vectorized, rowwise)
        np.testing.assert_array_equal(
          analysis.make_decision(p_values),
          np.array([analysis.make_decision(row) for row in p_values]),
        )

  def test_level_change_recomputes_threshold(self):
    analysis = CombinationTest(5, method="Stouffer", level=0.05)
    analysis.change_level(0.01)

    self.assertEqual(analysis.level, 0.01)
    self.assertAlmostEqual(analysis.threshold, norm.ppf(0.99))

  def test_explicit_weights_are_coerced_and_validated(self) -> None:
    expected = np.array([0.2, 0.3, 0.5])
    for values in (expected, expected.tolist(), (value for value in expected)):
      with self.subTest(input_type=type(values).__name__):
        analysis = CombinationTest(values, method="Stouffer")
        assert_allclose(analysis.weights, expected)
        self.assertFalse(analysis.weights.flags.writeable)

    for invalid in (
      [0.2, 0.3, 0.4],
      [0.2, 0.8, 0.0],
      [0.2, 0.9, -0.1],
      [0.2, 0.3, np.nan],
      [[0.5, 0.5]],
    ):
      with self.subTest(invalid=invalid), self.assertRaises(ValueError):
        CombinationTest(invalid, method="Stouffer")

  def test_method_level_count_and_pvalue_shapes_are_validated(self) -> None:
    for count in (True, 0, 2.5, np.inf):
      with self.subTest(count=count), self.assertRaises(ValueError):
        CombinationTest(count)
    with self.assertRaises(ValueError):
      CombinationTest(2, method="unknown")
    with self.assertRaises(ValueError):
      CombinationTest(2, level=1.0)

    analysis = CombinationTest(2, method="Stouffer")
    for invalid in ([0.1], [0.1, 1.1], [0.1, np.nan]):
      with self.subTest(pvalues=invalid), self.assertRaises(ValueError):
        analysis.get_global_score(invalid)

  def test_scalar_and_batch_return_shapes_are_explicit(self) -> None:
    single = CombinationTest(1)
    self.assertIsInstance(single.get_global_score(0.2), float)
    self.assertIsInstance(single.get_global_p(0.2), float)

    multiple = CombinationTest(2, method="Stouffer")
    self.assertIsInstance(multiple.get_global_score([0.1, 0.2]), float)
    batch = multiple.get_global_score([[0.1, 0.2], [0.3, 0.4]])
    self.assertEqual(batch.shape, (2,))


if __name__ == "__main__":
  unittest.main()
