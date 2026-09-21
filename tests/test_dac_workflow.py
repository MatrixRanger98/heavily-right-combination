"""General DAC input construction and parity with direct inference APIs."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import f, t

from heavily_right import MetaAnalysisMD
from heavily_right.dac import DacFit, dac_studies, fit_dac


class DacConstructionTests(unittest.TestCase):
  def setUp(self):
    self.sample = np.random.default_rng(91842).normal(size=(40, 5))

  def test_default_partition_is_singleton_columns_using_all_observations(self):
    inputs = dac_studies(self.sample)
    self.assertEqual(inputs.parameter_names, ("theta1", "theta2", "theta3", "theta4", "theta5"))
    self.assertEqual(len(inputs.studies), 5)
    for coordinate, study in enumerate(inputs.studies):
      self.assertEqual(study.df, 39)
      assert_allclose(study.estimate, [self.sample[:, coordinate].mean()])
      assert_allclose(study.covariance, [[self.sample[:, coordinate].var(ddof=1) / 40]])
      assert_array_equal(study.projection, np.eye(5)[[coordinate]])

  def test_nondividing_block_size_retains_smaller_last_block_without_wraparound(self):
    inputs = dac_studies(self.sample, block_size=2)
    self.assertEqual([study.estimate.size for study in inputs.studies], [2, 2, 1])
    assert_array_equal(np.concatenate([study.projection for study in inputs.studies]), np.eye(5))

  def test_named_irregular_partition_preserves_coordinate_order_and_covariance(self):
    inputs = dac_studies(
      self.sample, blocks=[["e", "b"], ["a"], ["d", "c"]],
      parameter_names=["a", "b", "c", "d", "e"],
    )
    for study, indices in zip(inputs.studies, ([4, 1], [0], [3, 2]), strict=True):
      block = self.sample[:, indices]
      centered = block - block.mean(axis=0)
      assert_allclose(study.estimate, block.mean(axis=0))
      assert_allclose(study.covariance, centered.T @ centered / (40 * 39))
      assert_array_equal(study.projection, np.eye(5)[indices])

  def test_constructed_block_pvalues_match_hotelling_calibration(self):
    inputs = dac_studies(self.sample, blocks=[[4, 1], [0, 3, 2]])
    point = np.array([0.03, -0.02, 0.01, 0.04, -0.01])
    analysis = MetaAnalysisMD(2, dim=5)
    studies = inputs.studies
    actual = analysis.get_p_vector(
      point,
      [study.estimate for study in studies],
      [study.covariance for study in studies],
      [np.arange(2), np.arange(2, 5)],
      df=39,
      projs=np.concatenate([study.projection for study in studies]),
    )
    expected = []
    for study in studies:
      dimension = study.estimate.size
      residual = study.projection @ point - study.estimate
      statistic = residual @ np.linalg.solve(study.covariance, residual)
      expected.append(f.sf(statistic * (40 - dimension) / (dimension * 39), dimension, 40 - dimension))
    assert_allclose(actual, expected, rtol=1e-13)

  def test_bad_partitions_fail_instead_of_repeating_or_omitting_coordinates(self):
    invalid = (
      [], [[0, 1], [2, 3]], [[0, 1], [1, 2, 3, 4]],
      [[0, 0], [1, 2, 3, 4]], [[], [0, 1, 2, 3, 4]],
      [[-1, 0], [1, 2, 3]], [[0, 1, 2, 3, 5]],
      [[0.0, 1, 2, 3, 4]], [[False, 1, 2, 3, 4]],
      [["missing", "theta2", "theta3", "theta4", "theta5"]],
      "theta1", [0, 1, 2, 3, 4],
    )
    for blocks in invalid:
      with self.subTest(blocks=blocks), self.assertRaises(ValueError):
        dac_studies(self.sample, blocks=blocks)
    with self.assertRaisesRegex(ValueError, "not both"):
      dac_studies(self.sample, blocks=[[0, 1, 2, 3, 4]], block_size=2)

  def test_block_size_requires_a_positive_integer(self):
    for size in (0, -1, 1.0, 1.5, True, np.nan, "2"):
      with self.subTest(size=size), self.assertRaises(ValueError):
        dac_studies(self.sample, block_size=size)
    self.assertEqual(len(dac_studies(self.sample, block_size=np.int64(10)).studies), 1)

  def test_invalid_samples_and_parameter_names_fail_early(self):
    invalid_samples = (
      np.ones((1, 3)), np.ones((3, 0)), np.arange(10),
      [[1, 2], [3, np.nan]], [[1, 2], [3, np.inf]],
      [["1", "2"], ["3", "4"]], np.ones((3, 2), dtype=complex),
      [[1, 2], [3]],
    )
    for sample in invalid_samples:
      with self.subTest(sample=sample), self.assertRaises(ValueError):
        dac_studies(sample)
    for names in (["a"] * 5, ["a", "b"], ["a", "b", "c", "d", ""], "abcde"):
      with self.subTest(names=names), self.assertRaises(ValueError):
        dac_studies(self.sample, parameter_names=names)

  def test_singular_covariance_and_insufficient_hotelling_df_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "positive.definite"):
      dac_studies(np.column_stack([np.arange(8), np.arange(8)]), block_size=2)
    with self.assertRaisesRegex(ValueError, "degrees of freedom"):
      dac_studies(np.array([[1, 0], [0, 1]]), block_size=2)


class DacInferenceTests(unittest.TestCase):
  def setUp(self):
    self.sample = np.random.default_rng(18943).normal(size=(60, 5))

  def test_fit_retains_original_partition_and_sample_size(self):
    fitted = fit_dac(
      self.sample, blocks=[[4, 1], [0], [3, 2]],
      parameter_names=("a", "b", "c", "d", "e"),
    )
    self.assertIsInstance(fitted, DacFit)
    self.assertEqual(fitted.sample_size, 60)
    self.assertEqual(fitted.blocks, ((4, 1), (0,), (3, 2)))
    self.assertEqual(fitted.parameter_names, ("a", "b", "c", "d", "e"))
    self.assertEqual(len(fitted.inputs.studies), 3)
    self.assertEqual(len(fitted.active_inputs.studies), 3)
    self.assertEqual(fitted.original_indices, (0, 1, 2))
    assert_allclose(fitted.weights, np.full(3, 1 / 3))
    self.assertEqual(fitted.threshold, fitted.fit.threshold)
    assert_allclose(fitted.estimate, self.sample.mean(axis=0), atol=1e-10)
    self.assertTrue(fitted.contains(fitted.estimate))
    interval = fitted.support_interval([1, 0.1, -0.2, 0.3, 0.5])
    self.assertTrue(interval.success)
    self.assertLess(interval.lower, interval.upper)

  def test_equal_block_fit_matches_legacy_direct_api(self):
    sample = self.sample[:, :4]
    weights = [0.4, 0.6]
    fitted = fit_dac(sample, block_size=2, weights=weights, level=0.1)
    direct = MetaAnalysisMD(weights, dim=4, level=0.1)
    studies = fitted.inputs.studies
    means = np.stack([study.estimate for study in studies])
    covariance = np.stack([study.covariance for study in studies])
    direction = [1, -0.2, 0.3, 0.4]
    expected = direct.support_interval(direction, means, covariance, 2, df=59)
    actual = fitted.support_interval(direction)
    self.assertTrue(expected.success and actual.success)
    assert_allclose([actual.lower, actual.upper], [expected.lower, expected.upper], atol=2e-7)
    point = sample.mean(axis=0) + 0.03
    direct_score = direct.get_global_score(direct.get_p_vector(point, means, covariance, 2, df=59))
    assert_allclose(fitted.score(point), direct_score, rtol=1e-12)

  def test_single_block_matches_analytic_hotelling_ellipsoid(self):
    sample = self.sample[:, :3]
    fitted = fit_dac(sample, block_size=3)
    direction = np.array([1.0, -0.5, 0.2])
    result = fitted.support_interval(direction)
    covariance = np.cov(sample, rowvar=False, ddof=1) / 60
    quadratic_cutoff = 3 * 59 / 57 * f.isf(0.05, 3, 57)
    center = direction @ sample.mean(axis=0)
    radius = np.sqrt(quadratic_cutoff * (direction @ covariance @ direction))
    self.assertTrue(result.success)
    assert_allclose([result.lower, result.upper], [center - radius, center + radius], atol=1e-10)

  def test_scalar_sample_matches_student_t_interval(self):
    sample = self.sample[:, [0]]
    fitted = fit_dac(sample)
    interval = fitted.support_interval([1.0])
    mean = sample[:, 0].mean()
    radius = t.isf(0.025, 59) * sample[:, 0].std(ddof=1) / np.sqrt(60)
    self.assertEqual(fitted.blocks, ((0,),))
    self.assertTrue(interval.success)
    assert_allclose([interval.lower, interval.upper], [mean - radius, mean + radius], atol=1e-10)

  def test_ehmp_irregular_blocks_produce_certified_support(self):
    fitted = fit_dac(self.sample, block_size=2, method="EHMP")
    result = fitted.support_interval([0, 0, 0, 0, 1])
    self.assertTrue(result.success)
    self.assertTrue(fitted.contains(fitted.estimate))


if __name__ == "__main__":
  unittest.main()
