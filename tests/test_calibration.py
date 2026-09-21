"""Regression tests for the exact one-sample Hotelling calibration."""

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import f, t

from heavily_right.calibration import covariance_of_mean, standard_error_of_mean
from heavily_right.meta_analysis import meta_analysis_1d, meta_analysis_md


class CalibrationTest(unittest.TestCase):
  def setUp(self):
    self.rng = np.random.default_rng(20260823)

  def test_standard_error_uses_unbiased_sample_variance(self):
    sample = self.rng.normal(size=(17, 4))
    expected = sample.std(axis=0, ddof=1) / np.sqrt(sample.shape[0])
    assert_allclose(standard_error_of_mean(sample), expected)

  def test_block_covariance_equals_scatter_over_n_n_minus_one(self):
    sample = self.rng.normal(size=(19, 3, 5))
    centered = sample - sample.mean(axis=0)
    expected = np.einsum("nbi,nbj->bij", centered, centered) / (19 * 18)
    assert_allclose(covariance_of_mean(sample), expected)

  def test_univariate_p_value_matches_two_sided_t_test(self):
    sample = self.rng.normal(size=31)
    point = 0.25
    mean = np.array([sample.mean()])
    standard_error = np.array([standard_error_of_mean(sample)])
    degrees_freedom = sample.size - 1

    analysis = meta_analysis_1d(1, method="HCauchy", level=0.05)
    actual = analysis.get_p_vector(point, mean, standard_error, degrees_freedom)[0]
    statistic = abs(mean[0] - point) / standard_error[0]
    expected = 2 * t.sf(statistic, degrees_freedom)
    self.assertAlmostEqual(actual, expected, places=14)

  def test_multivariate_p_value_matches_hotelling_f_transform(self):
    num_sample = 40
    dimension = 3
    sample = self.rng.normal(size=(num_sample, dimension))
    mean = sample.mean(axis=0)
    covariance = covariance_of_mean(sample)
    degrees_freedom = num_sample - 1

    analysis = meta_analysis_md(1, dim=dimension, method="HCauchy", level=0.05)
    actual = analysis.get_p_vector(
      np.zeros(dimension),
      mean[np.newaxis, :],
      covariance[np.newaxis, :, :],
      sub_dim=dimension,
      df=degrees_freedom,
    )[0]

    statistic = mean @ np.linalg.inv(covariance) @ mean
    f_statistic = statistic * (num_sample - dimension) / (
      dimension * (num_sample - 1)
    )
    expected = f.sf(f_statistic, dimension, num_sample - dimension)
    self.assertAlmostEqual(actual, expected, places=14)

  def test_rejects_samples_with_fewer_than_two_observations(self):
    with self.assertRaises(ValueError):
      standard_error_of_mean(np.ones((1, 2)))
    with self.assertRaises(ValueError):
      covariance_of_mean(np.ones((1, 2)))

  def test_simulation_paths_use_shared_calibration_helpers(self):
    code_root = Path(__file__).resolve().parents[1]
    paths = (
      code_root / "reproduction" / "multidimensional" / "dac.py",
      code_root / "reproduction" / "network_meta_analysis" / "simulation.py",
    )
    for path in paths:
      source = path.read_text()
      with self.subTest(path=path.name):
        self.assertNotIn(".std(0)", source)
        self.assertNotIn("num_sample**2", source)
        self.assertIn("standard_error_of_mean", source)


if __name__ == "__main__":
  unittest.main()
