"""Tests for the dependency-free bivariate copula samplers."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from reproduction.one_dimensional.copula_coverage import sample_amh, sample_fgm


class CopulaSamplingTest(unittest.TestCase):
  def test_zero_parameter_is_independent_uniform_sampling(self):
    for sampler in (sample_amh, sample_fgm):
      sample = sampler(0, 20_000, np.random.default_rng(2025))
      with self.subTest(sampler=sampler.__name__):
        assert_allclose(sample.mean(axis=0), [0.5, 0.5], atol=0.01)
        self.assertLess(abs(np.corrcoef(sample.T)[0, 1]), 0.02)

  def test_samples_remain_in_unit_square_at_parameter_extremes(self):
    for sampler in (sample_amh, sample_fgm):
      for parameter in (-0.9, 0.9):
        sample = sampler(parameter, 5_000, np.random.default_rng(2025))
        with self.subTest(sampler=sampler.__name__, parameter=parameter):
          self.assertTrue(np.all((0 <= sample) & (sample <= 1)))
          assert_allclose(sample.mean(axis=0), [0.5, 0.5], atol=0.02)


if __name__ == "__main__":
  unittest.main()
