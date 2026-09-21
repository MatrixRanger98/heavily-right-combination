"""Reproduction speedups must not change draws, their order, or RNG state."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, multivariate_t

from heavily_right.combination import CombinationTest
from heavily_right.correlation import cor_ar_1, cor_fixed
from reproduction.multidimensional import coverage as md_coverage
from reproduction.one_dimensional import (
  copula_coverage,
  false_positive_rate,
  interval_width,
  normal_coverage,
  power_comparison,
)
from reproduction.one_dimensional.common import build_band_figure
from reproduction.sampling import CachedMultivariateNormal, CachedMultivariateT


class CachedSamplingTests(unittest.TestCase):
  def test_normal_samples_and_following_rng_state_match_scipy(self):
    for rng_class in (np.random.RandomState, np.random.default_rng):
      for dimension in (1, 7, 30, 500):
        for correlation in (cor_ar_1, cor_fixed):
          with self.subTest(rng=rng_class, dimension=dimension, cov=correlation):
            rng_old, rng_new = rng_class(2025), rng_class(2025)
            mean = np.linspace(-0.5, 0.3, dimension)
            covariance = correlation(dimension, 0.6)
            original = multivariate_normal(mean=mean, cov=covariance)
            cached = CachedMultivariateNormal(mean, covariance)
            for size in (1, 5, (3, 2)):
              np.testing.assert_array_equal(
                cached.rvs(size, rng_new), original.rvs(size, rng_old),
              )
            np.testing.assert_array_equal(
              rng_new.standard_normal(20), rng_old.standard_normal(20),
            )

  def test_global_randomstate_is_preserved(self):
    state = np.random.get_state()
    try:
      mean = np.arange(5) / 10
      covariance = cor_fixed(5, 0.3)
      np.random.seed(2025)
      expected = multivariate_normal(mean, covariance).rvs(size=(4, 2))
      following = np.random.standard_normal(10)
      np.random.seed(2025)
      actual = CachedMultivariateNormal(mean, covariance).rvs(size=(4, 2))
      np.testing.assert_array_equal(actual, expected)
      np.testing.assert_array_equal(np.random.standard_normal(10), following)
    finally:
      np.random.set_state(state)

  def test_student_radial_order_and_following_rng_state_match_scipy(self):
    for rng_class in (np.random.RandomState, np.random.default_rng):
      for correlation in (cor_ar_1, cor_fixed):
        with self.subTest(rng=rng_class, cov=correlation):
          rng_old, rng_new = rng_class(2025), rng_class(2025)
          location = np.linspace(-0.5, 0.3, 7)
          covariance = correlation(7, 0.6)
          cached = CachedMultivariateT(location, covariance, 10)
          for size in (1, 5, (3, 2)):
            expected = multivariate_t.rvs(
              location, covariance, 10, size=size, random_state=rng_old,
            )
            np.testing.assert_array_equal(cached.rvs(size, rng_new), expected)
          np.testing.assert_array_equal(
            rng_new.standard_normal(20), rng_old.standard_normal(20),
          )

  def test_factorization_is_cached_and_invalid_covariance_rejected(self):
    with patch("numpy.linalg.svd", wraps=np.linalg.svd) as factorize:
      sampler = CachedMultivariateNormal(np.zeros(3), cor_fixed(3, 0.5))
      rng = np.random.default_rng(2025)
      sampler.rvs(10, rng)
      sampler.rvs(10, rng)
    self.assertEqual(factorize.call_count, 1)
    with self.assertRaises(np.linalg.LinAlgError):
      CachedMultivariateNormal(np.zeros(2), np.array([[1, 2], [2, 1]]))


class SimulationEquivalenceTests(unittest.TestCase):
  def setUp(self):
    self.random_state = np.random.get_state()

  def tearDown(self):
    np.random.set_state(self.random_state)

  def test_normal_experiment_smoke_arrays_keep_original_streams(self):
    cases = (
      (normal_coverage, lambda: normal_coverage.simulate_coverage(cor_ar_1, "test")),
      (false_positive_rate, false_positive_rate.simulate),
      (power_comparison, power_comparison.simulate),
      (interval_width, interval_width.simulate),
      (md_coverage, lambda: md_coverage.simulate(10)),
    )
    state = np.random.get_state()
    try:
      with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), contextlib.redirect_stdout(io.StringIO()):
        for module, simulate in cases:
          with self.subTest(module=module.__name__):
            np.random.seed(2025)
            actual = simulate()
            actual_next = np.random.standard_normal(20)
            np.random.seed(2025)
            with patch.object(module, "CachedMultivariateNormal", multivariate_normal):
              expected = simulate()
            expected_next = np.random.standard_normal(20)
            if isinstance(actual, tuple):
              for new, old in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(new, old)
            else:
              np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_next, expected_next)
    finally:
      np.random.set_state(state)

  def test_student_smoke_array_keeps_original_stream(self):
    with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), contextlib.redirect_stdout(io.StringIO()):
      actual, parameters = copula_coverage.simulate_student(cor_fixed, "test")
      with patch.object(copula_coverage, "CachedMultivariateT", multivariate_t):
        expected, expected_parameters = copula_coverage.simulate_student(cor_fixed, "test")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(parameters, expected_parameters)

  def test_coverage_calibrates_each_level_only_once(self):
    cases = (
      (normal_coverage, normal_coverage.simulate_coverage),
      (copula_coverage, copula_coverage.simulate_student),
    )
    with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), contextlib.redirect_stdout(io.StringIO()):
      for module, simulate in cases:
        with self.subTest(module=module.__name__), patch.object(
          module, "CombinationTest", wraps=CombinationTest,
        ) as construct:
          simulate(cor_fixed, "test")
        self.assertEqual(construct.call_count, 2)

  def test_copula_rejection_panels_use_one_minus_coverage(self):
    coverage = np.full((2, 2, 2), 0.95)
    parameters = np.array([0.0, 0.6])
    with (
      patch.object(copula_coverage, "simulate_student", return_value=(coverage, parameters)),
      patch.object(copula_coverage, "simulate_bivariate_copula", return_value=(coverage, parameters)),
      patch.object(copula_coverage.ArtifactStore, "for_experiment"),
      patch.object(copula_coverage.np, "save"),
      patch.object(copula_coverage, "save_pdf"),
      patch.object(copula_coverage, "build_band_figure") as build,
    ):
      copula_coverage.main()
    self.assertEqual(build.call_count, 6)
    self.assertTrue(all(call.kwargs["complement_offset"] == 1.0 for call in build.call_args_list))
    figure = build_band_figure(coverage, parameters, complement_offset=1.0)
    try:
      for line in figure.axes[0].lines:
        np.testing.assert_allclose(line.get_ydata(), 0.05, rtol=0, atol=1e-15)
    finally:
      plt.close(figure)


if __name__ == "__main__":
  unittest.main()
