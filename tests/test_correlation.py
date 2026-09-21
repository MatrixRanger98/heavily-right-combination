"""Tests for reusable simulation correlation matrices."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from heavily_right.correlation import ar1_correlation, equicorrelation
from reproduction.manifest import get_experiment


class CorrelationTest(unittest.TestCase):
  def test_ar1_entries_follow_distance_power(self):
    actual = ar1_correlation(4, 0.5)
    expected = np.array(
      [
        [1.0, 0.5, 0.25, 0.125],
        [0.5, 1.0, 0.5, 0.25],
        [0.25, 0.5, 1.0, 0.5],
        [0.125, 0.25, 0.5, 1.0],
      ]
    )
    assert_allclose(actual, expected)

  def test_equicorrelation_has_unit_diagonal_and_fixed_off_diagonal(self):
    actual = equicorrelation(5, 0.3)
    assert_allclose(np.diag(actual), np.ones(5))
    assert_allclose(actual[np.triu_indices(5, k=1)], np.full(10, 0.3))

  def test_simulation_scripts_use_shared_correlation_helpers(self):
    experiment_ids = (
      "one-dimensional/normal-coverage",
      "one-dimensional/false-positive-rate",
      "one-dimensional/interval-width",
      "one-dimensional/power",
      "one-dimensional/copulas",
      "multidimensional/coverage",
      "multidimensional/contours",
      "network-meta-analysis/simulation",
    )
    for experiment_id in experiment_ids:
      path = get_experiment(experiment_id).script_path
      source = path.read_text()
      with self.subTest(path=path.name):
        self.assertIn("from heavily_right.correlation import", source)
        self.assertNotIn("def cor_fixed", source)
        self.assertNotIn("def cor_ar_1", source)

    dac_source = (
      get_experiment("multidimensional/dac-normal").script_path.parent / "dac.py"
    ).read_text()
    self.assertIn("from heavily_right.correlation import", dac_source)
    self.assertNotIn("def cor_fixed", dac_source)
    self.assertNotIn("def cor_ar_1", dac_source)


if __name__ == "__main__":
  unittest.main()
