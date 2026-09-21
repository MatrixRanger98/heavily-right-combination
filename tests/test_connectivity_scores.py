import unittest

import numpy as np

from heavily_right.combination import CombinationTest
from reproduction.illustrations.connectivity_scores import (
  ESTIMATES,
  individual_p_values,
  score_curves,
  thresholds,
)


class ConnectivityScoreTests(unittest.TestCase):
  def test_individual_p_values_use_paper_estimates_and_standard_error(self):
    p_values = individual_p_values(ESTIMATES)
    self.assertEqual(p_values.shape, (2, 2))
    np.testing.assert_allclose(np.diag(p_values), np.ones(2))

  def test_score_curves_match_combination_api(self):
    for theta in (-0.1, 0.0, 0.1):
      p_values = individual_p_values(theta)
      expected_cct = CombinationTest(2, method="Cauchy").get_global_score(p_values)
      expected_hcct = CombinationTest(2, method="HCauchy").get_global_score(p_values)
      actual_cct, actual_hcct = score_curves(theta)
      self.assertAlmostEqual(float(actual_cct), float(expected_cct))
      self.assertAlmostEqual(float(actual_hcct), float(expected_hcct))

  def test_thresholds_match_combination_api(self):
    cct, hcct = thresholds()
    self.assertAlmostEqual(cct, CombinationTest(2, method="Cauchy").threshold)
    self.assertAlmostEqual(hcct, CombinationTest(2, method="HCauchy").threshold)


if __name__ == "__main__":
  unittest.main()
