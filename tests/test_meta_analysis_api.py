"""Public input-contract tests for the meta-analysis compatibility API."""

import unittest

import numpy as np

from heavily_right.meta_analysis import MetaAnalysis1D, MetaAnalysisMD


class MetaAnalysisInputTests(unittest.TestCase):
  def test_multidimensional_parameter_dimension_is_a_finite_integer(self) -> None:
    for dimension in (True, 1, 2.5, np.nan, np.inf):
      with self.subTest(dimension=dimension), self.assertRaises(ValueError):
        MetaAnalysisMD(1, dim=dimension)

  def test_one_dimensional_degrees_of_freedom_accept_a_list(self) -> None:
    analysis = MetaAnalysis1D(2)
    p_values = analysis.get_p_vector(
      point=0.0,
      theta_hat=[0.1, -0.2],
      sigma=[0.05, 0.08],
      df=[19, 24],
    )
    self.assertEqual(p_values.shape, (2,))

  def test_multidimensional_support_example_accepts_list_inputs(self) -> None:
    analysis = MetaAnalysisMD(1, dim=2)
    result = analysis.support_interval(
      direction=[1.0, 0.0],
      xi_hat=[np.array([0.2, -0.1])],
      Sigma=[np.array([[0.04, 0.01], [0.01, 0.09]])],
      sub_dim=2,
      df=[29],
    )
    self.assertTrue(result.success)


if __name__ == "__main__":
  unittest.main()
