"""Public inference checks beyond the internal support solver."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import chi2

from heavily_right import EmptySetNumericalError, MetaAnalysisMD


class MultivariateValidationTests(unittest.TestCase):
  def test_single_block_level_below_public_floor_is_not_certified(self):
    with self.assertRaisesRegex(EmptySetNumericalError, "p-value floor"):
      MetaAnalysisMD(1, dim=2, level=1.0e-151).support_interval(
        [1.0, 0.0], [[0.0, 0.0]], np.eye(2), 2,
      )

  def test_legacy_helpers_are_explicitly_uncertified(self):
    analysis = MetaAnalysisMD(1, dim=2)
    with patch.object(analysis, "_legacy_simultaneous_interval", return_value=(-1.0, 1.0, None, None)):
      with self.assertWarnsRegex(RuntimeWarning, "without endpoint certificates"):
        analysis.simultaneous_interval([1.0, 0.0], [[0.0, 0.0]], np.eye(2), 2, solver="legacy")
    with self.assertWarnsRegex(DeprecationWarning, "not a validated score"):
      analysis.get_dif_vector([0.1, 0.2], [[0.0, 0.0]], np.eye(2), 2)

  def test_fractional_scalar_and_vector_degrees_of_freedom_agree(self):
    analysis = MetaAnalysisMD(2, dim=2)
    estimates = np.array([[0.0, 0.0], [0.02, -0.01]])
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    arguments = ([1.0, 0.3], estimates, covariance, 2)
    scalar = analysis.support_interval(*arguments, df=2.5)
    vector = analysis.support_interval(*arguments, df=[2.5, 2.5])
    self.assertTrue(scalar.success and vector.success)
    assert_allclose([scalar.lower, scalar.upper], [vector.lower, vector.upper], atol=1.0e-10)
    assert_allclose(
      analysis.get_p_vector([0.3, 0.1], estimates, covariance, 2, df=2.5),
      analysis.get_p_vector([0.3, 0.1], estimates, covariance, 2, df=[2.5, 2.5]),
      atol=1.0e-14,
    )

  def test_mixed_block_dimensions_support_batch_p_values_and_minimum(self):
    analysis = MetaAnalysisMD(2, dim=2)
    estimates = [np.array([0.0]), np.array([0.02, -0.01])]
    covariance = [np.array([[1.0]]), np.eye(2)]
    indices = [np.array([0]), np.array([0, 1])]
    points = np.array([[0.3, 0.1], [-0.1, 0.2]])
    actual = analysis.get_p_vector(points, estimates, covariance, indices)
    expected = np.column_stack([
      chi2.sf(points[:, 0]**2, 1),
      chi2.sf(np.sum((points - estimates[1])**2, axis=1), 2),
    ])
    assert_allclose(actual, expected, atol=1.0e-14)
    minimum = analysis.find_minimizer(estimates, covariance, indices)
    gradient_minimum = analysis.find_minimizer(
      estimates, covariance, indices, method="BFGS", require_grad=True,
    )
    assert_allclose(minimum, gradient_minimum, atol=2.0e-6)
    result = analysis.support_interval([1.0, 0.3], estimates, covariance, indices)
    self.assertTrue(result.success)

  def test_nonminimal_inactive_coordinates_cannot_shrink_support(self):
    analysis = MetaAnalysisMD(4, dim=2)
    estimates = np.array([[0.0], [0.1], [0.0], [0.1]])
    covariance = np.ones((4, 1, 1))
    indices = np.array([[0], [0], [1], [1]])
    actual = analysis.support_interval(
      [1.0, 0.0], estimates, covariance, indices,
      x0=np.array([0.0, 1.0]), find_min=False,
    )
    reference = analysis.support_interval([1.0, 0.0], estimates, covariance, indices)
    self.assertTrue(actual.success and reference.success)
    self.assertEqual(actual.active_dimension, 1)
    assert_allclose([actual.lower, actual.upper], [reference.lower, reference.upper], atol=1.0e-7)
    self.assertIsNotNone(actual.minimum_diagnostics)

  def test_invalid_iteration_configuration_is_rejected_before_analytic_path(self):
    analysis = MetaAnalysisMD(1, dim=2)
    for options in (
      {"maxiter": 0}, {"maxiter": True}, {"newton_maxiter": 0.5},
      {"dense_newton_max_dimension": 0}, {"ftol": np.nan},
      {"x0": [np.nan, 0.0]},
    ):
      with self.subTest(options=options), self.assertRaises(ValueError):
        analysis.support_interval([1.0, 0.0], [[0.0, 0.0]], np.eye(2), 2, **options)


if __name__ == "__main__":
  unittest.main()
