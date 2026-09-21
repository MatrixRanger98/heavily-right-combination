"""Tests for reusable heavy-right score transforms."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose

from heavily_right.scores import half_cauchy_score, reciprocal_score


class ScoreTransformTests(unittest.TestCase):
  def test_half_cauchy_score_support_and_shape(self) -> None:
    values = np.array([[0.0, 0.5, 1.0]])
    scores = half_cauchy_score(values)
    self.assertEqual(np.asarray(scores).shape, values.shape)
    self.assertTrue(math.isinf(np.asarray(scores)[0, 0]))
    assert_allclose(np.asarray(scores)[0, 1:], [1.0, 0.0], atol=1.0e-15)

  def test_reciprocal_score_support_and_scalar_contract(self) -> None:
    self.assertEqual(reciprocal_score(0.5), 2.0)
    scores = np.asarray(reciprocal_score([0.0, 0.25, 1.0]))
    self.assertTrue(math.isinf(scores[0]))
    assert_allclose(scores[1:], [4.0, 1.0])

  def test_invalid_pvalues_are_rejected(self) -> None:
    for values in ([-0.1], [1.1], [np.nan], [np.inf]):
      with self.subTest(values=values), self.assertRaises(ValueError):
        half_cauchy_score(values)
      with self.assertRaises(ValueError):
        reciprocal_score(values)


if __name__ == "__main__":
  unittest.main()
