"""Regression checks for recovered blockwise geometry illustrations."""

import unittest

import numpy as np

from reproduction.illustrations.hcct_cct_geometry import (
    Q_C,
    Q_H,
    cct_half_width,
    f_score,
)


class GeometryTests(unittest.TestCase):
    def test_boundary_satisfies_cct_score(self) -> None:
        y = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
        x = cct_half_width(y)
        np.testing.assert_allclose((f_score(x) + f_score(y)) / 4, Q_C, rtol=1e-8)
        self.assertEqual(Q_H, 13.68751)

    def test_equal_area_tubes_are_inside_horn(self) -> None:
        for level in (32, 64, 256, 1024):
            right = 1 / (4 * level)
            left = 1 / (8 * level)
            self.assertAlmostEqual((right - left) * level, 1 / 8)
            self.assertLess(right, cct_half_width(2 * level))


if __name__ == "__main__":
    unittest.main()
