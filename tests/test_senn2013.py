"""Portable parity checks against the retained R/netmeta output."""

import json
import unittest
from pathlib import Path

import numpy as np

from reproduction.network_meta_analysis.senn2013 import (
  ADJUSTED_STANDARD_ERRORS,
  DATA,
  DESIGN_MATRIX,
  STANDARD_ERRORS,
  TREATMENTS,
  wls_interval_width_matrix,
)


class Senn2013Tests(unittest.TestCase):
  def test_snapshot_and_design_have_expected_shapes(self):
    self.assertEqual(len(DATA), 28)
    self.assertEqual(DESIGN_MATRIX.shape, (28, 9))
    self.assertEqual(tuple(TREATMENTS), (
      "acar", "benf", "metf", "migl", "piog", "rosi", "sita", "sulf", "vild"
    ))
    np.testing.assert_array_equal(
      DESIGN_MATRIX[2],
      np.array([-1, 0, 1, 0, 0, 0, 0, 0, 0]),
    )

  def test_only_multiarm_trial_standard_errors_are_adjusted(self):
    changed = np.flatnonzero(~np.isclose(ADJUSTED_STANDARD_ERRORS, STANDARD_ERRORS))
    np.testing.assert_array_equal(changed, np.array([2, 26, 27]))
    np.testing.assert_allclose(
      ADJUSTED_STANDARD_ERRORS[changed],
      np.array([0.38837788, 0.41252517, 0.82419108]),
      rtol=0,
      atol=5e-9,
    )

  def test_python_wls_widths_match_legacy_netmeta_output(self):
    fixture_path = Path(__file__).resolve().parent / "fixtures/nma_wls_widths.json"
    fixture = json.loads(fixture_path.read_text())
    source = np.asarray(fixture["values"], dtype=float)
    non_reference = [index for index in range(10) if index != 5]
    expected = source[np.ix_(non_reference, non_reference)].copy()
    for index in range(9):
      source_index = index if index < 5 else index + 1
      expected[index, index] = source[5, source_index]

    np.testing.assert_allclose(
      wls_interval_width_matrix(),
      expected,
      rtol=0,
      atol=1e-8,
    )


if __name__ == "__main__":
  unittest.main()
