"""Complete non-CAtr threshold-table support and calibration round trips."""

from __future__ import annotations

import unittest

import numpy as np

from reproduction.numerical.independence_thresholds import METHODS, SCENARIOS, evaluate


class ThresholdTableTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.rows = evaluate()

  def test_every_non_catr_cell_is_emitted(self) -> None:
    self.assertEqual(len(self.rows), 20)
    self.assertEqual({(row["scenario"], row["method"]) for row in self.rows},
                     {(scenario.name, method) for scenario in SCENARIOS for method in METHODS})
    for row in self.rows:
      with self.subTest(scenario=row["scenario"], method=row["method"]):
        self.assertTrue(np.isfinite(row["threshold"]))
        self.assertLess(abs(row["survival_round_trip_residual"]), 1e-7)
        self.assertTrue(row["within_printed_rounding"])

  def test_previous_recipe_is_distinct_from_weight_aware_extension(self) -> None:
    previous = {row["scenario"]: row for row in self.rows if row["method"] == "wilson_previous"}
    for name in ("two-equal", "five-equal", "twenty-six-equal"):
      self.assertAlmostEqual(previous[name]["threshold"], previous[name]["weight_aware_hmp_threshold"])
    for name in ("two-weighted", "five-weighted"):
      self.assertGreater(previous[name]["threshold"], previous[name]["weight_aware_hmp_threshold"])


if __name__ == "__main__":
  unittest.main()
