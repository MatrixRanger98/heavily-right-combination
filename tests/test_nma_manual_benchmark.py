"""Historical manual plotting benchmark is distinct from simulated estimates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from reproduction.artifacts import ArtifactStore
from reproduction.network_meta_analysis import simulation
from reproduction.network_meta_analysis.manual_benchmark import (
  HISTORICAL_WIDTHS,
  select_wls_ma_widths,
)


def _summary() -> dict[str, np.ndarray]:
  rhos = np.arange(0, 1, .1)
  return {
    "rhos": rhos,
    "coverage_hcct": np.full(10, .94), "coverage_wls": np.full(10, .9),
    "coverage_wls_ma": np.full(10, .95),
    "width_hcct_1": np.full(10, .5), "width_hcct_2": np.full(10, .6),
    "width_wls_1": np.full(10, .3), "width_wls_2": np.full(10, .4),
    "width_wls_ma_1": np.full(10, .7), "width_wls_ma_2": np.full(10, .8),
    "wls_ma_multiplier": np.full(10, 6.),
    "empty_hcct_count": np.zeros(10), "replicates_per_rho": np.full(10, 100),
  }


class NmaManualBenchmarkTests(unittest.TestCase):
  def test_default_matches_both_historical_width_arrays_without_mutating_inputs(self) -> None:
    results = _summary()
    before = {key: value.copy() for key, value in results.items()}
    widths, policy = select_wls_ma_widths(results, profile="paper")
    np.testing.assert_array_equal(widths, HISTORICAL_WIDTHS)
    self.assertEqual(widths[0][-1], .64964069)
    self.assertEqual(widths[1][-1], .84543146)
    self.assertEqual(policy["selected_width_policy"], "historical-manual")
    self.assertIsNone(policy["fallback_reason"])
    self.assertIn("not estimated", policy["width_qualification"])
    for key in results:
      np.testing.assert_array_equal(results[key], before[key])

  def test_unmatched_grid_or_smoke_uses_explicit_empirical_fallback(self) -> None:
    for profile, shortened in (("smoke", False), ("paper", True)):
      results = _summary()
      if shortened:
        results = {key: value[:2] for key, value in results.items()}
      widths, policy = select_wls_ma_widths(results, profile=profile)
      self.assertEqual(policy["selected_width_policy"], "empirical")
      self.assertTrue(policy["fallback_reason"])
      for index, values in enumerate(widths, start=1):
        np.testing.assert_array_equal(values, results[f"width_wls_ma_{index}"])
      self.assertIsNone(policy["historical_source"])

  def test_historical_500_replication_qualification_does_not_claim_100(self) -> None:
    results = _summary()
    results["replicates_per_rho"] = np.full(10, 500)
    _, policy = select_wls_ma_widths(results, profile="paper")
    self.assertEqual(policy["selected_width_policy"], "historical-manual")
    self.assertEqual(policy["replicates_per_rho"], [500] * 10)
    self.assertNotIn("100-replication", policy["width_qualification"])
    self.assertIn("not estimated from these saved replications", policy["width_qualification"])

  def test_explicit_empirical_choice_and_invalid_arguments(self) -> None:
    results = _summary()
    widths, policy = select_wls_ma_widths(results, profile="paper", policy="empirical")
    self.assertIsNone(policy["fallback_reason"])
    np.testing.assert_array_equal(widths[0], results["width_wls_ma_1"])
    with self.assertRaisesRegex(ValueError, "width policy"):
      select_wls_ma_widths(results, profile="paper", policy="other")
    with self.assertRaisesRegex(ValueError, "profile"):
      select_wls_ma_widths(results, profile="other")
    results["width_wls_ma_1"][0] = np.nan
    with self.assertRaisesRegex(ValueError, "finite"):
      select_wls_ma_widths(results, profile="paper", policy="empirical")

  def test_render_saves_actual_plotted_values_and_labels_without_simulating(self) -> None:
    results = _summary()
    figures = []

    def capture(figure, _store, filename, **_kwargs):
      figures.append((filename, figure))

    try:
      with tempfile.TemporaryDirectory() as directory:
        store = ArtifactStore(simulation.EXPERIMENT_ID, "manual", Path(directory))
        with patch.object(simulation, "simulate", side_effect=AssertionError("no simulation")), \
             patch.object(simulation, "save_pdf", side_effect=capture):
          simulation.render(results, store, profile="paper")
        policy = json.loads((store.run_dir / "diagnostics/nma-simulation-plot-policy.json").read_text())
        self.assertEqual(policy["selected_width_policy"], "historical-manual")
        with np.load(store.run_dir / "data/nma-simulation-plotted-summary.npz") as plotted:
          self.assertNotIn("wls_ma_multiplier", plotted.files)
          np.testing.assert_array_equal(plotted["empirical_wls_ma_multiplier"],
                                        results["wls_ma_multiplier"])
          for key in results:
            if key not in {"width_wls_ma_1", "width_wls_ma_2", "wls_ma_multiplier"}:
              np.testing.assert_array_equal(plotted[key], results[key])
          np.testing.assert_array_equal(plotted["width_wls_ma_1"], HISTORICAL_WIDTHS[0])
          np.testing.assert_array_equal(plotted["width_wls_ma_2"], HISTORICAL_WIDTHS[1])
      self.assertEqual(len(figures), 3)
      coverage_labels = [text.get_text() for text in figures[0][1].axes[0].get_legend().texts]
      self.assertIn("in-sample oracle", coverage_labels[-1])
      for _, figure in figures[1:]:
        labels = [text.get_text() for text in figure.axes[0].get_legend().texts]
        self.assertEqual(labels[-1], "WLS-MA (historical manual)")
      self.assertTrue(np.all(results["width_wls_ma_1"] == .7))
    finally:
      for _, figure in figures:
        plt.close(figure)


if __name__ == "__main__":
  unittest.main()
