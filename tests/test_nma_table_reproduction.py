"""NMA table completeness, calibration, and empirical-oracle regressions."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from heavily_right.empty_sets import EmptyRegionError, EmptySetNumericalError
from heavily_right.meta_analysis import MetaAnalysisMD, SupportIntervalResult
from heavily_right.minimum import MinimumDiagnostics
from heavily_right.support import SupportDiagnostics
from reproduction.artifacts import ArtifactStore
from reproduction.manifest import get_experiment
from reproduction.network_meta_analysis import simulation
from reproduction.network_meta_analysis import treatment_estimates as tables
from reproduction.network_meta_analysis.senn2013 import (
  ADJUSTED_STANDARD_ERRORS,
  DESIGN_MATRIX,
  ESTIMATES,
  TREATMENTS,
  wls_point_estimate,
)


def _support_fixture() -> SupportIntervalResult:
  diagnostic = SupportDiagnostics(True, "fixture", 1., 1., 0., 0., 1., 1, 1, 0, "fixture")
  return SupportIntervalResult(-1., 1., np.zeros(9), np.zeros(9), diagnostic, diagnostic,
                               False, 9, np.zeros(9), minimum_score=0.)


class NmaTableReproductionTests(unittest.TestCase):
  def test_selected_seed_matches_registry_and_saved_simulation_diagnostics(self) -> None:
    self.assertEqual(tables.SEED, 2025)
    self.assertEqual(get_experiment(tables.EXPERIMENT_ID).seed, tables.SEED)
    self.assertEqual(self.simulation_diagnostics["seed"], tables.SEED)

  def test_selected_seed_matches_all_updated_table_cells_at_original_precision(self) -> None:
    self.assertEqual(len(self.simulation_rows), 72)
    for row in self.simulation_rows:
      with self.subTest(rho=row["rho"], method=row["method"], treatment=row["treatment"]):
        printed = re.sub(r"^(-?)0\.", r"\1.", f"{row['estimate']:#.3g}")
        self.assertEqual(row["paper_estimate"], printed)
        self.assertTrue(row["within_paper_rounding"])
        self.assertLessEqual(abs(row["difference_from_paper"]),
                             row["paper_rounding_half_unit"] + 1e-12)
    self.assertFalse(self.simulation_diagnostics["original_realization_recovered"])
    self.assertEqual(self.simulation_diagnostics["reference_snapshot"],
                     tables.REFERENCE_SNAPSHOT)
    self.assertIn("seed-2025 publication", tables.REFERENCE_SNAPSHOT)
    self.assertIn("historical table", self.simulation_diagnostics["comparison_qualification"])

  @classmethod
  def setUpClass(cls) -> None:
    cls.simulation_rows, cls.simulation_diagnostics, cls.inputs = tables.evaluate_simulation()
    with warnings.catch_warnings(record=True):
      cls.real_rows, cls.real_diagnostics = tables.evaluate_real_data()

  def test_seeded_table_inputs_preserve_global_rng_and_correct_mean_calibration(self) -> None:
    state_before = np.random.get_state()
    second = tables.simulation_inputs()
    state_after = np.random.get_state()
    np.testing.assert_array_equal(state_before[1], state_after[1])
    self.assertEqual(state_before[0], state_after[0])
    self.assertEqual(state_before[2:], state_after[2:])
    self.assertEqual(second["samples"].shape, (4, 100, 28))
    for name, value in self.inputs.items():
      np.testing.assert_array_equal(second[name], value)
    np.testing.assert_allclose(
      second["estimated_mean_variances"], second["samples"].var(axis=1, ddof=1) / 100,
      rtol=2e-14,
    )
    np.testing.assert_allclose(
      np.diagonal(second["true_mean_covariances"], axis1=1, axis2=2), 0.01,
    )
    for seed in (-1, True, 1.5):
      with self.assertRaises(ValueError):
        tables.simulation_inputs(seed)

  def test_simulation_table_has_all_scenarios_methods_treatments_and_valid_minima(self) -> None:
    expected = [(rho, method, str(treatment)) for rho in tables.RHOS
                for method in ("WLS", "HCCT") for treatment in TREATMENTS]
    self.assertEqual([(row["rho"], row["method"], row["treatment"])
                      for row in self.simulation_rows], expected)
    self.assertEqual(len(self.simulation_rows), 72)
    diagnostics = self.simulation_diagnostics
    self.assertEqual(diagnostics["replicates_per_rho"], 1)
    self.assertEqual(diagnostics["degrees_freedom"], 99)
    self.assertFalse(diagnostics["original_realization_recovered"])
    analysis = MetaAnalysisMD(28, dim=9)
    for index, case in enumerate(diagnostics["cases"]):
      p_values = analysis.get_p_vector(
        case["hcct_point"], self.inputs["means"][index],
        self.inputs["estimated_mean_variances"][index],
        sub_dim=1, df=99, projs=DESIGN_MATRIX,
      )
      self.assertAlmostEqual(float(analysis.get_global_score(p_values)), case["hcct_score"])
      numerical = case["minimum_diagnostics"]
      self.assertLessEqual(numerical["optimality_gap_bound"],
                           numerical["optimality_gap_tolerance"])
      self.assertLess(case["hcct_score"], case["hcct_threshold"])

  def test_wls_point_estimate_uses_inverse_variance_not_ols(self) -> None:
    design = np.array([[1., 0.], [0., 1.], [1., -1.]])
    observed = np.array([0.2, -0.7, 0.1])
    errors = np.array([0.1, 0.3, 0.8])
    expected = np.linalg.solve(design.T @ (design / errors[:, None]**2),
                               design.T @ (observed / errors**2))
    actual = wls_point_estimate(observed, errors, design)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    self.assertGreater(np.linalg.norm(actual - np.linalg.lstsq(design, observed, rcond=None)[0]),
                       0.1)
    with self.assertRaises(ValueError):
      wls_point_estimate(observed, [0., 1., 1.], design)
    with self.assertRaises(ValueError):
      wls_point_estimate(observed, errors, np.ones((3, 2)))

  def test_real_table_keeps_adjusted_wls_and_explicit_hcct_intervention(self) -> None:
    self.assertEqual(len(self.real_rows), 18)
    trace = self.real_diagnostics["empty_set_adjustment"]
    self.assertEqual([row["original_index"] for row in trace["removals"]], [0, 6])
    self.assertEqual(len(trace["final_original_indices"]), 26)
    wls = [row for row in self.real_rows if row["method"] == "WLS"]
    hcct = [row for row in self.real_rows if row["method"] == "HCCT"]
    np.testing.assert_allclose([row["estimate"] for row in wls],
                               wls_point_estimate(ESTIMATES, ADJUSTED_STANDARD_ERRORS))
    self.assertTrue(all(row["within_paper_rounding"] for row in wls))
    self.assertTrue(all(row["within_paper_rounding"] for row in hcct))
    self.assertEqual(self.real_diagnostics["reference_snapshot"], tables.REFERENCE_SNAPSHOT)
    self.assertIn("pre-update Table 2 reference",
                  self.real_diagnostics["comparison_qualification"])
    self.assertTrue(all(row["study_count"] == 28 for row in wls))
    self.assertTrue(all(row["study_count"] == 26 and row["removed_original_indices"] == "0;6"
                        for row in hcct))

  def test_dated_printed_values_match_active_paper_when_checkout_available(self) -> None:
    paper = Path(__file__).resolve().parents[2] / "Paper/content/main.tex"
    if not paper.is_file():
      self.skipTest("The source distribution intentionally excludes Paper")
    text = re.sub(r"(?<!\\)%[^\n]*", "", paper.read_text())
    for label, expected in (
      ("tab:thetaressim", list(tables.PAPER_SIMULATION.values())),
      ("tab:thetares", list(tables.PAPER_REAL.values())),
    ):
      end = text.index(r"\label{" + label + "}")
      block = text[text.rfind(r"\begin{table}", 0, end):end]
      printed = re.findall(r"(?:&|\s)(?:WLS|HCCT)\s*&([^\n]+?)\\\\", block)
      actual = [tuple(value.strip() for value in row.split("&")) for row in printed]
      self.assertEqual(actual, expected)

  def test_table_csv_and_strict_json_preserve_current_comparisons(self) -> None:
    with tempfile.TemporaryDirectory() as temporary:
      artifacts = ArtifactStore(tables.EXPERIMENT_ID, "test", Path(temporary))
      tables._write_csv(artifacts, tables.REAL_OUTPUT, self.real_rows)
      path = artifacts.run_dir / "data" / tables.REAL_OUTPUT
      with path.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
      self.assertEqual(len(saved), 18)
      self.assertEqual(saved[0]["paper_estimate"], "-0.827")
      self.assertTrue(all(row["within_paper_rounding"] == "True" for row in saved))
      with self.assertRaises(FileExistsError):
        tables._write_csv(artifacts, tables.REAL_OUTPUT, self.real_rows)
    payload = tables._json_safe({"simulation": self.simulation_diagnostics,
                                 "real_data": self.real_diagnostics})
    json.dumps(payload, allow_nan=False)

  def test_reused_minimum_keeps_second_direction_endpoints_and_certificates(self) -> None:
    analysis = MetaAnalysisMD(28, dim=9)
    arguments = {
      "xi_hat": self.inputs["means"][0],
      "Sigma": self.inputs["estimated_mean_variances"][0],
      "sub_dim": 1, "df": 99, "projs": DESIGN_MATRIX, "method": "Powell",
    }
    first = analysis.support_interval(np.eye(9)[0], **arguments)
    independent = analysis.support_interval(np.eye(9)[1], **arguments)
    reused = analysis.support_interval(np.eye(9)[1], x0=first.minimum_point, find_min=False,
                                       **arguments)
    self.assertTrue(first.success and independent.success and reused.success)
    np.testing.assert_allclose([reused.lower, reused.upper],
                               [independent.lower, independent.upper], rtol=0, atol=2e-7)
    for endpoint in (reused.lower_diagnostics, reused.upper_diagnostics):
      self.assertGreater(endpoint.eta, 0.)
      self.assertLess(endpoint.kkt_residual, 5e-6)


class NmaEmpiricalOracleTests(unittest.TestCase):
  def test_configuration_records_reduced_paper_workload_and_unchanged_smoke(self) -> None:
    with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}):
      paper = simulation.simulation_configuration("paper")
    self.assertEqual(paper["replicates_per_rho"], 100)
    self.assertEqual(paper["sample_size"], 100)
    self.assertEqual(paper["degrees_freedom"], 99)
    self.assertEqual(paper["seed"], 2025)
    self.assertEqual(paper["workload_revision"], "nma-100-replicates-2026-09-21")
    np.testing.assert_array_equal(paper["rhos"], np.arange(0, 1, .1))
    smoke = simulation.simulation_configuration("smoke")
    self.assertEqual(smoke["replicates_per_rho"], 2)
    self.assertEqual(smoke["sample_size"], 30)
    self.assertEqual(smoke["degrees_freedom"], 29)
    with self.assertRaises(ValueError):
      simulation.simulation_configuration("unknown")

  def test_oracle_uses_empirical_maximum_errors_and_records_actual_coverage(self) -> None:
    estimates = np.arange(1., 21.)[:, None] * np.array([[1., 0.5]])
    errors = np.ones_like(estimates)
    oracle = simulation.calibrate_wls_oracle(estimates, errors, np.zeros(2))
    self.assertEqual(oracle["multiplier"], 19.)
    self.assertEqual(oracle["coverage"], 0.95)
    np.testing.assert_array_equal(oracle["maximum_standardized_errors"], np.arange(1., 21.))
    np.testing.assert_array_equal(oracle["mean_widths"], [38., 38.])
    # A small profile or a tied empirical quantile need not give exactly 95%.
    tied = simulation.calibrate_wls_oracle(np.ones((2, 2)), np.ones((2, 2)), np.zeros(2))
    self.assertEqual(tied["coverage"], 1.)
    for invalid in (np.zeros((2, 2)), np.full((2, 2), np.nan)):
      with self.assertRaises(ValueError):
        simulation.calibrate_wls_oracle(np.ones((2, 2)), invalid, np.zeros(2))

  def test_simulation_saves_raw_replicates_and_empirical_not_literal_oracle(self) -> None:
    state = np.random.get_state()
    try:
      np.random.seed(simulation.SEED)
      reference_rng = np.random.RandomState(simulation.SEED)
      reference_means = np.array([
        [simulation.multivariate_normal(
          mean=DESIGN_MATRIX @ simulation.THETA, cov=simulation.cor_fixed(28, rho),
        ).rvs(size=30, random_state=reference_rng).mean(axis=0) for _ in range(2)]
        for rho in (0., 0.6)
      ])
      with tempfile.TemporaryDirectory() as temporary:
        artifacts = ArtifactStore(simulation.EXPERIMENT_ID, "test", Path(temporary))
        with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), \
             patch.object(simulation, "_is_covered", return_value=True), \
             patch.object(simulation.MetaAnalysisMD, "support_interval",
                          return_value=_support_fixture()):
          summary = simulation.simulate(artifacts)
        with np.load(artifacts.run_dir / "data/nma-simulation-replicates.npz") as raw:
          self.assertEqual(raw["wls_estimates"].shape, (2, 2, 9))
          np.testing.assert_array_equal(raw["xi_hat"], reference_means)
          for rho_index in range(2):
            for replicate in range(2):
              expected = wls_point_estimate(raw["xi_hat"][rho_index, replicate],
                                           np.sqrt(raw["estimated_mean_variances"]
                                                   [rho_index, replicate]))
              np.testing.assert_allclose(raw["wls_estimates"][rho_index, replicate], expected)
            expected = simulation.calibrate_wls_oracle(raw["wls_estimates"][rho_index],
                                                       raw["wls_standard_errors"][rho_index],
                                                       simulation.THETA)
            self.assertEqual(summary["wls_ma_multiplier"][rho_index], expected["multiplier"])
            self.assertEqual(summary["coverage_wls_ma"][rho_index], 1.)
            np.testing.assert_allclose(summary["width_wls_ma_1"][rho_index],
                                       expected["mean_widths"][0])
          np.testing.assert_array_equal(raw["widths_hcct"], 2.)
        audit = json.loads((artifacts.run_dir / "diagnostics/nma-simulation-calibration.json")
                            .read_text())
        self.assertIn("in-sample oracle", audit["oracle_qualification"])
        self.assertEqual(audit["degrees_freedom"], 29)
        self.assertEqual(audit["configuration"], simulation.simulation_configuration("smoke"))
        events = audit["coverage_events"]
        self.assertEqual(events["hcct"]["event"], "truth_in_joint_score_region")
        for method in ("wls", "wls_ma"):
          self.assertEqual(events[method]["event"], "all_coordinate_intervals_cover")
          self.assertEqual(events[method]["coordinate_count"], 9)
        self.assertEqual(events["wls"]["comparison"], "strict_interval_containment")
        self.assertEqual(events["wls_ma"]["comparison"], "less_than_or_equal")
        relation = events["relation"]
        self.assertTrue(relation["hcct_region_implies_own_coordinate_support_box"])
        self.assertFalse(relation["reverse_implication_valid_in_general"])
        self.assertFalse(relation["hcct_coordinate_box_coverage_evaluated"])
        self.assertFalse(relation["like_for_like_coverage_events"])
        self.assertIn("not the WLS boxes", relation["qualification"])
    finally:
      np.random.set_state(state)

  def test_coverage_figure_labels_differing_events_without_relabeling_widths(self) -> None:
    rhos = np.array([0., 0.6])
    series = (np.array([0.95, 0.97]),) * 3
    coverage = simulation.build_comparison_figure(
      rhos, series, ylabel="coverage of stated event", legend_outside=True,
      coverage_events=True,
    )
    widths = simulation.build_comparison_figure(
      rhos, series, ylabel="widths of simultaneous interval",
    )
    try:
      self.assertEqual(coverage.axes[0].get_ylabel(), "coverage of stated event")
      self.assertEqual(
        [text.get_text() for text in coverage.axes[0].get_legend().get_texts()],
        ["HCCT joint region", "WLS simultaneous box", "WLS-MA box (in-sample oracle)"],
      )
      caption = " ".join(text.get_text() for text in coverage.texts)
      self.assertIn("Coverage events differ", caption)
      self.assertIn("9-interval box coverage is not evaluated", caption)
      self.assertEqual(widths.axes[0].get_ylabel(), "widths of simultaneous interval")
      self.assertEqual(
        [text.get_text() for text in widths.axes[0].get_legend().get_texts()],
        ["HCCT", "WLS", "WLS-MA (in-sample oracle)"],
      )
      self.assertFalse(widths.texts)
      coverage.canvas.draw()
      widths.canvas.draw()
      renderer = coverage.canvas.get_renderer()
      axis_bounds = coverage.axes[0].get_window_extent(renderer)
      legend_bounds = coverage.axes[0].get_legend().get_window_extent(renderer)
      self.assertGreater(legend_bounds.y0, axis_bounds.y1)
      for caption_text in coverage.texts:
        self.assertLess(caption_text.get_window_extent(renderer).y1, axis_bounds.y0)
    finally:
      simulation.plt.close(coverage)
      simulation.plt.close(widths)

  def test_simulation_does_not_suppress_numerically_failed_intervals(self) -> None:
    state = np.random.get_state()
    try:
      with tempfile.TemporaryDirectory() as temporary:
        artifacts = ArtifactStore(simulation.EXPERIMENT_ID, "test", Path(temporary))
        with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), \
             patch.object(simulation.MetaAnalysisMD, "support_interval",
                          side_effect=EmptySetNumericalError("uncertified test region")), \
             self.assertRaisesRegex(EmptySetNumericalError, "uncertified"):
          simulation.simulate(artifacts)
        diagnostic = artifacts.run_dir / "diagnostics/nma-simulation-failure.json"
        self.assertTrue(diagnostic.is_file())
        self.assertEqual(json.loads(diagnostic.read_text())["completed_replicates"], 0)
        with np.load(artifacts.run_dir / "diagnostics/nma-simulation-partial-replicates.npz") as raw:
          self.assertTrue(np.isfinite(raw["xi_hat"][0, 0]).all())
          self.assertTrue(np.isnan(raw["covered_hcct"]).all())
    finally:
      np.random.set_state(state)

  def test_certified_empty_is_recorded_zero_width_without_dropping_replicate(self) -> None:
    certificate = MinimumDiagnostics(
      "fixture", 0, 1, 0., "fixture", feasible_set_score_lower_bound=20.,
      empty_certified=True,
    )
    cause = EmptyRegionError(point=np.zeros(9), p_values=np.full(28, 0.01),
                             minimum_score=21., threshold=16., minimum_diagnostics=certificate)
    wrapped = RuntimeError("the confidence region is empty or no feasible support start was found")
    wrapped.__cause__ = cause
    self.assertIs(simulation._certified_empty_region(wrapped), cause)
    self.assertIsNone(simulation._certified_empty_region(RuntimeError("empty")))
    for invalid in (
      replace(certificate, empty_certified=False),
      replace(certificate, feasible_set_score_lower_bound=None),
      replace(certificate, feasible_set_score_lower_bound=16.),
    ):
      incomplete = EmptyRegionError(point=np.zeros(9), p_values=np.full(28, 0.01),
                                     minimum_score=21., threshold=16., minimum_diagnostics=invalid)
      self.assertIsNone(simulation._certified_empty_region(incomplete))
    state = np.random.get_state()
    try:
      with tempfile.TemporaryDirectory() as temporary:
        artifacts = ArtifactStore(simulation.EXPERIMENT_ID, "test", Path(temporary))
        with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), \
             patch.object(simulation.MetaAnalysisMD, "support_interval", side_effect=wrapped):
          summary = simulation.simulate(artifacts)
        np.testing.assert_array_equal(summary["empty_hcct_count"], [2, 2])
        np.testing.assert_array_equal(summary["replicates_per_rho"], [2, 2])
        np.testing.assert_array_equal(summary["coverage_hcct"], [0., 0.])
        np.testing.assert_array_equal(summary["width_hcct_1"], [0., 0.])
        with np.load(artifacts.run_dir / "data/nma-simulation-replicates.npz") as raw:
          self.assertTrue(np.all(raw["empty_hcct"] == 1))
          self.assertTrue(np.isfinite(raw["wls_estimates"]).all())
        certificates = list((artifacts.run_dir / "diagnostics").glob("empty-region-*.json"))
        self.assertEqual(len(certificates), 4)
        self.assertEqual(json.loads(certificates[0].read_text())["minimum_diagnostics"]
                          ["feasible_set_score_lower_bound"], 20.)
    finally:
      np.random.set_state(state)

  def test_failed_support_endpoint_retains_complete_certificate(self) -> None:
    support = _support_fixture()
    failed = replace(support, lower_diagnostics=replace(
      support.lower_diagnostics, success=False, message="forced endpoint failure",
    ))
    state = np.random.get_state()
    try:
      with tempfile.TemporaryDirectory() as temporary:
        artifacts = ArtifactStore(simulation.EXPERIMENT_ID, "test", Path(temporary))
        with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), \
             patch.object(simulation.MetaAnalysisMD, "support_interval", return_value=failed), \
             self.assertRaisesRegex(RuntimeError, "forced endpoint failure"):
          simulation.simulate(artifacts)
        failure = json.loads((artifacts.run_dir / "diagnostics/nma-simulation-failure.json")
                              .read_text())
        saved = failure["failed_support_result"]
        self.assertFalse(saved["lower_diagnostics"]["success"])
        self.assertEqual(saved["lower_diagnostics"]["message"], "forced endpoint failure")
        self.assertEqual(saved["lower_point"], [0.] * 9)
    finally:
      np.random.set_state(state)


if __name__ == "__main__":
  unittest.main()
