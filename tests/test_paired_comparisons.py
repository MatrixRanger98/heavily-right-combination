"""Method comparisons must share samples without losing outcomes or provenance."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from heavily_right.combination import CombinationTest
from reproduction.one_dimensional import false_positive_rate, power_comparison
from reproduction.one_dimensional.common import (
  paired_rejection_decisions,
  two_sided_normal_pvalues,
)
from reproduction.sampling import CachedMultivariateNormal

METHODS = (
  "HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes",
)
WORKFLOWS = (false_positive_rate, power_comparison)


class PairedDecisionTests(unittest.TestCase):
  def test_all_methods_receive_the_same_matrix_without_filtering_draws(self):
    p_values = np.array([[.001, .002], [.4, .7], [.02, .03], [.99, .99]])
    original = p_values.copy()
    expected = np.array([[True, False, True, False], [False, False, True, True]])
    tests = [Mock(make_decision=Mock(return_value=row)) for row in expected]

    actual = paired_rejection_decisions(p_values, tests)

    self.assertEqual(actual.dtype, np.dtype(bool))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(p_values, original)
    for test in tests:
      test.make_decision.assert_called_once()
      self.assertIs(test.make_decision.call_args.args[0], p_values)

  def test_empty_or_nonmatrix_inputs_and_empty_method_lists_are_rejected(self):
    test = Mock()
    for values, tests in (
      (np.empty((0, 3)), [test]),
      (np.ones(3), [test]),
      (np.ones((2, 3, 4)), [test]),
      (np.ones((2, 3)), []),
    ):
      with self.subTest(shape=values.shape, methods=len(tests)):
        with self.assertRaises(ValueError):
          paired_rejection_decisions(values, tests)
    test.make_decision.assert_not_called()


class PairedComparisonTests(unittest.TestCase):
  def setUp(self):
    self.random_state = np.random.get_state()
    self.environment = patch.dict(os.environ, {"HCCT_PROFILE": "smoke"})
    self.environment.start()
    self.output = contextlib.redirect_stdout(io.StringIO())
    self.output.__enter__()

  def tearDown(self):
    self.output.__exit__(None, None, None)
    self.environment.stop()
    np.random.set_state(self.random_state)

  def test_smoke_samples_once_per_case_and_preserves_every_method_and_draw(self):
    original_draw = CachedMultivariateNormal.rvs
    original_decision = CombinationTest.make_decision

    for module in WORKFLOWS:
      with self.subTest(workflow=module.EXPERIMENT_ID):
        draws = []
        evaluations = []

        def record_draw(sampler, size=1, random_state=None, *, recorded=draws):
          result = original_draw(sampler, size, random_state)
          recorded.append(result.copy())
          return result

        def record_decision(test, p_values, *, recorded=evaluations):
          result = original_decision(test, p_values)
          recorded.append((test.method, p_values, result.copy(), test.threshold))
          return result

        diagnostics = {}
        np.random.seed(module.SEED)
        with (
          patch.object(CachedMultivariateNormal, "rvs", record_draw),
          patch.object(CombinationTest, "make_decision", record_decision),
        ):
          rates, rhos = module.simulate(diagnostics=diagnostics)

        # Four correlations, two cases: eight batches, not one per method.
        self.assertEqual(len(draws), 8)
        self.assertEqual(len(evaluations), 8 * len(METHODS))
        self.assertEqual(tuple(module.METHODS), METHODS)
        self.assertEqual(rates.shape, (9, 4, 2))
        self.assertEqual(diagnostics["rejections"].shape, (9, 4, 2, 200))
        self.assertEqual(diagnostics["rejections"].dtype, np.dtype(bool))
        np.testing.assert_array_equal(rhos, [0, .3, .6, .9])
        np.testing.assert_array_equal(diagnostics["rhos"], rhos)
        np.testing.assert_array_equal(diagnostics["methods"], METHODS)
        np.testing.assert_array_equal(diagnostics["axes"], ["method", "rho", "case", "draw"])
        np.testing.assert_array_equal(rates, diagnostics["rejections"].mean(axis=-1))
        self.assertEqual(diagnostics["study_count"].item(), 30)
        self.assertEqual(diagnostics["simulations_per_case"].item(), 200)
        self.assertEqual(diagnostics["level"].item(), .05)
        self.assertIn("shared-across-methods", diagnostics["sampling_scheme"].item())
        expected_cases = ["ar1", "equicorrelated"] if module is false_positive_rate else ["dense", "sparse"]
        np.testing.assert_array_equal(diagnostics["cases"], expected_cases)

        for case_index, draw in enumerate(draws):
          rho_index, scenario_index = divmod(case_index, 2)
          group = evaluations[case_index * 9:(case_index + 1) * 9]
          self.assertEqual([entry[0] for entry in group], list(METHODS))
          np.testing.assert_array_equal(group[0][1], two_sided_normal_pvalues(draw))
          for method_index, (_, inputs, decisions, threshold) in enumerate(group):
            self.assertIs(inputs, group[0][1])
            np.testing.assert_array_equal(
              diagnostics["rejections"][method_index, rho_index, scenario_index], decisions,
            )
            self.assertEqual(diagnostics["thresholds"][method_index], threshold)
          if case_index:
            self.assertIsNot(group[0][1], evaluations[(case_index - 1) * 9][1])
            self.assertFalse(np.array_equal(draw, draws[case_index - 1]))
        if module is power_comparison:
          np.testing.assert_array_equal(
            diagnostics["alternative_means"], np.stack(module.alternatives(30)),
          )

  def test_seed_repeatability_and_method_order_do_not_change_draws_or_rng_stream(self):
    for module in WORKFLOWS:
      with self.subTest(workflow=module.EXPERIMENT_ID):
        runs = []
        for methods in (METHODS, METHODS, METHODS[::-1]):
          diagnostics = {}
          np.random.seed(module.SEED)
          with patch.object(module, "METHODS", methods):
            rates, rhos = module.simulate(diagnostics=diagnostics)
          runs.append((rates, rhos, diagnostics, np.random.standard_normal(20)))

        baseline = runs[0]
        for actual in runs[1:]:
          np.testing.assert_array_equal(actual[1], baseline[1])
          np.testing.assert_array_equal(actual[3], baseline[3])
          for baseline_index, method in enumerate(METHODS):
            actual_index = actual[2]["methods"].tolist().index(method)
            np.testing.assert_array_equal(actual[0][actual_index], baseline[0][baseline_index])
            np.testing.assert_array_equal(
              actual[2]["rejections"][actual_index], baseline[2]["rejections"][baseline_index],
            )
            self.assertEqual(
              actual[2]["thresholds"][actual_index], baseline[2]["thresholds"][baseline_index],
            )

  def test_entrypoints_save_boolean_archives_with_seed_and_reconstructible_rates(self):
    for module, stem in (
      (false_positive_rate, "false-positive-rate"),
      (power_comparison, "power-comparison"),
    ):
      with self.subTest(workflow=module.EXPERIMENT_ID), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifacts = SimpleNamespace(data=lambda name, base=root: base / name)
        with (
          patch.object(module.ArtifactStore, "for_experiment", return_value=artifacts),
          patch.object(module, "render") as render,
        ):
          module.main()

        render.assert_called_once()
        rates = np.load(root / f"{stem}.npy", allow_pickle=False)
        with np.load(root / f"{stem}-paired.npz", allow_pickle=False) as archive:
          self.assertEqual(archive["rejections"].dtype, np.dtype(bool))
          self.assertEqual(archive["rejections"].shape, (9, 4, 2, 200))
          self.assertEqual(archive["seed"].item(), module.SEED)
          self.assertIn("RandomState/MT19937", archive["rng"].item())
          self.assertEqual(archive["thresholds"].shape, (9,))
          self.assertTrue(np.isfinite(archive["thresholds"]).all())
          np.testing.assert_array_equal(archive["methods"], METHODS)
          np.testing.assert_array_equal(rates, archive["rejections"].mean(axis=-1))
          np.testing.assert_array_equal(render.call_args.args[0], rates)
          np.testing.assert_array_equal(render.call_args.args[1], archive["rhos"])


if __name__ == "__main__":
  unittest.main()
