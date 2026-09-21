"""Bounded plot-only reuse tests: synthetic data, never simulation or retained runs."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np

from reproduction._replot_data import (
  CACHE_NAME,
  MODULES,
  SCHEMA_NAME,
  configuration_for,
  load_arrays,
  schema_for,
  sha256,
)
from reproduction.manifest import get_experiment
from reproduction.replot import _figures, main, replot


def _source(
  directory: Path, experiment: str = "one-dimensional/power", *, status: str = "completed",
  profile: str = "smoke", nma_replicates: int | None = None,
  configuration: dict | None = None,
) -> Path:
  """Make schema-complete fixtures with deterministic synthetic values, not draws."""
  source = directory / "source" / experiment
  source.mkdir(parents=True)
  schema = schema_for(experiment, profile, nma_replicates=nma_replicates)
  repeat = schema.document()["configuration_interpretation"].get("replicates_per_rho", 2)
  grouped: dict[str, dict[str, np.ndarray]] = {}
  for name, field in schema.fields.items():
    shape = tuple(len(schema.axes[axis]) for axis in field.axes)
    array = np.linspace(.2, .8, num=int(np.prod(shape))).reshape(shape)
    if field.kind == "count":
      array = np.full(shape, repeat if name == "replicates_per_rho" else 0)
    path = source / field.source
    path.parent.mkdir(exist_ok=True)
    if field.member is not None:
      grouped.setdefault(field.source, {})[field.member] = array
    else:
      np.save(path, array)
  for relative, arrays in grouped.items():
    np.savez(source / relative, rhos=schema.axes["rhos"], **arrays)
  if experiment == "network-meta-analysis/simulation":
    from reproduction.network_meta_analysis.simulation import COVERAGE_EVENTS

    arrays = grouped["data/nma-simulation-summary.npz"]
    calibration = {
      "schema_version": 1, "seed": 2025, "replicates_per_rho": repeat,
      "sample_size": 100 if profile == "paper" else 30,
      "degrees_freedom": 99 if profile == "paper" else 29,
      "rhos": schema.axes["rhos"].tolist(), "coverage_events": COVERAGE_EVENTS,
      "oracle_multipliers": arrays["wls_ma_multiplier"].tolist(),
      "oracle_empirical_coverage": arrays["coverage_wls_ma"].tolist(),
      "empty_hcct_counts": arrays["empty_hcct_count"].tolist(),
    }
    if configuration is not None:
      calibration["configuration"] = configuration
    (source / "diagnostics").mkdir()
    (source / "diagnostics/nma-simulation-calibration.json").write_text(json.dumps(calibration))
    # Diagnostic endpoint NaNs are allowed; this file is linked/hashed, not a plot input.
    np.savez(source / "data/nma-simulation-replicates.npz", hcct_support_points=np.array([np.nan]))
  entry = get_experiment(experiment)
  digest = sha256(entry.script_path)
  metadata = {
    "schema_version": 1, "experiment": experiment, "profile": profile,
    "declared_seed": 2025, "status": status, "finished_at": "2026-09-21T00:00:00+00:00",
    "script": entry.script, "script_sha256": digest, "source_sha256": {entry.script: digest},
  }
  if configuration is not None:
    metadata["configuration"] = configuration
  (source / "run.json").write_text(json.dumps(metadata))
  return source


def _hashes(source: Path) -> dict[str, str]:
  return {str(path.relative_to(source)): sha256(path) for path in source.rglob("*") if path.is_file()}


def _placeholder_render(schema, _arrays, store) -> None:
  for relative in _figures(schema):
    store.figure(Path(relative).name).write_bytes(b"temporary test fixture; not a figure result")
  if schema.experiment == "network-meta-analysis/simulation":
    np.savez(store.data("nma-simulation-plotted-summary.npz"), fixture=True)
    store.diagnostic("nma-simulation-plot-policy.json").write_text('{"fixture": true}')


class ReplotTests(unittest.TestCase):
  def test_nma_current_and_historical_workloads_survive_cache_chains(self) -> None:
    from reproduction.network_meta_analysis.simulation import simulation_configuration

    experiment = "network-meta-analysis/simulation"
    for profile, count, recorded in (("paper", 100, True), ("paper", 100, False),
                                     ("paper", 500, False), ("paper", 500, True),
                                     ("smoke", 2, True)):
      with self.subTest(profile=profile, count=count, recorded=recorded), \
           tempfile.TemporaryDirectory() as directory, \
           patch("reproduction.replot.render_saved", _placeholder_render):
        root = Path(directory)
        configuration = None
        if recorded:
          configuration = (simulation_configuration(profile) if count != 500
                           else configuration_for(experiment, profile, nma_replicates=count))
        source = _source(root, experiment, profile=profile, nma_replicates=count,
                         configuration=configuration)
        before = _hashes(source)
        first = replot(source, run_id="first", results_root=root / "out")
        second = replot(first, run_id="second", results_root=root / "out")
        self.assertEqual(_hashes(source), before)
        for target in (first, second):
          metadata = json.loads((target / "run.json").read_text())
          document = json.loads((target / "data" / SCHEMA_NAME).read_text())
          self.assertEqual(metadata["status"], "completed")
          self.assertEqual(metadata["source_recorded_configuration"], configuration)
          self.assertEqual(document["source_recorded_configuration"], configuration)
          self.assertEqual(document["configuration_interpretation"]["replicates_per_rho"], count)
          self.assertEqual(document["qualifications"]["replicates_per_rho"], count)
          self.assertEqual(document["schema_version"], 1)
          with np.load(target / "data" / CACHE_NAME, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["replicates_per_rho"],
                                          np.full(10 if profile == "paper" else 2, count))
    self.assertEqual(configuration_for(experiment, "paper")["replicates_per_rho"], 100)

  def test_nma_top_level_calibration_conflicts_rejected_before_output(self) -> None:
    experiment = "network-meta-analysis/simulation"
    conflicts = {
      "sample_size": 200, "degrees_freedom": 199, "level": .1,
      "parameter_dimension": 10, "study_count": 29, "seed": 2024,
      "profile": "smoke", "rhos": [0., .6],
    }
    for nested in (False, True):
      for key, value in conflicts.items():
        with self.subTest(nested=nested, key=key), tempfile.TemporaryDirectory() as directory:
          root = Path(directory)
          configuration = configuration_for(experiment, "paper") if nested else None
          source = _source(root, experiment, profile="paper", configuration=configuration)
          path = source / "diagnostics/nma-simulation-calibration.json"
          calibration = json.loads(path.read_text())
          calibration[key] = value
          path.write_text(json.dumps(calibration))
          before = _hashes(source)
          with self.assertRaises(ValueError):
            replot(source, run_id="invalid", results_root=root / "out")
          self.assertFalse((root / "out").exists())
          self.assertEqual(_hashes(source), before)

  def test_nma_cached_top_level_calibration_conflict_rejected_with_valid_hash(self) -> None:
    experiment = "network-meta-analysis/simulation"
    with tempfile.TemporaryDirectory() as directory, \
         patch("reproduction.replot.render_saved", _placeholder_render):
      root = Path(directory)
      source = _source(root, experiment, profile="paper", nma_replicates=500)
      first = replot(source, run_id="first", results_root=root / "out")
      path = first / "data" / SCHEMA_NAME
      document = json.loads(path.read_text())
      document["qualifications"].update(sample_size=200, degrees_freedom=199)
      path.write_text(json.dumps(document))
      metadata_path = first / "run.json"
      metadata = json.loads(metadata_path.read_text())
      metadata["cache_sha256"][path.relative_to(first).as_posix()] = sha256(path)
      metadata_path.write_text(json.dumps(metadata))
      before = _hashes(first)
      with self.assertRaisesRegex(ValueError, "sample_size"):
        replot(first, run_id="invalid", results_root=root / "out")
      self.assertFalse((root / "out/network-meta-analysis/simulation/invalid").exists())
      self.assertEqual(_hashes(first), before)

  def test_nma_completion_requires_plotted_summary_and_policy(self) -> None:
    experiment = "network-meta-analysis/simulation"
    for missing in ("data/nma-simulation-plotted-summary.npz",
                    "diagnostics/nma-simulation-plot-policy.json"):
      with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root, experiment, profile="paper")

        def incomplete_render(schema, arrays, store, *, missing_output=missing):
          _placeholder_render(schema, arrays, store)
          (store.run_dir / missing_output).unlink()

        before = _hashes(source)
        with patch("reproduction.replot.render_saved", incomplete_render), \
             self.assertRaisesRegex(RuntimeError, "expected output"):
          replot(source, run_id="incomplete", results_root=root / "out")
        metadata = json.loads((root / "out" / experiment / "incomplete/run.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertIn(missing, metadata["expected_outputs"])
        self.assertEqual(metadata["missing_expected_outputs"], [missing])
        self.assertEqual(_hashes(source), before)

  def test_nma_incomplete_or_conflicting_source_counts_rejected_before_output(self) -> None:
    experiment = "network-meta-analysis/simulation"
    for problem in ("unsupported", "mixed", "calibration", "recorded", "embedded",
                    "missing_recorded_count", "smoke", "boolean_calibration"):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        profile = "smoke" if problem == "smoke" else "paper"
        source = _source(root, experiment, profile=profile,
                         nma_replicates=2 if profile == "smoke" else 500, status="failed")
        summary_path = source / "data/nma-simulation-summary.npz"
        calibration_path = source / "diagnostics/nma-simulation-calibration.json"
        metadata_path = source / "run.json"
        with np.load(summary_path, allow_pickle=False) as saved:
          summary = dict(saved)
        calibration = json.loads(calibration_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        if problem == "unsupported":
          summary["replicates_per_rho"][:] = 499
          calibration["replicates_per_rho"] = 499
        elif problem == "mixed":
          summary["replicates_per_rho"][0] = 100
        elif problem == "calibration":
          calibration["replicates_per_rho"] = 100
        elif problem == "boolean_calibration":
          calibration["replicates_per_rho"] = True
        elif problem == "recorded":
          metadata["configuration"] = {"replicates_per_rho": 100}
        elif problem == "embedded":
          calibration["configuration"] = {"replicates_per_rho": 100}
        elif problem == "missing_recorded_count":
          metadata["configuration"] = {"seed": 2025}
        else:
          summary["replicates_per_rho"][:] = 100
          calibration["replicates_per_rho"] = 100
        np.savez(summary_path, **summary)
        calibration_path.write_text(json.dumps(calibration))
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
          replot(source, run_id="invalid", results_root=root / "out", allow_failed_source=True)
        self.assertFalse((root / "out").exists())

  def test_nma_cache_count_disagreements_rejected_even_with_updated_checksums(self) -> None:
    for problem in ("interpretation", "qualification", "recorded_sidecar", "recorded_metadata",
                    "summary"):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory, \
           patch("reproduction.replot.render_saved", _placeholder_render):
        root = Path(directory)
        source = _source(root, "network-meta-analysis/simulation", profile="paper", nma_replicates=500)
        first = replot(source, run_id="first", results_root=root / "out")
        document_path = first / "data" / SCHEMA_NAME
        cache_path = first / "data" / CACHE_NAME
        metadata_path = first / "run.json"
        document = json.loads(document_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        if problem == "interpretation":
          document["configuration_interpretation"]["replicates_per_rho"] = 100
        elif problem == "qualification":
          document["qualifications"]["replicates_per_rho"] = 100
        elif problem == "recorded_sidecar":
          document["source_recorded_configuration"] = {"replicates_per_rho": 100}
        elif problem == "recorded_metadata":
          metadata["source_recorded_configuration"] = {"replicates_per_rho": 100}
        else:
          with np.load(cache_path, allow_pickle=False) as saved:
            arrays = dict(saved)
          arrays["replicates_per_rho"][:] = 100
          np.savez_compressed(cache_path, **arrays)
        document_path.write_text(json.dumps(document))
        for path in (document_path, cache_path):
          metadata["cache_sha256"][path.relative_to(first).as_posix()] = sha256(path)
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
          replot(first, run_id="invalid", results_root=root / "out")
        self.assertFalse((root / "out/network-meta-analysis/simulation/invalid").exists())

  def test_failed_nma_recovery_requires_complete_summary_not_partial_evidence(self) -> None:
    for count in (100, 500):
      with self.subTest(count=count), tempfile.TemporaryDirectory() as directory, \
           patch("reproduction.replot.render_saved", _placeholder_render):
        root = Path(directory)
        source = _source(root, "network-meta-analysis/simulation", profile="paper",
                         nma_replicates=count, status="failed")
        with self.assertRaises(ValueError):
          replot(source, run_id="recovery", results_root=root / "out")
        target = replot(source, run_id="recovery", results_root=root / "out",
                        allow_failed_source=True)
        self.assertEqual(json.loads((target / "run.json").read_text())["status"], "completed")
        (source / "data/nma-simulation-summary.npz").unlink()
        with self.assertRaises(ValueError):
          replot(source, run_id="partial", results_root=root / "out", allow_failed_source=True)
        self.assertFalse((root / "out/network-meta-analysis/simulation/partial").exists())

  def test_all_seven_render_real_pdfs_without_sampling_or_rng_changes(self) -> None:
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
      root = Path(directory)
      sources = {experiment: _source(root, experiment) for experiment in MODULES}
      for module_name in MODULES.values():
        module = importlib.import_module(module_name)
        for name in ("simulate", "simulate_coverage", "simulate_student", "simulate_bivariate_copula"):
          if hasattr(module, name):
            stack.enter_context(patch.object(module, name, side_effect=AssertionError("must not simulate")))
      stack.enter_context(patch("numpy.random.seed", side_effect=AssertionError("must not reseed")))
      stack.enter_context(patch("numpy.random.default_rng", side_effect=AssertionError("must not create RNG")))
      stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
      before_rng = np.random.get_state()
      for experiment, source in sources.items():
        before_source = _hashes(source)
        with self.subTest(experiment=experiment):
          target = replot(source, run_id="plots", results_root=root / "results")
          self.assertEqual(_hashes(source), before_source)
          metadata = json.loads((target / "run.json").read_text())
          self.assertEqual(metadata["status"], "completed")
          self.assertEqual(metadata["mode"], "replot")
          self.assertFalse(metadata["fresh_simulation"])
          self.assertEqual(metadata["declared_seed"], 2025)
          self.assertEqual(metadata["missing_expected_outputs"], [])
          for relative in _figures(schema_for(experiment, "smoke")):
            self.assertTrue((target / relative).read_bytes().startswith(b"%PDF"))
          with np.load(target / "data" / CACHE_NAME, allow_pickle=False) as cache:
            self.assertEqual(cache["seed"].item(), 2025)
            self.assertEqual(cache["profile"].item(), "smoke")
          if experiment == "network-meta-analysis/simulation":
            document = json.loads((target / "data" / SCHEMA_NAME).read_text())
            self.assertFalse(document["qualifications"]["coverage_events"]["relation"]["like_for_like_coverage_events"])
      after_rng = np.random.get_state()
      self.assertEqual(before_rng[0], after_rng[0])
      np.testing.assert_array_equal(before_rng[1], after_rng[1])
      self.assertEqual(before_rng[2:], after_rng[2:])

  def test_cache_reuse_and_existing_destination_protection(self) -> None:
    with tempfile.TemporaryDirectory() as directory, patch("reproduction.replot.render_saved", _placeholder_render):
      root = Path(directory)
      source = _source(root)
      first = replot(source, run_id="first", results_root=root / "out")
      before = _hashes(first)
      second = replot(first, run_id="second", results_root=root / "out")
      self.assertEqual(_hashes(first), before)
      metadata = json.loads((second / "run.json").read_text())
      self.assertEqual(metadata["source_axes"], "validated normalized cache")
      self.assertEqual(metadata["source_run"], str(first))
      with self.assertRaises(FileExistsError):
        replot(source, run_id="first", results_root=root / "out")

  def test_plot_failure_keeps_complete_cache_and_requires_explicit_recovery(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = _source(root)
      target = root / "out/one-dimensional/power/failed"

      def fail_after_cache(_schema, _arrays, _store):
        self.assertTrue((target / "data" / CACHE_NAME).is_file())
        raise RuntimeError("test plot failure")

      with patch("reproduction.replot.render_saved", fail_after_cache), self.assertRaises(RuntimeError):
        replot(source, run_id="failed", results_root=root / "out")
      metadata = json.loads((target / "run.json").read_text())
      self.assertEqual(metadata["status"], "failed")
      self.assertIn("test plot failure", metadata["incident"])
      with self.assertRaises(ValueError):
        replot(target, run_id="recovered", results_root=root / "out")
      with patch("reproduction.replot.render_saved", _placeholder_render):
        recovered = replot(target, run_id="recovered", results_root=root / "out", allow_failed_source=True)
      self.assertEqual(json.loads((recovered / "run.json").read_text())["status"], "completed")

  def test_invalid_identity_or_active_source_rejected_before_output(self) -> None:
    for key, value in (("schema_version", 9), ("profile", "unknown"), ("declared_seed", 1),
                       ("status", "running"), ("finished_at", None), ("script_sha256", "wrong")):
      with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root)
        path = source / "run.json"
        metadata = json.loads(path.read_text())
        metadata[key] = value
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
          replot(source, run_id="invalid", results_root=root / "out", allow_failed_source=True)
        self.assertFalse((root / "out").exists())

  def test_partial_wrong_shape_nonfinite_and_out_of_range_rejected(self) -> None:
    for problem in ("missing", "shape", "nan", "range", "object"):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root, status="failed")
        data = source / "data/power-comparison.npy"
        if problem == "missing":
          data.unlink()
        else:
          values = np.load(data, allow_pickle=False)
          if problem == "shape":
            values = values[:, :2]
          elif problem == "nan":
            values[0, 0, 0] = np.nan
          elif problem == "range":
            values[0, 0, 0] = 1.01
          else:
            values = values.astype(object)
          np.save(data, values)
        with self.assertRaises(ValueError):
          replot(source, run_id="invalid", results_root=root / "out", allow_failed_source=True)
        self.assertFalse((root / "out").exists())

  def test_incomplete_copula_case_is_not_a_full_recovery(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = _source(root, "one-dimensional/copulas", status="timed-out")
      (source / "data/copula-coverage-ali-mikhail-haq-fisher.npy").unlink()
      with self.assertRaises(ValueError):
        replot(source, run_id="invalid", results_root=root / "out", allow_failed_source=True)

  def test_old_nma_missing_oracle_and_mismatched_event_metadata_rejected(self) -> None:
    for problem in ("missing_oracle", "wrong_events", "wrong_summary"):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = _source(root, "network-meta-analysis/simulation")
        summary = source / "data/nma-simulation-summary.npz"
        with np.load(summary, allow_pickle=False) as saved:
          arrays = dict(saved)
        if problem == "missing_oracle":
          del arrays["coverage_wls_ma"]
          np.savez(summary, **arrays)
        elif problem == "wrong_summary":
          arrays["wls_ma_multiplier"] *= 2
          np.savez(summary, **arrays)
        else:
          path = source / "diagnostics/nma-simulation-calibration.json"
          calibration = json.loads(path.read_text())
          calibration["coverage_events"] = {}
          path.write_text(json.dumps(calibration))
        with self.assertRaises(ValueError):
          replot(source, run_id="invalid", results_root=root / "out")

  def test_cache_checksum_and_semantic_axis_mismatch_rejected(self) -> None:
    for problem in ("checksum", "axis"):
      with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory, \
           patch("reproduction.replot.render_saved", _placeholder_render):
        root = Path(directory)
        first = replot(_source(root), run_id="first", results_root=root / "out")
        path = first / "data" / CACHE_NAME
        with np.load(path, allow_pickle=False) as saved:
          arrays = dict(saved)
        arrays["axis__rhos"] = arrays["axis__rhos"] + .01
        np.savez_compressed(path, **arrays)
        if problem == "axis":
          meta_path = first / "run.json"
          metadata = json.loads(meta_path.read_text())
          metadata["cache_sha256"][f"data/{CACHE_NAME}"] = sha256(path)
          meta_path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
          replot(first, run_id="second", results_root=root / "out")

  def test_symlink_destination_cannot_write_inside_source_or_protected_tree(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = _source(root)
      out = root / "out"
      (out / "one-dimensional").mkdir(parents=True)
      (out / "one-dimensional/power").symlink_to(source, target_is_directory=True)
      before = _hashes(source)
      with self.assertRaises(ValueError):
        replot(source, run_id="nested", results_root=out)
      self.assertEqual(_hashes(source), before)
      self.assertFalse((source / "nested").exists())
      project = root / "project"
      (project / "Code/reproduction").mkdir(parents=True)
      (project / "Code/reproduction/manifest.py").write_text("# fixture")
      (project / "Paper").mkdir()
      (out / "one-dimensional/power").unlink()
      (out / "one-dimensional/power").symlink_to(project / "Paper", target_is_directory=True)
      with self.assertRaises(ValueError):
        replot(source, run_id="protected", results_root=out)
      self.assertEqual(list((project / "Paper").iterdir()), [])

  def test_direct_nested_destination_and_unsafe_run_id_are_rejected(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = _source(root)
      for destination, run_id in ((source, "nested"), (root / "out", "../bad")):
        with self.subTest(destination=destination, run_id=run_id), self.assertRaises(ValueError):
          replot(source, run_id=run_id, results_root=destination)

  def test_cli_reuses_source_profile_not_environment_and_prints_no_simulation(self) -> None:
    with tempfile.TemporaryDirectory() as directory, \
         patch("reproduction.replot.render_saved", _placeholder_render), \
         patch.dict("os.environ", {"HCCT_PROFILE": "paper"}), \
         contextlib.redirect_stdout(io.StringIO()) as output:
      root = Path(directory)
      main([str(_source(root)), "--run-id", "cli", "--results-root", str(root / "out")])
      metadata = json.loads((root / "out/one-dimensional/power/cli/run.json").read_text())
      self.assertEqual(metadata["profile"], "smoke")
      self.assertIn("no simulation was run", output.getvalue())

  def test_every_profile_schema_has_matching_complete_field_shapes(self) -> None:
    for experiment in MODULES:
      for profile in ("paper", "smoke"):
        with self.subTest(experiment=experiment, profile=profile):
          schema = schema_for(experiment, profile)
          document = schema.document()
          self.assertEqual(document["seed"], 2025)
          self.assertIn("not proof", document["configuration_qualification"])
          for name, field in schema.fields.items():
            self.assertEqual(document["fields"][name]["shape"], [len(schema.axes[a]) for a in field.axes])

  def test_source_hashes_record_exact_loaded_data(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      source = _source(Path(directory))
      schema = schema_for("one-dimensional/power", "smoke")
      arrays, hashes, _ = load_arrays(source, json.loads((source / "run.json").read_text()), schema)
      self.assertEqual(arrays["power"].shape, (9, 4, 2))
      self.assertEqual(hashes["data/power-comparison.npy"], sha256(source / "data/power-comparison.npy"))


if __name__ == "__main__":
  unittest.main()
