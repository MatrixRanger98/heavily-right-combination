import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.artifacts import ArtifactStore
from reproduction.manifest import EXPERIMENTS, get_experiment


class ArtifactStoreTests(unittest.TestCase):
  def test_paths_are_grouped_by_experiment_and_run(self):
    with tempfile.TemporaryDirectory() as directory:
      with patch.dict(
        os.environ,
        {
          "HCCT_RESULTS_ROOT": directory,
          "HCCT_RUN_ID": "test-run",
          "HCCT_EXPERIMENT_ID": "one-dimensional/normal-coverage",
        },
        clear=False,
      ):
        store = ArtifactStore.for_experiment("one-dimensional/normal-coverage")
        path = store.figure("coverage-ar1.pdf")

      self.assertEqual(
        path,
        Path(directory).resolve()
        / "one-dimensional/normal-coverage/test-run/figures/coverage-ar1.pdf",
      )
      self.assertTrue(path.parent.is_dir())

  def test_existing_artifact_is_never_reused(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(
      os.environ,
      {"HCCT_RESULTS_ROOT": directory, "HCCT_RUN_ID": "test-run"},
      clear=False,
    ):
      store = ArtifactStore.for_experiment("numerical/distribution-precision")
      path = store.data("critical-values.txt")
      path.write_text("existing")
      with self.assertRaises(FileExistsError):
        store.data("critical-values.txt")

  def test_figure_requires_vector_pdf(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(
      os.environ,
      {"HCCT_RESULTS_ROOT": directory, "HCCT_RUN_ID": "test-run"},
      clear=False,
    ):
      store = ArtifactStore.for_experiment("illustrations/connectivity-regions")
      with self.assertRaises(ValueError):
        store.figure("acceptance-region.png")

  def test_runner_and_script_ids_must_agree(self):
    with patch.dict(
      os.environ,
      {"HCCT_EXPERIMENT_ID": "one-dimensional/power"},
      clear=False,
    ), self.assertRaises(ValueError):
      ArtifactStore.for_experiment("one-dimensional/interval-width")

  def test_protected_and_legacy_roots_are_rejected(self):
    # Simulate a source project independently of where the wheel is installed.
    with tempfile.TemporaryDirectory() as directory:
      project = Path(directory)
      manifest = project / "Code/reproduction/manifest.py"
      manifest.parent.mkdir(parents=True)
      manifest.write_text("# source project marker\n")
      for relative in ("Paper/results", "Archived/new", "Code/fig/new", "Code/data/new",
                       "Code/tmp/new", "Code/legacy/new", "Code/results/historical/new",
                       "Code/results/legacy/new"):
        root = project / relative
        with self.subTest(root=root), patch.dict(
          os.environ,
          {"HCCT_RESULTS_ROOT": str(root), "HCCT_RUN_ID": "test-run"},
          clear=False,
        ), self.assertRaises(ValueError):
          ArtifactStore.for_experiment("one-dimensional/power")

  def test_run_id_cannot_escape_its_experiment(self):
    with tempfile.TemporaryDirectory() as directory:
      for run_id in ("../escape", "nested/run"):
        with self.subTest(run_id=run_id), patch.dict(
          os.environ,
          {"HCCT_RESULTS_ROOT": directory, "HCCT_RUN_ID": run_id},
          clear=False,
        ), self.assertRaises(ValueError):
          ArtifactStore.for_experiment("one-dimensional/power")


class ExperimentManifestTests(unittest.TestCase):
  def test_ids_are_unique_and_scripts_exist(self):
    ids = [experiment.id for experiment in EXPERIMENTS]
    self.assertEqual(len(ids), len(set(ids)))
    for experiment in EXPERIMENTS:
      with self.subTest(experiment=experiment.id):
        if experiment.language != "python" and not (Path(__file__).resolve().parents[1] / "geogebra").exists():
          continue  # Optional external-tool sources are not shipped in the sdist.
        self.assertTrue(experiment.script_path.is_file())
        for alternate in experiment.alternate_script_paths:
          if alternate.suffix == ".py" or (Path(__file__).resolve().parents[1] / "wolfram").exists():
            self.assertTrue(alternate.is_file())

  def test_lookup(self):
    experiment = get_experiment("one-dimensional/normal-coverage")
    self.assertEqual(
      experiment.script,
      "reproduction/one_dimensional/normal_coverage.py",
    )

  def test_declared_outputs_are_unique_safe_and_typed(self):
    for experiment in EXPERIMENTS:
      with self.subTest(experiment=experiment.id):
        self.assertTrue(experiment.outputs)
        self.assertEqual(len(experiment.outputs), len(set(experiment.outputs)))
      for output in experiment.outputs:
        path = Path(output)
        with self.subTest(experiment=experiment.id, output=output):
          self.assertFalse(path.is_absolute())
          self.assertNotIn("..", path.parts)
          self.assertIn(path.parts[0], {"data", "diagnostics", "figures"})
          if path.parts[0] == "figures":
            self.assertEqual(path.suffix, ".pdf")

  def test_registered_python_experiments_are_import_safe_entry_points(self):
    for experiment in EXPERIMENTS:
      if experiment.language != "python":
        continue
      tree = ast.parse(experiment.script_path.read_text())
      has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in tree.body
      )
      has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
      )
      with self.subTest(experiment=experiment.id):
        self.assertTrue(has_main)
        self.assertTrue(has_main_guard)

  def test_registered_python_writers_do_not_target_legacy_output_folders(self):
    legacy_fragments = ("./tmp", "../tmp", "./fig", "../fig")
    writer_names = {"save", "savez", "savefig"}
    for experiment in EXPERIMENTS:
      if experiment.language != "python":
        continue
      tree = ast.parse(experiment.script_path.read_text())
      for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
          continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else ""
        first_argument = node.args[0]
        if name not in writer_names:
          continue
        with self.subTest(experiment=experiment.id, line=node.lineno):
          if isinstance(first_argument, ast.Constant):
            self.assertFalse(
              str(first_argument.value).startswith(legacy_fragments),
              "registered writers must use ArtifactStore paths",
            )
          self.assertIsInstance(first_argument, ast.Call)
          self.assertIsInstance(first_argument.func, ast.Attribute)
          self.assertIsInstance(first_argument.func.value, ast.Name)
          self.assertEqual(first_argument.func.value.id, "artifacts")
          self.assertIn(first_argument.func.attr, {"data", "diagnostic", "figure"})

  def test_declared_seeds_match_scripts(self):
    for experiment in EXPERIMENTS:
      if experiment.seed is None:
        continue
      tree = ast.parse(experiment.script_path.read_text())
      seeds = []
      constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
      }
      # A private Generator may receive an overridable function parameter
      # whose default is the experiment's declared seed. Resolve it within
      # that function only, rather than requiring global RNG mutation.
      for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef):
          continue
        arguments = definition.args.posonlyargs + definition.args.args
        defaults = definition.args.defaults
        local_seeds = {}
        default_arguments = arguments[-len(defaults):] if defaults else []
        for argument, default in zip(default_arguments, defaults, strict=True):
          if isinstance(default, ast.Name) and default.id in constants:
            local_seeds[argument.arg] = constants[default.id]
          elif isinstance(default, ast.Constant):
            local_seeds[argument.arg] = default.value
        for call in ast.walk(definition):
          if (isinstance(call, ast.Call) and call.args
              and isinstance(call.func, ast.Attribute) and call.func.attr == "default_rng"
              and isinstance(call.args[0], ast.Name) and call.args[0].id in local_seeds):
            seeds.append(local_seeds[call.args[0].id])
      for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
          continue
        function = node.func
        if (
          node.args
          and
          isinstance(function, ast.Attribute)
          and function.attr == "seed"
          and isinstance(function.value, ast.Attribute)
          and function.value.attr == "random"
        ):
          argument = node.args[0]
          if isinstance(argument, ast.Constant):
            seeds.append(argument.value)
          elif isinstance(argument, ast.Name) and argument.id in constants:
            seeds.append(constants[argument.id])
        if (
          node.args
          and
          isinstance(function, ast.Attribute)
          and function.attr == "default_rng"
          and isinstance(node.args[0], ast.Name)
          and node.args[0].id in constants
        ):
          seeds.append(constants[node.args[0].id])
        for keyword in node.keywords:
          if keyword.arg == "seed" and isinstance(keyword.value, ast.Constant):
            seeds.append(keyword.value.value)
      with self.subTest(experiment=experiment.id):
        self.assertIn(experiment.seed, seeds)


if __name__ == "__main__":
  unittest.main()
