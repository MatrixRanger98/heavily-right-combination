"""Runner provenance, failure persistence, and command-line behavior."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reproduction.run import ExperimentTimeout, main


class ReproductionRunnerTests(unittest.TestCase):
  def test_catalog_commands_are_read_only(self) -> None:
    for arguments in (["--list"], ["numerical/pvalue-examples", "--describe"]):
      with self.subTest(arguments=arguments), patch("pathlib.Path.mkdir") as mkdir:
        with contextlib.redirect_stdout(io.StringIO()) as stream:
          main(arguments)
        self.assertIn("numerical/pvalue-examples", stream.getvalue())
        mkdir.assert_not_called()

  def test_invalid_timeout_is_rejected_before_creating_run(self) -> None:
    for value in ("0", "-1", "nan", "inf"):
      with self.subTest(value=value), patch("pathlib.Path.mkdir") as mkdir:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
          main(["numerical/pvalue-examples", "--max-seconds", value])
        mkdir.assert_not_called()

  def test_success_failure_and_timeout_preserve_provenance_and_environment(self) -> None:
    for error, status in (
      (None, "completed"),
      (RuntimeError("test failure"), "failed"),
      (ExperimentTimeout("test timeout"), "timed-out"),
    ):
      with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
        arguments = [
          "numerical/pvalue-examples", "--run-id", "audit", "--profile", "smoke",
          "--results-root", directory,
        ]
        environment = dict(os.environ)
        with patch("reproduction.run.runpy.run_path", side_effect=error):
          if error is None:
            main(arguments)
          else:
            with self.assertRaises(type(error)):
              main(arguments)
        self.assertEqual(dict(os.environ), environment)
        run = Path(directory) / "numerical/pvalue-examples/audit"
        metadata = json.loads((run / "run.json").read_text())
        self.assertEqual(metadata["status"], status)
        self.assertEqual(metadata["profile"], "smoke")
        self.assertEqual(len(metadata["script_sha256"]), 64)
        self.assertIn("reproduction/run.py", metadata["source_sha256"])
        self.assertIn("heavily-right", metadata["packages"])
        self.assertEqual(metadata["produced_outputs"], [])
        self.assertTrue(metadata["missing_expected_outputs"])
        self.assertTrue((run / "logs/stderr.log").is_file())
        with self.assertRaises(SystemExit):
          main(arguments)

  def test_nma_configuration_is_recorded_before_simulation_starts(self) -> None:
    from reproduction.network_meta_analysis.simulation import simulation_configuration

    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "network-meta-analysis/simulation/reduced/run.json"

      def inspect_initial_metadata(*_args, **_kwargs):
        saved = json.loads(target.read_text())
        self.assertEqual(saved["status"], "running")
        self.assertEqual(saved["configuration"], simulation_configuration("paper"))
        raise RuntimeError("bounded provenance check; no simulation")

      with patch("reproduction.run.runpy.run_path", side_effect=inspect_initial_metadata):
        with self.assertRaisesRegex(RuntimeError, "bounded provenance check"):
          main(["network-meta-analysis/simulation", "--profile", "paper",
                "--run-id", "reduced", "--results-root", directory])
      saved = json.loads(target.read_text())
      self.assertEqual(saved["status"], "failed")
      self.assertEqual(saved["configuration"]["replicates_per_rho"], 100)


if __name__ == "__main__":
  unittest.main()
