"""Run a registered Python reproduction experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import runpy
import signal
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from io import TextIOBase
from pathlib import Path
from typing import TextIO

from reproduction.artifacts import (
  CODE_ROOT,
  DEFAULT_RESULTS_ROOT,
  _validate_results_root,
  _validate_run_id,
)
from reproduction.manifest import EXPERIMENTS, get_experiment
from reproduction.profiles import PAPER_PROFILE, PROFILES


class ExperimentTimeout(TimeoutError):
  """Raised when an experiment exceeds its requested wall-time limit."""


class _Tee(TextIOBase):
  def __init__(self, *streams: TextIO):
    self._streams = streams

  def write(self, text: str) -> int:
    for stream in self._streams:
      stream.write(text)
    return len(text)

  def flush(self) -> None:
    for stream in self._streams:
      if not stream.closed:
        stream.flush()

  def isatty(self) -> bool:
    return any(stream.isatty() for stream in self._streams)


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
  path.write_text(json.dumps(metadata, indent=2) + "\n")


def _package_versions() -> dict[str, str]:
  installed = {}
  for package in ("heavily-right", "matplotlib", "numpy", "pandas", "scipy", "seaborn"):
    try:
      installed[package] = version(package)
    except PackageNotFoundError:
      installed[package] = "not-installed"
  return installed


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("experiment", nargs="?", choices=sorted(item.id for item in EXPERIMENTS))
  parser.add_argument("--list", action="store_true", help="List experiment IDs and runtime guidance.")
  parser.add_argument("--describe", action="store_true", help="Print the selected experiment as JSON.")
  parser.add_argument("--run-id", help="Stable label for this run; defaults to a UTC timestamp.")
  parser.add_argument(
    "--profile",
    choices=PROFILES,
    default=PAPER_PROFILE,
    help="Use the documented paper-profile settings or a small end-to-end smoke workload.",
  )
  parser.add_argument(
    "--max-seconds",
    type=float,
    help="Stop and record the run if it exceeds this wall-time limit.",
  )
  parser.add_argument(
    "--results-root",
    type=Path,
    default=DEFAULT_RESULTS_ROOT,
    help="Root directory for immutable run folders.",
  )
  return parser


@contextmanager
def _time_limit(seconds: float | None):
  if seconds is None:
    yield
    return
  if seconds <= 0:
    raise ValueError("max_seconds must be positive")

  def expire(_signum, _frame):
    raise ExperimentTimeout(f"experiment exceeded {seconds:g} seconds")

  previous_handler = signal.signal(signal.SIGALRM, expire)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler)


def main(argv: list[str] | None = None) -> None:
  parser = _parser()
  args = parser.parse_args(argv)
  if args.list:
    for item in EXPERIMENTS:
      print(f"{item.id}\t{item.language}\t{item.paper_runtime}")
    return
  if args.experiment is None:
    parser.error("an experiment ID is required unless --list is used")
  experiment = get_experiment(args.experiment)
  if args.describe:
    print(json.dumps(asdict(experiment), indent=2))
    return
  if experiment.language != "python":
    raise SystemExit(
      f"{experiment.id} uses {experiment.language}; see its Markdown instructions instead"
    )
  if not experiment.runner_enabled:
    raise SystemExit(f"{experiment.id} is not runner-enabled: {experiment.notes}")
  if args.max_seconds is not None and (
    not math.isfinite(args.max_seconds) or args.max_seconds <= 0
  ):
    parser.error("--max-seconds must be finite and positive")
  if args.max_seconds is not None and not hasattr(signal, "SIGALRM"):
    parser.error("--max-seconds requires POSIX interval timers; use Linux/macOS or WSL")

  run_id = _validate_run_id(
    args.run_id or datetime.now(UTC).strftime("%Y%m%dt%H%M%S.%fz")
  )
  results_root = _validate_results_root(args.results_root)
  run_dir = results_root / experiment.id / run_id
  if run_dir.exists():
    raise SystemExit(f"refusing to reuse existing run directory: {run_dir}")
  run_dir.mkdir(parents=True)
  metadata: dict[str, object] = {
    "schema_version": 1,
    "status": "running",
    "experiment": experiment.id,
    "run_id": run_id,
    "script": experiment.script,
    "script_sha256": hashlib.sha256(experiment.script_path.read_bytes()).hexdigest(),
    "declared_seed": experiment.seed,
    "profile": args.profile,
    "max_seconds": args.max_seconds,
    "expected_outputs": list(experiment.outputs),
    "paper_runtime": experiment.paper_runtime,
    "started_at": datetime.now(UTC).isoformat(),
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "packages": _package_versions(),
    "source_sha256": {
      str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
      for directory in ("heavily_right", "reproduction")
      for path in sorted((CODE_ROOT / directory).rglob("*.py"))
    },
    "bundled_input_sha256": {
      str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted((CODE_ROOT / "reproduction").rglob("*.csv"))
    },
  }
  metadata_path = run_dir / "run.json"
  if experiment.id == "network-meta-analysis/simulation":
    from reproduction.network_meta_analysis.simulation import simulation_configuration

    metadata["configuration"] = simulation_configuration(args.profile)
  _write_metadata(metadata_path, metadata)

  environment_keys = ("HCCT_EXPERIMENT_ID", "HCCT_RUN_ID", "HCCT_RESULTS_ROOT", "HCCT_PROFILE")
  previous_environment = {key: os.environ.get(key) for key in environment_keys}
  os.environ["HCCT_EXPERIMENT_ID"] = experiment.id
  os.environ["HCCT_RUN_ID"] = run_id
  os.environ["HCCT_RESULTS_ROOT"] = str(results_root)
  os.environ["HCCT_PROFILE"] = args.profile
  log_dir = run_dir / "logs"
  log_dir.mkdir()
  try:
    with (
      (log_dir / "stdout.log").open("w") as stdout_log,
      (log_dir / "stderr.log").open("w") as stderr_log,
      contextlib.redirect_stdout(_Tee(sys.stdout, stdout_log)),
      contextlib.redirect_stderr(_Tee(sys.stderr, stderr_log)),
      _time_limit(args.max_seconds),
    ):
      runpy.run_path(experiment.script_path, run_name="__main__")
  except ExperimentTimeout as error:
    metadata["status"] = "timed-out"
    metadata["incident"] = str(error)
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    _write_metadata(metadata_path, metadata)
    raise
  except BaseException as error:
    metadata["status"] = "failed"
    metadata["incident"] = f"{type(error).__name__}: {error}"
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    _write_metadata(metadata_path, metadata)
    raise
  finally:
    for key, value in previous_environment.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value
    metadata["produced_outputs"] = sorted(
      str(path.relative_to(run_dir))
      for path in run_dir.rglob("*")
      if path.is_file() and path != metadata_path and path.parent != log_dir
    )
    metadata["missing_expected_outputs"] = sorted(
      set(experiment.outputs) - set(metadata["produced_outputs"])
    )
    _write_metadata(metadata_path, metadata)
  metadata["status"] = "completed"
  metadata["finished_at"] = datetime.now(UTC).isoformat()
  _write_metadata(metadata_path, metadata)


if __name__ == "__main__":
  main()
