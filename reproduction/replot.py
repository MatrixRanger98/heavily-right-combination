"""Rebuild long-experiment figures from complete saved data, without simulating."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reproduction._replot_data import (
  CACHE_NAME,
  COPULA_LABELS,
  MODULES,
  SCHEMA_NAME,
  SCHEMA_VERSION,
  SEED,
  PlotSchema,
  load_arrays,
  schema_for,
  sha256,
  source_file,
)
from reproduction.artifacts import (
  CODE_ROOT,
  DEFAULT_RESULTS_ROOT,
  ArtifactStore,
  _validate_results_root,
  _validate_run_id,
)
from reproduction.manifest import get_experiment
from reproduction.run import _package_versions, _Tee, _write_metadata


def render_saved(schema: PlotSchema, arrays: dict[str, np.ndarray], store: ArtifactStore) -> None:
  """Use exactly the rendering functions called after each original simulation."""
  module = importlib.import_module(MODULES[schema.experiment])
  axes = schema.axes
  if schema.experiment == "one-dimensional/false-positive-rate":
    module.render(arrays["rates"], axes["rhos"], store)
  elif schema.experiment == "one-dimensional/power":
    module.render(arrays["power"], axes["rhos"], store)
  elif schema.experiment == "one-dimensional/interval-width":
    module.render(arrays["widths"], store)
  elif schema.experiment == "one-dimensional/normal-coverage":
    for label in ("ar1", "equicorrelated"):
      module.render(arrays[label], axes["rhos"], label, store)
  elif schema.experiment == "one-dimensional/copulas":
    for label in COPULA_LABELS:
      parameters = axes["rhos"] if label.startswith("student-t-") else axes["parameters"]
      module.render(arrays[label], parameters, label, store)
  elif schema.experiment == "multidimensional/coverage":
    for count in axes["study_counts"]:
      module.render(arrays[f"studies_{count}"], tuple(axes["dimensions"]), axes["rhos"], count, store)
  else:
    module.render({"rhos": axes["rhos"], **arrays}, store, profile=schema.profile)


def _figures(schema: PlotSchema) -> list[str]:
  if schema.experiment == "multidimensional/coverage":
    return [f"figures/multivariate-coverage-{n}-studies.pdf" for n in schema.axes["study_counts"]]
  return [path for path in get_experiment(schema.experiment).outputs if path.startswith("figures/")]


def replot(
  source_run: Path, *, run_id: str, results_root: Path = DEFAULT_RESULTS_ROOT,
  allow_failed_source: bool = False,
) -> Path:
  """Validate all data first; retain a new cache and diagnostics even if plotting fails."""
  source = source_run.resolve()
  metadata_path = source_file(source, "run.json")
  metadata_hash = sha256(metadata_path)
  source_metadata = json.loads(metadata_path.read_text())
  if source_metadata.get("schema_version") != 1:
    raise ValueError("unsupported source run metadata schema")
  schema = schema_for(source_metadata.get("experiment"), source_metadata.get("profile"))
  experiment = get_experiment(schema.experiment)
  if source_metadata.get("declared_seed") != SEED or experiment.seed != SEED:
    raise ValueError("source declared seed does not match the experiment")
  status = source_metadata.get("status")
  if status != "completed" and not (allow_failed_source and status in {"failed", "timed-out"}):
    raise ValueError("source must be completed; failed/timed-out recovery requires --allow-failed-source")
  if not source_metadata.get("finished_at"):
    raise ValueError("source is not a finished immutable run")
  if source_metadata.get("mode") not in {None, "replot"}:
    raise ValueError("unsupported source run mode")
  if source_metadata.get("mode") != "replot":
    if (source_metadata.get("script") != experiment.script
        or source_metadata.get("source_sha256", {}).get(experiment.script)
        != source_metadata.get("script_sha256")
        or not isinstance(source_metadata.get("script_sha256"), str)):
      raise ValueError("source run has missing or inconsistent generator provenance")
  arrays, hashes, qualifications = load_arrays(source, source_metadata, schema)
  if schema.experiment == "network-meta-analysis/simulation":
    schema = schema_for(schema.experiment, schema.profile,
                        nma_replicates=qualifications["replicates_per_rho"])
  hashes["run.json"] = metadata_hash
  # Check the file snapshots again before writing. This is integrity checking,
  # not proof that old NPY files were never edited before this invocation.
  if any(sha256(source_file(source, path)) != digest for path, digest in hashes.items()):
    raise ValueError("source data changed while it was being read")
  root = _validate_results_root(results_root)
  store = ArtifactStore(schema.experiment, _validate_run_id(run_id), root)
  target = _validate_results_root(store.run_dir)
  if target == source or target.is_relative_to(source):
    raise ValueError("a replot run must not be written inside its source run")
  if target != store.run_dir:
    raise ValueError("replot destination contains a symlinked experiment/run directory")
  if any((parent / "run.json").is_file() for parent in target.parents):
    raise ValueError("a new replot run must not be nested inside another immutable run")
  target.mkdir(parents=True, exist_ok=False)
  expected = [f"data/{CACHE_NAME}", f"data/{SCHEMA_NAME}", *_figures(schema)]
  if schema.experiment == "network-meta-analysis/simulation":
    expected += ["data/nma-simulation-plotted-summary.npz",
                 "diagnostics/nma-simulation-plot-policy.json"]
  metadata: dict[str, Any] = {
    "schema_version": 1, "mode": "replot", "experiment": schema.experiment,
    "run_id": store.run_id, "script": "reproduction/replot.py",
    "script_sha256": sha256(Path(__file__)), "declared_seed": SEED, "profile": schema.profile,
    "fresh_simulation": False, "started_at": datetime.now(UTC).isoformat(), "status": "running",
    "source_run": str(source), "source_run_status": status,
    "source_recovery_requested": allow_failed_source, "source_files_sha256": hashes,
    "source_generator_sha256": source_metadata.get("source_generator_sha256",
                                                    source_metadata.get("source_sha256", {})),
    "source_recorded_configuration": source_metadata.get("source_recorded_configuration",
                                                         source_metadata.get("configuration")),
    "source_axes": "validated normalized cache" if source_metadata.get("mode") == "replot"
                   else "reconstructed from documented paper/smoke schema for legacy NPY; NMA rho checked",
    "expected_outputs": expected, "packages": _package_versions(),
    "python_version": platform.python_version(), "platform": platform.platform(),
    "source_sha256": {
      str(path.relative_to(CODE_ROOT)): sha256(path)
      for path in sorted((CODE_ROOT / "reproduction").rglob("*.py"))
    },
    "qualification": "Plot-only reuse of saved simulation data; not an independent Monte Carlo replication.",
  }
  result_metadata = target / "run.json"
  _write_metadata(result_metadata, metadata)
  logs = target / "logs"
  logs.mkdir()
  try:
    with ((logs / "stdout.log").open("x") as stdout,
          (logs / "stderr.log").open("x") as stderr,
          contextlib.redirect_stdout(_Tee(sys.stdout, stdout)),
          contextlib.redirect_stderr(_Tee(sys.stderr, stderr))):
      cache = store.data(CACHE_NAME)
      np.savez_compressed(cache, schema_version=SCHEMA_VERSION, experiment_id=schema.experiment,
                          profile=schema.profile, seed=SEED, **arrays,
                          **{f"axis__{name}": values for name, values in schema.axes.items()})
      document = {**schema.document(), "qualifications": qualifications,
                  "source_recorded_configuration": metadata["source_recorded_configuration"],
                  "source_generator_sha256": metadata["source_generator_sha256"]}
      document_path = store.data(SCHEMA_NAME)
      with document_path.open("x") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")
      metadata["cache_sha256"] = {
        str(path.relative_to(target)): sha256(path) for path in (cache, document_path)
      }
      _write_metadata(result_metadata, metadata)
      render_saved(schema, arrays, store)
      if any(not (target / name).is_file() for name in expected):
        raise RuntimeError("plotting did not produce every expected output")
      if any(sha256(source_file(source, path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("source changed during rendering; new run cannot be certified")
    metadata["status"] = "completed"
  except BaseException as error:
    metadata["status"] = "failed"
    metadata["incident"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    metadata["produced_outputs"] = sorted(
      str(path.relative_to(target)) for path in target.rglob("*")
      if path.is_file() and path != result_metadata and path.parent != logs
    )
    metadata["missing_expected_outputs"] = sorted(set(expected) - set(metadata["produced_outputs"]))
    _write_metadata(result_metadata, metadata)
  return target


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source_run", type=Path, help="Finished immutable run directory containing run.json")
  parser.add_argument("--run-id", required=True, help="New immutable run ID; existing destinations are rejected")
  parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
  parser.add_argument("--allow-failed-source", action="store_true",
                      help="Recover a failed/timed-out source only if all required plotting data are complete")
  args = parser.parse_args(argv)
  import matplotlib

  matplotlib.use("Agg")
  path = replot(args.source_run, run_id=args.run_id, results_root=args.results_root,
                allow_failed_source=args.allow_failed_source)
  print(f"[replot] completed {path}; no simulation was run", flush=True)


if __name__ == "__main__":
  main()
