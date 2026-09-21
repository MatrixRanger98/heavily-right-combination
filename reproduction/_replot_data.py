"""Versioned, pickle-free plotting-data schemas for the long simulations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
SEED = 2025
CACHE_NAME = "replot-cache.npz"
SCHEMA_NAME = "replot-cache-schema.json"
MODULES = {
  "one-dimensional/false-positive-rate": "reproduction.one_dimensional.false_positive_rate",
  "one-dimensional/power": "reproduction.one_dimensional.power_comparison",
  "one-dimensional/normal-coverage": "reproduction.one_dimensional.normal_coverage",
  "one-dimensional/interval-width": "reproduction.one_dimensional.interval_width",
  "one-dimensional/copulas": "reproduction.one_dimensional.copula_coverage",
  "multidimensional/coverage": "reproduction.multidimensional.coverage",
  "network-meta-analysis/simulation": "reproduction.network_meta_analysis.simulation",
}
COPULA_LABELS = (
  "student-t-ar1", "student-t-equicorrelated", "ali-mikhail-haq-hcauchy",
  "ali-mikhail-haq-fisher", "farlie-gumbel-morgenstern-hcauchy",
  "farlie-gumbel-morgenstern-fisher",
)
METHODS = ("HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes")


@dataclass(frozen=True)
class Field:
  source: str
  member: str | None
  axes: tuple[str, ...]
  description: str
  kind: str = "probability"


@dataclass(frozen=True)
class PlotSchema:
  experiment: str
  profile: str
  axes: dict[str, np.ndarray]
  fields: dict[str, Field]
  nma_replicates: int | None = None

  def document(self) -> dict[str, Any]:
    return {
      "schema_version": SCHEMA_VERSION, "experiment_id": self.experiment,
      "profile": self.profile, "seed": SEED,
      "axes": {name: values.tolist() for name, values in self.axes.items()},
      "fields": {
        name: {"source": field.source, "member": field.member,
               "axes": list(field.axes), "shape": [len(self.axes[a]) for a in field.axes],
               "description": field.description, "kind": field.kind}
        for name, field in self.fields.items()
      },
      "configuration_interpretation": configuration_for(
        self.experiment, self.profile, nma_replicates=self.nma_replicates),
      "configuration_qualification": (
        "Current documented profile interpretation, not proof of original m/n/df settings. "
        "Legacy NPY does not store configuration; retain original run/code hashes and any "
        "source-recorded settings. Validation is structural, not numerical recalibration."
      ),
      "scope": "Saved simulation summaries/samples for plotting; not fresh Monte Carlo draws.",
    }


def _nma_replicate_count(profile: str, value: Any) -> int:
  """Accept only complete supported workloads, never an arbitrary partial count."""
  count = np.asarray(value)
  allowed = {100, 500} if profile == "paper" else {2}
  if (count.shape != () or count.dtype.kind not in "iuf" or not np.isfinite(count)
      or count.item() not in allowed):
    raise ValueError("unsupported or incomplete NMA replicate count for the source profile")
  return int(count.item())


def configuration_for(
  experiment: str, profile: str, *, nma_replicates: int | None = None,
) -> dict[str, Any]:
  paper = profile == "paper"
  if experiment in {"one-dimensional/false-positive-rate", "one-dimensional/power"}:
    return {"num_study": 500 if paper else 30, "draws_per_case": 10_000 if paper else 200,
            "level": .05}
  if experiment == "one-dimensional/interval-width":
    return {"num_study": 500 if paper else 30, "draws_per_case": 10_000 if paper else 40,
            "level": .05}
  if experiment in {"one-dimensional/normal-coverage", "one-dimensional/copulas"}:
    return {"num_study": 500 if paper else (20 if experiment.endswith("copulas") else 30),
            "draws_per_repeat": 10_000 if paper else 200, "repeats": 50 if paper else 2,
            "levels": [.05, .01],
            **({"student_t_df": 10} if experiment.endswith("copulas") else {})}
  if experiment == "multidimensional/coverage":
    return {"study_counts": [500, 10] if paper else [30, 10],
            "dimensions": [2, 5, 10, 25] if paper else [2, 5],
            "draws_per_repeat": 10_000 if paper else 100, "repeats": 10 if paper else 1,
            "levels": [.05, .01]}
  if nma_replicates is None:
    from reproduction.network_meta_analysis.simulation import simulation_configuration

    nma_replicates = simulation_configuration(profile)["replicates_per_rho"]
  count = _nma_replicate_count(profile, nma_replicates)
  return {"sample_size": 100 if paper else 30, "degrees_freedom": 99 if paper else 29,
          "replicates_per_rho": count, "level": .05,
          "parameter_dimension": 9}


def schema_for(
  experiment: str, profile: str, *, nma_replicates: int | None = None,
) -> PlotSchema:
  """Return a profile schema; an explicit NMA count preserves supported old workloads."""
  if experiment not in MODULES or profile not in {"paper", "smoke"}:
    raise ValueError("unsupported replot experiment or profile")
  paper = profile == "paper"
  axes: dict[str, np.ndarray] = {}
  fields: dict[str, Field] = {}

  def field(name: str, stem: str, dims: tuple[str, ...], description: str,
            kind: str = "probability", member: str | None = None) -> None:
    fields[name] = Field(f"data/{stem}", member, dims, description, kind)

  if experiment in {"one-dimensional/false-positive-rate", "one-dimensional/power"}:
    axes = {"methods": np.asarray(METHODS),
            "rhos": np.arange(0, 1, 0.1) if paper else np.array([0., .3, .6, .9]),
            "series": np.asarray(("ar1", "equicorrelated") if experiment.endswith("rate")
                                 else ("dense", "sparse"))}
    name = "rates" if experiment.endswith("rate") else "power"
    stem = "false-positive-rate.npy" if name == "rates" else "power-comparison.npy"
    field(name, stem, ("methods", "rhos", "series"), "Empirical rejection probability")
  elif experiment == "one-dimensional/interval-width":
    axes = {"dependence": np.asarray(("ar1", "equicorrelated")),
            "rhos": np.array([0., .3, .6, .9]), "replicates": np.arange(10_000 if paper else 40)}
    field("widths", "interval-width.npy", ("dependence", "rhos", "replicates"),
          "HCCT upper minus lower interval endpoint for each replicate", "nonnegative")
  elif experiment in {"one-dimensional/normal-coverage", "one-dimensional/copulas"}:
    axes = {"replicates": np.arange(50 if paper else 2), "levels": np.array([.05, .01]),
            "rhos": np.arange(0, 1, .1) if paper else np.array([0., .6])}
    labels = ("ar1", "equicorrelated")
    if experiment.endswith("copulas"):
      labels = COPULA_LABELS
      axes["parameters"] = np.arange(-.9, 1, .2) if paper else np.array([-.6, 0., .6])
    for label in labels:
      stem = "copula-coverage" if experiment.endswith("copulas") else "normal-coverage"
      axis = "parameters" if label.startswith(("ali-", "farlie-")) else "rhos"
      field(label, f"{stem}-{label}.npy", ("replicates", axis, "levels"),
            "Empirical nonrejection/coverage per repeat; copula plots show its complement")
  elif experiment == "multidimensional/coverage":
    axes = {"dimensions": np.array([2, 5, 10, 25] if paper else [2, 5]),
            "rhos": np.arange(0, 1, .1) if paper else np.array([0., .6]),
            "levels": np.array([.05, .01]), "study_counts": np.array([500, 10] if paper else [30, 10])}
    for count in axes["study_counts"]:
      field(f"studies_{count}", f"multivariate-coverage-{count}-studies.npy",
            ("dimensions", "rhos", "levels"), "Coverage averaged over Monte Carlo repeats")
  else:
    axes = {"rhos": np.arange(0, 1, .1) if paper else np.array([0., .6])}
    for name in ("coverage_hcct", "coverage_wls", "coverage_wls_ma",
                 "width_hcct_1", "width_hcct_2", "width_wls_1", "width_wls_2",
                 "width_wls_ma_1", "width_wls_ma_2", "wls_ma_multiplier",
                 "empty_hcct_count", "replicates_per_rho"):
      kind = "probability" if name.startswith("coverage") else "nonnegative"
      if name in {"empty_hcct_count", "replicates_per_rho"}:
        kind = "count"
      field(name, "nma-simulation-summary.npz", ("rhos",),
            "NMA saved summary; read retained coverage-event and in-sample-oracle qualifications",
            kind, member=name)
  if nma_replicates is not None:
    if experiment != "network-meta-analysis/simulation":
      raise ValueError("NMA replicate counts apply only to the NMA simulation")
    nma_replicates = _nma_replicate_count(profile, nma_replicates)
  return PlotSchema(experiment, profile, axes, fields, nma_replicates)


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file(source: Path, relative: str) -> Path:
  candidate = source / relative
  if not candidate.is_file() or not candidate.resolve().is_relative_to(source.resolve()):
    raise ValueError(f"missing or externally linked source file: {relative}")
  return candidate


def validate_arrays(schema: PlotSchema, arrays: dict[str, np.ndarray]) -> None:
  if set(arrays) != set(schema.fields):
    raise ValueError("saved arrays do not match the complete experiment schema")
  for name, field in schema.fields.items():
    value = arrays[name]
    shape = tuple(len(schema.axes[axis]) for axis in field.axes)
    if value.dtype.kind not in "biuf" or value.shape != shape or not np.isfinite(value).all():
      raise ValueError(f"{name}: expected finite numeric array of shape {shape}")
    if np.any(value < 0) or (field.kind == "probability" and np.any(value > 1)):
      raise ValueError(f"{name}: values outside their statistical range")
    if field.kind == "count" and np.any(value != np.floor(value)):
      raise ValueError(f"{name}: counts must be integers")
  if schema.experiment == "network-meta-analysis/simulation":
    expected = _nma_replicate_count(schema.profile, arrays["replicates_per_rho"][0])
    if (np.any(arrays["replicates_per_rho"] != expected)
        or (schema.nma_replicates is not None and schema.nma_replicates != expected)):
      raise ValueError("NMA summary has incomplete or wrong-profile replicate counts")
    if np.any(arrays["empty_hcct_count"] > expected) or np.any(arrays["wls_ma_multiplier"] <= 0):
      raise ValueError("invalid NMA empty counts or oracle multipliers")


def _validate_nma_recorded_configuration(
  configuration: Any, schema: PlotSchema, count: int,
) -> None:
  """Check available top-level/nested settings without inventing legacy metadata."""
  if configuration is None:
    return
  if (not isinstance(configuration, dict)
      or _nma_replicate_count(schema.profile, configuration.get("replicates_per_rho")) != count):
    raise ValueError("NMA source-recorded configuration disagrees with summary replicate count")
  expected = configuration_for(schema.experiment, schema.profile, nma_replicates=count)
  expected.update(seed=SEED, profile=schema.profile, study_count=28,
                  rhos=schema.axes["rhos"].tolist())
  for name, value in expected.items():
    if name in configuration and configuration[name] != value:
      raise ValueError(f"NMA source-recorded configuration disagrees with source profile: {name}")


def load_arrays(
  source: Path, metadata: dict[str, Any], schema: PlotSchema,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
  """Read complete raw results or a validated normalized cache, without writing."""
  arrays: dict[str, np.ndarray] = {}
  hashes: dict[str, str] = {}
  qualifications: dict[str, Any] = {}
  document: dict[str, Any] | None = None
  if metadata.get("mode") == "replot":
    cache = source_file(source, f"data/{CACHE_NAME}")
    document_path = source_file(source, f"data/{SCHEMA_NAME}")
    for path in (cache, document_path):
      relative = path.relative_to(source).as_posix()
      hashes[relative] = sha256(path)
      if metadata.get("cache_sha256", {}).get(relative) != hashes[relative]:
        raise ValueError(f"replot cache checksum mismatch: {relative}")
    document = json.loads(document_path.read_text())
    with np.load(cache, allow_pickle=False) as saved:
      expected_keys = {*schema.fields, *(f"axis__{key}" for key in schema.axes),
                       "schema_version", "experiment_id", "profile", "seed"}
      if set(saved.files) != expected_keys:
        raise ValueError("replot cache has missing or unknown fields")
      for key, expected in (("schema_version", SCHEMA_VERSION), ("experiment_id", schema.experiment),
                            ("profile", schema.profile), ("seed", SEED)):
        if saved[key].shape != () or saved[key].item() != expected:
          raise ValueError(f"replot cache identity mismatch: {key}")
      for key, axis in schema.axes.items():
        if not np.array_equal(saved[f"axis__{key}"], axis):
          raise ValueError(f"replot cache axis mismatch: {key}")
      arrays = {name: saved[name] for name in schema.fields}
    qualifications = document.get("qualifications", {})
  else:
    loaded: dict[str, dict[str, np.ndarray] | np.ndarray] = {}
    for field in schema.fields.values():
      if field.source in loaded:
        continue
      path = source_file(source, field.source)
      hashes[field.source] = sha256(path)
      if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as saved:
          loaded[field.source] = {name: saved[name] for name in saved.files}
      else:
        loaded[field.source] = np.load(path, allow_pickle=False)
    for name, field in schema.fields.items():
      value = loaded[field.source]
      if field.member is not None:
        if not isinstance(value, dict) or field.member not in value:
          raise ValueError(f"saved NMA summary lacks {field.member}; cannot invent oracle data")
        value = value[field.member]
      arrays[name] = np.asarray(value)
    if schema.experiment == "network-meta-analysis/simulation":
      summary = loaded["data/nma-simulation-summary.npz"]
      if not isinstance(summary, dict) or not np.array_equal(summary.get("rhos"), schema.axes["rhos"]):
        raise ValueError("NMA summary correlation axis does not match the profile")
      relative = "diagnostics/nma-simulation-calibration.json"
      path = source_file(source, relative)
      hashes[relative] = sha256(path)
      qualifications = json.loads(path.read_text())
      # Raw endpoint diagnostics can contain NaNs for certified empty regions.
      # They are audit evidence, not plotting inputs: hash, do not reject them.
      raw = source_file(source, "data/nma-simulation-replicates.npz")
      hashes[raw.relative_to(source).as_posix()] = sha256(raw)
  validate_arrays(schema, arrays)
  if schema.experiment == "network-meta-analysis/simulation":
    from reproduction.network_meta_analysis.simulation import COVERAGE_EVENTS

    # Resolve the workload from retained evidence before comparing a cache sidecar.
    # A legacy version-1 paper cache means 500, not today's default of 100.
    count = _nma_replicate_count(schema.profile, arrays["replicates_per_rho"][0])
    schema = schema_for(schema.experiment, schema.profile, nma_replicates=count)
    if (qualifications.get("schema_version") != 1 or qualifications.get("seed") != SEED
        or qualifications.get("coverage_events") != COVERAGE_EVENTS
        or _nma_replicate_count(schema.profile, qualifications.get("replicates_per_rho")) != count
        or not np.array_equal(qualifications.get("rhos"), schema.axes["rhos"])):
      raise ValueError("NMA cache lacks the original coverage-event qualifications")
    for configuration in (
      qualifications,
      metadata.get("configuration"), metadata.get("source_recorded_configuration"),
      document.get("source_recorded_configuration") if document is not None else None,
      qualifications.get("configuration"),
    ):
      _validate_nma_recorded_configuration(configuration, schema, count)
    for key, name in (("oracle_multipliers", "wls_ma_multiplier"),
                      ("oracle_empirical_coverage", "coverage_wls_ma"),
                      ("empty_hcct_counts", "empty_hcct_count")):
      if not np.array_equal(qualifications.get(key), arrays[name]):
        raise ValueError(f"NMA saved calibration disagrees with summary: {key}")
  if document is not None:
    for name, expected in schema.document().items():
      if document.get(name) != expected:
        raise ValueError(f"replot cache schema mismatch: {name}")
  return arrays, hashes, qualifications
