"""Explicit, audited point-estimate counterparts for main-paper Tables 1 and 2.

The simulation table is one seeded sample at each of four correlations, not an
average over the coverage experiment. The author-approved seed-2025 realization
replaces the original table, whose full realization was not recovered. Both
profiles retain the complete table settings, independently of the figure
experiment's global random stream. No manuscript file is read or modified at
runtime; printed comparison values are a dated post-update reference snapshot.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from decimal import Decimal
from typing import Any

import numpy as np

from heavily_right.calibration import standard_error_of_mean
from heavily_right.correlation import cor_fixed
from heavily_right.meta_analysis import MetaAnalysisMD
from heavily_right.minimum import minimize_score_region
from heavily_right.projected_region import ProjectedScoreRegion
from reproduction.artifacts import ArtifactStore
from reproduction.network_meta_analysis.real_data import fit_with_empty_set_adjustment
from reproduction.network_meta_analysis.senn2013 import (
  ADJUSTED_STANDARD_ERRORS,
  DESIGN_MATRIX,
  ESTIMATES,
  STANDARD_ERRORS,
  TREATMENTS,
  wls_point_estimate,
)
from reproduction.network_meta_analysis.simulation import THETA

EXPERIMENT_ID = "network-meta-analysis/treatment-estimates"
SEED = 2025
SAMPLE_SIZE = 100
RHOS = (0.0, 0.3, 0.6, 0.9)
SIMULATION_OUTPUT = "nma-simulation-treatment-estimates.csv"
REAL_OUTPUT = "nma-real-data-treatment-estimates.csv"
INPUT_OUTPUT = "nma-simulation-treatment-inputs.npz"
DIAGNOSTIC_OUTPUT = "nma-treatment-estimates.json"

# Printed precision is retained to distinguish rounding from discrepancies.
# The immutable selected-run CSVs retain the pre-update printed comparisons.
REFERENCE_SNAPSHOT = (
  "Paper/content/main.tex, 2026-09-21 after author-approved Table 1 seed-2025 "
  "publication and Table 2 HCCT update; tab:thetaressim and tab:thetares updated"
)
PAPER_SIMULATION = {
  (0.0, "WLS"): (".00975", "-.461", "-.984", ".0109", "-.590", "-1.07", ".0170", "-.529", "-.999"),
  (0.0, "HCCT"): ("-.0479", "-.454", "-.973", "-.00505", "-.570", "-1.06", ".0170", "-.575", "-.999"),
  (0.3, "WLS"): (".0296", "-.366", "-.978", "-.0156", "-.517", "-.995", "-.00318", "-.380", "-.979"),
  (0.3, "HCCT"): (".0350", "-.366", "-.985", ".00739", "-.513", "-1.01", "-.00318", "-.394", "-.979"),
  (0.6, "WLS"): ("-.0797", "-.532", "-1.05", "-.0880", "-.595", "-1.11", ".0292", "-.538", "-1.17"),
  (0.6, "HCCT"): ("-.0821", "-.526", "-1.05", "-.0895", "-.594", "-1.12", ".0292", "-.531", "-1.17"),
  (0.9, "WLS"): ("-.0506", "-.567", "-1.06", "-.0401", "-.599", "-1.07", "-.0294", "-.428", "-1.12"),
  (0.9, "HCCT"): ("-.0539", "-.566", "-1.07", "-.0271", "-.604", "-1.07", "-.0294", "-.422", "-1.12"),
}
PAPER_REAL = {
  "WLS": ("-0.827", "-0.905", "-1.11", "-0.944", "-1.07", "-1.20", "-0.57", "-0.439", "-0.7"),
  "HCCT": ("-0.804", "-0.829", "-1.01", "-1.02", "-1.02", "-1.31", "-0.57", "-0.404", "-0.7"),
}


def simulation_inputs(seed: int = SEED) -> dict[str, np.ndarray]:
  """Draw exactly one n=100 sample per rho, using a private seeded generator."""
  if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
    raise ValueError("seed must be a nonnegative integer")
  rng = np.random.default_rng(seed)
  samples = np.array([
    rng.multivariate_normal(DESIGN_MATRIX @ THETA, cor_fixed(len(ESTIMATES), rho),
                            size=SAMPLE_SIZE)
    for rho in RHOS
  ])
  means = samples.mean(axis=1)
  variances = np.array([standard_error_of_mean(sample)**2 for sample in samples])
  return {
    "rhos": np.array(RHOS), "samples": samples, "means": means,
    "estimated_mean_variances": variances, "design": DESIGN_MATRIX,
    "true_theta": THETA,
    "true_mean_covariances": np.array([
      cor_fixed(len(ESTIMATES), rho) / SAMPLE_SIZE for rho in RHOS
    ]),
  }


def _comparison_rows(
  point: np.ndarray, *, method: str, printed: tuple[str, ...], rho: float | None,
  study_count: int, removed: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
  rows = []
  for index, (treatment, estimate, reference) in enumerate(
    zip(TREATMENTS, point, printed, strict=True)
  ):
    half_unit = 0.5 * 10.0**Decimal(reference).as_tuple().exponent
    difference = float(estimate) - float(reference)
    rows.append({
      "rho": "" if rho is None else rho,
      "method": method, "treatment": str(treatment), "estimate": float(estimate),
      "true_effect": "" if rho is None else float(THETA[index]),
      "paper_estimate": reference, "difference_from_paper": difference,
      "paper_rounding_half_unit": half_unit,
      "within_paper_rounding": abs(difference) <= half_unit + 1.0e-12,
      "study_count": study_count,
      "removed_original_indices": ";".join(str(value) for value in removed),
    })
  return rows


def evaluate_simulation(
  seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
  """Compute both table estimators with the corrected estimated-SE t model."""
  inputs = simulation_inputs(seed)
  rows: list[dict[str, Any]] = []
  cases = []
  count, dimension = DESIGN_MATRIX.shape
  analysis = MetaAnalysisMD(count, dim=dimension, method="HCauchy", level=0.05)
  for rho, mean, variance in zip(
    RHOS, inputs["means"], inputs["estimated_mean_variances"], strict=True,
  ):
    region = ProjectedScoreRegion(
      dimension=dimension, weights=analysis.weights, method="HCauchy",
      xi_hat=mean[:, None], sigma=variance[:, None, None],
      sub_dim=np.arange(count)[:, None], degrees_freedom=np.full(count, SAMPLE_SIZE - 1),
      projections=DESIGN_MATRIX, equal_sub_dim=True,
    )
    minimum = minimize_score_region(region, cutoff=float(analysis.threshold))
    if not minimum.success:
      raise RuntimeError(f"unvalidated HCCT table minimum at rho={rho}: {minimum.diagnostics}")
    hcct = region.point_from_standardized(minimum.point)
    wls = wls_point_estimate(mean, np.sqrt(variance))
    for method, point in (("WLS", wls), ("HCCT", hcct)):
      rows.extend(_comparison_rows(
        point, method=method, printed=PAPER_SIMULATION[rho, method], rho=rho,
        study_count=count,
      ))
    cases.append({
      "rho": rho, "wls_point": wls, "hcct_point": hcct,
      "hcct_score": minimum.score, "hcct_threshold": float(analysis.threshold),
      "minimum_diagnostics": asdict(minimum.diagnostics),
      "empty_set_intervention": "disabled; the point estimator uses all 28 comparisons",
    })
  diagnostics = {
    "seed": int(seed), "rng": "numpy.random.default_rng / PCG64",
    "sample_size": SAMPLE_SIZE, "degrees_freedom": SAMPLE_SIZE - 1,
    "rhos": RHOS, "replicates_per_rho": 1,
    "interpretation": "single sample per scenario, not a Monte Carlo mean",
    "calibration": "estimated variance of mean W/[n(n-1)]; marginal Student t p-values",
    "reconstruction_assumption": (
      "n=100 raw normal observations per study, matching the active figure generator; "
      "true mean covariance is 0.01*cor_fixed(28,rho). The manuscript only specifies "
      "this mean covariance, not the extra variance-estimation model."
    ),
    "original_realization_recovered": False,
    "reference_snapshot": REFERENCE_SNAPSHOT,
    "comparison_qualification": (
      "Comparisons now use the author-approved seed-2025 Table 1, rounded to three "
      "significant digits. The immutable selected-run CSV predates that update and "
      "retains comparisons against the historical table. The full original seed/sample "
      "was not recovered; matching the updated table is not recovery of the historical "
      "realization. This table uses inverse-estimated-variance WLS, as does the corrected "
      "figure generator; older figure results used an unweighted point estimate."
    ),
    "cases": cases,
  }
  return rows, diagnostics, inputs


def evaluate_real_data(
  *, artifacts: ArtifactStore | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Use the unchanged figure model: adjusted WLS and sequentially adjusted HCCT."""
  result = fit_with_empty_set_adjustment(artifacts=artifacts)
  if not result.success:
    raise RuntimeError("real-data table fit has uncertified initial support endpoints")
  trace = result.empty_set_diagnostics
  if trace is None or not trace.resolved or trace.last_global_minimizer is None:
    raise RuntimeError("real-data table fit lacks a resolved minimum/intervention trace")
  wls = wls_point_estimate(ESTIMATES, ADJUSTED_STANDARD_ERRORS)
  hcct = np.asarray(trace.last_global_minimizer, dtype=float)
  removed = tuple(removal.original_index for removal in trace.removals)
  rows = _comparison_rows(
    wls, method="WLS", printed=PAPER_REAL["WLS"], rho=None, study_count=len(ESTIMATES),
  )
  rows.extend(_comparison_rows(
    hcct, method="HCCT", printed=PAPER_REAL["HCCT"], rho=None,
    study_count=len(trace.final_original_indices), removed=removed,
  ))
  diagnostics = {
    "degrees_freedom": None, "calibration": "reported standard errors; Gaussian p-values",
    "reference_snapshot": REFERENCE_SNAPSHOT,
    "comparison_qualification": (
      "Comparisons use the updated Table 2 printed values. The immutable selected-run "
      "CSV and diagnostic discrepancy summary predate that update and retain the "
      "pre-update Table 2 reference; their estimate values are unchanged."
    ),
    "wls_model": "all 28 comparisons; existing multi-arm-adjusted standard errors",
    "wls_standard_errors": ADJUSTED_STANDARD_ERRORS,
    "hcct_model": "reported unadjusted comparison standard errors; equal retained weights",
    "hcct_original_standard_errors": STANDARD_ERRORS,
    "wls_point": wls, "hcct_point": hcct,
    "minimum_diagnostics": asdict(result.minimum_diagnostics),
    "empty_set_adjustment": trace.to_dict(),
    "qualification": (
      "HCCT retains the figure workflow's data-dependent study removal; its original "
      "nominal coverage does not automatically transfer after selection. WLS uses all "
      "28 comparisons, so the methods do not use the same final study set. Printed "
      "rounding is checked cellwise; any remaining discrepancies are retained."
    ),
  }
  return rows, diagnostics


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def _write_csv(artifacts: ArtifactStore, name: str, rows: list[dict[str, Any]]) -> None:
  with artifacts.data(name).open("x", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  simulation_rows, simulation_diagnostics, inputs = evaluate_simulation()
  np.savez(artifacts.data(INPUT_OUTPUT), **inputs)
  _write_csv(artifacts, SIMULATION_OUTPUT, simulation_rows)
  real_rows, real_diagnostics = evaluate_real_data(artifacts=artifacts)
  _write_csv(artifacts, REAL_OUTPUT, real_rows)
  report = {
    "schema_version": 1, "experiment": EXPERIMENT_ID,
    "reference_snapshot": REFERENCE_SNAPSHOT,
    "simulation": simulation_diagnostics, "real_data": real_diagnostics,
    "discrepancy_summary": {
      name: {
        "cells": len(rows),
        "cells_outside_printed_rounding": sum(not row["within_paper_rounding"] for row in rows),
        "max_absolute_difference": max(abs(row["difference_from_paper"]) for row in rows),
        "max_excess_over_printed_rounding": max(
          max(0., abs(row["difference_from_paper"]) - row["paper_rounding_half_unit"])
          for row in rows
        ),
        "by_method": {
          method: {
            "cells_outside_printed_rounding": sum(
              not row["within_paper_rounding"] for row in rows if row["method"] == method
            ),
            "max_absolute_difference": max(
              abs(row["difference_from_paper"]) for row in rows if row["method"] == method
            ),
          }
          for method in ("WLS", "HCCT")
        },
      }
      for name, rows in (("simulation", simulation_rows), ("real_data", real_rows))
    },
  }
  with artifacts.diagnostic(DIAGNOSTIC_OUTPUT).open("x", encoding="utf-8") as output:
    json.dump(_json_safe(report), output, indent=2, allow_nan=False)
    output.write("\n")
  print(json.dumps(report["discrepancy_summary"], indent=2), flush=True)


if __name__ == "__main__":
  main()
