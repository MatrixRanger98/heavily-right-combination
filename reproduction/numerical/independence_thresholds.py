"""Main-table independence thresholds, excluding the requested CAtr columns.

The Wilson previous-work column uses the equal-weight, cardinality-based Landau
recipe displayed in the paper, including for its two unequal-weight examples.
The extra weight-aware HMP cutoff is recorded separately, not substituted for
that historical comparator. Exact columns use the canonical finite-m laws.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import halfcauchy

from heavily_right.null_laws import HalfCauchyMean, HarmonicMean, Landau, euler_gamma
from reproduction.artifacts import ArtifactStore

EXPERIMENT_ID = "numerical/independence-thresholds"
OUTPUT = "independence-thresholds.csv"
DIAGNOSTICS = "independence-thresholds.json"
LEVEL = 0.05


@dataclass(frozen=True)
class ThresholdScenario:
  """One row of main table tab:wilson; manuscript entries are rounded to .01."""

  name: str
  weights: tuple[float, ...]
  paper_values: tuple[float, float, float, float]


SCENARIOS = (
  ThresholdScenario("two-equal", (0.5, 0.5), (23.57, 21.73, 12.71, 13.69)),
  ThresholdScenario("two-weighted", (0.8, 0.2), (23.57, 21.19, 12.71, 13.39)),
  ThresholdScenario("five-equal", (0.2,) * 5, (24.48, 23.51, 12.71, 14.74)),
  ThresholdScenario("five-weighted", (0.6, 0.1, 0.1, 0.1, 0.1),
                    (24.48, 22.64, 12.71, 14.24)),
  ThresholdScenario("twenty-six-equal", (1 / 26,) * 26, (26.13, 25.85, 12.71, 16.19)),
)
METHODS = ("wilson_previous", "ehmp_exact", "gui_half_cauchy_proxy", "hcct_exact")
TableValue = str | int | float | bool


def evaluate() -> list[dict[str, TableValue]]:
  """Return all twenty non-CAtr cells, with round-trip and discrepancy evidence."""
  rows: list[dict[str, TableValue]] = []
  for scenario in SCENARIOS:
    weights = np.asarray(scenario.weights)
    m = len(weights)
    cardinality_location = np.log(m) + 1 - euler_gamma
    entropy_location = -float(weights @ np.log(weights)) + 1 - euler_gamma
    previous = float(Landau.ppf(1 - LEVEL, cardinality_location, np.pi / 2))
    weight_aware = float(Landau.ppf(1 - LEVEL, entropy_location, np.pi / 2))
    values = (
      previous,
      float(HarmonicMean.ppf(1 - LEVEL, weights)),
      float(halfcauchy.isf(LEVEL)),
      float(HalfCauchyMean.ppf(1 - LEVEL, weights)),
    )
    for method, threshold, paper in zip(METHODS, values, scenario.paper_values, strict=True):
      if method == "ehmp_exact":
        probability, error = HarmonicMean.sf(threshold, weights, precision=True)
      elif method == "hcct_exact":
        probability, error = HalfCauchyMean.sf(threshold, weights, precision=True)
      elif method == "wilson_previous":
        probability = Landau.sf(threshold, cardinality_location, np.pi / 2)
        error = 0.0
      else:
        probability, error = halfcauchy.sf(threshold), 0.0
      residual = float(probability - LEVEL)
      if not np.isfinite(threshold) or abs(residual) > max(1e-7, 10 * float(error)):
        raise ArithmeticError(f"Unreliable threshold round trip: {scenario.name}/{method}")
      difference = threshold - paper
      rows.append({
        "scenario": scenario.name,
        "num_studies": m,
        "weights": ";".join(format(value, ".17g") for value in weights),
        "level": LEVEL,
        "method": method,
        "threshold": threshold,
        "paper_threshold": paper,
        "difference_from_printed": difference,
        "within_printed_rounding": bool(abs(difference) <= 0.005),
        "reference_survival_at_threshold": float(probability),
        "survival_round_trip_residual": residual,
        "estimated_quadrature_error": float(error),
        "weight_aware_hmp_threshold": weight_aware,
      })
  return rows


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  rows = evaluate()
  path = artifacts.data(OUTPUT)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  diagnostics = {
    "schema_version": 1,
    "paper_label": "tab:wilson",
    "level": LEVEL,
    "seed": None,
    "stochastic": False,
    "omitted_at_user_request": ["Fang/CAtr previous-work and exact columns"],
    "wilson_previous_recipe": "Landau(loc=log(m)+1-gamma, scale=pi/2)",
    "weight_aware_hmp_recipe": "Landau(loc=-sum(w*log(w))+1-gamma, scale=pi/2)",
    "gui_proxy_recipe": "HalfCauchy(scale=1) 0.95 quantile",
    "exact_recipe": "Canonical HarmonicMean/ HalfCauchyMean ppf(.95, weights)",
    "round_trip_tolerance": "max(1e-7, 10 * estimated quadrature error)",
    "error_caveat": "Quadrature errors are estimates, not rigorous bounds; zero for approximation/proxy rows does not quantify approximation error.",
    "manuscript_values": "Read-only transcription of the current main table, rounded to .01.",
    "cells": len(rows),
    "outside_printed_rounding": [row for row in rows if not row["within_printed_rounding"]],
  }
  artifacts.diagnostic(DIAGNOSTICS).write_text(json.dumps(diagnostics, indent=2) + "\n")
  print(json.dumps(diagnostics, indent=2), flush=True)
  print(f"[artifact] {path}", flush=True)


if __name__ == "__main__":
  main()
