"""Bounded rewrite of the oldest truncated-Cauchy threshold exploration.

The recovered script requested several hundred million variates at once and did
not record its results. This version is deterministic and bounded; it is useful
for auditing the calculation but is not an exact Monte Carlo reproduction of
the unpublished exploratory run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.stats import cauchy

from reproduction.artifacts import ArtifactStore
from reproduction.profiles import profile_value
from reproduction.progress import Progress

EXPERIMENT_ID = "numerical/legacy-thresholds"
OUTPUT = "legacy-thresholds.csv"
SEED = 2025


@dataclass(frozen=True)
class Scenario:
  name: str
  weights: np.ndarray
  threshold: float


SCENARIOS = (
  Scenario("five-equal", np.full(5, 0.2), 6.362),
  Scenario("five-weighted", np.array([0.6, 0.1, 0.1, 0.1, 0.1]), 6.365),
  Scenario("twenty-six-equal", np.full(26, 1 / 26), 6.86),
)


def simulate_scenario(
  scenario: Scenario,
  *,
  draws: int,
  rng: np.random.Generator,
) -> dict[str, str | int | float]:
  lower = cauchy.ppf(0.01)
  sample = np.maximum(
    cauchy.rvs(size=(draws, len(scenario.weights)), random_state=rng),
    lower,
  )
  statistics = sample @ scenario.weights
  return {
    "scenario": scenario.name,
    "draws": draws,
    "seed": SEED,
    "quantile_0.95": float(np.quantile(statistics, 0.95)),
    "reference_threshold": scenario.threshold,
    "exceedance_rate": float(np.mean(statistics > scenario.threshold)),
  }


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  draws = profile_value(paper=1_000_000, smoke=20_000)
  rng = np.random.default_rng(SEED)
  progress = Progress("bounded legacy thresholds", len(SCENARIOS))
  rows = []
  for index, scenario in enumerate(SCENARIOS, start=1):
    row = simulate_scenario(scenario, draws=draws, rng=rng)
    rows.append(row)
    print(row, flush=True)
    progress.update(index)

  path = artifacts.data(OUTPUT)
  with path.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[artifact] {path}", flush=True)


if __name__ == "__main__":
  main()
