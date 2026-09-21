"""Benchmark the dense-KKT/SLSQP support-solver regime switch."""

from __future__ import annotations

import csv
from time import perf_counter

import numpy as np

from heavily_right.meta_analysis import MetaAnalysisMD
from reproduction.artifacts import ArtifactStore
from reproduction.profiles import profile_value

EXPERIMENT_ID = "numerical/support-solver-benchmark"
OUTPUT = "support-solver-benchmark.csv"
SEED = 20260901
DENSE_NEWTON_MAX_DIMENSION = 250


def _overlapping_blocks(dimension: int, component_size: int) -> np.ndarray:
  if dimension % component_size:
    raise ValueError("dimension must be divisible by component_size")
  blocks = []
  for offset in range(0, dimension, component_size):
    starts = list(range(0, component_size - 3, 3))
    if starts[-1] != component_size - 4:
      starts.append(component_size - 4)
    blocks.extend(
      offset + np.arange(start, start + 4) for start in starts
    )
  return np.array(blocks)


def evaluate() -> list[dict[str, int | float | str | bool]]:
  """Time connected q=4 systems below and above the dense-Newton cap."""
  scenarios = profile_value(
    paper=((50, 50), (170, 170), (260, 260), (2000, 50)),
    smoke=((20, 20), (260, 260), (400, 20)),
  )
  rng = np.random.default_rng(SEED)
  rows: list[dict[str, int | float | str | bool]] = []
  for dimension, component_size in scenarios:
    sub_dim = _overlapping_blocks(dimension, component_size)
    num_blocks = len(sub_dim)
    direction = np.zeros(dimension)
    direction[:component_size] = rng.normal(size=component_size)
    direction /= np.linalg.norm(direction)
    analysis = MetaAnalysisMD(
      num_blocks,
      dim=dimension,
      method="HCauchy",
      level=0.05,
    )
    started = perf_counter()
    result = analysis.support_interval(
      direction,
      np.zeros((num_blocks, 4)),
      np.eye(4),
      sub_dim,
      df=np.full(num_blocks, 40),
      dense_newton_max_dimension=DENSE_NEWTON_MAX_DIMENSION,
    )
    elapsed_seconds = perf_counter() - started
    diagnostics = result.upper_diagnostics
    row = {
      "dimension": dimension,
      "component_size": component_size,
      "blocks": num_blocks,
      "active_dimension": result.active_dimension,
      "dense_newton_max_dimension": DENSE_NEWTON_MAX_DIMENSION,
      "solver": diagnostics.solver,
      "elapsed_seconds": elapsed_seconds,
      "iterations": diagnostics.iterations,
      "evaluations": diagnostics.evaluations,
      "line_search_steps": diagnostics.line_search_steps,
      "boundary_error": diagnostics.boundary_error,
      "kkt_residual": diagnostics.kkt_residual,
      "success": result.success,
    }
    rows.append(row)
    print(row, flush=True)
  return rows


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  rows = evaluate()
  path = artifacts.data(OUTPUT)
  with path.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[artifact] {path}", flush=True)


if __name__ == "__main__":
  main()
