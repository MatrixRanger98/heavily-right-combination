"""Finite-mean precision tables and correctly parameterized Landau comparisons.

The long-format CSV retains individual PDF/CDF measurements; the two wide CSVs
have one row for each point printed in the corresponding supplement table.
Elapsed times and estimated absolute numerical errors are not fixed references.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypedDict

import numpy as np

from heavily_right.null_laws import HalfCauchyMean, HarmonicMean, Landau, euler_gamma
from reproduction.artifacts import ArtifactStore
from reproduction.profiles import profile_value

EXPERIMENT_ID = "numerical/distribution-precision"
OUTPUT = "distribution-precision.csv"
HCCT_OUTPUT = "distribution-precision-hcct.csv"
EHMP_OUTPUT = "distribution-precision-ehmp.csv"
DistributionName = Literal["half-cauchy-mean", "harmonic-mean"]
Cell = int | float | str


class PrecisionRow(TypedDict):
  distribution: DistributionName
  paper_label: str
  quantity: str
  sample_size: int
  x: float
  exact: float
  error_estimate: float
  elapsed_ms: float
  landau_location: float
  landau_scale: float
  landau_approximation: float
  landau_minus_exact: float


POINTS = (
  (2, 0.2),
  (2, 2),
  (2, 10),
  (2, 50),
  (10, 1),
  (10, 4),
  (10, 10),
  (10, 50),
  (100, 2),
  (100, 5),
  (100, 10),
  (100, 50),
  (1000, 4),
  (1000, 7),
  (1000, 10),
  (1000, 50),
)
# The EHMP table omits these two support-boundary/below-support points.
EHMP_POINTS = tuple(point for point in POINTS if point not in ((2, 0.2), (10, 1)))


@dataclass(frozen=True)
class PrecisionTable:
  name: DistributionName
  distribution: type[HalfCauchyMean] | type[HarmonicMean]
  label: str
  output: str
  points: tuple[tuple[int, float], ...]


TABLES = (
  PrecisionTable("half-cauchy-mean", HalfCauchyMean, "tab:precision", HCCT_OUTPUT, POINTS),
  PrecisionTable("harmonic-mean", HarmonicMean, "tab:precisionehmp", EHMP_OUTPUT, EHMP_POINTS),
)


def landau_parameters(sample_size: int, distribution: DistributionName) -> tuple[float, float]:
  """Return the paper's S1 location and scale for an equally weighted mean.

  Half-Cauchy uses mu=(2/pi)*(log(m)+1-gamma), c=1. Pareto(1,1)
  reciprocal scores use mu=log(m)+1-gamma, c=pi/2. In particular,
  the scale is not itself the coefficient of the logarithmic location shift.
  These conventions agree with the canonical CombinationTest calibration.
  """
  if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
    raise ValueError("sample_size must be a positive integer")
  if distribution == "half-cauchy-mean":
    scale = 1.0
  elif distribution == "harmonic-mean":
    scale = float(np.pi / 2)
  else:
    raise ValueError(f"unknown precision-table distribution: {distribution!r}")
  location = 2 * scale / np.pi * (1 - euler_gamma + np.log(sample_size))
  return float(location), scale


def evaluate() -> list[PrecisionRow]:
  """Measure only the requested profile's paper points, without writing files."""
  rows: list[PrecisionRow] = []
  for table in TABLES:
    points = profile_value(paper=table.points, smoke=table.points[:2])
    for sample_size, x in points:
      location, scale = landau_parameters(sample_size, table.name)
      for quantity in ("pdf", "cdf"):
        exact_function = getattr(table.distribution, quantity)
        approximate_function = getattr(Landau, quantity)
        started = perf_counter()
        value, error = exact_function(x, sample_size, precision=True)
        elapsed_ms = 1000 * (perf_counter() - started)
        approximation = float(approximate_function(x, location, scale))
        row: PrecisionRow = {
          "distribution": table.name,
          "paper_label": table.label,
          "quantity": quantity,
          "sample_size": sample_size,
          "x": float(x),
          "exact": float(value),
          "error_estimate": float(error),
          "elapsed_ms": elapsed_ms,
          "landau_location": location,
          "landau_scale": scale,
          "landau_approximation": approximation,
          "landau_minus_exact": approximation - float(value),
        }
        rows.append(row)
        print(row, flush=True)
  return rows


def table_rows(rows: Sequence[PrecisionRow], distribution: DistributionName) -> list[dict[str, Cell]]:
  """Pair PDF/CDF measurements for display, preserving order and diagnostics.

  The sign convention is approximation minus exact. Timings are converted
  from milliseconds to seconds, matching the manuscript table headings.
  Missing or duplicate measurements are rejected rather than silently filled.
  """
  grouped: dict[tuple[int, float], dict[str, PrecisionRow]] = {}
  for row in rows:
    if row["distribution"] != distribution:
      continue
    key = (row["sample_size"], row["x"])
    pair = grouped.setdefault(key, {})
    quantity = row["quantity"]
    if quantity not in ("pdf", "cdf") or quantity in pair:
      raise ValueError(f"invalid or duplicate precision measurement: {key}, {quantity}")
    pair[quantity] = row
  if not grouped:
    raise ValueError(f"no measurements for {distribution!r}")

  result: list[dict[str, Cell]] = []
  for (sample_size, x), pair in grouped.items():
    if set(pair) != {"pdf", "cdf"}:
      raise ValueError(f"both PDF and CDF measurements are required at {(sample_size, x)}")
    pdf, cdf = pair["pdf"], pair["cdf"]
    if ((pdf["paper_label"], pdf["landau_location"], pdf["landau_scale"])
        != (cdf["paper_label"], cdf["landau_location"], cdf["landau_scale"])):
      raise ValueError(f"inconsistent PDF/CDF parameters at {(sample_size, x)}")
    combined: dict[str, Cell] = {
      "distribution": distribution,
      "paper_label": pdf["paper_label"],
      "sample_size": sample_size,
      "x": x,
      "landau_location": pdf["landau_location"],
      "landau_scale": pdf["landau_scale"],
    }
    for quantity, row in pair.items():
      combined[quantity] = row["exact"]
      combined[f"{quantity}_error_estimate"] = row["error_estimate"]
      combined[f"{quantity}_elapsed_s"] = row["elapsed_ms"] / 1000
      combined[f"{quantity}_landau"] = row["landau_approximation"]
      combined[f"{quantity}_landau_minus_exact"] = row["landau_minus_exact"]
    result.append(combined)
  return result


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  rows = evaluate()
  outputs: list[tuple[str, Sequence[PrecisionRow] | Sequence[dict[str, Cell]]]] = [(OUTPUT, rows)]
  outputs.extend((table.output, table_rows(rows, table.name)) for table in TABLES)
  for filename, measurements in outputs:
    path = artifacts.data(filename)
    with path.open("w", newline="") as output:
      writer = csv.DictWriter(output, fieldnames=list(measurements[0]))
      writer.writeheader()
      writer.writerows(measurements)
    print(f"[artifact] {path}", flush=True)


if __name__ == "__main__":
  main()
