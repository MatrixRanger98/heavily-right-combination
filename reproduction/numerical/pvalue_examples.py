"""All non-CAtr columns of the paper's heterogeneous p-value examples."""

from __future__ import annotations

import csv
import warnings

from heavily_right.combination import CombinationMethod, CombinationTest
from reproduction.artifacts import ArtifactStore

EXPERIMENT_ID = "numerical/pvalue-examples"
OUTPUT = "pvalue-examples.csv"
EXAMPLES = (
  (0.02, 0.03, 0.96),
  (0.02, 0.03, 0.98),
  (0.02, 0.03, 0.99),
  (0.015, 0.9, 0.96),
  (0.02, 0.02, 0.8, 0.98),
  (0.01, 0.05, 0.3, 0.5, 0.99),
)
METHODS: tuple[CombinationMethod, ...] = (
  "Fisher", "Stouffer", "Bonferroni", "Cauchy", "HCauchy", "EHMP",
)


def evaluate() -> list[dict[str, str | float | int]]:
  rows: list[dict[str, str | float | int]] = []
  for p_values in EXAMPLES:
    row: dict[str, str | float | int] = {
      "p_values": ";".join(str(value) for value in p_values),
      "num_tests": len(p_values),
    }
    for method in METHODS:
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Equal weights are used for Fisher's test\\.",
                                category=UserWarning)
        row[method.lower()] = float(
          CombinationTest(len(p_values), method=method).get_global_p(p_values)
        )
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
