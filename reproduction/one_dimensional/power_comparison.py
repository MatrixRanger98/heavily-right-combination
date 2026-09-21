"""Power comparison for dense and sparse normal alternatives."""

from __future__ import annotations

import warnings

import numpy as np

from heavily_right.combination import CombinationTest
from heavily_right.correlation import cor_fixed
from reproduction.artifacts import ArtifactStore
from reproduction.one_dimensional.common import (
  build_method_figure,
  paired_rejection_decisions,
  two_sided_normal_pvalues,
)
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress
from reproduction.sampling import CachedMultivariateNormal

EXPERIMENT_ID = "one-dimensional/power"
SEED = 2025
METHODS = ("HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes")


def signal_count(num_study: int, sparsity: float) -> int:
  return int(num_study ** (1 - sparsity))


def signal_value(num_study: int, strength: float) -> float:
  return float(np.sqrt(2 * strength * np.log(num_study)))


def alternatives(num_study: int) -> tuple[np.ndarray, np.ndarray]:
  dense = np.zeros(num_study)
  sparse = np.zeros(num_study)
  dense[: signal_count(num_study, 0)] = signal_value(num_study, 0.1)
  sparse[: signal_count(num_study, 0.3)] = signal_value(num_study, 0.3)
  return dense, sparse


def simulate(
  *, diagnostics: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Compare all methods on shared samples within each rho/alternative case."""
  num_run = profile_value(paper=10_000, smoke=200)
  num_study = profile_value(paper=500, smoke=30)
  rhos = profile_value(paper=np.arange(0, 1, 0.1), smoke=np.array([0.0, 0.3, 0.6, 0.9]))
  rejections = np.zeros((len(METHODS), len(rhos), 2, num_run), dtype=bool)
  points = alternatives(num_study)
  progress = Progress("power paired correlation/alternative cases", 2 * len(rhos))
  completed = 0

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    tests = [CombinationTest(num_study, method, level=0.05) for method in METHODS]
  for rho_index, rho in enumerate(rhos):
    covariance = cor_fixed(num_study, float(rho))
    distributions = [CachedMultivariateNormal(point, covariance) for point in points]
    for alternative_index, distribution in enumerate(distributions):
      p_values = two_sided_normal_pvalues(distribution.rvs(size=num_run))
      rejections[:, rho_index, alternative_index] = paired_rejection_decisions(p_values, tests)
      completed += 1
      progress.update(completed)
  if diagnostics is not None:
    diagnostics.update(
      rejections=rejections, rhos=rhos, methods=np.asarray(METHODS),
      cases=np.asarray(["dense", "sparse"]),
      thresholds=np.asarray([test.threshold for test in tests]),
      study_count=np.asarray(num_study), simulations_per_case=np.asarray(num_run),
      level=np.asarray(0.05), alternative_means=np.stack(points),
      axes=np.asarray(["method", "rho", "case", "draw"]),
      sampling_scheme=np.asarray("shared-across-methods; distinct batches across rho/case"),
    )
  return rejections.mean(axis=-1), rhos


def render(power: np.ndarray, rhos: np.ndarray, artifacts: ArtifactStore) -> None:
  """Render saved powers without sampling or changing random state."""
  for index, label in enumerate(("dense", "sparse")):
    figure = build_method_figure(
      power,
      rhos,
      METHODS,
      series=index,
      ylabel="power",
    )
    axis = figure.axes[0]
    # The paired estimates can coincide: retain both colors without moving data.
    for line in axis.lines:
      if line.get_label() == "HCauchy":
        line.set(linewidth=2.5, zorder=3)
      elif line.get_label() == "EHMP":
        line.set(linestyle="--", zorder=4)
    axis.legend()
    save_pdf(figure, artifacts, f"power-comparison-{label}.pdf")


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  diagnostics: dict[str, np.ndarray] = {}
  power, rhos = simulate(diagnostics=diagnostics)
  np.save(artifacts.data("power-comparison.npy"), power)
  np.savez_compressed(
    artifacts.data("power-comparison-paired.npz"), seed=SEED,
    rng="RandomState/MT19937; rho-major, case-minor", **diagnostics,
  )
  render(power, rhos, artifacts)


if __name__ == "__main__":
  main()
