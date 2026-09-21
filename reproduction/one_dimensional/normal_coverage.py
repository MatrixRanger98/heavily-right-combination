"""Normal-theory coverage under AR(1) and equicorrelated dependence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from heavily_right.combination import CombinationTest
from heavily_right.correlation import cor_ar_1, cor_fixed
from reproduction.artifacts import ArtifactStore
from reproduction.one_dimensional.common import (
  build_band_figure,
  coverage_at_levels,
  two_sided_normal_pvalues,
)
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress
from reproduction.sampling import CachedMultivariateNormal

EXPERIMENT_ID = "one-dimensional/normal-coverage"
SEED = 2025


def simulate_coverage(correlation: Callable[[int, float], np.ndarray], label: str) -> tuple[np.ndarray, np.ndarray]:
  repeat = profile_value(paper=50, smoke=2)
  num_run = profile_value(paper=10_000, smoke=200)
  num_study = profile_value(paper=500, smoke=30)
  rhos = profile_value(paper=np.arange(0, 1, 0.1), smoke=np.array([0.0, 0.6]))
  coverage = np.zeros((repeat, len(rhos), 2))
  progress = Progress(f"normal coverage ({label})", len(rhos) * repeat)
  tests = [CombinationTest(num_study, "HCauchy", level) for level in (0.05, 0.01)]

  for rho_index, rho in enumerate(rhos):
    distribution = CachedMultivariateNormal(
      mean=np.zeros(num_study),
      cov=correlation(num_study, float(rho)),
    )
    for replicate in range(repeat):
      estimates = distribution.rvs(size=num_run)
      coverage[replicate, rho_index] = coverage_at_levels(
        two_sided_normal_pvalues(estimates), tests=tests,
      )
      progress.update(rho_index * repeat + replicate + 1)
  return coverage, rhos


def render(coverage: np.ndarray, rhos: np.ndarray, label: str, artifacts: ArtifactStore) -> None:
  """Render one saved dependence case without new Monte Carlo draws."""
  save_pdf(build_band_figure(coverage, rhos), artifacts, f"normal-coverage-{label}.pdf")


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  for label, correlation in (("ar1", cor_ar_1), ("equicorrelated", cor_fixed)):
    coverage, rhos = simulate_coverage(correlation, label)
    np.save(artifacts.data(f"normal-coverage-{label}.npy"), coverage)
    render(coverage, rhos, label, artifacts)


if __name__ == "__main__":
  main()
