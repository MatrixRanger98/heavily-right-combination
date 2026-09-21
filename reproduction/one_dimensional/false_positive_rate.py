"""False-positive-rate comparison under normal dependence."""

from __future__ import annotations

import warnings

import numpy as np

from heavily_right.combination import CombinationTest
from heavily_right.correlation import cor_ar_1, cor_fixed
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

EXPERIMENT_ID = "one-dimensional/false-positive-rate"
SEED = 2025
METHODS = ("HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes")


def simulate(
  *, diagnostics: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Compare all methods on shared samples within each rho/dependence case."""
  num_run = profile_value(paper=10_000, smoke=200)
  num_study = profile_value(paper=500, smoke=30)
  rhos = profile_value(paper=np.arange(0, 1, 0.1), smoke=np.array([0.0, 0.3, 0.6, 0.9]))
  rejections = np.zeros((len(METHODS), len(rhos), 2, num_run), dtype=bool)
  progress = Progress("false-positive paired correlation/dependence cases", 2 * len(rhos))
  completed = 0

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    tests = [CombinationTest(num_study, method, level=0.05) for method in METHODS]
  for rho_index, rho in enumerate(rhos):
    point = np.zeros(num_study)
    distributions = (
      CachedMultivariateNormal(mean=point, cov=cor_ar_1(num_study, float(rho))),
      CachedMultivariateNormal(mean=point, cov=cor_fixed(num_study, float(rho))),
    )
    for dependence_index, distribution in enumerate(distributions):
      p_values = two_sided_normal_pvalues(distribution.rvs(size=num_run))
      rejections[:, rho_index, dependence_index] = paired_rejection_decisions(p_values, tests)
      completed += 1
      progress.update(completed)
  if diagnostics is not None:
    diagnostics.update(
      rejections=rejections, rhos=rhos, methods=np.asarray(METHODS),
      cases=np.asarray(["ar1", "equicorrelated"]),
      thresholds=np.asarray([test.threshold for test in tests]),
      study_count=np.asarray(num_study), simulations_per_case=np.asarray(num_run),
      level=np.asarray(0.05),
      axes=np.asarray(["method", "rho", "case", "draw"]),
      sampling_scheme=np.asarray("shared-across-methods; distinct batches across rho/case"),
    )
  return rejections.mean(axis=-1), rhos


def render(rates: np.ndarray, rhos: np.ndarray, artifacts: ArtifactStore) -> None:
  """Render saved rates without sampling or changing random state."""
  for index, (label, ylim) in enumerate(
    (("ar1", (0, 0.15)), ("equicorrelated", (0, 0.35)))
  ):
    figure = build_method_figure(
      rates,
      rhos,
      METHODS,
      series=index,
      ylabel="false positive rate",
      ylim=ylim,
      reference=0.05,
    )
    axis = figure.axes[0]
    # The paired estimates can coincide: retain both colors without moving data.
    for line in axis.lines:
      if line.get_label() == "HCauchy":
        line.set(linewidth=2.5, zorder=3)
      elif line.get_label() == "EHMP":
        line.set(linestyle="--", zorder=4)
    axis.legend()
    save_pdf(figure, artifacts, f"false-positive-rate-{label}.pdf")


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  diagnostics: dict[str, np.ndarray] = {}
  rates, rhos = simulate(diagnostics=diagnostics)
  np.save(artifacts.data("false-positive-rate.npy"), rates)
  np.savez_compressed(
    artifacts.data("false-positive-rate-paired.npz"), seed=SEED,
    rng="RandomState/MT19937; rho-major, case-minor", **diagnostics,
  )
  render(rates, rhos, artifacts)


if __name__ == "__main__":
  main()
