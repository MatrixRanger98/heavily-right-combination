"""Multidimensional HCCT coverage for 10 and 500 studies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import chi2

from heavily_right.correlation import cor_fixed
from heavily_right.meta_analysis import MetaAnalysisMD
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress
from reproduction.sampling import CachedMultivariateNormal

EXPERIMENT_ID = "multidimensional/coverage"
SEED = 2025
LEVELS = (0.05, 0.01)
COLORS = ("coral", "darkseagreen", "orchid", "steelblue")


def settings() -> tuple[int, int, tuple[int, ...], tuple[int, ...], np.ndarray]:
  return (
    profile_value(paper=10, smoke=1),
    profile_value(paper=10_000, smoke=100),
    profile_value(paper=(500, 10), smoke=(30, 10)),
    profile_value(paper=(2, 5, 10, 25), smoke=(2, 5)),
    profile_value(paper=np.arange(0, 1, 0.1), smoke=np.array([0.0, 0.6])),
  )


def simulate(num_study: int) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
  repeat, num_run, _, dimensions, rhos = settings()
  coverage = np.zeros((repeat, len(dimensions), len(rhos), len(LEVELS)))
  progress = Progress(
    f"multidimensional coverage ({num_study} studies)",
    len(dimensions) * len(rhos),
  )
  completed = 0
  point = np.zeros(num_study)

  for dimension_index, dimension in enumerate(dimensions):
    analyses = [
      MetaAnalysisMD(num_study, dim=dimension, method="HCauchy", level=level)
      for level in LEVELS
    ]
    for rho_index, rho in enumerate(rhos):
      distribution = CachedMultivariateNormal(
        mean=point,
        cov=cor_fixed(num_study, float(rho)),
      )
      for replicate in range(repeat):
        estimates = np.swapaxes(
          distribution.rvs(size=(num_run, dimension)),
          1,
          2,
        )
        p_values = np.maximum(
          chi2.sf(np.sum(estimates**2, axis=-1), dimension),
          1e-150,
        )
        coverage[replicate, dimension_index, rho_index] = [
          1 - np.mean(analysis.make_decision(p_values))
          for analysis in analyses
        ]
      completed += 1
      progress.update(completed)
  return coverage.mean(axis=0), dimensions, rhos


def build_figure(
  coverage: np.ndarray,
  dimensions: tuple[int, ...],
  rhos: np.ndarray,
) -> plt.Figure:
  x_new = np.arange(rhos[0], rhos[-1], (rhos[1] - rhos[0]) / 5)
  degree = min(3, len(rhos) - 1)
  curves = make_interp_spline(rhos, coverage, k=degree, axis=1)(x_new)
  figure, axis = plt.subplots()
  lines = []
  labels = []
  for dimension_index, dimension in enumerate(dimensions):
    color = COLORS[dimension_index]
    for level_index, level in enumerate(LEVELS):
      (line,) = axis.plot(
        x_new,
        curves[dimension_index, :, level_index],
        color=color,
        linestyle="-" if level_index == 0 else (0, (5, 1)),
      )
      lines.append(line)
      labels.append(f"dim={dimension} p={level:g}")
  box = axis.get_position()
  axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axis.set(xlabel=r"$\rho$", ylabel="coverage")
  axis.legend(lines, labels, loc="lower left", bbox_to_anchor=(0, 0.45), fontsize=8)
  return figure


def render(
  coverage: np.ndarray, dimensions: tuple[int, ...], rhos: np.ndarray,
  num_study: int, artifacts: ArtifactStore,
) -> None:
  """Render a saved coverage summary without simulating study statistics."""
  save_pdf(build_figure(coverage, dimensions, rhos), artifacts,
           f"multivariate-coverage-{num_study}-studies.pdf", tight=True)


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  _, _, study_counts, _, _ = settings()
  for num_study in study_counts:
    coverage, dimensions, rhos = simulate(num_study)
    label = f"{num_study}-studies"
    np.save(artifacts.data(f"multivariate-coverage-{label}.npy"), coverage)
    render(coverage, dimensions, rhos, num_study, artifacts)


if __name__ == "__main__":
  main()
