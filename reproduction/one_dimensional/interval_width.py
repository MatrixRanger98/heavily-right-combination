"""HCCT confidence-interval widths under two dependence structures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from heavily_right.correlation import cor_ar_1, cor_fixed
from heavily_right.meta_analysis import MetaAnalysis1D
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress
from reproduction.sampling import CachedMultivariateNormal

EXPERIMENT_ID = "one-dimensional/interval-width"
SEED = 2025
RHOS = np.array([0.0, 0.3, 0.6, 0.9])


def simulate() -> np.ndarray:
  num_run = profile_value(paper=10_000, smoke=40)
  num_study = profile_value(paper=500, smoke=30)
  analysis = MetaAnalysis1D(num_study, "HCauchy", level=0.05)
  widths = np.zeros((2, len(RHOS), num_run))
  progress = Progress("interval-width correlation cases", len(RHOS))

  for rho_index, rho in enumerate(RHOS):
    point = np.zeros(num_study)
    distributions = (
      CachedMultivariateNormal(mean=point, cov=cor_ar_1(num_study, rho)),
      CachedMultivariateNormal(mean=point, cov=cor_fixed(num_study, rho)),
    )
    samples = [distribution.rvs(size=num_run) for distribution in distributions]
    for dependence_index, sample in enumerate(samples):
      for run_index, estimates in enumerate(sample):
        lower, upper, _ = analysis.confidence_interval(estimates, 1.0)
        widths[dependence_index, rho_index, run_index] = upper - lower
    progress.update(rho_index + 1)
  return widths


def build_figure(widths: np.ndarray, *, dependence_index: int) -> plt.Figure:
  """Use the same scoped white-grid style for both panels, in any call order."""
  data = pd.DataFrame(
    {rf"$\rho={rho:g}$": widths[dependence_index, index] for index, rho in enumerate(RHOS)}
  )
  with sns.axes_style("whitegrid"):
    figure, axis = plt.subplots()
    sns.kdeplot(data, ax=axis)
    axis.set(xlabel="width", ylabel="density")
    axis.set_xlim((0.2, 3) if dependence_index == 0 else (0.4, 4.1))
  return figure


def render(widths: np.ndarray, artifacts: ArtifactStore) -> None:
  """Render the saved width sample without fitting intervals again."""
  for index, label in enumerate(("ar1", "equicorrelated")):
    save_pdf(
      build_figure(widths, dependence_index=index),
      artifacts,
      f"interval-width-{label}.pdf",
    )


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  widths = simulate()
  np.save(artifacts.data("interval-width.npy"), widths)
  render(widths, artifacts)


if __name__ == "__main__":
  main()
