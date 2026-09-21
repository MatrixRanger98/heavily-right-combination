"""Two-dimensional slices of HCCT confidence regions by correlation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from heavily_right.correlation import cor_fixed
from heavily_right.meta_analysis import MetaAnalysisMD
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress

EXPERIMENT_ID = "multidimensional/contours"
SEED = 2024
COLORS = ("darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki")


def build_figure(
  *,
  dimension: int,
  num_study: int,
  rho: float,
) -> plt.Figure:
  correlation = cor_fixed(num_study, rho)
  estimates = multivariate_normal.rvs(
    mean=np.zeros(num_study),
    cov=correlation,
    size=dimension,
  ).T
  analysis = MetaAnalysisMD(num_study, dim=dimension, method="HCauchy", level=0.05)
  direction_1 = np.zeros(dimension)
  direction_2 = np.zeros(dimension)
  direction_1[0] = 1
  direction_2[1] = 1

  figure, axis = plt.subplots()
  analysis.area_slice(
    point=np.zeros(dimension),
    direction_1=direction_1,
    direction_2=direction_2,
    xi_hat=estimates,
    Sigma=np.eye(dimension),
    sub_dim=dimension,
    df=None,
    projs=None,
    xrange=(-4, 4.1),
    yrange=(-3, 3.1),
    ax=axis,
    colors=COLORS,
  )
  axis.scatter([0], [0], s=50, color="red", marker="*", label="True Value")
  axis.scatter(
    estimates[:, 0],
    estimates[:, 1],
    s=0.5,
    color="orange",
    label="Estimates from\n Each Study",
  )
  axis.legend(loc="upper left")
  axis.set(xlim=(-4, 4.1), ylim=(-3, 3.1))
  return figure


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  num_study = profile_value(paper=500, smoke=30)
  rhos = profile_value(paper=(0.0, 0.3, 0.6, 0.9), smoke=(0.0, 0.6))
  dimensions = (2, 10)
  progress = Progress("multidimensional contour panels", len(dimensions) * len(rhos))
  completed = 0
  for dimension in dimensions:
    for rho in rhos:
      save_pdf(
        build_figure(dimension=dimension, num_study=num_study, rho=rho),
        artifacts,
        f"multivariate-contours-dimension-{dimension}-rho-{rho:g}.pdf",
      )
      completed += 1
      progress.update(completed)


if __name__ == "__main__":
  main()
