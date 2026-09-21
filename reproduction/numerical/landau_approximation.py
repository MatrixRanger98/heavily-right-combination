"""Landau approximations to finite Half-Cauchy sample means."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy

from heavily_right.null_laws import HalfCauchyMean, Landau, euler_gamma
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress

EXPERIMENT_ID = "numerical/landau-approximation"
SAMPLE_SIZES = (1, 10, 100, 1000)


def _vectorize(function: Callable[[float], float], x: np.ndarray) -> np.ndarray:
  return np.fromiter((function(value) for value in x), dtype=float, count=len(x))


def _landau_density(x: np.ndarray, sample_size: int) -> np.ndarray:
  location = 2 / np.pi * (1 - euler_gamma + np.log(sample_size))
  return Landau.pdf(x, location, 1)


def build_density_figure(sample_size: int) -> plt.Figure:
  ranges = {
    1: (-2.5, 8.0),
    10: (-1.0, 9.0),
    100: (0.0, 10.0),
    1000: (1.0, 11.0),
  }
  lower, upper = ranges[sample_size]
  step = profile_value(paper=0.01, smoke=0.05)
  x = np.arange(lower, upper, step)
  exact = np.zeros_like(x)
  positive = x >= 0
  if sample_size == 1:
    exact[positive] = 2 * cauchy.pdf(x[positive])
  else:
    exact[positive] = _vectorize(
      lambda value: HalfCauchyMean.pdf(value, sample_size),
      x[positive],
    )

  figure, axis = plt.subplots()
  axis.plot(x, exact, color="b", label=f"Half-Cauchy mean, m={sample_size}")
  if sample_size == 1:
    axis.axvline(0, linewidth=1, linestyle="--")
  axis.plot(
    x,
    _landau_density(x, sample_size),
    color="r",
    label="Landau approximation",
  )
  axis.legend()
  return figure


def build_cdf_figure(sample_size: int = 1000) -> plt.Figure:
  step = profile_value(paper=0.1, smoke=0.25)
  exact_x = np.arange(2, 11, step)
  approximation_x = np.arange(1, 11, 0.01)
  location = 2 / np.pi * (1 - euler_gamma + np.log(sample_size))

  figure, axis = plt.subplots()
  axis.plot(
    exact_x,
    _vectorize(lambda value: HalfCauchyMean.cdf(value, sample_size), exact_x),
    color="b",
    label=f"Half-Cauchy mean, m={sample_size}",
  )
  axis.plot(
    approximation_x,
    Landau.cdf(approximation_x, location, 1),
    color="r",
    label="Landau approximation",
  )
  axis.legend()
  return figure


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  sizes = profile_value(paper=SAMPLE_SIZES, smoke=(1, 10))
  progress = Progress("Landau density panels", len(sizes))
  for index, sample_size in enumerate(sizes, start=1):
    save_pdf(
      build_density_figure(sample_size),
      artifacts,
      f"landau-approximation-density-sum-{sample_size}.pdf",
    )
    progress.update(index)
  if 1000 in sizes:
    save_pdf(
      build_cdf_figure(),
      artifacts,
      "landau-approximation-cdf-sum-1000.pdf",
    )


if __name__ == "__main__":
  main()
