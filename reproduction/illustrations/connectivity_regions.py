"""Two-dimensional CCT and HCCT connectivity regions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import cauchy, norm

from heavily_right.null_laws import HalfCauchyMean
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value

EXPERIMENT_ID = "illustrations/connectivity-regions"
OUTPUT = "connectivity-regions.pdf"
ESTIMATES = np.array([[0.0, 0.21], [0.21, 0.0], [-0.10, -0.10]])


def cct_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Return the equal-weight CCT score for the three bivariate studies."""
  distances = [np.hypot(x - ex, y - ey) * np.sqrt(50) for ex, ey in ESTIMATES]
  return np.mean(
    [np.tan(np.pi * (0.5 - 2 * norm.cdf(-distance))) for distance in distances],
    axis=0,
  )


def hcct_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Return the equal-weight HCCT score for the three bivariate studies."""
  distances = [np.hypot(x - ex, y - ey) * np.sqrt(50) for ex, ey in ESTIMATES]
  return np.mean(
    [np.tan(np.pi * (0.5 - norm.cdf(-distance))) for distance in distances],
    axis=0,
  )


def build_figure() -> plt.Figure:
  delta = profile_value(paper=0.0004, smoke=0.004)
  coordinates = np.arange(-0.18, 0.28, delta)
  x, y = np.meshgrid(coordinates, coordinates)

  figure, axis = plt.subplots()
  cct_contour = axis.contour(
    x,
    y,
    cct_score(x, y),
    [cauchy.ppf(0.95)],
    colors=["red"],
    linewidths=2.1,
  )
  hcct_contour = axis.contour(
    x,
    y,
    hcct_score(x, y),
    [HalfCauchyMean.ppf(0.95, 3)],
    colors=["#0d98ba"],
    linewidths=1.4,
  )
  cct_handles, _ = cct_contour.legend_elements()
  hcct_handles, _ = hcct_contour.legend_elements()
  axis.legend(cct_handles + hcct_handles, ["CCT", "HCCT"])
  return figure


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  save_pdf(build_figure(), artifacts, OUTPUT)


if __name__ == "__main__":
  main()
