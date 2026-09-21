"""CCT and HCCT score curves for the one-dimensional connectivity example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from heavily_right.combination import CombinationTest
from heavily_right.scores import half_cauchy_score
from reproduction.artifacts import ArtifactStore

ESTIMATES = np.array([-0.125, 0.125])
STANDARD_ERROR = 0.1
LEVEL = 0.05


def individual_p_values(
  theta: float | np.ndarray,
  *,
  estimates: np.ndarray = ESTIMATES,
  standard_error: float = STANDARD_ERROR,
) -> np.ndarray:
  """Return the two-sided normal p-values at candidate parameter values."""
  theta_array = np.asarray(theta, dtype=float)
  standardized = np.abs(theta_array[..., np.newaxis] - estimates) / standard_error
  return 2 * norm.sf(standardized)


def score_curves(theta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Evaluate equal-weight CCT and HCCT scores over ``theta``."""
  p_values = individual_p_values(theta)
  with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    cct = np.mean(1 / np.tan(np.pi * p_values), axis=-1)
    hcct = np.mean(np.asarray(half_cauchy_score(p_values)), axis=-1)
  return cct, hcct


def thresholds() -> tuple[float, float]:
  """Return the exact independent-study 95% CCT and HCCT thresholds."""
  cct = CombinationTest(2, method="Cauchy", level=LEVEL).threshold
  hcct = CombinationTest(2, method="HCauchy", level=LEVEL).threshold
  return float(cct), float(hcct)


def build_figure() -> plt.Figure:
  """Build the score-and-threshold figure without saving it."""
  theta = np.linspace(-0.145, 0.145, 6001)
  cct, hcct = score_curves(theta)
  cct_threshold, hcct_threshold = thresholds()

  fig, ax = plt.subplots(figsize=(6, 4.5))
  (cct_line,) = ax.plot(theta, cct, color="red", linewidth=2.4, label="CCT score")
  (hcct_line,) = ax.plot(
    theta,
    hcct,
    color="#0d98ba",
    linewidth=1.8,
    label="HCCT score",
  )
  cct_cutoff = ax.axhline(
    cct_threshold,
    color="red",
    linestyle=(0, (4, 3)),
    linewidth=1.0,
    label="95% coverage threshold",
  )
  hcct_cutoff = ax.axhline(
    hcct_threshold,
    color="#0d98ba",
    linestyle=(0, (4, 3)),
    linewidth=1.0,
    label="95% coverage threshold",
  )

  ax.set_xlim(-0.145, 0.145)
  ax.set_ylim(-15, 32)
  ax.spines["left"].set_position("zero")
  ax.spines["bottom"].set_position("zero")
  ax.spines["right"].set_visible(False)
  ax.spines["top"].set_visible(False)
  ax.legend(
    [cct_line, cct_cutoff, hcct_line, hcct_cutoff],
    [
      "CCT score",
      "95% coverage threshold",
      "HCCT score",
      "95% coverage threshold",
    ],
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    frameon=False,
  )
  fig.subplots_adjust(bottom=0.27)
  return fig


def main() -> None:
  artifacts = ArtifactStore.for_experiment("illustrations/connectivity-scores")
  figure = build_figure()
  figure.savefig(
    artifacts.figure("connectivity-scores.pdf"),
    format="pdf",
    bbox_inches="tight",
  )
  plt.close(figure)


if __name__ == "__main__":
  main()
