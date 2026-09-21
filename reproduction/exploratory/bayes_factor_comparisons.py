"""Compare oracle, harmonic-mean, and half-Cauchy Bayes-factor curves."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf

EXPERIMENT_ID = "exploratory/bayes-factor-comparisons"


def beta_one_oracle(p_values: np.ndarray) -> np.ndarray:
  """Oracle curve from the recovered beta-one calculation."""
  transition = 1 - 1 / np.e
  below = 1 / (np.e * (1 - p_values) * np.log(1 / (1 - p_values)))
  return np.where(p_values < transition, below, 1)


def goodman_oracle(p_values: np.ndarray) -> np.ndarray:
  return np.exp(norm.ppf(p_values / 2) ** 2 / 2)


def edwards_oracle(p_values: np.ndarray) -> np.ndarray:
  return np.exp(norm.ppf(p_values) ** 2 / 2)


def edwards_two_oracle(p_values: np.ndarray) -> np.ndarray:
  quantiles = norm.ppf(p_values / 2)
  return np.exp(quantiles**2 / 2) / (np.sqrt(np.e) * np.abs(quantiles))


def comparison_curves(
  p_values: np.ndarray,
  *,
  oracle: Callable[[np.ndarray], np.ndarray],
  scale: float,
  endpoint: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  oracle_values = oracle(p_values)
  harmonic_mean = 1 / (scale * p_values) + 1 - 1 / (scale * endpoint)
  half_cauchy = (
    np.pi / (2 * scale * np.tan(p_values * np.pi / 2))
    + 1
    - np.pi / (2 * scale * np.tan(endpoint * np.pi / 2))
  )
  return oracle_values, harmonic_mean, half_cauchy


def build_figure(
  p_values: np.ndarray,
  curves: tuple[np.ndarray, np.ndarray, np.ndarray],
  *,
  title: str,
) -> plt.Figure:
  figure, axis = plt.subplots()
  for values, label in zip(
    curves,
    ("oracle", "harmonic mean", "half-Cauchy"),
    strict=True,
  ):
    axis.plot(p_values, values, label=label)
  axis.set(
    xscale="log",
    yscale="log",
    xlabel="p-value (log scale)",
    ylabel="Bayes factor (log scale)",
    title=title,
  )
  axis.legend()
  return figure


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  settings = (
    (
      "beta-one",
      np.logspace(-3, -0.01, 1000),
      beta_one_oracle,
      np.e,
      1.0,
      "Beta-one comparison",
    ),
    (
      "goodman",
      np.logspace(-3, -0.01, 1000),
      goodman_oracle,
      5.0,
      1.0,
      "Goodman comparison",
    ),
    (
      "edwards",
      np.logspace(-3, np.log10(0.5), 1000),
      edwards_oracle,
      5.0,
      0.5,
      "Edwards comparison",
    ),
    (
      "edwards-two",
      np.logspace(-3, -0.1, 1000),
      edwards_two_oracle,
      5.0,
      0.5,
      "Edwards comparison with tail correction",
    ),
  )
  for label, p_values, oracle, scale, endpoint, title in settings:
    curves = comparison_curves(
      p_values,
      oracle=oracle,
      scale=scale,
      endpoint=endpoint,
    )
    save_pdf(
      build_figure(p_values, curves, title=title),
      artifacts,
      f"bayes-factor-comparisons-{label}.pdf",
      tight=True,
    )


if __name__ == "__main__":
  main()
