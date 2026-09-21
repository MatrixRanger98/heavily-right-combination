"""Acceptance contours for six two-p-value combination rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import cauchy, chi2, norm

from heavily_right.null_laws import HalfCauchyMean, HarmonicMean
from heavily_right.scores import half_cauchy_score, reciprocal_score
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value

EXPERIMENT_ID = "illustrations/two-pvalue-regions"
METHODS = ("fisher", "stouffer", "bonferroni", "cauchy", "hcauchy", "harmonic")
COLORS = [
  "#dff2cb",
  "#e1e6bf",
  "#e4dbb2",
  "#e6cfa6",
  "#e9c39a",
  "#ebb88d",
  "#eeac81",
  "#f0a075",
  "#f39568",
  "#f5895c",
  "#ee8268",
  "#e67c75",
  "#df7581",
  "#d86f8e",
  "#d0689a",
  "#c962a7",
  "#c25bb3",
  "#ba55c0",
  "#b34ecc",
]


@dataclass(frozen=True)
class RegionSpec:
  name: str
  surface: Callable[[np.ndarray, np.ndarray], np.ndarray]
  levels: np.ndarray
  labels: np.ndarray
  bracket_low: float = 0.0
  reverse_colors: bool = True
  skip_first_label: bool = False


def _fisher(x, y):
  with np.errstate(divide="ignore"):
    return chi2.sf(-2 * np.log(x) - 2 * np.log(y), 4)


def _stouffer(x, y):
  return norm.sf(-norm.ppf(x) - norm.ppf(y), scale=np.sqrt(2))


def _bonferroni(x, y):
  return 2 * np.minimum(x, y)


def _cauchy(x, y):
  score = 0.5 * np.tan(np.pi * (0.5 - x)) + 0.5 * np.tan(np.pi * (0.5 - y))
  return cauchy.sf(score)


def _hcauchy_score(x, y):
  return 0.5 * np.asarray(half_cauchy_score(x)) + 0.5 * np.asarray(
    half_cauchy_score(y)
  )


def _harmonic_score(x, y):
  return 0.5 * np.asarray(reciprocal_score(x)) + 0.5 * np.asarray(
    reciprocal_score(y)
  )


def region_specs() -> tuple[RegionSpec, ...]:
  probabilities = np.arange(0.05, 1, 0.05)
  return (
    RegionSpec("fisher", _fisher, probabilities, probabilities),
    RegionSpec("stouffer", _stouffer, probabilities, probabilities),
    RegionSpec(
      "bonferroni",
      _bonferroni,
      probabilities,
      probabilities,
      skip_first_label=True,
    ),
    RegionSpec("cauchy", _cauchy, probabilities, probabilities),
    RegionSpec(
      "hcauchy",
      _hcauchy_score,
      np.array([HalfCauchyMean.ppf(p, 2) for p in probabilities]),
      1 - probabilities,
      reverse_colors=False,
      skip_first_label=True,
    ),
    RegionSpec(
      "harmonic",
      _harmonic_score,
      np.array([HarmonicMean.ppf(p, 2) for p in probabilities]),
      1 - probabilities,
      bracket_low=0.01,
      reverse_colors=False,
      skip_first_label=True,
    ),
  )


def build_figure(spec: RegionSpec) -> plt.Figure:
  delta = profile_value(paper=0.0005, smoke=0.005)
  coordinates = np.arange(delta, 1, delta)
  x, y = np.meshgrid(coordinates, coordinates)
  colors = COLORS[::-1] if spec.reverse_colors else COLORS

  figure, axis = plt.subplots()
  axis.set_aspect(1)
  contours = axis.contour(
    x,
    y,
    spec.surface(x, y),
    spec.levels,
    colors=colors,
    linewidths=1.5 if spec.name == "bonferroni" else 2,
  )
  label_map = {
    level: "0.05" if np.isclose(label, 0.05) else f"{label:.1f}"
    for level, label in zip(contours.levels, spec.labels, strict=True)
  }
  locations = []
  for level, label in zip(contours.levels, spec.labels, strict=True):
    if spec.name in {"hcauchy", "harmonic"}:
      root = root_scalar(
        lambda value, label=label, level=level: spec.surface(label, value) - level,
        bracket=[spec.bracket_low, 1],
        method="brentq",
      ).root
    else:
      root = root_scalar(
        lambda value, label=label: spec.surface(label, value) - label,
        bracket=[spec.bracket_low, 1],
        method="brentq",
      ).root
    locations.append((label, root))

  indices = list(range(1 if spec.skip_first_label else 0, len(locations), 2))
  if not spec.skip_first_label:
    indices.insert(0, 0)
  if spec.name in {"hcauchy", "harmonic"}:
    indices.append(len(locations) - 1)
  indices = sorted(set(indices))
  axis.clabel(
    contours,
    levels=contours.levels,
    fmt=label_map,
    inline=True,
    fontsize=10 if spec.skip_first_label else 12.5,
    manual=[locations[index] for index in indices],
  )
  axis.set(xlabel=r"$p_1$", ylabel=r"$p_2$")
  axis.set_xticks(np.arange(0, 1.1, 0.2))
  axis.set_yticks(np.arange(0, 1.1, 0.2))
  return figure


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  for spec in region_specs():
    save_pdf(
      build_figure(spec),
      artifacts,
      f"pvalue-regions-{spec.name}.pdf",
      tight=True,
    )


if __name__ == "__main__":
  main()
