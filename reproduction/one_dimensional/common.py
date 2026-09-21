"""Shared calculations and plots for one-dimensional simulations."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

from heavily_right.combination import CombinationTest


def two_sided_normal_pvalues(estimates: np.ndarray) -> np.ndarray:
  """Convert unit-standard-error estimates to two-sided normal p-values."""
  return 2 * np.maximum(norm.sf(np.abs(estimates)), 1e-150)


def coverage_at_levels(
  p_values: np.ndarray,
  *,
  method: str = "HCauchy",
  levels: Sequence[float] = (0.05, 0.01),
  tests: Sequence[CombinationTest] | None = None,
) -> np.ndarray:
  """Return non-rejection rates, optionally reusing already calibrated tests."""
  if tests is None:
    tests = [CombinationTest(p_values.shape[-1], method, level) for level in levels]
  return np.array(
    [
      1 - np.mean(test.make_decision(p_values))
      for test in tests
    ]
  )


def rejection_rate(p_values: np.ndarray, *, method: str, level: float = 0.05) -> float:
  """Return one vectorized empirical rejection rate."""
  test = CombinationTest(p_values.shape[-1], method, level)
  return float(np.mean(test.make_decision(p_values)))


def paired_rejection_decisions(
  p_values: np.ndarray, tests: Sequence[CombinationTest],
) -> np.ndarray:
  """Evaluate every method on the same draws; return (method, draw) indicators.

  No outcomes are filtered, repaired, or resampled. Inverting a rejecting test
  at the true parameter gives noncoverage, including an empty confidence set.
  """
  if p_values.ndim != 2 or p_values.shape[0] == 0 or not tests:
    raise ValueError("paired comparisons require a nonempty draw matrix and tests")
  return np.stack([test.make_decision(p_values) for test in tests])


def smooth(values: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
  """Spline-interpolate along the first axis, using a safe order for smoke grids."""
  degree = min(3, len(x_old) - 1)
  return make_interp_spline(x_old, values, k=degree, axis=0)(x_new)


def build_band_figure(
  coverage: np.ndarray,
  x_old: np.ndarray,
  *,
  xlabel: str = r"$\rho$",
  ylabel: str = "coverage",
  complement_offset: float | None = None,
) -> plt.Figure:
  """Build the repeated-simulation mean curves and 95% Monte Carlo bands."""
  x_new = np.arange(x_old[0], x_old[-1], (x_old[1] - x_old[0]) / 5)
  upper = smooth(np.quantile(coverage, 0.975, axis=0), x_old, x_new)
  lower = smooth(np.quantile(coverage, 0.025, axis=0), x_old, x_new)
  mean = smooth(coverage.mean(axis=0), x_old, x_new)
  if complement_offset is not None:
    upper, lower = complement_offset - lower, complement_offset - upper
    mean = complement_offset - mean

  figure, axis = plt.subplots()
  axis.fill_between(x_new, upper[:, 0], lower[:, 0], color="skyblue", alpha=0.5)
  axis.fill_between(x_new, upper[:, 1], lower[:, 1], color="bisque", alpha=0.5)
  (line_05,) = axis.plot(x_new, mean[:, 0], color="cornflowerblue")
  (line_01,) = axis.plot(x_new, mean[:, 1], color="darkgoldenrod")
  axis.set(xlabel=xlabel, ylabel=ylabel)
  axis.legend([line_05, line_01], ["p=.05", "p=.01"], loc="lower right", fontsize=8)
  return figure


def build_method_figure(
  values: np.ndarray,
  x_old: np.ndarray,
  methods: Sequence[str],
  *,
  series: int,
  ylabel: str,
  ylim: tuple[float, float] | None = None,
  reference: float | None = None,
) -> plt.Figure:
  """Build a smoothed method-comparison curve panel."""
  x_new = np.arange(x_old[0], x_old[-1], (x_old[1] - x_old[0]) / 10)
  curves = smooth(np.moveaxis(values[:, :, series], 0, -1), x_old, x_new).T
  figure, axis = plt.subplots()
  axis.set_prop_cycle(cycler("color", plt.rcParams["axes.prop_cycle"].by_key()["color"]))
  plotted = (0, 1, *range(3, len(methods)))
  for index in plotted:
    axis.plot(x_new, curves[index], label=methods[index], linewidth=1.5)
  if reference is not None:
    axis.axhline(reference, linestyle="--", linewidth=0.8, color="grey")
  axis.set(xlabel=r"$\rho$", ylabel=ylabel, ylim=ylim)
  axis.legend()
  return figure
