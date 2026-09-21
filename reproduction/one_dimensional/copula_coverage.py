"""HCCT and Fisher behavior under multivariate-t and bivariate copula dependence."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import t

from heavily_right.combination import CombinationTest
from heavily_right.correlation import cor_ar_1, cor_fixed
from reproduction.artifacts import ArtifactStore
from reproduction.one_dimensional.common import build_band_figure, coverage_at_levels
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress
from reproduction.sampling import CachedMultivariateT

EXPERIMENT_ID = "one-dimensional/copulas"
SEED = 2025


def simulate_student(
  correlation: Callable[[int, float], np.ndarray],
  label: str,
) -> tuple[np.ndarray, np.ndarray]:
  repeat = profile_value(paper=50, smoke=2)
  num_run = profile_value(paper=10_000, smoke=200)
  num_study = profile_value(paper=500, smoke=20)
  rhos = profile_value(paper=np.arange(0, 1, 0.1), smoke=np.array([0.0, 0.6]))
  coverage = np.zeros((repeat, len(rhos), 2))
  progress = Progress(f"multivariate-t coverage ({label})", len(rhos) * repeat)
  rng = np.random.default_rng(SEED)
  tests = [CombinationTest(num_study, "HCauchy", level) for level in (0.05, 0.01)]

  for rho_index, rho in enumerate(rhos):
    distribution = CachedMultivariateT(
      loc=np.zeros(num_study),
      shape=correlation(num_study, float(rho)),
      df=10,
    )
    for replicate in range(repeat):
      estimates = distribution.rvs(
        size=num_run,
        random_state=rng,
      )
      p_values = 2 * np.maximum(t.sf(np.abs(estimates), df=10), 1e-150)
      coverage[replicate, rho_index] = coverage_at_levels(p_values, tests=tests)
      progress.update(rho_index * repeat + replicate + 1)
  return coverage, rhos


def sample_fgm(
  parameter: float,
  size: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """Sample the Farlie-Gumbel-Morgenstern copula by conditional inversion."""
  u = rng.uniform(size=size)
  probability = rng.uniform(size=size)
  coefficient = parameter * (1 - 2 * u)
  discriminant = np.maximum((1 + coefficient) ** 2 - 4 * coefficient * probability, 0)
  denominator = 1 + coefficient + np.sqrt(discriminant)
  v = np.where(np.abs(coefficient) < 1e-12, probability, 2 * probability / denominator)
  return np.column_stack((u, v))


def sample_amh(
  parameter: float,
  size: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """Sample the Ali-Mikhail-Haq copula by conditional inversion."""
  u = rng.uniform(size=size)
  probability = rng.uniform(size=size)
  if abs(parameter) < 1e-12:
    return np.column_stack((u, probability))

  a = parameter * (1 - u)
  b = 1 - a
  quadratic = probability * a**2 - parameter
  linear = 2 * probability * a * b - (1 - parameter)
  constant = probability * b**2
  discriminant = np.maximum(linear**2 - 4 * quadratic * constant, 0)
  root_1 = (-linear - np.sqrt(discriminant)) / (2 * quadratic)
  root_2 = (-linear + np.sqrt(discriminant)) / (2 * quadratic)
  linear_root = -constant / linear
  use_linear = np.abs(quadratic) < 1e-12
  root = np.where((0 <= root_1) & (root_1 <= 1), root_1, root_2)
  v = np.where(use_linear, linear_root, root)
  return np.column_stack((u, np.clip(v, 0, 1)))


def simulate_bivariate_copula(
  sampler: Callable[[float, int, np.random.Generator], np.ndarray],
  *,
  copula_name: str,
  method: str,
) -> tuple[np.ndarray, np.ndarray]:
  repeat = profile_value(paper=50, smoke=2)
  num_run = profile_value(paper=10_000, smoke=200)
  num_study = profile_value(paper=500, smoke=20)
  parameters = profile_value(
    paper=np.arange(-0.9, 1, 0.2),
    smoke=np.array([-0.6, 0.0, 0.6]),
  )
  coverage = np.zeros((repeat, len(parameters), 2))
  tests = [CombinationTest(num_study, method, level) for level in (0.05, 0.01)]
  progress = Progress(f"{copula_name} ({method})", len(parameters))
  rng = np.random.default_rng(SEED)

  for parameter_index, parameter in enumerate(parameters):
    for replicate in range(repeat):
      paired_sample = sampler(
        float(parameter),
        num_study * num_run // 2,
        rng,
      )
      p_values = paired_sample.reshape(num_run, num_study)
      coverage[replicate, parameter_index] = [
        1 - np.mean(test.make_decision(p_values)) for test in tests
      ]
    progress.update(parameter_index + 1)
  return coverage, parameters


def render(
  coverage: np.ndarray, parameters: np.ndarray, label: str, artifacts: ArtifactStore,
) -> None:
  """Render saved coverage as false-positive rates; no copula sampling."""
  figure = build_band_figure(
    coverage, parameters, xlabel=r"$\rho$" if label.startswith("student-t-") else r"$\theta$",
    ylabel="false positive rate", complement_offset=1.0,
  )
  if label.endswith("-fisher"):
    figure.axes[0].set_yticks(np.arange(0, 0.1, 0.01))
  save_pdf(figure, artifacts, f"copula-coverage-{label}.pdf")


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)

  for label, correlation in (("student-t-ar1", cor_ar_1), ("student-t-equicorrelated", cor_fixed)):
    coverage, parameters = simulate_student(correlation, label)
    np.save(artifacts.data(f"copula-coverage-{label}.npy"), coverage)
    render(coverage, parameters, label, artifacts)

  copulas = (
    ("ali-mikhail-haq", sample_amh),
    ("farlie-gumbel-morgenstern", sample_fgm),
  )
  for copula_name, factory in copulas:
    for method in ("HCauchy", "Fisher"):
      label = f"{copula_name}-{method.lower()}"
      coverage, parameters = simulate_bivariate_copula(
        factory,
        copula_name=copula_name,
        method=method,
      )
      np.save(artifacts.data(f"copula-coverage-{label}.npy"), coverage)
      render(coverage, parameters, label, artifacts)


if __name__ == "__main__":
  main()
