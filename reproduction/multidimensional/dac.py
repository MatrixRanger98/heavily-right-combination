"""Shared divide-and-combine slice, width, and coverage experiments."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from heavily_right.calibration import covariance_of_mean, standard_error_of_mean
from heavily_right.correlation import cor_fixed
from heavily_right.meta_analysis import MetaAnalysisMD
from reproduction.artifacts import ArtifactStore
from reproduction.plotting import save_pdf
from reproduction.profiles import profile_value
from reproduction.progress import Progress

COLORS = ("darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki")


@dataclass(frozen=True)
class DacConfig:
  experiment_id: str
  slug: str
  distribution: str
  center: str
  seed: int
  coverage_levels: tuple[float, ...] = (0.05,)


@dataclass(frozen=True)
class BlockEstimates:
  mean: np.ndarray
  covariance: np.ndarray
  num_study: int
  block_size: int


def experiment_settings() -> tuple[int, float, int, tuple[int, ...], tuple[tuple[int, int], ...]]:
  dimension = profile_value(paper=100, smoke=10)
  return (
    dimension,
    0.6,
    profile_value(paper=1000, smoke=50),
    profile_value(paper=(1, 5, 25, 100), smoke=(1, 5, 10)),
    profile_value(paper=((0, 50), (0, 1)), smoke=((0, 5), (0, 1))),
  )


def generate_sample(
  config: DacConfig,
  *,
  dimension: int,
  rho: float,
  num_sample: int,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  covariance = cor_fixed(dimension, rho)
  normal_sample = multivariate_normal.rvs(
    mean=np.zeros(dimension),
    cov=covariance,
    size=num_sample,
    random_state=rng,
  )
  if config.distribution == "normal":
    return normal_sample, np.zeros(dimension), covariance
  if config.distribution == "lognormal":
    return np.exp(normal_sample), np.exp(np.diag(covariance) / 2), covariance
  raise ValueError(f"unknown DAC distribution: {config.distribution}")


def estimate_blocks(sample: np.ndarray, block_size: int) -> BlockEstimates:
  dimension = sample.shape[1]
  if dimension % block_size:
    raise ValueError("slice block sizes must divide the parameter dimension")
  num_study = dimension // block_size
  if block_size == 1:
    return BlockEstimates(
      mean=sample.mean(axis=0),
      covariance=standard_error_of_mean(sample) ** 2,
      num_study=num_study,
      block_size=block_size,
    )
  blocked = sample.reshape(sample.shape[0], num_study, block_size)
  return BlockEstimates(
    mean=blocked.mean(axis=0),
    covariance=covariance_of_mean(blocked),
    num_study=num_study,
    block_size=block_size,
  )


def find_estimate(
  analysis: MetaAnalysisMD,
  estimates: BlockEstimates,
  *,
  degrees_freedom: int,
  initial: np.ndarray,
) -> np.ndarray:
  options = {"x0": initial} if estimates.block_size >= 25 else {}
  return analysis.find_minimizer(
    xi_hat=estimates.mean,
    Sigma=estimates.covariance,
    sub_dim=estimates.block_size,
    df=degrees_freedom,
    projs=None,
    **options,
  )


def build_slice_figure(
  config: DacConfig,
  *,
  analysis: MetaAnalysisMD,
  estimates: BlockEstimates,
  point_estimate: np.ndarray,
  true_mean: np.ndarray,
  coordinates: tuple[int, int],
  degrees_freedom: int,
) -> plt.Figure:
  dimension = len(point_estimate)
  first, second = coordinates
  direction_1 = np.zeros(dimension)
  direction_2 = np.zeros(dimension)
  direction_1[first] = 1
  direction_2[second] = 1
  center = point_estimate if config.center == "estimate" else true_mean
  x_range = (-0.305, 0.305) if config.distribution == "normal" else (0.97, 2.22)
  y_range = (-0.22, 0.22) if config.distribution == "normal" else (1.17, 2.12)

  figure, axis = plt.subplots()
  analysis.area_slice(
    point=center,
    direction_1=direction_1,
    direction_2=direction_2,
    xi_hat=estimates.mean,
    Sigma=estimates.covariance,
    sub_dim=estimates.block_size,
    df=degrees_freedom,
    projs=None,
    xrange=x_range,
    yrange=y_range,
    ax=axis,
    colors=COLORS,
  )
  estimate_label = "Estimate" if config.center == "estimate" else "Projected Estimate"
  truth_label = "Projected True Value" if config.center == "estimate" else "True Value"
  axis.scatter(
    [point_estimate[first]],
    [point_estimate[second]],
    s=30,
    color="blue",
    marker="^",
    label=estimate_label,
  )
  axis.scatter(
    [true_mean[first]],
    [true_mean[second]],
    s=50,
    color="red",
    marker="*",
    label=truth_label,
  )
  axis.legend()
  return figure


def pad_and_block(sample: np.ndarray, block_size: int) -> np.ndarray:
  dimension = sample.shape[1]
  num_study = int(np.ceil(dimension / block_size))
  extra = block_size * num_study - dimension
  padded = np.concatenate((sample, sample[:, :extra]), axis=1) if extra else sample
  return padded.reshape(sample.shape[0], num_study, block_size)


def direction_widths(
  sample: np.ndarray,
  *,
  degrees_freedom: int,
) -> tuple[np.ndarray, np.ndarray]:
  dimension = sample.shape[1]
  block_sizes = profile_value(paper=np.arange(2, 26), smoke=np.arange(2, 6))
  widths = np.zeros(len(block_sizes))
  direction = np.zeros(dimension)
  direction[:2] = 1 / np.sqrt(2)
  progress = Progress("DAC direction widths", len(block_sizes))

  for index, block_size in enumerate(block_sizes):
    blocked = pad_and_block(sample, int(block_size))
    num_study = blocked.shape[1]
    analysis = MetaAnalysisMD(num_study, dim=dimension, method="HCauchy", level=0.05)
    mean = blocked.mean(axis=0)
    covariance = covariance_of_mean(blocked)
    point_estimate = analysis.find_minimizer(
      xi_hat=mean,
      Sigma=covariance,
      sub_dim=int(block_size),
      df=degrees_freedom,
      projs=None,
      x0=sample.mean(axis=0),
    )
    lower, upper, *_ = analysis.simultaneous_interval(
      direction=direction,
      xi_hat=mean,
      Sigma=covariance,
      sub_dim=int(block_size),
      df=degrees_freedom,
      projs=None,
      x0=point_estimate,
      method="Powell",
      find_min=False,
    )
    widths[index] = upper - lower
    progress.update(index + 1)
  return np.asarray(block_sizes), widths


def coverage_rates(
  config: DacConfig,
  *,
  dimension: int,
  rho: float,
  num_sample: int,
  block_sizes: tuple[int, ...],
  rng: np.random.Generator,
) -> np.ndarray:
  total = profile_value(paper=2000, smoke=5)
  counts = np.zeros((len(config.coverage_levels), len(block_sizes)))
  analyses = {
    (level, block_size): MetaAnalysisMD(
      dimension // block_size,
      dim=dimension,
      method="HCauchy",
      level=level,
    )
    for level in config.coverage_levels
    for block_size in block_sizes
  }
  progress = Progress("DAC coverage replicates", total)

  for replicate in range(total):
    sample, true_mean, _ = generate_sample(
      config,
      dimension=dimension,
      rho=rho,
      num_sample=num_sample,
      rng=rng,
    )
    for block_index, block_size in enumerate(block_sizes):
      estimates = estimate_blocks(sample, block_size)
      for level_index, level in enumerate(config.coverage_levels):
        analysis = analyses[level, block_size]
        p_values = analysis.get_p_vector(
          true_mean,
          estimates.mean,
          estimates.covariance,
          block_size,
          num_sample - 1,
        )
        counts[level_index, block_index] += (
          analysis.get_global_score(p_values) <= analysis.threshold
        )
    if replicate == 0 or (replicate + 1) % max(1, total // 20) == 0:
      progress.update(replicate + 1)
  return counts / total


def run_slices(config: DacConfig) -> None:
  np.random.seed(config.seed)
  rng = np.random.default_rng(config.seed)
  artifacts = ArtifactStore.for_experiment(config.experiment_id)
  dimension, rho, num_sample, block_sizes, coordinate_pairs = experiment_settings()
  sample, true_mean, _ = generate_sample(
    config,
    dimension=dimension,
    rho=rho,
    num_sample=num_sample,
    rng=rng,
  )
  progress = Progress("DAC slice panels", len(block_sizes) * len(coordinate_pairs))
  completed = 0
  for block_size in block_sizes:
    estimates = estimate_blocks(sample, block_size)
    analysis = MetaAnalysisMD(
      estimates.num_study,
      dim=dimension,
      method="HCauchy",
      level=0.05,
    )
    point_estimate = find_estimate(
      analysis,
      estimates,
      degrees_freedom=num_sample - 1,
      initial=sample.mean(axis=0),
    )
    for coordinates in coordinate_pairs:
      save_pdf(
        build_slice_figure(
          config,
          analysis=analysis,
          estimates=estimates,
          point_estimate=point_estimate,
          true_mean=true_mean,
          coordinates=coordinates,
          degrees_freedom=num_sample - 1,
        ),
        artifacts,
        (
          f"{config.slug}-slice-block-{block_size}-coordinates-"
          f"{coordinates[0] + 1}-{coordinates[1] + 1}.pdf"
        ),
      )
      completed += 1
      progress.update(completed)



def run_diagnostics(config: DacConfig) -> None:
  np.random.seed(config.seed)
  rng = np.random.default_rng(config.seed)
  artifacts = ArtifactStore.for_experiment(config.experiment_id)
  dimension, rho, num_sample, block_sizes, _ = experiment_settings()
  sample, _, _ = generate_sample(
    config,
    dimension=dimension,
    rho=rho,
    num_sample=num_sample,
    rng=rng,
  )
  width_blocks, widths = direction_widths(sample, degrees_freedom=num_sample - 1)
  np.savez(
    artifacts.data(f"{config.slug}-direction-widths.npz"),
    block_sizes=width_blocks,
    widths=widths,
  )
  rates = coverage_rates(
    config,
    dimension=dimension,
    rho=rho,
    num_sample=num_sample,
    block_sizes=block_sizes,
    rng=rng,
  )
  np.savez(
    artifacts.data(f"{config.slug}-coverage.npz"),
    levels=np.asarray(config.coverage_levels),
    block_sizes=np.asarray(block_sizes),
    coverage=rates,
  )


__all__ = ["DacConfig", "estimate_blocks", "run_diagnostics", "run_slices"]
