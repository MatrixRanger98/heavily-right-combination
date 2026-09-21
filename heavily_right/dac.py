"""Divide-and-combine inference for explicitly partitioned sample coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .calibration import covariance_of_mean
from .empty_sets import EmptySetDiagnostics, EmptySetPolicy
from .workflows import Study, StudyCollection, WorkflowFit, WorkflowInterval, fit_studies

Coordinate = int | str
Partition = tuple[tuple[int, ...], ...]


def _coordinate_names(sample: Any, names: Sequence[str] | None, dimension: int) -> tuple[str, ...]:
  if names is None:
    names = getattr(sample, "columns", None)
  if names is None:
    return tuple(f"theta{index + 1}" for index in range(dimension))
  if isinstance(names, (str, bytes)):
    raise ValueError("parameter_names must contain one unique name per sample column")
  try:
    result = tuple(names)
  except TypeError as error:
    raise ValueError("parameter_names must contain one unique name per sample column") from error
  if (
    len(result) != dimension
    or any(not isinstance(name, str) or not name for name in result)
    or len(set(result)) != dimension
  ):
    raise ValueError("parameter_names must contain one unique, nonempty name per sample column")
  return result


def _coordinate_partition(
  dimension: int,
  names: tuple[str, ...],
  blocks: Iterable[Iterable[Coordinate]] | None,
  block_size: int | None,
) -> Partition:
  if blocks is not None and block_size is not None:
    raise ValueError("supply blocks or block_size, not both")
  if blocks is None:
    size = 1 if block_size is None else block_size
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
      raise ValueError("block_size must be a positive integer")
    return tuple(tuple(range(start, min(start + int(size), dimension)))
                 for start in range(0, dimension, int(size)))
  if isinstance(blocks, (str, bytes)):
    raise ValueError("blocks must be a nonempty sequence of coordinate groups")
  try:
    groups = tuple(blocks)
  except TypeError as error:
    raise ValueError("blocks must be a nonempty sequence of coordinate groups") from error
  if not groups:
    raise ValueError("blocks must be a nonempty sequence of coordinate groups")
  positions = {name: index for index, name in enumerate(names)}
  partition = []
  for group in groups:
    if isinstance(group, (str, bytes)):
      raise ValueError("each block must be a nonempty sequence of coordinate indices or names")
    try:
      coordinates = tuple(group)
    except TypeError as error:
      raise ValueError("each block must be a nonempty sequence of coordinate indices or names") from error
    if not coordinates:
      raise ValueError("each block must be nonempty")
    indices = []
    for coordinate in coordinates:
      if isinstance(coordinate, str):
        if coordinate not in positions:
          raise ValueError(f"unknown block coordinate name: {coordinate!r}")
        index = positions[coordinate]
      elif isinstance(coordinate, Integral) and not isinstance(coordinate, (bool, np.bool_)):
        index = int(coordinate)
        if not 0 <= index < dimension:
          raise ValueError("block coordinate indices must be zero-based and within the sample columns")
      else:
        raise ValueError("block coordinates must be integer indices or parameter names")
      indices.append(index)
    partition.append(tuple(indices))
  flattened = [index for group in partition for index in group]
  if len(flattened) != dimension or sorted(flattened) != list(range(dimension)):
    raise ValueError("blocks must partition all sample columns exactly once")
  return tuple(partition)


def _prepare_dac(
  sample: ArrayLike,
  blocks: Iterable[Iterable[Coordinate]] | None,
  block_size: int | None,
  parameter_names: Sequence[str] | None,
) -> tuple[StudyCollection, int, Partition]:
  try:
    array = np.asarray(sample)
  except (TypeError, ValueError) as error:
    raise ValueError("sample must be a finite numeric observation-by-coordinate matrix") from error
  if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
    raise ValueError("sample must be a matrix with at least two observations and one coordinate")
  if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
    raise ValueError("sample must contain only finite real numeric values")
  array = np.asarray(array, dtype=float)
  sample_size, dimension = array.shape
  names = _coordinate_names(sample, parameter_names, dimension)
  partition = _coordinate_partition(dimension, names, blocks, block_size)
  identity = np.eye(dimension)
  studies = []
  for index, coordinates in enumerate(partition):
    if len(coordinates) >= sample_size:
      raise ValueError(
        f"block {index + 1} needs more observations than coordinates "
        "for positive-definite covariance and Hotelling degrees of freedom"
      )
    block = array[:, coordinates]
    studies.append(Study(
      estimate=block.mean(axis=0),
      covariance=covariance_of_mean(block),
      projection=identity[list(coordinates)],
      df=sample_size - 1,
      label=f"block{index + 1}",
    ))
  return StudyCollection(tuple(studies), names), sample_size, partition


def dac_studies(
  sample: ArrayLike,
  *,
  blocks: Iterable[Iterable[Coordinate]] | None = None,
  block_size: int | None = None,
  parameter_names: Sequence[str] | None = None,
) -> StudyCollection:
  """Construct sample-mean studies by partitioning columns, not observations.

  ``sample`` has shape ``(n, d)`` with at least two observations. Supply an
  exact partition in ``blocks`` using zero-based indices or parameter names,
  or use ``block_size`` for consecutive groups. A final smaller group is
  retained. Omitting both arguments creates one study per coordinate.

  Each block uses all observations, covariance of its mean
  ``W / [n * (n - 1)]``, and Hotelling degrees of freedom ``n - 1``.
  ``parameter_names`` defaults to sample column labels when present, or
  ``theta1``, ``theta2``, and so on. Blocks preserve their supplied order.
  Singular block covariance matrices are rejected.

  Finite combination calibration assumes independent block p-values;
  partitioning correlated columns does not establish that assumption.

  Returns a :class:`~heavily_right.workflows.StudyCollection` with one study
  per block, labeled ``block1``, ``block2``, and so on. Invalid partitions,
  nonfinite samples, insufficient observations, and singular covariance
  raise ``ValueError``. One-column samples are supported as two-dimensional
  arrays with shape ``(n, 1)``.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right.dac import dac_studies
  >>> sample = np.array([[0., 1.], [1., 0.], [2., 1.],
  ...                    [1., 2.], [0., 0.], [2., 2.]])
  >>> studies = dac_studies(sample, blocks=[["x", "y"]], parameter_names=["x", "y"])
  >>> block = studies.studies[0]
  >>> np.testing.assert_allclose(block.estimate, [1., 1.])
  >>> np.testing.assert_allclose(block.covariance, np.cov(sample, rowvar=False) / 6)
  >>> assert block.df == 5 and studies.dimension == 2
  """
  return _prepare_dac(sample, blocks, block_size, parameter_names)[0]


@dataclass(frozen=True)
class DacFit:
  """A fitted workflow retaining its sample size and coordinate partition.

  ``fit`` exposes the common workflow result. Solver controls are supplied
  to :func:`fit_dac` and reused by :meth:`support_interval`.

  Examples
  --------
  >>> from heavily_right.dac import DacFit, fit_dac
  >>> fitted = fit_dac([[0.], [1.], [2.], [1.]])
  >>> assert isinstance(fitted, DacFit)
  >>> assert fitted.sample_size == 4 and fitted.blocks == ((0,),)
  """

  fit: WorkflowFit
  sample_size: int
  blocks: Partition

  @property
  def inputs(self) -> StudyCollection:
    return self.fit.inputs

  @property
  def active_inputs(self) -> StudyCollection:
    return self.fit.active_inputs

  @property
  def parameter_names(self) -> tuple[str, ...]:
    return self.inputs.parameter_names

  @property
  def estimate(self) -> NDArray[np.float64]:
    return self.fit.estimate

  @property
  def original_indices(self) -> tuple[int, ...]:
    return self.fit.original_indices

  @property
  def weights(self) -> NDArray[np.float64]:
    return self.fit.weights

  @property
  def threshold(self) -> float:
    return self.fit.threshold

  @property
  def empty_set_diagnostics(self) -> EmptySetDiagnostics | None:
    return self.fit.empty_set_diagnostics

  def support_interval(self, direction: ArrayLike) -> WorkflowInterval:
    """Project the fitted joint mean region onto a finite nonzero direction.

    Directions follow ``parameter_names``; scalar fits also accept a scalar.
    The returned :class:`~heavily_right.workflows.WorkflowInterval` contains
    the scalar bounds, endpoint points in original parameter coordinates,
    ``success``, and ``native_result``. Check ``success`` before using bounds.
    The retained blocks and weights stay fixed for subsequent projections.

    Examples
    --------
    >>> import numpy as np
    >>> from heavily_right.dac import fit_dac
    >>> fitted = fit_dac([[0.], [1.], [2.], [1.]])
    >>> interval = fitted.support_interval(1.0)
    >>> reverse = fitted.support_interval(-1.0)
    >>> assert interval.success and interval.lower < 1.0 < interval.upper
    >>> np.testing.assert_allclose([reverse.lower, reverse.upper],
    ...                            [-interval.upper, -interval.lower])
    """
    return self.fit.support_interval(direction)

  def score(self, point: ArrayLike) -> float:
    """Evaluate the retained blocks' score at a candidate mean vector.

    Examples
    --------
    >>> from heavily_right.dac import fit_dac
    >>> fitted = fit_dac([[0.], [1.], [2.], [1.]])
    >>> assert fitted.score([1.0]) <= fitted.threshold
    """
    return self.fit.score(point)

  def contains(self, point: ArrayLike) -> bool:
    """Test whether a candidate mean vector belongs to the fitted region.

    Examples
    --------
    >>> from heavily_right.dac import fit_dac
    >>> fitted = fit_dac([[0.], [1.], [2.], [1.]])
    >>> assert fitted.contains(fitted.estimate)
    """
    return self.fit.contains(point)


def fit_dac(
  sample: ArrayLike,
  *,
  blocks: Iterable[Iterable[Coordinate]] | None = None,
  block_size: int | None = None,
  parameter_names: Sequence[str] | None = None,
  weights: ArrayLike | None = None,
  method: str = "HCauchy",
  level: float = 0.05,
  empty_set_policy: EmptySetPolicy | None = None,
  **solver_options: Any,
) -> DacFit:
  """Fit divide-and-combine inference using the shared study workflow.

  Partition arguments and covariance calibration follow :func:`dac_studies`.
  Weights follow block order. ``solver_options`` are passed to
  :func:`fit_studies` and retained for later support intervals. The returned
  fit exposes ``estimate``, ``score``, ``contains``, and ``support_interval``,
  together with the original partition and number of observations. Active
  ``weights`` and ``original_indices`` retain the shared fit's bookkeeping.

  ``method`` is ``"HCauchy"`` or ``"EHMP"``; ``level`` is the significance
  level. Weights must be positive and sum to one. An empty-set policy is
  optional and off by default; any retained set is fixed for later queries.
  Removal cannot discard a block needed to identify a mean coordinate.
  Numerical failure and unresolved emptiness raise typed errors where
  available, with diagnostics; neither yields a successful interval.

  Examples
  --------
  Retaining both coordinates in one block preserves their covariance.

  >>> import numpy as np
  >>> from heavily_right.dac import fit_dac
  >>> sample = np.array([[0., 1.], [1., 0.], [2., 1.],
  ...                    [1., 2.], [0., 0.], [2., 2.]])
  >>> fitted = fit_dac(sample, blocks=[[0, 1]], parameter_names=["x", "y"])
  >>> interval = fitted.support_interval([1.0, -1.0])
  >>> assert interval.success and interval.lower < 0.0 < interval.upper
  >>> assert fitted.contains(fitted.estimate)
  """
  inputs, sample_size, partition = _prepare_dac(sample, blocks, block_size, parameter_names)
  fit = fit_studies(
    inputs, weights=weights, method=method, level=level,
    empty_set_policy=empty_set_policy, **solver_options,
  )
  return DacFit(fit, sample_size, partition)


__all__ = ["DacFit", "dac_studies", "fit_dac"]
