"""Validated study summaries and fitted front ends over the numerical engines.

This layer owns data validation and retained-study bookkeeping, not new
statistical or optimization algorithms. It never reads or writes files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .empty_sets import (
  EmptySetDiagnostics,
  EmptySetNumericalError,
  EmptySetPolicy,
  EmptySetResolutionError,
)
from .multivariate import MetaAnalysisMD, SupportIntervalResult
from .univariate import ConfidenceIntervalResult, MetaAnalysis1D

FloatArray = NDArray[np.float64]
NativeInterval = SupportIntervalResult | ConfidenceIntervalResult


def _array(value: ArrayLike, name: str) -> FloatArray:
  raw = np.asarray(value)
  if raw.dtype.kind not in "iuf":
    raise ValueError(f"{name} must contain real numeric values")
  result = np.array(raw, dtype=float, copy=True)
  if not np.isfinite(result).all():
    raise ValueError(f"{name} must contain finite values")
  result.setflags(write=False)
  return result


@dataclass(frozen=True, eq=False)
class Study:
  """One estimate of ``projection @ theta`` and its covariance of estimation.

  ``covariance`` is not an individual-observation covariance. ``df=None``
  selects known/asymptotic covariance; finite df selects Hotelling calibration.
  Inputs are defensively copied. Scalar summaries may use scalar arrays.
  """

  estimate: ArrayLike
  covariance: ArrayLike
  projection: ArrayLike
  df: float | None = None
  label: str | None = None

  def __post_init__(self) -> None:
    estimate = _array(self.estimate, "estimate")
    covariance = _array(self.covariance, "covariance")
    projection = _array(self.projection, "projection")
    if estimate.ndim == 0:
      estimate = estimate.reshape(1)
    if covariance.ndim == 0:
      covariance = covariance.reshape(1, 1)
    if projection.ndim == 1:
      projection = projection.reshape(1, -1)
    if estimate.ndim != 1 or not estimate.size:
      raise ValueError("estimate must be a nonempty vector")
    q = estimate.size
    if covariance.shape != (q, q):
      raise ValueError("covariance shape must match the estimate dimension")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if np.max(np.abs(covariance - covariance.T)) > 1e-12 * scale:
      raise ValueError("covariance must be symmetric")
    covariance = _array(covariance / 2 + covariance.T / 2, "covariance")
    try:
      np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
      raise ValueError("covariance must be positive definite") from error
    if projection.ndim != 2 or projection.shape[0] != q or not projection.shape[1]:
      raise ValueError("projection must have one row per estimate and at least one column")
    if not np.any(projection):
      raise ValueError("projection must measure at least one parameter")
    if self.df is not None:
      if isinstance(self.df, (bool, np.bool_)) or not np.isscalar(self.df):
        raise ValueError("df must be a finite scalar greater than q - 1")
      df = float(self.df)
      if not np.isfinite(df) or df <= q - 1:
        raise ValueError("df must be a finite scalar greater than q - 1")
      object.__setattr__(self, "df", df)
    if self.label is not None and (not isinstance(self.label, str) or not self.label.strip()):
      raise ValueError("label must be a nonempty string or None")
    object.__setattr__(self, "estimate", estimate)
    object.__setattr__(self, "covariance", covariance)
    object.__setattr__(self, "projection", projection)


@dataclass(frozen=True, eq=False)
class StudyCollection:
  """An identifiable set of study blocks with named common parameters."""

  studies: tuple[Study, ...]
  parameter_names: tuple[str, ...]

  def __post_init__(self) -> None:
    if isinstance(self.parameter_names, (str, bytes)):
      raise ValueError("parameter_names must be a sequence of nonempty strings")
    studies = tuple(self.studies)
    names = tuple(self.parameter_names)
    if not studies or any(not isinstance(study, Study) for study in studies):
      raise ValueError("studies must be a nonempty collection of Study objects")
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
      raise ValueError("parameter_names must be nonempty strings")
    if len(set(names)) != len(names):
      raise ValueError("parameter_names must be unique")
    if any(study.projection.shape[1] != len(names) for study in studies):
      raise ValueError("study projections must match parameter_names")
    studies = tuple(
      study if study.label is not None else replace(study, label=f"study-{index + 1}")
      for index, study in enumerate(studies)
    )
    if len({study.label for study in studies}) != len(studies):
      raise ValueError("study labels must be unique")
    design = np.vstack([study.projection for study in studies])
    # Normalize columns without squaring huge entries; this is an
    # identifiability check, not a guarantee of numerical conditioning.
    column_scale = np.max(np.abs(design), axis=0)
    if np.any(column_scale == 0) or np.linalg.matrix_rank(design / column_scale) < len(names):
      raise ValueError("study projections do not identify all parameters")
    object.__setattr__(self, "studies", studies)
    object.__setattr__(self, "parameter_names", names)

  @property
  def dimension(self) -> int:
    return len(self.parameter_names)


@dataclass(frozen=True, eq=False)
class WorkflowInterval:
  """A named-workflow projection with the original engine result retained.

  Endpoint points always remain in parameter coordinates, even when the
  scalar bounds refer to a contrast or a scaled direction.
  """

  lower: float
  upper: float
  lower_point: FloatArray
  upper_point: FloatArray
  native_result: NativeInterval

  @property
  def success(self) -> bool:
    finite = bool(
      np.isfinite([self.lower, self.upper]).all()
      and np.isfinite(self.lower_point).all() and np.isfinite(self.upper_point).all()
      and self.lower <= self.upper
    )
    if isinstance(self.native_result, SupportIntervalResult):
      return finite and self.native_result.success and bool(
        np.isfinite(self.native_result.minimum_point).all()
      )
    return finite and self.native_result.empty_set_diagnostics.resolved

  @property
  def empty_set_diagnostics(self) -> EmptySetDiagnostics | None:
    return self.native_result.empty_set_diagnostics


class WorkflowFitError(EmptySetNumericalError):
  """Initial fit projection failed; its complete native result is retained."""

  def __init__(self, result: NativeInterval) -> None:
    super().__init__(
      "initial workflow support interval was not certified; inspect error.result",
      result.empty_set_diagnostics,
    )
    self.result = result


class WorkflowQueryError(EmptySetNumericalError):
  """A later direction failed, without changing the fitted retained studies.

  ``diagnostics`` retains the original intervention trace; ``query_diagnostics``
  describes the failed solve on the active studies. Original and local indices
  therefore remain explicitly distinct.
  """

  def __init__(
    self, direction: FloatArray, error: EmptySetNumericalError,
    diagnostics: EmptySetDiagnostics | None,
  ) -> None:
    super().__init__(
      "support query failed; retained study set is unchanged: " + str(error),
      diagnostics if diagnostics is not None else error.diagnostics,
    )
    self.direction = _array(direction, "direction")
    self.query_diagnostics = error.diagnostics


def _engine_inputs(inputs: StudyCollection) -> dict[str, Any]:
  studies = inputs.studies
  dfs = [study.df for study in studies]
  if any(value is None for value in dfs) and not all(value is None for value in dfs):
    raise ValueError("mixing known covariance (df=None) and finite-df studies is not yet supported")
  degrees = None if dfs[0] is None else np.array(dfs, dtype=float)
  if inputs.dimension == 1:
    if any(study.estimate.size != 1 or study.projection[0, 0] == 0 for study in studies):
      raise ValueError("one-parameter workflows currently require scalar study summaries")
    factors = np.array([study.projection[0, 0] for study in studies])
    return {
      "theta_hat": np.array([study.estimate[0] for study in studies]) / factors,
      "sigma": np.sqrt([study.covariance[0, 0] for study in studies]) / np.abs(factors),
      "df": degrees,
    }
  sizes = [study.estimate.size for study in studies]
  offsets = np.r_[0, np.cumsum(sizes)]
  return {
    "xi_hat": [study.estimate for study in studies],
    "Sigma": [study.covariance for study in studies],
    "sub_dim": [np.arange(offsets[j], offsets[j + 1]) for j in range(len(studies))],
    "projs": np.vstack([study.projection for study in studies]),
    "df": degrees,
  }


def _output_snapshot(value: ArrayLike) -> FloatArray:
  # Failed native results may legitimately contain NaN/Inf diagnostics.
  # Preserve those for inspection; success checks, not input validation,
  # decide whether a returned interval can be used.
  snapshot = np.array(value, dtype=float, copy=True)
  snapshot.setflags(write=False)
  return snapshot


def _wrap_interval(result: NativeInterval, direction: FloatArray) -> WorkflowInterval:
  if isinstance(result, SupportIntervalResult):
    # Native result dataclasses are frozen but their NumPy members are not.
    # Never expose arrays owned by the cached initial solve to caller mutation.
    result = replace(
      result, lower_point=_output_snapshot(result.lower_point),
      upper_point=_output_snapshot(result.upper_point),
      minimum_point=_output_snapshot(result.minimum_point),
    )
    return WorkflowInterval(
      result.lower, result.upper, _output_snapshot(result.lower_point),
      _output_snapshot(result.upper_point), result,
    )
  first, second = (result.lower, result.upper) if direction[0] > 0 else (result.upper, result.lower)
  with np.errstate(over="ignore", invalid="ignore"):
    lower, upper = float(direction[0] * first), float(direction[0] * second)
  return WorkflowInterval(
    lower, upper, _output_snapshot([first]), _output_snapshot([second]), result,
  )


@dataclass(frozen=True, eq=False)
class WorkflowFit:
  """A fitted region whose retained study set is fixed for every later contrast.

  Fitting certifies the first coordinate's support interval and resolves any
  requested empty-set intervention once. Other directions still receive full
  endpoint checks; no additional studies are removed during later queries.
  """

  inputs: StudyCollection
  active_inputs: StudyCollection
  estimate: FloatArray
  original_indices: tuple[int, ...]
  empty_set_diagnostics: EmptySetDiagnostics | None
  _analysis: MetaAnalysis1D | MetaAnalysisMD
  _initial_result: NativeInterval
  _solver_options: Mapping[str, Any]

  @property
  def parameter_names(self) -> tuple[str, ...]:
    return self.inputs.parameter_names

  @property
  def weights(self) -> FloatArray:
    """Normalized weights of the retained study blocks."""
    return self._analysis.weights.copy()

  @property
  def threshold(self) -> float:
    return float(self._analysis.threshold)

  def score(self, point: ArrayLike) -> float:
    """Evaluate the fitted (possibly adjusted) region's original score."""
    value = _array(point, "point")
    if value.ndim == 0 and self.inputs.dimension == 1:
      value = value.reshape(1)
    if value.shape != (self.inputs.dimension,):
      raise ValueError("point must match the parameter dimension")
    argument = float(value[0]) if self.inputs.dimension == 1 else value
    pvalues = self._analysis.get_p_vector(argument, **_engine_inputs(self.active_inputs))
    return float(self._analysis.get_global_score(pvalues))

  def contains(self, point: ArrayLike) -> bool:
    return self.score(point) <= self.threshold

  def support_interval(self, direction: ArrayLike) -> WorkflowInterval:
    """Project onto a direction; always inspect ``result.success``."""
    vector = _array(direction, "direction")
    if vector.ndim == 0 and self.inputs.dimension == 1:
      vector = vector.reshape(1)
    if vector.shape != (self.inputs.dimension,) or not np.any(vector):
      raise ValueError("direction must be nonzero and match the parameter dimension")
    initial_direction = np.eye(1, self.inputs.dimension)[0]
    if self.inputs.dimension == 1 or np.array_equal(vector, initial_direction):
      return _wrap_interval(self._initial_result, vector)
    assert isinstance(self._analysis, MetaAnalysisMD)
    options = dict(self._solver_options)
    options.update(x0=self.estimate, find_min=False)
    try:
      result = self._analysis.support_interval(
        vector, **_engine_inputs(self.active_inputs), **options,
        study_labels=[study.label for study in self.active_inputs.studies],
      )
    except EmptySetNumericalError as error:
      raise WorkflowQueryError(vector, error, self.empty_set_diagnostics) from error
    result = replace(result, empty_set_diagnostics=self.empty_set_diagnostics)
    return _wrap_interval(result, vector)


def fit_studies(
  inputs: StudyCollection,
  *,
  weights: Sequence[float] | None = None,
  method: str = "HCauchy",
  level: float = 0.05,
  empty_set_policy: EmptySetPolicy | None = None,
  **solver_options: Any,
) -> WorkflowFit:
  """Fit supplied summaries, preserving native calibration and diagnostics.

  Weights must be positive and sum to one; they are weights of study blocks,
  not individual contrast rows. MD controls are ``ftol``, ``maxiter``,
  ``newton_maxiter``, ``dense_newton_max_dimension``, and ``x0``. Scalar fits
  use the validated scalar root solver and accept no MD controls.

  Examples
  --------
  >>> from heavily_right import Study, StudyCollection, fit_studies
  >>> inputs = StudyCollection(
  ...     (Study([0.2], [[0.04]], [[1.]], label="study A"),), ("effect",),
  ... )
  >>> fitted = fit_studies(inputs)
  >>> interval = fitted.support_interval([1.])
  >>> assert interval.success and fitted.contains(fitted.estimate)
  >>> assert fitted.original_indices == (0,)
  """
  if not isinstance(inputs, StudyCollection):
    raise TypeError("inputs must be a StudyCollection")
  if method not in {"HCauchy", "EHMP"}:
    raise ValueError("general confidence-region workflows support HCauchy or EHMP")
  allowed = {"ftol", "maxiter", "newton_maxiter", "dense_newton_max_dimension", "x0"}
  if set(solver_options) - allowed:
    raise ValueError(f"unknown solver options: {sorted(set(solver_options) - allowed)}")
  if inputs.dimension == 1 and solver_options:
    raise ValueError("multidimensional solver options do not apply to scalar fits")
  options = dict(solver_options)
  if "x0" in options:
    options["x0"] = _array(options["x0"], "x0")
  arguments = _engine_inputs(inputs)
  count = len(inputs.studies)
  arg = count if weights is None else _array(weights, "weights")
  if weights is not None and arg.shape != (count,):
    raise ValueError("weights must contain one value per study block")
  labels = [study.label for study in inputs.studies]
  if inputs.dimension == 1:
    analysis = MetaAnalysis1D(arg, method=method, level=level)
    result = analysis.confidence_interval_result(
      **arguments, empty_set_policy=empty_set_policy, study_labels=labels,
    )
    if not result.empty_set_diagnostics.resolved:
      raise EmptySetResolutionError(result.empty_set_diagnostics)
    estimate = np.array([result.estimate])
  else:
    analysis = MetaAnalysisMD(arg, dim=inputs.dimension, method=method, level=level)
    result = analysis.support_interval(
      np.eye(1, inputs.dimension)[0], **arguments,
      empty_set_policy=empty_set_policy, study_labels=labels, **options,
    )
    estimate = result.minimum_point
  if not _wrap_interval(result, np.eye(1, inputs.dimension)[0]).success:
    raise WorkflowFitError(result)
  diagnostics = result.empty_set_diagnostics
  retained = tuple(range(count)) if diagnostics is None else diagnostics.final_original_indices
  active_inputs = StudyCollection(tuple(inputs.studies[j] for j in retained), inputs.parameter_names)
  if len(retained) != count:
    retained_weights = analysis.weights[list(retained)].copy()
    retained_weights /= retained_weights.sum()
    analysis = (
      MetaAnalysis1D(retained_weights, method=method, level=level)
      if inputs.dimension == 1 else
      MetaAnalysisMD(retained_weights, dim=inputs.dimension, method=method, level=level)
    )
  fit = WorkflowFit(
    inputs, active_inputs, _array(estimate, "estimate"), retained, diagnostics,
    analysis, result, MappingProxyType(options),
  )
  # Do not let a certified internal minimum silently reconstruct outside the
  # public confidence set (large translations can lose input-unit precision).
  if not fit.contains(fit.estimate):
    raise EmptySetNumericalError(
      "the fitted estimate is not feasible in input coordinates; recenter/rescale the data",
      diagnostics,
    )
  return fit


__all__ = [
  "Study", "StudyCollection", "WorkflowFit", "WorkflowFitError", "WorkflowInterval",
  "WorkflowQueryError", "fit_studies",
]
