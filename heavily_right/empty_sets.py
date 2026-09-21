"""Opt-in diagnostics and policies for empty confidence sets."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
  from .minimum import MinimumDiagnostics

EmptySetStrategy = Literal["remove-smallest-p"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmptySetPolicy:
  """Configure an explicit attempt to repair an empty confidence set.

  The only current strategy removes the study with the smallest individual
  p-value at the global-score minimizer, refits, and repeats up to
  ``max_removals`` times. Passing no policy keeps handling off.

  Examples
  --------
  >>> from heavily_right import EmptySetPolicy, MetaAnalysis1D
  >>> result = MetaAnalysis1D(2).confidence_interval_result(
  ...     [-3., 3.], [.5, .5], study_labels=["left", "right"],
  ...     empty_set_policy=EmptySetPolicy(max_removals=1, warn=False),
  ... )
  >>> trace = result.empty_set_diagnostics
  >>> assert trace.resolved and len(trace.removals) == 1
  >>> assert trace.removals[0].study_label in {"left", "right"}
  >>> assert trace.to_dict()["final_state"] == "nonempty"

  Data-dependent removal changes the procedure; the original nominal coverage
  guarantee does not automatically transfer to the selected-study region.
  """

  strategy: EmptySetStrategy = "remove-smallest-p"
  max_removals: int = 1
  min_remaining: int = 1
  warn: bool = True

  def __post_init__(self) -> None:
    if self.strategy != "remove-smallest-p":
      raise ValueError("unsupported empty-set strategy")
    for name in ("max_removals", "min_remaining"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if not isinstance(self.warn, bool):
      raise ValueError("warn must be a boolean")


@dataclass(frozen=True)
class CandidateEvaluation:
  """An evaluated deletion, including candidates rejected for lost rank."""

  original_index: int
  study_label: str | None
  individual_pvalue: float
  priority: int
  eligible: bool
  selected: bool
  reason: str
  retained_projection_rank: int | None = None


@dataclass(frozen=True)
class EmptySetEvaluation:
  """One fitted configuration, before any next deletion is performed."""

  step: int
  original_indices: tuple[int, ...]
  weights: tuple[float, ...]
  individual_pvalues: tuple[float, ...] | None
  minimum_global_score: float | None
  threshold: float | None
  global_minimizer: tuple[float, ...] | None
  empty: bool | None
  candidates: tuple[CandidateEvaluation, ...] = ()
  failure_message: str | None = None
  minimum_diagnostics: MinimumDiagnostics | None = None


@dataclass(frozen=True)
class StudyRemoval:
  """One study removed while attempting to make a confidence set nonempty."""

  step: int
  original_index: int
  study_label: str | None
  individual_pvalue: float
  minimum_global_score: float
  threshold: float
  global_minimizer: tuple[float, ...]
  remaining_original_indices: tuple[int, ...]
  remaining_weights: tuple[float, ...] = ()

  @property
  def description(self) -> str:
    label = "" if self.study_label is None else f" ({self.study_label!r})"
    location = _format_point(self.global_minimizer)
    return (
      f"removed study at original zero-based index {self.original_index}{label}; "
      f"individual p-value={self.individual_pvalue:.6g} at the global "
      f"minimizer {location}; minimum score={self.minimum_global_score:.6g}, "
      f"threshold={self.threshold:.6g}"
    )


@dataclass(frozen=True)
class EmptySetDiagnostics:
  """Structured record of empty-set detection and any study removals."""

  enabled: bool
  encountered: bool
  resolved: bool
  strategy: EmptySetStrategy | None
  original_study_count: int
  final_original_indices: tuple[int, ...]
  removals: tuple[StudyRemoval, ...]
  final_minimum_score: float | None
  final_threshold: float | None
  last_global_minimizer: tuple[float, ...] | None
  message: str
  policy: EmptySetPolicy | None = None
  method: str = ""
  significance_level: float | None = None
  parameter_dimension: int | None = None
  original_weights: tuple[float, ...] = ()
  final_weights: tuple[float, ...] = ()
  study_labels: tuple[str | None, ...] = ()
  evaluations: tuple[EmptySetEvaluation, ...] = ()
  schema_version: int = 1
  final_state: Literal["nonempty", "empty", "unknown"] = "unknown"
  failure_type: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Return a JSON-compatible record; nonfinite numbers become JSON null."""
    return _json_safe(asdict(self))

  def to_json(self, *, indent: int = 2) -> str:
    """Serialize the complete trace with strict JSON (no NaN/Infinity)."""
    return json.dumps(self.to_dict(), indent=indent, allow_nan=False) + "\n"

  def save_json(self, path: str | Path) -> Path:
    """Persist diagnostics exclusively, refusing to overwrite any existing file.

    The caller chooses and creates the output directory, for example using
    ``ArtifactStore.diagnostic``. The inference routines never write files.
    """
    target = Path(path)
    payload = self.to_json()
    with target.open("x", encoding="utf-8") as stream:
      stream.write(payload)
    return target


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, (float, np.floating)):
    return float(value) if np.isfinite(value) else None
  if isinstance(value, np.integer):
    return int(value)
  return value


class EmptySetNumericalError(RuntimeError):
  """A numerical failure that does not justify classifying a set as empty."""

  def __init__(
    self, message: str, diagnostics: EmptySetDiagnostics | None = None,
  ) -> None:
    super().__init__(message)
    self.diagnostics = diagnostics


def numerical_failure_diagnostics(
  *,
  error: Exception,
  policy: EmptySetPolicy | None,
  original_weights: Sequence[float],
  active_indices: Sequence[int],
  active_weights: Sequence[float],
  labels: tuple[str | None, ...],
  method: str,
  level: float,
  dimension: int,
  threshold: float | None,
  removals: Sequence[StudyRemoval],
  evaluations: Sequence[EmptySetEvaluation],
) -> EmptySetDiagnostics:
  """Preserve prior actions and explicitly mark an unclassified failed stage."""
  original = tuple(float(value) for value in original_weights)
  retained = tuple(int(value) for value in active_indices)
  weights = tuple(float(value) for value in active_weights)
  failed_stage = EmptySetEvaluation(
    step=len(removals),
    original_indices=retained,
    weights=weights,
    individual_pvalues=None,
    minimum_global_score=None,
    threshold=threshold,
    global_minimizer=None,
    empty=None,
    failure_message=str(error),
  )
  return EmptySetDiagnostics(
    enabled=policy is not None,
    encountered=any(evaluation.empty is True for evaluation in evaluations),
    resolved=False,
    strategy=None if policy is None else policy.strategy,
    original_study_count=len(original),
    final_original_indices=retained,
    removals=tuple(removals),
    final_minimum_score=None,
    final_threshold=threshold,
    last_global_minimizer=None,
    message=(
      "numerical failure; the final confidence set was not classified; "
      f"{len(removals)} earlier removal(s) retained in the trace: {error}"
    ),
    policy=policy,
    method=method,
    significance_level=float(level),
    parameter_dimension=dimension,
    original_weights=original,
    final_weights=weights,
    study_labels=labels,
    evaluations=(*evaluations, failed_stage),
    final_state="unknown",
    failure_type=type(error).__name__,
  )


class EmptySetAdjustmentWarning(UserWarning):
  """Warning emitted for each opt-in study removal."""


class EmptySetResolutionError(RuntimeError):
  """Raised when an enabled policy cannot resolve an empty confidence set."""

  def __init__(self, diagnostics: EmptySetDiagnostics) -> None:
    super().__init__(diagnostics.message)
    self.diagnostics = diagnostics


class EmptyRegionError(RuntimeError):
  """Internal signal carrying the evidence for an empty region."""

  def __init__(
    self,
    *,
    point: FloatArray,
    p_values: FloatArray,
    minimum_score: float,
    threshold: float,
    minimum_diagnostics: MinimumDiagnostics | None = None,
  ) -> None:
    super().__init__("the confidence region is empty")
    self.point = point
    self.p_values = p_values
    self.minimum_score = minimum_score
    self.threshold = threshold
    self.minimum_diagnostics = minimum_diagnostics


def _format_point(point: tuple[float, ...]) -> str:
  """Format an optimizer location without flooding high-dimensional logs."""
  if len(point) <= 6:
    coordinates = ", ".join(f"{value:.6g}" for value in point)
    return f"[{coordinates}]"
  start = ", ".join(f"{value:.6g}" for value in point[:3])
  end = ", ".join(f"{value:.6g}" for value in point[-2:])
  return f"[{start}, ..., {end}] (dimension {len(point)})"


def validated_study_labels(
  labels: Sequence[str] | None,
  count: int,
) -> tuple[str | None, ...]:
  """Return one optional reporting label for each original study."""
  if labels is None:
    return (None,) * count
  if isinstance(labels, (str, bytes)):
    raise ValueError("study_labels must be a sequence of labels, not a string")
  result = tuple(str(label) for label in labels)
  if len(result) != count:
    raise ValueError("study_labels must match the original number of studies")
  return result


def warn_about_removal(removal: StudyRemoval, policy: EmptySetPolicy) -> None:
  """Emit the default visible report for an enabled empty-set intervention."""
  if policy.warn:
    warnings.warn(
      f"empty confidence set: {removal.description}",
      EmptySetAdjustmentWarning,
      stacklevel=3,
    )


__all__ = [
  "CandidateEvaluation",
  "EmptySetAdjustmentWarning",
  "EmptySetDiagnostics",
  "EmptySetEvaluation",
  "EmptySetNumericalError",
  "EmptySetPolicy",
  "EmptySetResolutionError",
  "EmptySetStrategy",
  "StudyRemoval",
]
