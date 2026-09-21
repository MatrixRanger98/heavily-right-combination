"""Opt-in empty-region resolution for multidimensional inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .empty_sets import (
  CandidateEvaluation,
  EmptyRegionError,
  EmptySetDiagnostics,
  EmptySetEvaluation,
  EmptySetNumericalError,
  EmptySetPolicy,
  EmptySetResolutionError,
  StudyRemoval,
  numerical_failure_diagnostics,
  validated_study_labels,
  warn_about_removal,
)

if TYPE_CHECKING:
  from .multivariate import MetaAnalysisMD, SupportIntervalResult


def _take_studies(
  values: Any,
  keep: np.ndarray,
  *,
  shared_first_axis: bool = False,
) -> Any:
  if values is None:
    return None
  if np.isscalar(values):
    return np.full(int(np.sum(keep)), values)
  if isinstance(values, np.ndarray):
    if shared_first_axis and values.shape[0] == 1:
      return values
    return values[keep]
  return [value for value, retain in zip(values, keep, strict=True) if retain]


def _retained_rank(
  sub_dimensions: Sequence[np.ndarray] | np.ndarray,
  projections: np.ndarray,
  keep: np.ndarray,
) -> int:
  retained = _take_studies(sub_dimensions, keep)
  if isinstance(retained, np.ndarray):
    projection_rows = retained.reshape(-1)
  else:
    projection_rows = np.concatenate(retained) if retained else np.array([], dtype=int)
  if projection_rows.size == 0:
    return 0
  return int(np.linalg.matrix_rank(projections[np.asarray(projection_rows, dtype=int)]))


class MultivariateEmptySetMixin:
  """Resolve empty regions only when an explicit policy is supplied."""

  def support_interval(
    self: MetaAnalysisMD,
    direction: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: float | Sequence[float | np.ndarray] | np.ndarray,
    df: float | Sequence[float] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    x0: np.ndarray | None = None,
    find_min: bool = True,
    method: str = "Powell",
    ftol: float = 1.0e-9,
    maxiter: int = 1000,
    newton_maxiter: int = 80,
    dense_newton_max_dimension: int = 250,
    empty_set_policy: EmptySetPolicy | None = None,
    study_labels: Sequence[str] | None = None,
    **kwargs: Any,
  ) -> SupportIntervalResult:
    """Project the region, optionally resolving emptiness by study removal.

    Handling is off when ``empty_set_policy`` is omitted. When enabled, each
    removal is reported and retained in the returned structured diagnostics.
    A candidate removal must preserve identifiability of the parameter space.
    """
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)
    labels = validated_study_labels(study_labels, self.num_test)
    if empty_set_policy is not None and not isinstance(empty_set_policy, EmptySetPolicy):
      raise TypeError("empty_set_policy must be an EmptySetPolicy or None")
    active = np.arange(self.num_test, dtype=int)
    removals: list[StudyRemoval] = []
    evaluations: list[EmptySetEvaluation] = []
    current_analysis = self
    current_xi = xi_hat
    current_sigma = Sigma
    current_sub_dim = sub_dim
    current_df = df
    current_projections = np.asarray(projs, dtype=float)
    current_equal = bool(equal_sub_dim)
    current_x0 = x0
    current_find_min = find_min

    def record(*, minimum_score: float, point: np.ndarray, resolved: bool, message: str) -> EmptySetDiagnostics:
      assert empty_set_policy is not None
      return EmptySetDiagnostics(
        enabled=True,
        encountered=any(evaluation.empty for evaluation in evaluations),
        resolved=resolved,
        strategy=empty_set_policy.strategy,
        original_study_count=self.num_test,
        final_original_indices=tuple(int(index) for index in active),
        removals=tuple(removals),
        final_minimum_score=float(minimum_score),
        final_threshold=float(current_analysis.threshold),
        last_global_minimizer=tuple(float(value) for value in point),
        message=message,
        policy=empty_set_policy,
        method=self.method,
        significance_level=float(self.level),
        parameter_dimension=self.dim,
        original_weights=tuple(float(weight) for weight in self.weights),
        final_weights=tuple(float(weight) for weight in current_analysis.weights),
        study_labels=labels,
        evaluations=tuple(evaluations),
        final_state="nonempty" if resolved else "empty",
      )

    def numerical_failure(
      error: Exception, *, weights: np.ndarray | None = None,
      threshold_known: bool = True,
    ) -> EmptySetNumericalError:
      diagnostics = numerical_failure_diagnostics(
        error=error, policy=empty_set_policy, original_weights=self.weights,
        active_indices=active,
        active_weights=current_analysis.weights if weights is None else weights,
        labels=labels, method=self.method, level=self.level, dimension=self.dim,
        threshold=float(current_analysis.threshold) if threshold_known else None,
        removals=removals, evaluations=evaluations,
      )
      return EmptySetNumericalError(diagnostics.message, diagnostics)

    while True:
      try:
        result = current_analysis._support_interval_once(
          direction=direction,
          xi_hat=current_xi,
          Sigma=current_sigma,
          sub_dim=current_sub_dim,
          df=current_df,
          projs=current_projections,
          format_input=False,
          equal_sub_dim=current_equal,
          x0=current_x0,
          find_min=current_find_min,
          method=method,
          ftol=ftol,
          maxiter=maxiter,
          newton_maxiter=newton_maxiter,
          dense_newton_max_dimension=dense_newton_max_dimension,
          **kwargs,
        )
      except EmptyRegionError as error:
        if empty_set_policy is None:
          raise RuntimeError(
            "the confidence region is empty or no feasible support start was found"
          ) from error
        if not np.isfinite(error.p_values).all():
          raise numerical_failure(
            EmptySetNumericalError("individual p-values are nonfinite")
          ) from error
        allowed = (
          len(removals) < empty_set_policy.max_removals
          and len(active) > empty_set_policy.min_remaining
        )
        candidates: list[CandidateEvaluation] = []
        selected_index: int | None = None
        selected_keep: np.ndarray | None = None
        for priority, candidate in enumerate(np.lexsort((active, error.p_values)), 1):
          keep = np.arange(len(active)) != int(candidate)
          try:
            rank = _retained_rank(current_sub_dim, current_projections, keep)
          except np.linalg.LinAlgError as rank_error:
            raise numerical_failure(rank_error) from rank_error
          eligible = allowed and rank == self.dim
          selected = eligible and selected_index is None
          if selected:
            selected_index = int(candidate)
            selected_keep = keep
          candidates.append(CandidateEvaluation(
            original_index=int(active[candidate]),
            study_label=labels[int(active[candidate])],
            individual_pvalue=float(error.p_values[candidate]),
            priority=priority,
            eligible=eligible,
            selected=selected,
            reason=(
              "policy limit reached" if not allowed
              else "removal loses parameter identifiability" if rank < self.dim
              else "selected: smallest admissible p-value, ties by original index" if selected
              else "eligible but lower priority"
            ),
            retained_projection_rank=rank,
          ))
        evaluations.append(EmptySetEvaluation(
          step=len(removals),
          original_indices=tuple(int(index) for index in active),
          weights=tuple(float(weight) for weight in current_analysis.weights),
          individual_pvalues=tuple(float(value) for value in error.p_values),
          minimum_global_score=float(error.minimum_score),
          threshold=float(error.threshold),
          global_minimizer=tuple(float(value) for value in error.point),
          empty=True,
          candidates=tuple(candidates),
          minimum_diagnostics=error.minimum_diagnostics,
        ))
        if selected_index is None or selected_keep is None:
          diagnostics = record(
            minimum_score=error.minimum_score,
            point=error.point,
            resolved=False,
            message=(
              "empty confidence set remains after the permitted study removals"
              if not allowed else
              "empty confidence set cannot be adjusted without losing parameter identifiability"
            ),
          )
          raise EmptySetResolutionError(diagnostics) from error

        remaining = active[selected_keep]
        weights = np.asarray(current_analysis.weights[selected_keep], dtype=float)
        weights /= weights.sum()
        removal = StudyRemoval(
          step=len(removals) + 1,
          original_index=int(active[selected_index]),
          study_label=labels[int(active[selected_index])],
          individual_pvalue=float(error.p_values[selected_index]),
          minimum_global_score=error.minimum_score,
          threshold=error.threshold,
          global_minimizer=tuple(float(value) for value in error.point),
          remaining_original_indices=tuple(int(index) for index in remaining),
          remaining_weights=tuple(float(weight) for weight in weights),
        )
        removals.append(removal)
        warn_about_removal(removal, empty_set_policy)
        active = remaining
        current_xi = _take_studies(current_xi, selected_keep)
        current_sigma = _take_studies(
          current_sigma,
          selected_keep,
          shared_first_axis=True,
        )
        current_sub_dim = _take_studies(current_sub_dim, selected_keep)
        current_df = _take_studies(current_df, selected_keep)
        try:
          current_analysis = type(self)(
            weights,
            dim=self.dim,
            method=current_analysis.method,
            level=current_analysis.level,
          )
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as refit_error:
          raise numerical_failure(
            refit_error, weights=weights, threshold_known=False,
          ) from refit_error
        current_x0 = error.point
        current_find_min = True
        continue
      except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise numerical_failure(error) from error

      if empty_set_policy is None:
        return result
      try:
        p_values = current_analysis.get_p_vector(
          result.minimum_point, current_xi, current_sigma, current_sub_dim,
          current_df, current_projections, format_input=False,
          equal_sub_dim=current_equal,
        )
        if not np.isfinite(p_values).all():
          raise EmptySetNumericalError("individual p-values are nonfinite")
      except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise numerical_failure(error) from error
      evaluations.append(EmptySetEvaluation(
        step=len(removals),
        original_indices=tuple(int(index) for index in active),
        weights=tuple(float(weight) for weight in current_analysis.weights),
        individual_pvalues=tuple(float(value) for value in p_values),
        minimum_global_score=float(result.minimum_score),
        threshold=float(current_analysis.threshold),
        global_minimizer=tuple(float(value) for value in result.minimum_point),
        empty=False,
        minimum_diagnostics=result.minimum_diagnostics,
      ))
      diagnostics = record(
        minimum_score=result.minimum_score,
        point=result.minimum_point,
        resolved=True,
        message=(
          "confidence set was nonempty"
          if not removals
          else f"resolved after removing {len(removals)} study/studies"
        ),
      )
      return replace(result, empty_set_diagnostics=diagnostics)


__all__ = ["MultivariateEmptySetMixin"]
