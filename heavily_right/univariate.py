"""One-dimensional combined-test confidence intervals."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import norm, t

from .combination import CombinationTest
from .empty_sets import (
  CandidateEvaluation,
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


@dataclass(frozen=True)
class ConfidenceIntervalResult:
  """A scalar confidence interval with optional empty-set diagnostics."""

  lower: float
  upper: float
  estimate: float
  empty_set_diagnostics: EmptySetDiagnostics


@dataclass(frozen=True)
class _IntervalState:
  lower: float
  upper: float
  estimate: float
  minimum_score: float
  empty: bool


@dataclass(frozen=True)
class _ScalarProblem:
  """Common dimensionless coordinates for minimization and inversion."""

  origin: float
  scale: float
  estimates: np.ndarray
  errors: np.ndarray
  df: np.ndarray | None

  def original(self, point: float) -> float:
    value = self.origin + self.scale * point
    if not np.isfinite(value):
      raise EmptySetNumericalError("scalar result is not representable in input units")
    return float(value)


class MetaAnalysis1D(CombinationTest):
  """Combined-test confidence intervals for a scalar parameter.

  Study ``sigma`` values are standard errors of estimates, not raw-data
  standard deviations. ``df=None`` uses normal calibration; finite df uses
  Student t. Automatic study removal is off unless a policy is supplied.

  Examples
  --------
  >>> import numpy as np
  >>> from scipy.stats import t
  >>> from heavily_right import MetaAnalysis1D
  >>> result = MetaAnalysis1D(1).confidence_interval_result([0.2], [0.1], df=19)
  >>> half_width = t.isf(0.025, 19) * 0.1
  >>> assert np.allclose([result.lower, result.upper], [0.2-half_width, 0.2+half_width])
  >>> assert result.empty_set_diagnostics.resolved
  """

  def __init__(
    self,
    arg: np.int_ | float | np.float64 | Sequence[float | np.float64] | np.ndarray,
    method: str = "HCauchy",
    level: float = 0.05,
    **kwargs: Any,
  ) -> None:
    super().__init__(arg, method, level, **kwargs)
    if method not in {"HCauchy", "EHMP"} and self.num_test != 1:
      warnings.warn(
        "confidence intervals are guaranteed only for HCauchy or EHMP",
        stacklevel=2,
      )
    self.__dim = 1

  @property
  def dim(self) -> int:
    return self.__dim

  def __format_input(
    self,
    theta_hat: Sequence[float | np.float64] | np.ndarray,
    sigma: Sequence[float | np.float64] | np.ndarray | float | np.float64,
    df: Sequence[float | np.float64] | np.ndarray | float | np.float64 | None,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # check and format theta_hat
    if isinstance(theta_hat, Iterable):
      theta_hat = np.array(theta_hat)
      if theta_hat.shape[0] != self.num_test:
        raise ValueError("The length of theta_hat should be the same as num_test.")
    else:
      raise TypeError("theta_hat should be an array of estimates.")

    # check and format sigma
    if isinstance(sigma, Iterable):
      sigma = np.array(sigma)
      if sigma.shape[0] != self.num_test:
        raise ValueError("The length of sigma should be the same as num_test.")
    elif isinstance(sigma, (float, int, np.floating, np.integer)):
      sigma = np.ones(self.num_test) * sigma
    else:
      raise TypeError("sigma should be either a float or an array of floats.")

    # check and format df
    if df is None:
      pass
    elif isinstance(df, Iterable):
      df = np.asarray(df, dtype=float)
      if df.shape[0] != self.num_test:
        raise ValueError("The length of df should be the same as num_test.")
    elif isinstance(df, (float, int, np.floating, np.integer)):
      df = np.full(self.num_test, df, dtype=float)
    else:
      raise TypeError("df should be a positive real scalar or array.")

    theta_hat = np.asarray(theta_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if theta_hat.shape != (self.num_test,) or not np.isfinite(theta_hat).all():
      raise ValueError("theta_hat must be a finite one-dimensional study vector")
    if sigma.shape != (self.num_test,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
      raise ValueError("sigma must contain one finite positive standard error per study")
    if df is not None and (
      df.shape != (self.num_test,) or not np.isfinite(df).all() or np.any(df <= 0)
    ):
      raise ValueError("df must contain one finite positive degrees of freedom per study")
    # return formatted theta_hat, sigma and df
    return theta_hat, sigma, df

  # get p-values from estimates for each study
  def get_p_vector(
    self,
    point: float | np.float64,
    theta_hat: Sequence[float | np.float64] | np.ndarray,
    sigma: Sequence[float | np.float64] | np.ndarray | float | np.float64,
    df: Sequence[float | np.float64] | np.ndarray | float | np.float64 | None = None,
    format_input: bool = True,
  ) -> np.ndarray:
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    # if df is None, use normal distribution
    if df is None:
      return 2 * np.maximum(norm.sf(np.abs(point - theta_hat) / sigma), 1e-150)
    # if df is given, use student t distribution
    else:
      return 2 * np.maximum(t.sf(np.abs(point - theta_hat) / sigma, df), 1e-150)

  # check if a point or points are covered by the confidence interval
  def check_cover(
    self,
    point: float | np.float64,
    theta_hat: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray | float | np.float64,
    df: Sequence[float] | np.ndarray | float | np.float64 | None = None,
    format_input: bool = True,
  ) -> bool:
    # format input as the default
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    return (
      self.get_global_score(
        self.get_p_vector(point, theta_hat, sigma, df, format_input=False)
      )
      <= self.threshold
    )

  def _prepare_scalar_problem(
    self, theta_hat: np.ndarray, sigma: np.ndarray, df: np.ndarray | None,
  ) -> _ScalarProblem:
    if (
      self.num_test > 1 and self.method in {"HCauchy", "EHMP"}
      and df is not None and np.any(df < 1)
    ):
      raise ValueError("multi-study HCauchy/EHMP interval inference requires df >= 1")
    origin = float(theta_hat[0])
    with np.errstate(over="ignore", invalid="ignore"):
      centered = theta_hat - origin
      scale = max(float(np.ptp(centered)), float(np.max(sigma)))
      estimates = centered / scale
      errors = sigma / scale
    if (
      not np.isfinite(scale) or scale <= 0
      or not np.isfinite(estimates).all() or not np.isfinite(errors).all()
      or np.any(errors <= 0)
    ):
      raise EmptySetNumericalError("scalar inputs cannot be reliably standardized")
    return _ScalarProblem(origin, scale, estimates, errors, df)

  def _scalar_score(self, point: float, problem: _ScalarProblem) -> float:
    return float(self.get_global_score(self.get_p_vector(
      point, problem.estimates, problem.errors, problem.df, format_input=False,
    )))

  def _scalar_subgradient(
    self, point: float, problem: _ScalarProblem,
  ) -> tuple[float, float]:
    """Closest-to-zero subgradient, divided by a common positive scale.

    The second result is the logarithm of that scale.  At a study estimate
    the score has a genuine cusp; its whole subgradient interval, rather
    than an artificial differentiable approximation, is used.
    """
    residual = (point - problem.estimates) / problem.errors
    radius = np.abs(residual)
    if problem.df is None:
      log_p = np.log(2.0) + norm.logsf(radius)
      log_density = norm.logpdf(radius)
    else:
      log_p = np.log(2.0) + t.logsf(radius, problem.df)
      log_density = t.logpdf(radius, problem.df)
    if self.method == "HCauchy":
      log_sine = np.empty_like(log_p)
      tiny = log_p < np.log(1.0e-7)
      log_sine[tiny] = np.log(np.pi / 2) + log_p[tiny]
      log_sine[~tiny] = np.log(np.sin(np.pi / 2 * np.exp(log_p[~tiny])))
      log_slopes = np.log(np.pi) + log_density - 2 * log_sine
    else:
      log_slopes = np.log(2.0) + log_density - 2 * log_p
    log_slopes += np.log(self.weights) - np.log(problem.errors)
    log_scale = float(np.max(log_slopes))
    if not np.isfinite(log_scale):
      raise EmptySetNumericalError("scalar score derivative is numerically saturated")
    slopes = np.exp(log_slopes - log_scale)
    signed = float(np.dot(np.sign(residual), slopes))
    cusp_radius = float(np.sum(slopes[residual == 0]))
    closest = max(signed - cusp_radius, 0.0) + min(signed + cusp_radius, 0.0)
    return closest, log_scale

  def _scalar_minimum(self, problem: _ScalarProblem) -> tuple[float, float, float]:
    """Return point, score, and a convex lower bound on the minimum.

    The lower bound is used before declaring an empty set.  Legacy methods
    retain their local-search semantics and have no such certificate.
    """
    lower = float(np.min(problem.estimates))
    upper = float(np.max(problem.estimates))
    if lower == upper:
      score = self._scalar_score(lower, problem)
      return lower, score, score
    func = lambda point: self._scalar_score(point, problem)
    result = minimize_scalar(
      func, bounds=(lower, upper), method="bounded",
      options={"xatol": 1.0e-14, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
      raise EmptySetNumericalError(
        f"global-score minimization failed; emptiness was not established: {result.message}"
      )
    point = float(result.x)
    if not lower <= point <= upper:
      raise EmptySetNumericalError("bounded scalar minimizer escaped the estimate hull")
    convex = self.method in {"HCauchy", "EHMP"}
    if convex:
      left, right = lower, upper
      # Independently verify/refine the bounded search using the monotone
      # analytic subgradient. This also locates nondifferentiable minima.
      for _ in range(100):
        slope, _ = self._scalar_subgradient(point, problem)
        if slope == 0.0:
          break
        if slope > 0:
          right = point
        else:
          left = point
        midpoint = left + (right - left) / 2
        if midpoint == left or midpoint == right:
          break
        point = midpoint
      candidates = [point, left, right]
      # Roundoff may leave a root immediately adjacent to a cusp. Test the
      # closest observed estimates exactly so its interval is not missed.
      nearest = np.argsort(np.abs(problem.estimates - point))[:2]
      candidates.extend(float(problem.estimates[index]) for index in nearest)
      # Scores can round to the same value on either side of a cusp. Use
      # stationarity to select its exact location, not a floating-point tie.
      point = min(candidates, key=lambda value: abs(self._scalar_subgradient(value, problem)[0]))
    score = func(point)
    p_values = self.get_p_vector(
      point, problem.estimates, problem.errors, problem.df, format_input=False,
    )
    if not np.isfinite(score) or np.any(p_values <= 2.0e-150):
      raise EmptySetNumericalError(
        "minimum global score is nonfinite or tail-saturated; emptiness was not established"
      )
    if not convex:
      return point, score, float("-inf")
    slope, log_scale = self._scalar_subgradient(point, problem)
    if abs(slope) > 1.0e-7:
      raise EmptySetNumericalError(
        "scalar minimum failed its derivative/subgradient certificate"
      )
    distance = max(point - lower, upper - point)
    if slope == 0 or distance == 0:
      correction = 0.0
    else:
      log_correction = log_scale + np.log(abs(slope)) + np.log(distance)
      correction = float(np.exp(log_correction)) if log_correction < 709 else float("inf")
    return point, score, score - correction

  def find_minimizer(
    self,
    theta_hat: Sequence[float | np.float64] | np.ndarray,
    sigma: Sequence[float | np.float64] | np.ndarray | float | np.float64,
    df: Sequence[float | np.float64] | np.ndarray | float | np.float64 | None = None,
    format_input: bool = True,
  ) -> float | np.float64:
    # format input as the default
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    problem = self._prepare_scalar_problem(theta_hat, sigma, df)
    point, _, _ = self._scalar_minimum(problem)
    return problem.original(point)

  def _confidence_interval_once(
    self,
    theta_hat: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray | None,
    method: str,
  ) -> _IntervalState:
    problem = self._prepare_scalar_problem(theta_hat, sigma, df)
    func = lambda point: self._scalar_score(point, problem)
    minimizer, minimum_score, minimum_lower_bound = self._scalar_minimum(problem)
    estimate = problem.original(minimizer)
    if minimum_score > self.threshold:
      if self.method in {"HCauchy", "EHMP"} and minimum_lower_bound <= self.threshold:
        raise EmptySetNumericalError("minimum error bound cannot establish an empty set")
      return _IntervalState(
        lower=estimate,
        upper=estimate,
        estimate=estimate,
        minimum_score=minimum_score,
        empty=True,
      )
    if method != "brentq":
      raise NotImplementedError("The root finding method is not implemented.")
    def endpoint(direction: float) -> float:
      step = float(np.max(problem.errors))
      inside = minimizer
      for _ in range(200):
        outside = minimizer + direction * step
        if not np.isfinite(outside) or outside == inside:
          raise EmptySetNumericalError("scalar boundary step is not numerically resolvable")
        score = func(outside)
        if not np.isfinite(score):
          raise EmptySetNumericalError("nonfinite scalar boundary-bracketing score")
        if score >= self.threshold:
          break
        inside = outside
        step *= 2
      else:
        raise EmptySetNumericalError("scalar boundary bracketing exceeded 200 expansions")
      result = root_scalar(
        lambda point: func(point) - self.threshold,
        bracket=sorted((inside, outside)), method="brentq",
        xtol=np.nextafter(0.0, 1.0), rtol=8 * np.finfo(float).eps, maxiter=200,
      )
      if not result.converged or not np.isfinite(result.root):
        raise EmptySetNumericalError("scalar boundary root did not converge")
      error = abs(func(float(result.root)) - self.threshold)
      if error > 1.0e-8 * max(abs(float(self.threshold)), np.finfo(float).tiny):
        raise EmptySetNumericalError("scalar boundary root failed its score-residual check")
      physical_point = problem.original(float(result.root))
      reconstructed = (physical_point - problem.origin) / problem.scale
      physical_error = abs(func(reconstructed) - self.threshold)
      if (
        not np.isfinite(physical_error)
        or physical_error > 1.0e-7 * max(abs(float(self.threshold)), np.finfo(float).tiny)
      ):
        raise EmptySetNumericalError(
          "scalar boundary loses accuracy when reconstructed in input units; "
          "recenter/rescale estimates and standard errors before inference"
        )
      return physical_point

    root_low = endpoint(-1.0)
    root_high = endpoint(1.0)
    if (
      not root_low <= estimate <= root_high
      or (minimum_score < self.threshold and root_low == root_high)
    ):
      raise EmptySetNumericalError(
        "scalar interval points are not representable in input units; "
        "recenter/rescale estimates and standard errors before inference"
      )
    return _IntervalState(
      lower=float(root_low),
      upper=float(root_high),
      estimate=estimate,
      minimum_score=minimum_score,
      empty=False,
    )

  def confidence_interval_result(
    self,
    theta_hat: Sequence[float | np.float64] | np.ndarray,
    sigma: Sequence[float | np.float64] | np.ndarray | float | np.float64,
    df: Sequence[float | np.float64] | np.ndarray | float | np.float64 | None = None,
    method: str = "brentq",
    format_input: bool = True,
    print_res: bool = False,
    empty_set_policy: EmptySetPolicy | None = None,
    study_labels: Sequence[str] | None = None,
  ) -> ConfidenceIntervalResult:
    """Return a confidence interval and a structured empty-set record.

    Empty-set adjustment is disabled unless an explicit policy is supplied.
    The enabled strategy removes the smallest-p study at the current global
    minimizer and visibly reports every removal.
    """
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    else:
      theta_hat = np.asarray(theta_hat, dtype=float)
      sigma = np.asarray(sigma, dtype=float)
      df = None if df is None else np.asarray(df)
    labels = validated_study_labels(study_labels, self.num_test)
    if empty_set_policy is not None and not isinstance(empty_set_policy, EmptySetPolicy):
      raise TypeError("empty_set_policy must be an EmptySetPolicy or None")
    if (
      empty_set_policy is not None and self.num_test > 1
      and self.method not in {"HCauchy", "EHMP"}
    ):
      raise ValueError(
        "automatic empty-set treatment requires convex HCauchy/EHMP inference; "
        "legacy methods do not certify a global minimum"
      )
    if method != "brentq":
      raise NotImplementedError("The root finding method is not implemented.")
    active = np.arange(self.num_test, dtype=int)
    removals: list[StudyRemoval] = []
    evaluations: list[EmptySetEvaluation] = []
    current_analysis = self
    current_theta = np.asarray(theta_hat, dtype=float)
    current_sigma = np.asarray(sigma, dtype=float)
    current_df = df

    def record(*, encountered: bool, resolved: bool, message: str) -> EmptySetDiagnostics:
      return EmptySetDiagnostics(
        enabled=empty_set_policy is not None,
        encountered=encountered,
        resolved=resolved,
        strategy=None if empty_set_policy is None else empty_set_policy.strategy,
        original_study_count=self.num_test,
        final_original_indices=tuple(int(index) for index in active),
        removals=tuple(removals),
        final_minimum_score=state.minimum_score,
        final_threshold=float(current_analysis.threshold),
        last_global_minimizer=(state.estimate,),
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
        state = current_analysis._confidence_interval_once(
          current_theta, current_sigma, current_df, method,
        )
        p_values = current_analysis.get_p_vector(
          state.estimate, current_theta, current_sigma, current_df, format_input=False
        )
        if not np.isfinite(p_values).all():
          raise EmptySetNumericalError("individual p-values are nonfinite")
      except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise numerical_failure(error) from error
      candidates: list[CandidateEvaluation] = []
      local_index: int | None = None
      if state.empty and empty_set_policy is not None:
        allowed = (
          len(removals) < empty_set_policy.max_removals
          and len(active) > empty_set_policy.min_remaining
        )
        order = np.lexsort((active, p_values))
        local_index = int(order[0]) if allowed else None
        for priority, candidate in enumerate(order, 1):
          candidates.append(CandidateEvaluation(
            original_index=int(active[candidate]),
            study_label=labels[int(active[candidate])],
            individual_pvalue=float(p_values[candidate]),
            priority=priority,
            eligible=allowed,
            selected=int(candidate) == local_index,
            reason=(
              "selected: smallest p-value, ties by original index"
              if int(candidate) == local_index
              else "eligible but lower priority" if allowed else "policy limit reached"
            ),
          ))
      evaluations.append(EmptySetEvaluation(
        step=len(removals),
        original_indices=tuple(int(index) for index in active),
        weights=tuple(float(weight) for weight in current_analysis.weights),
        individual_pvalues=tuple(float(value) for value in p_values),
        minimum_global_score=state.minimum_score,
        threshold=float(current_analysis.threshold),
        global_minimizer=(state.estimate,),
        empty=state.empty,
        candidates=tuple(candidates),
      ))
      if not state.empty:
        diagnostics = record(
          encountered=bool(removals),
          resolved=True,
          message=(
            "confidence set was nonempty"
            if not removals
            else f"resolved after removing {len(removals)} study/studies"
          ),
        )
        break
      if empty_set_policy is None:
        diagnostics = record(
          encountered=True,
          resolved=False,
          message="confidence set is empty; adjustment is disabled",
        )
        break
      cannot_remove = (
        len(removals) >= empty_set_policy.max_removals
        or len(active) <= empty_set_policy.min_remaining
      )
      if cannot_remove:
        diagnostics = record(
          encountered=True,
          resolved=False,
          message="empty confidence set remains after the permitted study removals",
        )
        raise EmptySetResolutionError(diagnostics)
      assert local_index is not None
      keep = np.arange(len(active)) != local_index
      remaining = active[keep]
      weights = np.asarray(current_analysis.weights[keep], dtype=float)
      weights /= weights.sum()
      removal = StudyRemoval(
        step=len(removals) + 1,
        original_index=int(active[local_index]),
        study_label=labels[int(active[local_index])],
        individual_pvalue=float(p_values[local_index]),
        minimum_global_score=state.minimum_score,
        threshold=current_analysis.threshold,
        global_minimizer=(state.estimate,),
        remaining_original_indices=tuple(int(index) for index in remaining),
        remaining_weights=tuple(float(weight) for weight in weights),
      )
      removals.append(removal)
      warn_about_removal(removal, empty_set_policy)
      active = remaining
      current_theta = current_theta[keep]
      current_sigma = current_sigma[keep]
      if current_df is not None:
        current_df = current_df[keep]
      try:
        current_analysis = MetaAnalysis1D(
          weights,
          method=current_analysis.method,
          level=current_analysis.level,
        )
      except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise numerical_failure(error, weights=weights, threshold_known=False) from error

    if print_res:
      if diagnostics.resolved:
        print(
          f"{1-current_analysis.level} confidence interval is "
          f"[{state.lower:.4f}, {state.upper:.4f}] with point estimate "
          f"{state.estimate:.4f}; {diagnostics.message}."
        )
      else:
        print(f"{1-self.level} confidence interval is empty; {diagnostics.message}.")
    return ConfidenceIntervalResult(
      lower=state.lower,
      upper=state.upper,
      estimate=state.estimate,
      empty_set_diagnostics=diagnostics,
    )

  def confidence_interval(
    self,
    theta_hat: Sequence[float | np.float64] | np.ndarray,
    sigma: Sequence[float | np.float64] | np.ndarray | float | np.float64,
    df: Sequence[float | np.float64] | np.ndarray | float | np.float64 | None = None,
    method: str = "brentq",
    format_input: bool = True,
    print_res: bool = False,
    empty_set_policy: EmptySetPolicy | None = None,
    study_labels: Sequence[str] | None = None,
    return_diagnostics: bool = False,
  ) -> (
    tuple[float, float, float]
    | tuple[float, float, float, EmptySetDiagnostics]
  ):
    """Return the historical tuple, optionally with empty-set diagnostics."""
    result = self.confidence_interval_result(
      theta_hat=theta_hat,
      sigma=sigma,
      df=df,
      method=method,
      format_input=format_input,
      print_res=print_res,
      empty_set_policy=empty_set_policy,
      study_labels=study_labels,
    )
    values = (result.lower, result.upper, result.estimate)
    if return_diagnostics:
      return (*values, result.empty_set_diagnostics)
    return values


__all__ = ["ConfidenceIntervalResult", "MetaAnalysis1D"]
