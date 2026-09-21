"""Support-function optimization for smooth convex confidence regions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, brentq, minimize

FloatArray = NDArray[np.float64]
Score = Callable[[FloatArray], float]
ScoreGradient = Callable[[FloatArray], tuple[float, FloatArray]]
ScoreGradientHessian = Callable[
  [FloatArray], tuple[float, FloatArray, FloatArray]
]
CandidateRefiner = Callable[
  [FloatArray, FloatArray, float, bool], "SupportRefinement | None"
]


@dataclass(frozen=True)
class SupportRefinement:
  """A candidate with a valid subgradient and refinement work counters."""

  point: FloatArray
  score: float
  gradient: FloatArray
  iterations: int = 0
  evaluations: int = 0
  line_search_steps: int = 0
  solver: str = "active-set-newton"
  attempts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SupportDiagnostics:
  """Numerical certificate for one support-point calculation."""

  success: bool
  solver: str
  score: float
  cutoff: float
  boundary_error: float
  kkt_residual: float
  eta: float
  iterations: int
  evaluations: int
  line_search_steps: int
  message: str
  certificate_kind: str = "smooth-kkt"
  score_scale: float = 1.0
  reconstruction_error: float = 0.0


@dataclass(frozen=True)
class SupportPoint:
  """A standardized support point and its numerical diagnostics."""

  point: FloatArray
  value: float
  diagnostics: SupportDiagnostics


@dataclass(frozen=True)
class _NewtonResult:
  point: FloatArray
  score: float
  gradient: FloatArray
  eta: float
  success: bool
  iterations: int
  line_search_steps: int
  message: str


def ray_boundary(
  score: Score,
  feasible_point: FloatArray,
  direction: FloatArray,
  cutoff: float,
  *,
  maximum_distance: float = 1.0e6,
) -> FloatArray:
  """Find a feasible radial starting point on ``score == cutoff``."""
  feasible_point = np.asarray(feasible_point, dtype=float)
  direction = np.asarray(direction, dtype=float)
  if (
    direction.ndim != 1
    or direction.size == 0
    or feasible_point.shape != direction.shape
    or not np.isfinite(feasible_point).all()
  ):
    raise ValueError("direction and feasible point must be finite matching vectors")
  if not np.isfinite(cutoff):
    raise ValueError("cutoff must be finite")
  if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
    raise ValueError("maximum_distance must be finite and positive")
  direction_norm = np.linalg.norm(direction)
  if not np.isfinite(direction_norm) or direction_norm == 0.0:
    raise ValueError("direction must be finite and nonzero")
  initial_score = score(feasible_point)
  if not np.isfinite(initial_score) or initial_score > cutoff:
    raise ValueError("the radial search requires a feasible starting point")
  unit = direction / direction_norm

  def difference(distance: float) -> float:
    return score(feasible_point + distance * unit) - cutoff

  upper = min(1.0, maximum_distance)
  while difference(upper) < 0.0:
    if upper >= maximum_distance:
      raise RuntimeError("failed to bracket the confidence-region boundary")
    upper = min(2.0 * upper, maximum_distance)
  distance = brentq(
    difference, 0.0, upper, xtol=np.finfo(float).tiny, rtol=1.0e-11, maxiter=256,
  )
  return feasible_point + distance * unit


def _kkt_residual(gradient: FloatArray, direction: FloatArray) -> float:
  gradient_norm_squared = float(gradient @ gradient)
  direction_norm = max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
  multiplier = float(direction @ gradient) / max(
    gradient_norm_squared, np.finfo(float).tiny
  )
  return float(np.linalg.norm(direction - multiplier * gradient) / direction_norm)


def _candidate_is_certified(
  *,
  optimizer_success: bool,
  value: float,
  eta: float,
  boundary_error: float,
  cutoff: float,
  kkt_residual: float,
) -> bool:
  """Return whether a proposed support point has a usable certificate."""
  return bool(
    optimizer_success
    and np.isfinite([value, eta, boundary_error, cutoff, kkt_residual]).all()
    and eta > 0.0
    and abs(boundary_error) <= 2.0e-8 * max(abs(cutoff), np.finfo(float).tiny)
    and kkt_residual <= 2.0e-6
  )


def _solve_kkt_newton(
  *,
  score_gradient_hessian: ScoreGradientHessian,
  initial_point: FloatArray,
  direction: FloatArray,
  cutoff: float,
  ftol: float,
  maxiter: int,
) -> _NewtonResult:
  """Solve the KKT equations from one supplied boundary approximation."""
  z = np.asarray(initial_point, dtype=float).copy()
  score_value, gradient, hessian = score_gradient_hessian(z)
  direction_norm_squared = max(
    float(direction @ direction),
    np.finfo(float).tiny,
  )
  direction_norm = np.sqrt(direction_norm_squared)
  eta = max(
    float(gradient @ direction) / direction_norm_squared,
    np.finfo(float).eps,
  )

  def merit(
    value: float,
    current_gradient: FloatArray,
    current_eta: float,
  ) -> float:
    stationarity = np.linalg.norm(current_gradient - current_eta * direction)
    return float(
      np.hypot(
        stationarity / direction_norm,
        (value - cutoff) / max(abs(cutoff), np.finfo(float).tiny),
      )
    )

  current_merit = merit(score_value, gradient, eta)
  iterations = 0
  line_search_steps = 0
  success = False
  message = (
    "dense KKT Newton disabled by caller"
    if maxiter == 0
    else "KKT Newton iteration did not converge"
  )
  if not np.isfinite(current_merit) or not np.isfinite(hessian).all():
    message = "KKT Newton start was nonfinite"
  else:
    for iteration in range(maxiter):
      iterations = iteration + 1
      if current_merit <= max(ftol, 2.0e-9):
        success = True
        message = "KKT Newton iteration converged"
        break
      kkt_matrix = np.empty((z.size + 1, z.size + 1), dtype=float)
      kkt_matrix[:-1, :-1] = hessian
      kkt_matrix[:-1, -1] = -direction
      kkt_matrix[-1, :-1] = gradient
      kkt_matrix[-1, -1] = 0.0
      residual = np.r_[gradient - eta * direction, score_value - cutoff]
      try:
        step = np.linalg.solve(kkt_matrix, -residual)
      except np.linalg.LinAlgError:
        try:
          step = np.linalg.lstsq(kkt_matrix, -residual, rcond=1.0e-12)[0]
        except np.linalg.LinAlgError:
          message = "KKT Newton linear system could not be solved"
          break
      if not np.isfinite(step).all():
        message = "KKT Newton step was nonfinite"
        break

      accepted = False
      step_scale = 1.0
      for _ in range(24):
        line_search_steps += 1
        candidate_eta = eta + step_scale * step[-1]
        if not np.isfinite(candidate_eta) or candidate_eta <= 0.0:
          step_scale *= 0.5
          continue
        candidate_z = z + step_scale * step[:-1]
        if not np.isfinite(candidate_z).all():
          step_scale *= 0.5
          continue
        candidate_score, candidate_gradient, candidate_hessian = (
          score_gradient_hessian(candidate_z)
        )
        candidate_merit = merit(
          candidate_score,
          candidate_gradient,
          candidate_eta,
        )
        if (
          np.isfinite(candidate_merit)
          and np.isfinite(candidate_hessian).all()
          and candidate_merit < current_merit
        ):
          z = candidate_z
          eta = candidate_eta
          score_value = candidate_score
          gradient = candidate_gradient
          hessian = candidate_hessian
          current_merit = candidate_merit
          accepted = True
          break
        step_scale *= 0.5
      if not accepted:
        message = (
          "KKT Newton line search stalled "
          f"(merit={current_merit:.3g}, score error="
          f"{score_value-cutoff:.3g}, eta={eta:.3g})"
        )
        break

  if maxiter > 0 and np.isfinite(hessian).all() and current_merit <= max(ftol, 2.0e-9):
    success = True
    message = "KKT Newton iteration converged"
  return _NewtonResult(
    point=z,
    score=float(score_value),
    gradient=gradient,
    eta=float(eta),
    success=success,
    iterations=iterations,
    line_search_steps=line_search_steps,
    message=message,
  )


def _solve_support_point_scaled(
  *,
  score: Score,
  score_gradient: ScoreGradient,
  score_gradient_hessian: ScoreGradientHessian,
  feasible_point: FloatArray,
  direction: FloatArray,
  cutoff: float,
  ftol: float = 1.0e-9,
  maxiter: int = 1000,
  newton_maxiter: int = 80,
  candidate_refiner: CandidateRefiner | None = None,
  nonsmooth_solver: CandidateRefiner | None = None,
) -> SupportPoint:
  """Maximize a linear functional over a smooth score sublevel set.

  Newton iterations solve ``grad G(z) = eta * direction`` and
  ``G(z) = cutoff``. A merit-function backtracking line search enforces a
  positive ``eta`` and strict residual reduction. SLSQP remains a conservative
  fallback when the KKT iteration is unavailable or stalls. If SLSQP does not
  return a certified point, a scaled quadratic-penalty continuation is the
  second fallback. Every path is checked against the same boundary and KKT
  certificate. For convex nonsmooth scores, an optional candidate refiner
  can provide a point and a valid subgradient for the final certificate.
  """
  if not 0.0 < ftol < 1.0:
    raise ValueError("ftol must lie strictly between zero and one")
  if maxiter < 1 or newton_maxiter < 0:
    raise ValueError("iteration limits must be nonnegative and maxiter positive")
  direction = np.asarray(direction, dtype=float)
  feasible_point = np.asarray(feasible_point, dtype=float)
  evaluations = 0

  def counted_score(point: FloatArray) -> float:
    nonlocal evaluations
    evaluations += 1
    return score(point)

  def counted_score_gradient_hessian(
    point: FloatArray,
  ) -> tuple[float, FloatArray, FloatArray]:
    nonlocal evaluations
    evaluations += 1
    try:
      return score_gradient_hessian(point)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
      # A numerical failure of Newton's derivative path must still allow the
      # independent first-derivative constrained methods to recover.
      return np.inf, np.full_like(point, np.nan), np.full((point.size, point.size), np.nan)

  start = ray_boundary(counted_score, feasible_point, direction, cutoff)
  direction_norm_squared = max(
    float(direction @ direction), np.finfo(float).tiny
  )
  direction_norm = np.sqrt(direction_norm_squared)
  certificate_kind = "smooth-kkt"

  def refined_candidate(
    point: FloatArray, current_score: float
  ) -> SupportRefinement | None:
    nonlocal iterations, evaluations, line_search_steps
    if candidate_refiner is None or not np.isfinite(point).all():
      return None
    refined = candidate_refiner(point, direction, cutoff, newton_maxiter > 0)
    if refined is None:
      return None
    iterations += refined.iterations
    evaluations += refined.evaluations
    line_search_steps += refined.line_search_steps
    refined_point, refined_score, refined_gradient = (
      refined.point, refined.score, refined.gradient
    )
    refined_eta = float(refined_gradient @ direction) / direction_norm_squared
    refined_value = float(direction @ refined_point)
    if (
      abs(refined_value - float(direction @ point))
      <= 1.0e-8 * max(1.0, abs(refined_value))
      and abs(refined_score - current_score)
      <= max(2.0e-7, 2.0e-8 * abs(cutoff))
      and _candidate_is_certified(
        optimizer_success=True,
        value=refined_value,
        eta=refined_eta,
        boundary_error=refined_score - cutoff,
        cutoff=cutoff,
        kkt_residual=_kkt_residual(refined_gradient, direction),
      )
    ):
      return refined
    return None

  if newton_maxiter > 0:
    newton = _solve_kkt_newton(
      score_gradient_hessian=counted_score_gradient_hessian,
      initial_point=start,
      direction=direction,
      cutoff=cutoff,
      ftol=ftol,
      maxiter=min(maxiter, newton_maxiter),
    )
  else:
    # This path is also the dimension cap: do not allocate/evaluate a dense
    # Hessian merely to discover that Newton is disabled.
    start_score, start_gradient = score_gradient(start)
    evaluations += 1
    newton = _NewtonResult(
      start, start_score, start_gradient,
      float(start_gradient @ direction) / direction_norm_squared,
      False, 0, 0, "dense KKT Newton disabled by caller",
    )
  z = newton.point
  score_value = newton.score
  gradient = newton.gradient
  eta = newton.eta
  iterations = newton.iterations
  line_search_steps = newton.line_search_steps
  newton_success = _candidate_is_certified(
    optimizer_success=newton.success,
    value=float(direction @ z),
    eta=eta,
    boundary_error=score_value - cutoff,
    cutoff=cutoff,
    kkt_residual=_kkt_residual(gradient, direction),
  )
  newton_message = newton.message

  if newton_success:
    solver = "kkt-newton"
    optimizer_success = True
    message = newton_message
  else:
    cached_point: FloatArray | None = None
    cached_value: tuple[float, FloatArray] | None = None
    score_scale = max(abs(cutoff), 1.0)
    unit_direction = direction / direction_norm

    def fallback_evaluation(point: FloatArray) -> tuple[float, FloatArray]:
      nonlocal cached_point, cached_value, evaluations
      point = np.asarray(point, dtype=float)
      if cached_point is None or not np.array_equal(point, cached_point):
        cached_point = point.copy()
        try:
          cached_value = score_gradient(point)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
          cached_value = (np.inf, np.full_like(point, np.nan))
        evaluations += 1
      if cached_value is None:
        raise RuntimeError("support-solver evaluation cache was not populated")
      return cached_value

    def objective(point: FloatArray) -> float:
      return -float(unit_direction @ point)

    def objective_gradient(point: FloatArray) -> FloatArray:
      del point
      return -unit_direction

    def constraint(point: FloatArray) -> float:
      return (cutoff - fallback_evaluation(point)[0]) / score_scale

    def constraint_gradient(point: FloatArray) -> FloatArray:
      return -fallback_evaluation(point)[1] / score_scale

    try:
      result = minimize(
        objective,
        z if np.isfinite(z).all() else start,
        jac=objective_gradient,
        constraints={
          "type": "ineq",
          "fun": constraint,
          "jac": constraint_gradient,
        },
        method="SLSQP",
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
      )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
      result = OptimizeResult(x=start, success=False, nit=0, message=str(error))
    z = np.asarray(result.x, dtype=float)
    score_value, gradient = fallback_evaluation(z)
    iterations += int(getattr(result, "nit", 0))
    eta = float(gradient @ direction) / direction_norm_squared
    solver = "slsqp"
    optimizer_success = bool(result.success)
    message = f"{newton_message}; SLSQP: {result.message}"
    refined = refined_candidate(z, score_value)
    if refined is not None:
      z, score_value, gradient = refined.point, refined.score, refined.gradient
      eta = float(gradient @ direction) / direction_norm_squared
      optimizer_success = True
      certificate_kind = "subgradient-kkt"
      message += "; certified with bounded scalar-cusp subgradients"

    boundary_error = float(score_value - cutoff)
    kkt_residual = _kkt_residual(gradient, direction)
    value = float(direction @ z)
    slsqp_certified = _candidate_is_certified(
      optimizer_success=optimizer_success,
      value=value,
      eta=eta,
      boundary_error=boundary_error,
      cutoff=cutoff,
      kkt_residual=kkt_residual,
    )
    if not slsqp_certified:
      message += (
        f" (boundary error={boundary_error:.3g}, "
        f"KKT residual={kkt_residual:.3g}, eta={eta:.3g})"
      )
      if newton_maxiter > 0 and np.isfinite(z).all():
        polish = _solve_kkt_newton(
          score_gradient_hessian=counted_score_gradient_hessian,
          initial_point=z,
          direction=direction,
          cutoff=cutoff,
          ftol=ftol,
          maxiter=min(maxiter, newton_maxiter),
        )
        iterations += polish.iterations
        line_search_steps += polish.line_search_steps
        polish_boundary_error = float(polish.score - cutoff)
        polish_kkt_residual = _kkt_residual(polish.gradient, direction)
        polish_value = float(direction @ polish.point)
        polish_certified = _candidate_is_certified(
          optimizer_success=polish.success,
          value=polish_value,
          eta=polish.eta,
          boundary_error=polish_boundary_error,
          cutoff=cutoff,
          kkt_residual=polish_kkt_residual,
        )
        message += f"; Newton restart after SLSQP: {polish.message}"
        if polish_certified:
          z = polish.point
          score_value = polish.score
          gradient = polish.gradient
          eta = polish.eta
          value = polish_value
          boundary_error = polish_boundary_error
          kkt_residual = polish_kkt_residual
          optimizer_success = True
          solver = "kkt-newton-after-slsqp"
          slsqp_certified = True

    if not slsqp_certified:
      penalty_point = start.copy()
      best_point = z.copy()
      best_score = score_value
      best_gradient = gradient
      best_eta = eta
      best_value = value
      best_boundary_error = boundary_error
      best_kkt_residual = kkt_residual
      best_merit = np.hypot(
        boundary_error / score_scale,
        kkt_residual,
      )
      if not np.isfinite(best_merit):
        best_merit = np.inf
      penalty_success = False
      penalty_messages: list[str] = []

      for coefficient in np.geomspace(1.0, 1.0e8, num=9):
        def penalty_evaluation(
          point: FloatArray,
          coefficient: float = float(coefficient),
        ) -> tuple[float, FloatArray]:
          value_at_point, gradient_at_point = fallback_evaluation(point)
          violation = max((value_at_point - cutoff) / score_scale, 0.0)
          objective_value = float(
            -unit_direction @ point + 0.5 * coefficient * violation**2
          )
          objective_gradient = (
            -unit_direction
            + coefficient * violation * gradient_at_point / score_scale
          )
          return objective_value, objective_gradient

        try:
          penalty_result = minimize(
            lambda point: penalty_evaluation(point)[0],
            penalty_point,
            jac=lambda point: penalty_evaluation(point)[1],
            method="L-BFGS-B",
            options={"ftol": ftol, "maxiter": maxiter},
          )
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
          penalty_messages.append(f"mu={coefficient:.0e}: {error}")
          continue
        iterations += int(getattr(penalty_result, "nit", 0))
        raw_point = np.asarray(penalty_result.x, dtype=float)
        raw_score, _ = fallback_evaluation(raw_point)
        penalty_messages.append(
          f"mu={coefficient:.0e}: {penalty_result.message}"
        )
        if not np.isfinite(raw_point).all() or not np.isfinite(raw_score):
          continue
        penalty_point = raw_point
        if raw_score <= cutoff:
          continue

        displacement = raw_point - feasible_point

        def line_difference(
          fraction: float,
          displacement: FloatArray = displacement,
        ) -> float:
          point = feasible_point + fraction * displacement
          return fallback_evaluation(point)[0] - cutoff

        try:
          fraction = brentq(
            line_difference,
            0.0,
            1.0,
            xtol=1.0e-11,
            rtol=1.0e-11,
          )
        except ValueError:
          continue
        candidate = feasible_point + fraction * displacement
        candidate_score, candidate_gradient = fallback_evaluation(candidate)
        candidate_eta = (
          float(candidate_gradient @ direction) / direction_norm_squared
        )
        candidate_value = float(direction @ candidate)
        candidate_boundary_error = float(candidate_score - cutoff)
        candidate_kkt_residual = _kkt_residual(
          candidate_gradient,
          direction,
        )
        candidate_merit = np.hypot(
          candidate_boundary_error / score_scale,
          candidate_kkt_residual,
        )
        if candidate_merit < best_merit:
          best_point = candidate
          best_score = candidate_score
          best_gradient = candidate_gradient
          best_eta = candidate_eta
          best_value = candidate_value
          best_boundary_error = candidate_boundary_error
          best_kkt_residual = candidate_kkt_residual
          best_merit = candidate_merit
        if _candidate_is_certified(
          optimizer_success=bool(penalty_result.success),
          value=candidate_value,
          eta=candidate_eta,
          boundary_error=candidate_boundary_error,
          cutoff=cutoff,
          kkt_residual=candidate_kkt_residual,
        ):
          # The certificate belongs to this candidate, even when an earlier
          # candidate had a smaller unsigned merit but a nonpositive eta.
          best_point = candidate
          best_score = candidate_score
          best_gradient = candidate_gradient
          best_eta = candidate_eta
          best_value = candidate_value
          best_boundary_error = candidate_boundary_error
          best_kkt_residual = candidate_kkt_residual
          penalty_success = True
          break

      z = best_point
      score_value = best_score
      gradient = best_gradient
      eta = best_eta
      value = best_value
      boundary_error = best_boundary_error
      kkt_residual = best_kkt_residual
      optimizer_success = penalty_success
      solver = "quadratic-penalty"
      message = (
        f"{message}; quadratic penalty: "
        + ("certified" if penalty_success else "no certified point")
        + "; "
        + " | ".join(penalty_messages)
      )
      if not penalty_success and newton_maxiter > 0:
        polish = _solve_kkt_newton(
          score_gradient_hessian=counted_score_gradient_hessian,
          initial_point=z,
          direction=direction,
          cutoff=cutoff,
          ftol=ftol,
          maxiter=min(maxiter, newton_maxiter),
        )
        iterations += polish.iterations
        line_search_steps += polish.line_search_steps
        polish_boundary_error = float(polish.score - cutoff)
        polish_kkt_residual = _kkt_residual(polish.gradient, direction)
        polish_value = float(direction @ polish.point)
        polish_certified = _candidate_is_certified(
          optimizer_success=polish.success,
          value=polish_value,
          eta=polish.eta,
          boundary_error=polish_boundary_error,
          cutoff=cutoff,
          kkt_residual=polish_kkt_residual,
        )
        message += f"; Newton restart after penalty: {polish.message}"
        if polish_certified:
          z = polish.point
          score_value = polish.score
          gradient = polish.gradient
          eta = polish.eta
          value = polish_value
          boundary_error = polish_boundary_error
          kkt_residual = polish_kkt_residual
          optimizer_success = True
          solver = "penalty-newton"

  if certificate_kind != "subgradient-kkt":
    refined = refined_candidate(z, score_value)
    if refined is not None:
      z, score_value, gradient = refined.point, refined.score, refined.gradient
      eta = float(gradient @ direction) / direction_norm_squared
      optimizer_success = True
      certificate_kind = "subgradient-kkt"
      message += "; certified with bounded scalar-cusp subgradients"

  boundary_error = float(score_value - cutoff)
  kkt_residual = _kkt_residual(gradient, direction)
  value = float(direction @ z)
  success = _candidate_is_certified(
    optimizer_success=optimizer_success,
    value=value,
    eta=eta,
    boundary_error=boundary_error,
    cutoff=cutoff,
    kkt_residual=kkt_residual,
  )
  if not success and nonsmooth_solver is not None:
    candidate = nonsmooth_solver(z, direction, cutoff, newton_maxiter > 0)
    if candidate is not None:
      iterations += candidate.iterations
      evaluations += candidate.evaluations
      line_search_steps += candidate.line_search_steps
      if candidate.attempts:
        message += "; nonsmooth attempts: " + ", ".join(candidate.attempts)
      candidate_eta = float(candidate.gradient @ direction) / direction_norm_squared
      candidate_residual = _kkt_residual(candidate.gradient, direction)
      if _candidate_is_certified(
        optimizer_success=True,
        value=float(direction @ candidate.point),
        eta=candidate_eta,
        boundary_error=candidate.score - cutoff,
        cutoff=cutoff,
        kkt_residual=candidate_residual,
      ):
        z, score_value, gradient = candidate.point, candidate.score, candidate.gradient
        eta = candidate_eta
        value = float(direction @ z)
        boundary_error = float(score_value - cutoff)
        kkt_residual = candidate_residual
        success = True
        solver = candidate.solver
        certificate_kind = "subgradient-kkt"
        message += "; bounded exact scalar-cusp active-set Newton solve certified"
  diagnostics = SupportDiagnostics(
    success=success,
    solver=solver,
    score=float(score_value),
    cutoff=float(cutoff),
    boundary_error=boundary_error,
    kkt_residual=kkt_residual,
    eta=float(eta),
    iterations=iterations,
    evaluations=evaluations,
    line_search_steps=line_search_steps,
    message=message,
    certificate_kind=certificate_kind,
  )
  return SupportPoint(point=z, value=value, diagnostics=diagnostics)


def solve_support_point(
  *,
  score: Score,
  score_gradient: ScoreGradient,
  score_gradient_hessian: ScoreGradientHessian,
  feasible_point: FloatArray,
  direction: FloatArray,
  cutoff: float,
  ftol: float = 1.0e-9,
  maxiter: int = 1000,
  newton_maxiter: int = 80,
  candidate_refiner: CandidateRefiner | None = None,
  nonsmooth_solver: CandidateRefiner | None = None,
) -> SupportPoint:
  """Compute a checked support point with automatic numerical scaling.

  The direction is normalized and score differences are measured in units of
  the initial feasible slack (or gradient scale at a boundary start). This
  leaves the feasible set unchanged and prevents score units from changing
  the meaning of the boundary certificate. Diagnostics are returned in the
  caller's original score and direction units, with ``score_scale`` retained.
  """
  for name, count, minimum in (("maxiter", maxiter, 1), ("newton_maxiter", newton_maxiter, 0)):
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < minimum:
      raise ValueError(f"{name} must be an integer of at least {minimum}")
  if not np.isfinite(ftol) or not 0.0 < ftol < 1.0:
    raise ValueError("ftol must lie strictly between zero and one")
  point = np.asarray(feasible_point, dtype=float)
  original_direction = np.asarray(direction, dtype=float)
  if (
    point.ndim != 1 or point.size == 0 or point.shape != original_direction.shape
    or not np.isfinite(point).all() or not np.isfinite(original_direction).all()
    or not np.isfinite(cutoff)
  ):
    raise ValueError("point, direction, and cutoff must be finite with matching vector shapes")
  magnitude = float(np.max(np.abs(original_direction)))
  if magnitude == 0.0:
    raise ValueError("direction must be nonzero")
  scaled_direction = original_direction / magnitude
  scaled_norm = float(np.linalg.norm(scaled_direction))
  unit_direction = scaled_direction / scaled_norm
  direction_norm = magnitude * scaled_norm
  if not np.isfinite(direction_norm):
    raise ValueError("direction magnitude exceeds floating-point range")
  base_value, base_gradient = score_gradient(point)
  base_gradient = np.asarray(base_gradient, dtype=float)
  if not np.isfinite(base_value) or base_value > cutoff:
    raise ValueError("a finite feasible starting point is required")
  if base_gradient.shape != point.shape or not np.isfinite(base_gradient).all():
    raise ValueError("the starting gradient must be finite and match the point")
  scale = abs(float(cutoff - base_value))
  if scale == 0.0:
    scale = float(np.linalg.norm(base_gradient))
  if not np.isfinite(scale) or scale == 0.0:
    return SupportPoint(
      point=point.copy(), value=float(original_direction @ point),
      diagnostics=SupportDiagnostics(
        success=False, solver="nonregular-start", score=float(base_value),
        cutoff=float(cutoff), boundary_error=float(base_value - cutoff),
        kkt_residual=1.0, eta=0.0, iterations=0, evaluations=1,
        line_search_steps=0,
        message="no strict feasible slack or nonzero boundary gradient; degenerate support requires a separate method",
      ),
    )

  def normalized_score(z: FloatArray) -> float:
    return float(1.0 + (score(z) - cutoff) / scale)

  def normalized_gradient(z: FloatArray) -> tuple[float, FloatArray]:
    value, gradient = score_gradient(z)
    return float(1.0 + (value - cutoff) / scale), np.asarray(gradient) / scale

  def normalized_hessian(z: FloatArray) -> tuple[float, FloatArray, FloatArray]:
    value, gradient, hessian = score_gradient_hessian(z)
    return float(1.0 + (value - cutoff) / scale), np.asarray(gradient) / scale, np.asarray(hessian) / scale

  def adapt_refiner(callback: CandidateRefiner | None) -> CandidateRefiner | None:
    if callback is None:
      return None

    def normalized_refiner(z: FloatArray, b: FloatArray, _cutoff: float, allow_newton: bool) -> SupportRefinement | None:
      candidate = callback(z, b, float(cutoff), allow_newton)
      if candidate is None:
        return None
      return replace(candidate, score=float(1.0 + (candidate.score - cutoff) / scale), gradient=candidate.gradient / scale)

    return normalized_refiner

  result = _solve_support_point_scaled(
    score=normalized_score, score_gradient=normalized_gradient,
    score_gradient_hessian=normalized_hessian, feasible_point=point,
    direction=unit_direction, cutoff=1.0, ftol=ftol, maxiter=maxiter,
    newton_maxiter=newton_maxiter, candidate_refiner=adapt_refiner(candidate_refiner),
    nonsmooth_solver=adapt_refiner(nonsmooth_solver),
  )
  value = float(original_direction @ result.point)
  original_score = float(score(result.point))
  eta = float(result.diagnostics.eta * scale / direction_norm)
  success = bool(
    result.diagnostics.success and np.isfinite([value, original_score, eta]).all()
    and eta > 0.0 and abs(original_score - cutoff) / scale <= 2.0e-8
  )
  diagnostics = replace(
    result.diagnostics, success=success, score=original_score, cutoff=float(cutoff),
    boundary_error=float(original_score - cutoff), eta=eta, score_scale=scale,
    evaluations=result.diagnostics.evaluations + 2,
  )
  return SupportPoint(point=result.point, value=value, diagnostics=diagnostics)


__all__ = [
  "SupportDiagnostics",
  "SupportPoint",
  "ray_boundary",
  "solve_support_point",
]
