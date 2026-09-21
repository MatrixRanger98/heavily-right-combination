"""Validated convex score minima and conservative empty-set lower bounds."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize
from scipy.stats import chi2, f

from .projected_region import ProjectedScoreRegion

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MinimumDiagnostics:
  solver: str
  iterations: int
  evaluations: int
  stationarity_residual: float
  message: str
  feasible_set_score_lower_bound: float | None = None
  empty_certified: bool = False
  component_indices: tuple[tuple[int, ...], ...] = ()
  component_weights: tuple[float, ...] = ()
  component_diagnostics: tuple[MinimumDiagnostics, ...] = ()
  minimum_value_lower_bound: float | None = None
  optimality_gap_bound: float | None = None
  optimality_gap_tolerance: float = 0.0
  cusp_anchor_correction: float = 0.0
  component_refinements: int = 0


@dataclass(frozen=True)
class ScoreMinimumResult:
  point: FloatArray
  score: float
  success: bool
  diagnostics: MinimumDiagnostics


def _representable_tails(region: ProjectedScoreRegion, point: FloatArray) -> bool:
  """A clipped tail and its artificial zero derivative cannot certify a minimum."""
  for block in region.blocks:
    residual = block.residual_at_center + block.projection_scaled @ point
    argument = float(residual @ block.precision @ residual) * block.argument_scale
    if block.degrees_freedom is None:
      probability = chi2.sf(argument, block.dimension)
    else:
      probability = f.sf(
        argument, block.dimension,
        block.degrees_freedom + 1.0 - block.dimension,
      )
    # Match the public multidimensional p-value floor, which is deliberately
    # much larger than machine underflow. A minimum of the unclipped kernel
    # cannot certify the semantics of a publicly clipped study score.
    if not np.isfinite(probability) or probability <= 1.0e-150:
      return False
  return True


def _cutoff_reaches_probability_floor(region: ProjectedScoreRegion, cutoff: float) -> bool:
  """Whether a requested domain can reach the public clipped-score regime."""
  if region._single_test:
    return False
  baseline = 1.0 if region.method == "EHMP" else 0.0
  for block in region.blocks:
    budget = cutoff - (1.0 - block.weight) * baseline
    if budget <= 0.0:
      continue
    inverse_score = block.weight / budget
    probability = (
      inverse_score if region.method == "EHMP"
      else 2.0 / np.pi * np.arctan(inverse_score)
    )
    if probability <= 1.0e-150:
      return True
  return False


def _minimum_subgradient(
  region: ProjectedScoreRegion, point: FloatArray,
) -> tuple[FloatArray, float, FloatArray, float]:
  """Find a minimum-norm valid subgradient, snapping only compatible near cusps."""
  scalar_blocks = tuple(block for block in region.blocks if block.dimension == 1)
  active = []
  rows = []
  residuals = []
  for block in scalar_blocks:
    precision_scale = np.sqrt(block.precision[0, 0])
    row = precision_scale * block.projection_scaled[0]
    residual = precision_scale * float(
      (block.residual_at_center + block.projection_scaled @ point)[0]
    )
    if abs(residual) <= 1.0e-8 * np.linalg.norm(row):
      active.append(block)
      rows.append(row)
      residuals.append(residual)
  selected = point.copy()
  if active:
    displacement = np.linalg.lstsq(np.asarray(rows), -np.asarray(residuals), rcond=1.0e-12)[0]
    if np.linalg.norm(displacement) <= 2.0e-8:
      proposed = point + displacement
      mismatch = np.asarray(rows) @ displacement + np.asarray(residuals)
      actual_residuals = np.array([
        np.sqrt(block.precision[0, 0])
        * float((block.residual_at_center + block.projection_scaled @ proposed)[0])
        for block in active
      ])
      tolerance = 64.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(proposed))
      if max(np.max(np.abs(mismatch)), np.max(np.abs(actual_residuals))) <= tolerance:
        selected = proposed
      else:
        active = []
    else:
      active = []
  value, gradient, _ = region._evaluate(selected, region.blocks, with_hessian=False)
  anchor_correction = 0.0
  if active and np.isfinite(gradient).all():
    columns = []
    for block, row in zip(active, rows, strict=True):
      residual = float((block.residual_at_center + block.projection_scaled @ selected)[0])
      scalar_value, derivative, _ = region._scalar_radial_derivatives(residual, block)
      gradient -= block.weight * derivative * np.sign(residual) * row
      cusp_value, cusp_derivative, _ = region._scalar_radial_derivatives(0.0, block)
      columns.append(block.weight * cusp_derivative * row)
      # A numerically near-zero residual may not equal zero exactly. Its
      # selected cusp subgradient is anchored at zero, so lower the tangent
      # intercept conservatively instead of pretending it is based here.
      anchor_correction += block.weight * (
        abs(scalar_value - cusp_value)
        + cusp_derivative * abs(residual) * np.sqrt(block.precision[0, 0])
      )
    matrix = np.column_stack(columns)
    fit = lsq_linear(matrix, -gradient, bounds=(-1.0, 1.0), tol=1.0e-13, max_iter=1000)
    if fit.success and np.isfinite(fit.x).all():
      gradient = gradient + matrix @ fit.x
    else:
      gradient = np.full_like(gradient, np.nan)
  return selected, float(value), np.asarray(gradient, dtype=float), float(anchor_correction)


def _feasible_domain_radius(region: ProjectedScoreRegion, cutoff: float) -> float | None:
  """Conservative radius enclosing every point satisfying the score cutoff.

  Each positive-weight study has an individual Mahalanobis upper bound. Their
  weighted sum bounds a positive-definite quadratic in standardized coordinates.
  Dropping its nonnegative constant yields a conservative enclosing ball.
  """
  if region._single_test or not np.isfinite(cutoff):
    return None
  baseline = 1.0 if region.method == "EHMP" else 0.0
  if cutoff <= baseline:
    return None
  information = np.zeros((region.dimension, region.dimension))
  linear = np.zeros(region.dimension)
  radius_sum = 0.0
  for block in region.blocks:
    allowed_score = (cutoff - (1.0 - block.weight) * baseline) / block.weight
    probability = (
      1.0 / allowed_score if region.method == "EHMP"
      else 2.0 / np.pi * np.arctan(1.0 / allowed_score)
    )
    # Below the public p-value floor the capped study score no longer
    # constrains its Mahalanobis radius at all. Such a block cannot justify
    # this quadratic domain or an uncapped convex tangent certificate.
    if not 1.0e-150 < probability < 1.0:
      return None
    if block.degrees_freedom is None:
      argument = chi2.isf(probability, block.dimension)
    else:
      argument = f.isf(
        probability, block.dimension,
        block.degrees_freedom + 1.0 - block.dimension,
      )
    radius_sum += block.weight * float(argument) / block.argument_scale
    matrix = block.projection_scaled
    information += block.weight * matrix.T @ block.precision @ matrix
    linear += block.weight * matrix.T @ block.precision @ block.residual_at_center
  eigenvalues = np.linalg.eigvalsh(information)
  eigen_floor = float(eigenvalues[0]) - (
    64.0 * np.finfo(float).eps * region.dimension * float(np.max(np.abs(eigenvalues)))
  )
  if not np.isfinite(radius_sum) or not np.isfinite(eigen_floor) or eigen_floor <= 0.0:
    return None
  linear_norm = float(np.linalg.norm(linear))
  radius = (linear_norm + np.sqrt(linear_norm**2 + eigen_floor * radius_sum)) / eigen_floor
  return float(radius * (1.0 + 1.0e-10))


def _minimum_gap(
  region: ProjectedScoreRegion, point: FloatArray, value: float,
  gradient: FloatArray, anchor_correction: float, cutoff: float | None,
) -> tuple[float | None, float]:
  """Bound value minus the true minimum, using a sublevel containing this point."""
  if not np.isfinite(value) or not np.isfinite(gradient).all():
    return None, np.inf
  baseline = 1.0 if region.method == "EHMP" else 0.0
  level = max(value, cutoff if cutoff is not None else value)
  bound = baseline
  radius = _feasible_domain_radius(region, level)
  if radius is not None and np.all(gradient == 0.0):
    bound = max(bound, value - anchor_correction)
  elif radius is not None:
    bound = max(bound, float(
      value - anchor_correction - gradient @ point - radius * np.linalg.norm(gradient)
    ))
  bound -= 1.0e-10 * max(1.0, abs(value), abs(bound))
  return float(bound), max(float(value - bound), 0.0)


def _gap_tolerance(
  value: float, cutoff: float | None, ftol: float, maximum_gap: float | None = None,
) -> float:
  tolerance = max(2.0e-6, 10.0 * ftol) * max(1.0, abs(value), abs(cutoff or 0.0))
  return tolerance if maximum_gap is None else min(tolerance, maximum_gap)


def _component_minima(
  region: ProjectedScoreRegion,
  initial_point: FloatArray | None,
  *, cutoff: float | None, maxiter: int, ftol: float, maximum_gap: float | None = None,
) -> ScoreMinimumResult:
  """Validate each independent component in its own normalized score units."""
  point = np.zeros(region.dimension)
  children = []
  coordinates = []
  masses = []
  lower_bounds = []
  minimum_bounds = []
  child_results = []
  local_problems = []
  success = True
  supplied = np.zeros(region.dimension) if initial_point is None else initial_point
  for label in np.unique(region._component_labels):
    indices = np.flatnonzero(region._component_labels == label)
    blocks = tuple(
      block for block in region.blocks
      if region._component_labels[block.coordinate_indices[0]] == label
    )
    mass = float(sum(block.weight for block in blocks))
    local_weights = np.array([block.weight / mass for block in blocks])
    projection_rows = []
    subdimensions = []
    offset = 0
    for block in blocks:
      projection_rows.append(block.projection_scaled[:, indices])
      subdimensions.append(np.arange(offset, offset + block.dimension))
      offset += block.dimension
    local = ProjectedScoreRegion(
      dimension=len(indices), weights=local_weights, method=region.method,
      xi_hat=[-block.residual_at_center for block in blocks],
      sigma=[np.linalg.inv(block.precision) for block in blocks],
      sub_dim=subdimensions,
      degrees_freedom=(
        None if blocks[0].degrees_freedom is None
        else np.array([block.degrees_freedom for block in blocks])
      ),
      projections=np.concatenate(projection_rows),
      equal_sub_dim=len({block.dimension for block in blocks}) == 1,
    )
    baseline = 1.0 if region.method == "EHMP" else 0.0
    local_cutoff = None if cutoff is None else (cutoff - (1.0 - mass) * baseline) / mass
    # A one-study child uses -p internally. Only its analytic minimizer is
    # reused; its score/cutoff are not the parent's HCCT/EHMP contribution.
    child = minimize_score_region(
      local, local.standardized_from_point(np.asarray(supplied)[indices]),
      cutoff=None if len(blocks) == 1 else local_cutoff,
      maxiter=maxiter, ftol=ftol,
    )
    point[indices] = local.point_from_standardized(child.point)
    children.append(child.diagnostics)
    child_results.append(child)
    local_problems.append((local, blocks, local_cutoff))
    coordinates.append(tuple(int(index) for index in indices))
    masses.append(mass)
    success = success and child.success
    if len(blocks) == 1 and child.success and cutoff is not None:
      lower_bounds.append(float(region._score_only(point, blocks)))
    else:
      bound = child.diagnostics.feasible_set_score_lower_bound
      lower_bounds.append(None if bound is None else mass * bound)
    if len(blocks) == 1 and child.success:
      minimum_bounds.append(float(region._score_only(point, blocks)))
    else:
      bound = child.diagnostics.minimum_value_lower_bound
      minimum_bounds.append(None if bound is None else mass * bound)
  score = float(region.score(point))
  success = bool(success and np.isfinite(score) and _representable_tails(region, point))
  minimum_bound = None
  gap_bound = np.inf
  gap_tolerance = _gap_tolerance(score, cutoff, ftol, maximum_gap)
  if success and all(bound is not None for bound in minimum_bounds):
    minimum_bound = float(sum(minimum_bounds))
    minimum_bound -= 1.0e-10 * max(1.0, abs(score), abs(minimum_bound))
    gap_bound = max(score - minimum_bound, 0.0)
  initial_gap = gap_bound
  refinements = 0
  # A child's normalized cutoff is larger by 1/mass. Its otherwise valid
  # local tolerance can therefore spend the entire parent's error budget.
  # Refine only when individually validated children fail the aggregate test;
  # never relax that test or reinterpret an invalid child as valid.
  if success and np.isfinite(gap_bound) and gap_bound > gap_tolerance:
    unrounded_bound = float(sum(minimum_bounds))
    rounding_reserve = 1.0e-10 * max(1.0, abs(score), abs(unrounded_bound))
    reconstructed_score = sum(
      float(region._score_only(point, blocks)) if len(blocks) == 1 else mass * child.score
      for mass, child, (_, blocks, _) in zip(masses, child_results, local_problems, strict=True)
    )
    reconstruction_reserve = max(0.0, score - reconstructed_score)
    available = gap_tolerance - rounding_reserve - reconstruction_reserve
    multistudy_count = sum(len(blocks) > 1 for _, blocks, _ in local_problems)
    if available > 0.0 and multistudy_count:
      # Allocate half the remaining parent budget equally in PARENT units.
      # The other half leaves headroom for recomposition/rounding changes.
      # The final aggregate certificate is still recomputed and required.
      allocation = available / (2.0 * multistudy_count)
      for index, (mass, old, (local, blocks, local_cutoff)) in enumerate(
        zip(masses, child_results, local_problems, strict=True)
      ):
        if len(blocks) == 1 or mass * old.diagnostics.optimality_gap_bound <= allocation:
          continue
        remaining = maxiter - old.diagnostics.iterations
        if remaining <= 0:
          continue
        refined = minimize_score_region(
          local, old.point, cutoff=local_cutoff, maxiter=remaining, ftol=ftol,
          _maximum_gap=allocation / mass,
        )
        refinements += 1
        accepted = (
          refined.success
          and refined.diagnostics.optimality_gap_bound < old.diagnostics.optimality_gap_bound
        )
        selected = refined if accepted else old
        diagnostics = replace(
          selected.diagnostics,
          iterations=old.diagnostics.iterations + refined.diagnostics.iterations,
          evaluations=old.diagnostics.evaluations + refined.diagnostics.evaluations,
          message=(selected.diagnostics.message + "; parent gap-budget refinement "
                   + ("accepted" if accepted else "not accepted: " + refined.diagnostics.message)),
        )
        selected = replace(selected, diagnostics=diagnostics)
        child_results[index] = selected
        children[index] = diagnostics
        if accepted:
          indices = np.asarray(coordinates[index], dtype=int)
          point[indices] = local.point_from_standardized(refined.point)
          bound = diagnostics.minimum_value_lower_bound
          minimum_bounds[index] = None if bound is None else mass * bound
          bound = diagnostics.feasible_set_score_lower_bound
          lower_bounds[index] = None if bound is None else mass * bound
      score = float(region.score(point))
      success = bool(
        all(child.success for child in child_results)
        and np.isfinite(score) and _representable_tails(region, point)
      )
      gap_tolerance = _gap_tolerance(score, cutoff, ftol, maximum_gap)
      minimum_bound, gap_bound = None, np.inf
      if success and all(bound is not None for bound in minimum_bounds):
        minimum_bound = float(sum(minimum_bounds))
        minimum_bound -= 1.0e-10 * max(1.0, abs(score), abs(minimum_bound))
        gap_bound = max(score - minimum_bound, 0.0)
  children_valid = success
  success = bool(success and gap_bound <= gap_tolerance)
  lower_bound = None
  if success and cutoff is not None and all(bound is not None for bound in lower_bounds):
    lower_bound = float(sum(lower_bounds))
    lower_bound -= 1.0e-10 * max(1.0, abs(lower_bound), abs(score))
  empty_certified = bool(
    lower_bound is not None and cutoff is not None
    and lower_bound > cutoff + 1.0e-8 * max(1.0, abs(cutoff))
  )
  return ScoreMinimumResult(
    point=point, score=score, success=success,
    diagnostics=MinimumDiagnostics(
      solver="independent-component-minima",
      iterations=sum(child.iterations for child in children),
      evaluations=sum(child.evaluations for child in children) + 1,
      stationarity_residual=max(child.stationarity_residual for child in children),
      message=(
        "each disconnected component minimized with independently normalized weights; "
        + ("all component minima and aggregate gap validated" if success else
           "component minima validated but aggregate gap exceeds tolerance" if children_valid else
           "a component minimum or reconstructed score was not validated")
        + f"; aggregate minimum-value gap={gap_bound:.3g} (tolerance={gap_tolerance:.3g})"
        + (f"; parent gap-budget refinements={refinements} (initial gap={initial_gap:.3g})"
           if refinements else "")
        + "; " + " | ".join(child.message for child in children)
      ),
      feasible_set_score_lower_bound=lower_bound,
      empty_certified=empty_certified,
      component_indices=tuple(coordinates), component_weights=tuple(masses),
      component_diagnostics=tuple(children),
      minimum_value_lower_bound=minimum_bound,
      optimality_gap_bound=float(gap_bound), optimality_gap_tolerance=gap_tolerance,
      cusp_anchor_correction=sum(
        mass * child.cusp_anchor_correction for mass, child in zip(masses, children, strict=True)
      ),
      component_refinements=refinements,
    ),
  )


def minimize_score_region(
  region: ProjectedScoreRegion,
  initial_point: FloatArray | None = None,
  *,
  cutoff: float | None = None,
  maxiter: int = 1000,
  ftol: float = 1.0e-9,
  _maximum_gap: float | None = None,
) -> ScoreMinimumResult:
  """Minimize the score in standardized coordinates and validate its minimum.

  Scalar absolute residuals use an epigraph representation. No optimizer status
  flag alone certifies a minimum. For empty-set decisions, an additional convex
  tangent lower bound over a necessary bounded feasible domain must exceed the
  requested cutoff. Clipped tails invalidate every first-order certificate.
  """
  if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
    raise ValueError("maxiter must be a positive integer")
  if not np.isfinite(ftol) or not 0.0 < ftol < 1.0:
    raise ValueError("ftol must lie strictly between zero and one")
  if cutoff is not None and not np.isfinite(cutoff):
    raise ValueError("cutoff must be finite")
  if _maximum_gap is not None and (not np.isfinite(_maximum_gap) or _maximum_gap <= 0.0):
    raise ValueError("internal absolute gap budget must be finite and positive")
  if initial_point is not None:
    initial_point = np.asarray(initial_point, dtype=float)
    if initial_point.shape != (region.dimension,) or not np.isfinite(initial_point).all():
      raise ValueError("initial_point must be a finite standardized parameter vector")
  if cutoff is not None and _cutoff_reaches_probability_floor(region, cutoff):
    point = np.zeros(region.dimension)
    return ScoreMinimumResult(
      point=point, score=float(region.score(point)), success=False,
      diagnostics=MinimumDiagnostics(
        solver="unsupported-clipped-score-domain", iterations=0, evaluations=1,
        stationarity_residual=np.inf,
        message=(
          "requested cutoff permits a study p-value at or below the public "
          "probability floor 1e-150; the clipped-score domain cannot be "
          "certified by the convex internal score; minimum not validated"
        ),
      ),
    )
  if np.unique(region._component_labels).size > 1:
    return _component_minima(
      region, initial_point, cutoff=cutoff, maxiter=maxiter, ftol=ftol, maximum_gap=_maximum_gap,
    )
  start = np.zeros(region.dimension)
  if initial_point is not None:
    supplied = np.asarray(initial_point, dtype=float)
    if supplied.shape != start.shape or not np.isfinite(supplied).all():
      raise ValueError("initial_point must be a finite standardized parameter vector")
    if region.score(supplied) < region.score(start):
      start = supplied.copy()
  scale = max(abs(cutoff) if cutoff is not None else 1.0, 1.0)
  objective_scale = max(scale, abs(float(region.score(start))))
  scalar_blocks = tuple(block for block in region.blocks if block.dimension == 1)
  smooth_blocks = tuple(block for block in region.blocks if block.dimension != 1)
  count = len(scalar_blocks)
  evaluations = 0
  iterations = 0
  optimizer_message = "analytic compatible-study center"
  solver = "analytic-center"
  point = start
  # For one study the score need not be convex, but its Mahalanobis minimizer
  # is analytically the generalized least-squares center.
  analytic = region._single_test or region.centrally_symmetric
  prevalidated = False
  if not analytic and _representable_tails(region, start):
    try:
      checked_point, checked_value, checked_gradient, correction = _minimum_subgradient(region, start)
      evaluations += 1
      residual = np.linalg.norm(checked_gradient) / max(scale, abs(checked_value), 1.0)
      _, gap = _minimum_gap(region, checked_point, checked_value, checked_gradient, correction, cutoff)
      if (
        np.isfinite(residual) and residual <= max(2.0e-6, 10.0 * ftol)
        and gap <= _gap_tolerance(checked_value, cutoff, ftol, _maximum_gap)
      ):
        point = checked_point
        prevalidated = True
        solver = "validated-start"
        optimizer_message = "supplied/GLS point already satisfies minimum conditions"
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
      pass
  if analytic:
    point = np.zeros(region.dimension)
  elif not prevalidated:
    # A supplied point may have lower score than GLS while being arbitrarily
    # distant along a weakly weighted direction. Once its gap check fails,
    # begin optimization at the data-based center with finite geometric scale.
    point = start = np.zeros(region.dimension)
    objective_scale = max(scale, abs(float(region.score(start))))
    rows = np.array([
      np.sqrt(block.precision[0, 0]) * block.projection_scaled[0]
      for block in scalar_blocks
    ]).reshape(count, region.dimension)
    offsets = np.array([
      np.sqrt(block.precision[0, 0]) * block.residual_at_center[0]
      for block in scalar_blocks
    ])
    initial = np.r_[start, np.abs(rows @ start + offsets)]

    def objective(lifted: FloatArray) -> tuple[float, FloatArray]:
      nonlocal evaluations
      evaluations += 1
      value, gradient, _ = region._evaluate(
        lifted[:region.dimension], smooth_blocks, with_hessian=False,
      )
      full_gradient = np.r_[gradient, np.zeros(count)]
      for index, block in enumerate(scalar_blocks):
        standardized = max(float(lifted[region.dimension + index]), 0.0)
        block_value, derivative, _ = region._scalar_radial_derivatives(
          standardized / np.sqrt(block.precision[0, 0]), block,
        )
        value += block.weight * block_value
        full_gradient[region.dimension + index] = block.weight * derivative
      return float(value / objective_scale), full_gradient / objective_scale

    constraints = ()
    if count:
      constraint_matrix = np.block([
        [-rows, np.eye(count)],
        [rows, np.eye(count)],
      ])
      constraints = (LinearConstraint(constraint_matrix, np.r_[offsets, -offsets], np.inf),)
    optimizer_messages = []
    # A large initial incompatibility score may fall by orders of magnitude.
    # At most two rescaled restarts improve the final gradient resolution,
    # sharing the same total iteration budget rather than resetting it.
    for _ in range(3):
      remaining = int(maxiter) - iterations
      if remaining <= 0:
        break
      try:
        optimized = minimize(
          objective, initial, jac=True, method="SLSQP",
          bounds=Bounds(np.r_[np.full(region.dimension, -np.inf), np.zeros(count)], np.inf),
          constraints=constraints,
          options={"maxiter": remaining, "ftol": min(ftol, 1.0e-14)},
        )
        iterations += int(getattr(optimized, "nit", 0))
        optimizer_messages.append(str(optimized.message))
        candidate = np.asarray(optimized.x[:region.dimension], dtype=float)
        if np.isfinite(candidate).all() and region.score(candidate) <= region.score(point):
          point = candidate
        if _representable_tails(region, point):
          checked_point, checked_value, checked_gradient, correction = _minimum_subgradient(region, point)
          residual = np.linalg.norm(checked_gradient) / max(scale, abs(checked_value), 1.0)
          _, gap = _minimum_gap(region, checked_point, checked_value, checked_gradient, correction, cutoff)
          if (
            residual <= max(2.0e-6, 10.0 * ftol)
            and gap <= _gap_tolerance(checked_value, cutoff, ftol, _maximum_gap)
          ):
            point = checked_point
            break
        else:
          break
        initial = np.r_[point, np.abs(rows @ point + offsets)]
        objective_scale = max(scale, abs(float(region.score(point))))
      except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        optimizer_messages.append(f"optimizer numerical failure: {error}")
        break
    optimizer_message = "; ".join(optimizer_messages)
    solver = "convex-epigraph-slsqp" if count else "smooth-slsqp"

  tail_valid = _representable_tails(region, point)
  anchor_correction = 0.0
  if tail_valid:
    try:
      point, value, gradient, anchor_correction = _minimum_subgradient(region, point)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
      value, gradient = float(region.score(point)), np.full(region.dimension, np.nan)
      optimizer_message += f"; subgradient failure: {error}"
  else:
    value, gradient = float(region.score(point)), np.full(region.dimension, np.nan)
    optimizer_message += "; clipped/underflowed tails cannot certify a minimum"
  tail_valid = tail_valid and _representable_tails(region, point)
  evaluations += 1
  stationarity = float(np.linalg.norm(gradient) / max(scale, abs(value), 1.0))
  if analytic and tail_valid:
    minimum_bound, gap_bound = value, 0.0
  else:
    minimum_bound, gap_bound = _minimum_gap(
      region, point, value, gradient, anchor_correction, cutoff,
    )
  gap_tolerance = _gap_tolerance(value, cutoff, ftol, _maximum_gap)
  success = bool(
    tail_valid and np.isfinite(value) and np.isfinite(stationarity)
    and stationarity <= max(2.0e-6, 10.0 * ftol)
    and gap_bound <= gap_tolerance
  )
  lower_bound = None
  empty_certified = False
  if success and cutoff is not None:
    if analytic:
      lower_bound = value
    else:
      radius = _feasible_domain_radius(region, cutoff)
      if radius is not None:
        lower_bound = float(value - anchor_correction - gradient @ point - radius * np.linalg.norm(gradient))
        lower_bound -= 1.0e-10 * max(1.0, abs(value), abs(lower_bound))
    if lower_bound is not None:
      empty_certified = lower_bound > cutoff + 1.0e-8 * max(1.0, abs(cutoff))
  message = (
    f"{optimizer_message}; normalized stationarity={stationarity:.3g}; "
    f"minimum-value gap bound={gap_bound:.3g} (tolerance={gap_tolerance:.3g}); "
    + ("minimum validated" if success else "minimum not validated")
  )
  return ScoreMinimumResult(
    point=np.asarray(point, dtype=float), score=float(value), success=success,
    diagnostics=MinimumDiagnostics(
      solver=solver, iterations=iterations, evaluations=evaluations,
      stationarity_residual=stationarity, message=message,
      feasible_set_score_lower_bound=lower_bound, empty_certified=bool(empty_certified),
      minimum_value_lower_bound=minimum_bound,
      optimality_gap_bound=float(gap_bound), optimality_gap_tolerance=gap_tolerance,
      cusp_anchor_correction=anchor_correction,
    ),
  )


__all__ = ["MinimumDiagnostics", "ScoreMinimumResult", "minimize_score_region"]
