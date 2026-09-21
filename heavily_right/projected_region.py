"""Analytic scores and derivatives for projected confidence regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space
from scipy.optimize import lsq_linear, minimize
from scipy.special import beta as beta_function
from scipy.special import gamma
from scipy.stats import chi2, f, norm, t

from .support import SupportRefinement, _solve_kkt_newton

FloatArray = NDArray[np.float64]


@dataclass
class _RefinementWork:
  iterations: int = 0
  evaluations: int = 0
  line_search_steps: int = 0


@dataclass(frozen=True)
class _RawProjectionBlock:
  projection: FloatArray
  estimate: FloatArray
  precision: FloatArray
  coordinate_indices: NDArray[np.int64]
  dimension: int
  weight: float
  degrees_freedom: float | None
  argument_scale: float


@dataclass(frozen=True)
class _ProjectionBlock:
  projection_scaled: FloatArray
  residual_at_center: FloatArray
  precision: FloatArray
  coordinate_indices: NDArray[np.int64]
  dimension: int
  weight: float
  degrees_freedom: float | None
  argument_scale: float


class ProjectedScoreComponent:
  """One connected coordinate component of a projected score region."""

  def __init__(
    self,
    *,
    parent: ProjectedScoreRegion,
    global_indices: NDArray[np.int64],
    blocks: tuple[_ProjectionBlock, ...],
    constant_score: float,
    base_z: FloatArray,
  ) -> None:
    self.parent = parent
    self.global_indices = global_indices
    self.blocks = blocks
    self.constant_score = float(constant_score)
    self.base_z = np.asarray(base_z, dtype=float)
    self.dimension = len(global_indices)

  def full_standardized_point(self, local_delta: FloatArray) -> FloatArray:
    full = self.base_z.copy()
    full[self.global_indices] += np.asarray(local_delta, dtype=float)
    return full

  def _score_gradient_hessian(
    self,
    local_delta: FloatArray,
  ) -> tuple[float, FloatArray, FloatArray]:
    score, gradient, hessian = self.parent._evaluate(
      np.asarray(local_delta, dtype=float),
      self.blocks,
      initial_score=self.constant_score,
    )
    if hessian is None:
      raise RuntimeError("internal Hessian evaluation was unexpectedly disabled")
    return score, gradient, hessian

  def _score_gradient(
    self,
    local_delta: FloatArray,
  ) -> tuple[float, FloatArray]:
    score, gradient, _ = self.parent._evaluate(
      np.asarray(local_delta, dtype=float),
      self.blocks,
      initial_score=self.constant_score,
      with_hessian=False,
    )
    return score, gradient

  def score(self, local_delta: FloatArray) -> float:
    return self.parent._score_only(
      np.asarray(local_delta, dtype=float),
      self.blocks,
      initial_score=self.constant_score,
    )

  def certify_support_candidate(
    self,
    local_delta: FloatArray,
    direction: FloatArray,
    cutoff: float,
    allow_dense_newton: bool = True,
  ) -> SupportRefinement | None:
    """Resolve numerical scalar cusps using valid bounded subgradients.

    A scalar transformed score has a subgradient interval at zero residual.
    Only hyperplanes within 1e-8 of the candidate in standardized-coordinate
    distance are nominated. Projection onto them must succeed to floating-point precision, move
    the initial point by at most 1e-8. For small components an optional
    Newton polish operates within these exact hyperplanes. The bounded
    subgradient certificate then checks positive-multiplier stationarity.
    The caller independently checks boundary and support-value changes.
    """
    point = np.asarray(local_delta, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if not np.isfinite(point).all() or not np.isfinite(direction).all():
      return None
    rows = []
    residuals = []
    cusp_blocks = []
    for block in self.blocks:
      if block.dimension != 1:
        continue
      scale = np.sqrt(block.precision[0, 0])
      residual = float(
        (block.residual_at_center + block.projection_scaled @ point)[0]
      )
      standardized_residual = scale * residual
      row = scale * block.projection_scaled[0]
      row_norm = float(np.linalg.norm(row))
      # Nomination is scale invariant and uses the same geometric bound as
      # the joint projection; a large projection-row norm must not make a
      # nearby cusp invisible merely because its scalar residual is larger.
      if row_norm > 0.0 and abs(standardized_residual) / row_norm <= 1.0e-8:
        rows.append(row)
        residuals.append(standardized_residual)
        cusp_blocks.append(block)
    if not cusp_blocks:
      return None
    work = _RefinementWork()
    candidate = self._solve_cusp_manifold(
      point, direction, cutoff, tuple(cusp_blocks), allow_dense_newton,
      maximum_initial_displacement=1.0e-8,
      work=work,
    )
    if candidate is not None:
      return replace(candidate, iterations=work.iterations, evaluations=work.evaluations,
                     line_search_steps=work.line_search_steps)
    if work.evaluations:
      return SupportRefinement(point, np.nan, np.full_like(point, np.nan),
                               work.iterations, work.evaluations, work.line_search_steps)
    return None

  def solve_nonsmooth_support_candidate(
    self,
    local_delta: FloatArray,
    direction: FloatArray,
    cutoff: float,
    allow_dense_newton: bool = True,
  ) -> SupportRefinement | None:
    """Try bounded exact active-manifold solves for nonsmooth support points.

    This is a new optimization solve, not numerical snapping. At most fifteen
    subsets of the four closest scalar residual hyperplanes are tried. Every
    accepted candidate must have an original-score boundary and a valid bounded
    subgradient certificate. Multiple simultaneous cusps are handled jointly.
    """
    if not allow_dense_newton or self.dimension > 250:
      return None
    point = np.asarray(local_delta, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if not np.isfinite(point).all() or not np.isfinite(direction).all():
      return None
    ranked = []
    for index, block in enumerate(self.blocks):
      if block.dimension != 1:
        continue
      norm = np.linalg.norm(block.projection_scaled[0])
      if norm > 0.0:
        residual = abs(float((block.residual_at_center + block.projection_scaled @ point)[0]))
        ranked.append((residual / norm, index, block))
    nominated = [entry[2] for entry in sorted(ranked, key=lambda item: item[:2])[:4]]
    work = _RefinementWork()
    attempts = []
    for count in range(1, len(nominated) + 1):
      for active_blocks in combinations(nominated, count):
        attempts.append(f"active set with {count} cusp(s)")
        candidate = self._solve_cusp_manifold(
          point, direction, cutoff, active_blocks, True,
          maximum_initial_displacement=None,
          work=work,
        )
        if candidate is not None and abs(candidate.score - cutoff) <= (
          max(2.0e-7, 2.0e-8 * abs(cutoff))
        ):
          return replace(candidate, iterations=work.iterations, evaluations=work.evaluations,
                         line_search_steps=work.line_search_steps, attempts=tuple(attempts))
    # A lift of |residual| supplies active-set discovery when several cusps
    # are involved or a nearby original-space candidate identifies the wrong
    # set. Certification still uses the original, unlifted score.
    attempts.append("epigraph discovery")
    lifted = self._epigraph_candidate(point, direction, cutoff, work=work)
    if lifted is not None:
      lifted_point, active_blocks, iterations, evaluations = lifted
      if active_blocks:
        candidate = self._solve_cusp_manifold(
          lifted_point, direction, cutoff, active_blocks, True,
          maximum_initial_displacement=None,
          work=work,
        )
        if candidate is not None and abs(candidate.score - cutoff) <= (
          max(2.0e-7, 2.0e-8 * abs(cutoff))
        ):
          return SupportRefinement(
            candidate.point, candidate.score, candidate.gradient,
            work.iterations, work.evaluations, work.line_search_steps,
            solver="epigraph-active-set-newton", attempts=tuple(attempts),
          )
    return SupportRefinement(
      point, np.nan, np.full_like(point, np.nan),
      work.iterations, work.evaluations, work.line_search_steps,
      attempts=tuple(attempts),
    )

  def _epigraph_candidate(
    self,
    point: FloatArray,
    direction: FloatArray,
    cutoff: float,
    *,
    work: _RefinementWork,
  ) -> tuple[FloatArray, tuple[_ProjectionBlock, ...], int, int] | None:
    """Discover multiple scalar cusps with a smooth convex epigraph problem."""
    scalar_blocks = tuple(block for block in self.blocks if block.dimension == 1)
    smooth_blocks = tuple(block for block in self.blocks if block.dimension != 1)
    count = len(scalar_blocks)
    if not count or self.dimension + count > 500:
      return None
    rows = np.array([
      np.sqrt(block.precision[0, 0]) * block.projection_scaled[0]
      for block in scalar_blocks
    ])
    offsets = np.array([
      np.sqrt(block.precision[0, 0]) * block.residual_at_center[0]
      for block in scalar_blocks
    ])
    initial = np.r_[point, np.abs(rows @ point + offsets)]
    unit_direction = direction / np.linalg.norm(direction)
    objective_gradient = np.r_[-unit_direction, np.zeros(count)]
    score_scale = max(abs(cutoff), 1.0)
    evaluations = 0
    cached_point = None
    cached_value = None

    def evaluate(lifted: FloatArray) -> tuple[float, FloatArray]:
      nonlocal cached_point, cached_value, evaluations
      if cached_point is not None and np.array_equal(lifted, cached_point):
        return cached_value
      evaluations += 1
      work.evaluations += 1
      value, gradient, _ = self.parent._evaluate(
        lifted[:self.dimension], smooth_blocks,
        initial_score=self.constant_score, with_hessian=False,
      )
      radial_gradient = np.empty(count)
      for index, block in enumerate(scalar_blocks):
        residual = lifted[self.dimension + index] / np.sqrt(block.precision[0, 0])
        radial_value, derivative, _ = self.parent._scalar_radial_derivatives(residual, block)
        value += block.weight * radial_value
        radial_gradient[index] = block.weight * derivative
      cached_point = lifted.copy()
      cached_value = float(value), np.r_[gradient, radial_gradient]
      return cached_value

    linear_jacobian = np.block([[-rows, np.eye(count)], [rows, np.eye(count)]])

    def linear_constraints(lifted: FloatArray) -> FloatArray:
      residual = rows @ lifted[:self.dimension] + offsets
      radial = lifted[self.dimension:]
      return np.r_[radial - residual, radial + residual]

    try:
      result = minimize(
        lambda lifted: float(objective_gradient @ lifted), initial,
        jac=lambda lifted: objective_gradient,
        bounds=[(None, None)] * self.dimension + [(0.0, None)] * count,
        constraints=[
          {
            "type": "ineq",
            "fun": lambda lifted: (cutoff - evaluate(lifted)[0]) / score_scale,
            "jac": lambda lifted: -evaluate(lifted)[1] / score_scale,
          },
          {"type": "ineq", "fun": linear_constraints,
           "jac": lambda lifted: linear_jacobian},
        ],
        method="SLSQP", options={"ftol": 1.0e-11, "maxiter": 1000},
      )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
      return None
    if not np.isfinite(result.x).all():
      return None
    work.iterations += int(result.nit)
    # This nominates equalities for a new exact solve; it is not the final
    # feasibility/stationarity acceptance tolerance.
    active_blocks = tuple(
      block for index, block in enumerate(scalar_blocks)
      if result.x[self.dimension + index] <= np.sqrt(1.0e-11)
    )
    return result.x[:self.dimension], active_blocks, int(result.nit), evaluations

  def _solve_cusp_manifold(
    self,
    point: FloatArray,
    direction: FloatArray,
    cutoff: float,
    cusp_blocks: tuple[_ProjectionBlock, ...],
    allow_dense_newton: bool,
    *,
    maximum_initial_displacement: float | None,
    work: _RefinementWork,
  ) -> SupportRefinement | None:
    rows = [
      np.sqrt(block.precision[0, 0]) * block.projection_scaled[0]
      for block in cusp_blocks
    ]
    residuals = [
      np.sqrt(block.precision[0, 0])
      * float((block.residual_at_center + block.projection_scaled @ point)[0])
      for block in cusp_blocks
    ]
    matrix = np.asarray(rows)
    try:
      displacement = np.linalg.lstsq(
        matrix, -np.asarray(residuals), rcond=1.0e-12
      )[0]
    except np.linalg.LinAlgError:
      return None
    snapped = point + displacement
    if (
      not np.isfinite(snapped).all()
      or (
        maximum_initial_displacement is not None
        and np.linalg.norm(displacement) > maximum_initial_displacement
      )
    ):
      return None
    if np.max(np.abs(matrix @ displacement + residuals)) > (
      64.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(snapped))
    ):
      return None
    refinement_iterations = 0
    refinement_evaluations = 0
    refinement_line_search_steps = 0
    if allow_dense_newton and self.dimension <= 250:
      try:
        tangent = null_space(matrix, rcond=1.0e-12)
      except np.linalg.LinAlgError:
        return None
      tangent_direction = tangent.T @ direction
      if tangent.shape[1] and np.linalg.norm(tangent_direction) > 1.0e-14:
        cusp_ids = {id(block) for block in cusp_blocks}
        smooth_blocks = tuple(
          block for block in self.blocks if id(block) not in cusp_ids
        )
        cusp_score = sum(
          block.weight * self.parent._scalar_radial_derivatives(0.0, block)[0]
          for block in cusp_blocks
        )

        def smooth_derivatives(delta: FloatArray) -> tuple[float, FloatArray, FloatArray]:
          nonlocal refinement_evaluations
          refinement_evaluations += 1
          work.evaluations += 1
          value, gradient, hessian = self.parent._evaluate(
            snapped + tangent @ delta,
            smooth_blocks,
            initial_score=self.constant_score + cusp_score,
          )
          if hessian is None:
            raise RuntimeError("tangent Hessian was unexpectedly unavailable")
          return value, tangent.T @ gradient, tangent.T @ hessian @ tangent

        try:
          polished = _solve_kkt_newton(
            score_gradient_hessian=smooth_derivatives,
            initial_point=np.zeros(tangent.shape[1]),
            direction=tangent_direction,
            cutoff=cutoff,
            ftol=1.0e-11,
            maxiter=80,
          )
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
          return None
        refinement_iterations = polished.iterations
        refinement_line_search_steps = polished.line_search_steps
        work.iterations += polished.iterations
        work.line_search_steps += polished.line_search_steps
        if polished.success:
          snapped = snapped + tangent @ polished.point
    score, gradient = self._score_gradient(snapped)
    work.evaluations += 1
    columns = []
    for block, row in zip(cusp_blocks, matrix, strict=True):
      residual = float(
        (block.residual_at_center + block.projection_scaled @ snapped)[0]
      )
      if abs(residual * np.sqrt(block.precision[0, 0])) > (
        64.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(snapped))
      ):
        return None
      _, derivative, _ = self.parent._scalar_radial_derivatives(residual, block)
      gradient -= block.weight * derivative * np.sign(residual) * row
      _, cusp_derivative, _ = self.parent._scalar_radial_derivatives(0.0, block)
      columns.append(block.weight * cusp_derivative * row)
    subgradient_matrix = np.column_stack(columns)
    system = np.column_stack([subgradient_matrix, -direction])
    if not np.isfinite(system).all() or not np.isfinite(gradient).all():
      return None
    try:
      result = lsq_linear(
        system,
        -gradient,
        bounds=(
          np.r_[-np.ones(len(columns)), np.finfo(float).eps],
          np.r_[np.ones(len(columns)), np.inf],
        ),
        tol=1.0e-13,
        max_iter=1000,
      )
    except (ValueError, np.linalg.LinAlgError):
      return None
    if not result.success or not np.isfinite(result.x).all():
      return None
    work.iterations += int(result.nit)
    certified_gradient = gradient + subgradient_matrix @ result.x[:-1]
    residual = np.linalg.norm(certified_gradient - result.x[-1] * direction)
    if result.x[-1] <= 0.0 or residual > (
      1.0e-7 * max(1.0, np.linalg.norm(certified_gradient))
    ):
      return None
    return SupportRefinement(
      snapped, float(score), certified_gradient,
      refinement_iterations, refinement_evaluations, refinement_line_search_steps,
    )


class ProjectedScoreRegion:
  """Standardized projected scores, smooth away from scalar residual cusps.

  A one-dimensional block depends on the absolute standardized residual.
  At exactly zero the gradient routine chooses the zero subgradient, while
  the Hessian is marked unavailable so Newton can defer to its fallback.
  """

  def __init__(
    self,
    *,
    dimension: int,
    weights: FloatArray,
    method: str,
    xi_hat: Sequence[np.ndarray] | np.ndarray,
    sigma: Sequence[np.ndarray] | np.ndarray,
    sub_dim: Sequence[np.ndarray] | np.ndarray,
    degrees_freedom: Sequence[float] | np.ndarray | None,
    projections: FloatArray,
    equal_sub_dim: bool,
  ) -> None:
    if (
      isinstance(dimension, (bool, np.bool_))
      or not isinstance(dimension, (int, np.integer))
      or dimension < 1
    ):
      raise ValueError("dimension must be a positive integer")
    weights = np.asarray(weights, dtype=float)
    projections = np.asarray(projections, dtype=float)
    if (
      weights.ndim != 1 or weights.size == 0
      or not np.isfinite(weights).all() or np.any(weights <= 0.0)
      or not np.isclose(weights.sum(), 1.0, rtol=1.0e-10, atol=1.0e-12)
    ):
      raise ValueError("weights must be positive, finite, and sum to one")
    if (
      projections.ndim != 2 or projections.shape[1] != dimension
      or not np.isfinite(projections).all()
    ):
      raise ValueError("projections must be a finite matrix matching dimension")
    if len(xi_hat) != len(weights) or len(sub_dim) != len(weights):
      raise ValueError("each study needs an estimate and projection-row indices")
    if len(sigma) not in ({1, len(weights)} if equal_sub_dim else {len(weights)}):
      raise ValueError("covariances must match the number of studies")
    if degrees_freedom is not None and (
      len(degrees_freedom) != len(weights)
      or not np.isfinite(degrees_freedom).all()
      or np.any(np.asarray(degrees_freedom) <= 0.0)
    ):
      raise ValueError("degrees of freedom must be positive and match studies")
    if method not in {"HCauchy", "EHMP"} and len(weights) != 1:
      raise NotImplementedError(
        "analytic support derivatives are implemented for HCauchy and EHMP"
      )
    self.dimension = int(dimension)
    self.weights = np.asarray(weights, dtype=float)
    self.method = method
    self._single_test = len(self.weights) == 1

    raw_blocks: list[_RawProjectionBlock] = []
    for index, weight in enumerate(self.weights):
      raw_indices = np.asarray(sub_dim[index])
      if (
        raw_indices.ndim != 1 or raw_indices.size == 0
        or not np.isfinite(raw_indices).all()
        or np.any(raw_indices != np.floor(raw_indices))
        or np.any(raw_indices < 0) or np.any(raw_indices >= len(projections))
      ):
        raise ValueError("projection-row indices must be valid nonempty integer vectors")
      row_indices = raw_indices.astype(int)
      projection = np.asarray(projections[row_indices, :], dtype=float)
      estimate = np.asarray(xi_hat[index], dtype=float)
      covariance_index = index
      if equal_sub_dim and len(sigma) == 1:
        covariance_index = 0
      covariance = np.atleast_2d(np.asarray(sigma[covariance_index], dtype=float))
      if estimate.shape != (len(row_indices),) or not np.isfinite(estimate).all():
        raise ValueError("study estimates must be finite and match block dimensions")
      if (
        covariance.shape != (len(row_indices), len(row_indices))
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=0.0)
      ):
        raise ValueError("study covariance must be finite, symmetric, and match its block")
      try:
        np.linalg.cholesky(covariance)
      except np.linalg.LinAlgError as error:
        raise ValueError("study covariance must be positive definite") from error
      precision = np.linalg.inv(covariance)
      block_dimension = len(row_indices)
      block_df = None
      argument_scale = 1.0
      if degrees_freedom is not None:
        block_df = float(degrees_freedom[index])
        denominator_df = block_df + 1.0 - block_dimension
        if denominator_df <= 0.0:
          raise ValueError("Hotelling residual degrees of freedom must be positive")
        argument_scale = denominator_df / (block_dimension * block_df)
        if not self._single_test:
          if block_dimension == 1:
            required_df = 1.0
          elif self.method == "EHMP":
            required_df = float(block_dimension)
          elif block_dimension == 2:
            required_df = 2.5
          else:
            required_df = float(block_dimension + 1)
          if block_df < required_df:
            raise ValueError(
              "support certification requires a proven convex-score regime: "
              f"{self.method}, block dimension {block_dimension}, df >= {required_df:g}"
            )
      coordinate_indices = np.flatnonzero(
        np.any(projection != 0.0, axis=0)
      ).astype(np.int64)
      if coordinate_indices.size == 0:
        raise ValueError("a positive-weight projection has empty coordinate support")
      raw_blocks.append(
        _RawProjectionBlock(
          projection=projection,
          estimate=estimate,
          precision=precision,
          coordinate_indices=coordinate_indices,
          dimension=block_dimension,
          weight=float(weight),
          degrees_freedom=block_df,
          argument_scale=argument_scale,
        )
      )

    self._component_labels = self._build_component_labels(
      tuple(block.coordinate_indices for block in raw_blocks)
    )
    self.center = np.zeros(self.dimension)
    scale = np.ones(self.dimension)
    for label in np.unique(self._component_labels):
      global_indices = np.flatnonzero(self._component_labels == label)
      component_information = np.zeros(
        (global_indices.size, global_indices.size)
      )
      component_right_hand_side = np.zeros(global_indices.size)
      for block in raw_blocks:
        if self._component_labels[block.coordinate_indices[0]] != label:
          continue
        local_projection = block.projection[:, global_indices]
        component_information += (
          block.weight
          * local_projection.T
          @ block.precision
          @ local_projection
        )
        component_right_hand_side += (
          block.weight
          * local_projection.T
          @ block.precision
          @ block.estimate
        )
      if np.linalg.matrix_rank(component_information) < global_indices.size:
        raise ValueError(
          "positive-weight projections do not span the parameter space"
        )
      self.center[global_indices] = np.linalg.solve(
        component_information,
        component_right_hand_side,
      )
      approximate_covariance = np.linalg.inv(component_information)
      scale[global_indices] = np.sqrt(
        np.maximum(np.diag(approximate_covariance), 0.0)
      )
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
      raise ValueError("coordinate scales must be finite and strictly positive")
    self.scale = scale
    blocks = []
    maximum_standardized_residual = 0.0
    for block in raw_blocks:
      residual = block.projection @ self.center - block.estimate
      maximum_standardized_residual = max(
        maximum_standardized_residual,
        float(np.sqrt(max(0.0, residual @ block.precision @ residual))),
      )
      blocks.append(
        _ProjectionBlock(
          projection_scaled=block.projection * self.scale,
          residual_at_center=residual,
          precision=block.precision,
          coordinate_indices=block.coordinate_indices,
          dimension=block.dimension,
          weight=block.weight,
          degrees_freedom=block.degrees_freedom,
          argument_scale=block.argument_scale,
        )
      )
    self.blocks = tuple(blocks)
    # Absolute-coordinate scaling would make inconsistent studies appear
    # symmetric after a large common translation, falsely reflecting an
    # endpoint. The Mahalanobis residual is translation/unit invariant.
    self.centrally_symmetric = maximum_standardized_residual <= 1.0e-10

  def _build_component_labels(
    self,
    coordinate_blocks: tuple[NDArray[np.int64], ...],
  ) -> NDArray[np.int64]:
    parent = np.arange(self.dimension, dtype=np.int64)

    def find(index: int) -> int:
      while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = int(parent[index])
      return index

    def union(first: int, second: int) -> None:
      first_root = find(first)
      second_root = find(second)
      if first_root != second_root:
        parent[second_root] = first_root

    for indices in coordinate_blocks:
      for index in indices[1:]:
        union(int(indices[0]), int(index))
    roots = np.array([find(index) for index in range(self.dimension)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)

  def point_from_standardized(self, z: FloatArray) -> FloatArray:
    return self.center + self.scale * np.asarray(z, dtype=float)

  def standardized_from_point(self, point: FloatArray) -> FloatArray:
    return (np.asarray(point, dtype=float) - self.center) / self.scale

  def component(
    self,
    direction: FloatArray,
    *,
    base_z: FloatArray,
  ) -> ProjectedScoreComponent:
    """Restrict the score to components touched by ``direction``."""
    direction = np.asarray(direction, dtype=float)
    base_z = np.asarray(base_z, dtype=float)
    if direction.shape != (self.dimension,) or base_z.shape != (self.dimension,):
      raise ValueError("direction and base point must match the region dimension")
    active_coordinates = np.flatnonzero(direction != 0.0)
    if active_coordinates.size == 0:
      raise ValueError("direction must be nonzero")
    active_labels = np.unique(self._component_labels[active_coordinates])
    active_mask = np.isin(self._component_labels, active_labels)
    global_indices = np.flatnonzero(active_mask).astype(np.int64)
    local_lookup = np.full(self.dimension, -1, dtype=np.int64)
    local_lookup[global_indices] = np.arange(global_indices.size)
    active_blocks = []
    inactive_score = 0.0
    for block in self.blocks:
      is_active = bool(np.any(active_mask[block.coordinate_indices]))
      residual_at_base = (
        block.residual_at_center + block.projection_scaled @ base_z
      )
      if not is_active:
        inactive_score += block.weight * self._block_value(
          residual_at_base, block
        )
        continue
      if not np.all(active_mask[block.coordinate_indices]):
        raise RuntimeError("a projection crosses detected component boundaries")
      local_coordinates = local_lookup[block.coordinate_indices]
      active_blocks.append(
        _ProjectionBlock(
          projection_scaled=block.projection_scaled[:, global_indices],
          residual_at_center=residual_at_base,
          precision=block.precision,
          coordinate_indices=local_coordinates,
          dimension=block.dimension,
          weight=block.weight,
          degrees_freedom=block.degrees_freedom,
          argument_scale=block.argument_scale,
        )
      )
    return ProjectedScoreComponent(
      parent=self,
      global_indices=global_indices,
      blocks=tuple(active_blocks),
      constant_score=inactive_score,
      base_z=base_z,
    )

  @staticmethod
  def _f_density_derivative(
    argument: float,
    dfn: int,
    dfd: float,
    density: float,
  ) -> float:
    """Derivative of the F density with its q=4 limit at zero."""
    if argument <= 1.0e-12:
      if dfn == 2:
        return float(-density * (dfn + dfd) / dfd)
      if dfn == 4:
        return float(
          (dfn / dfd) ** (dfn / 2.0)
          / beta_function(dfn / 2.0, dfd / 2.0)
        )
      return 0.0
    log_derivative = (
      (dfn / 2.0 - 1.0) / argument
      - (dfn + dfd)
      / 2.0
      * (dfn / dfd)
      / (1.0 + dfn * argument / dfd)
    )
    return float(density * log_derivative)

  @staticmethod
  def _chi2_density_derivative(
    argument: float,
    degrees_freedom: int,
    density: float,
  ) -> float:
    """Derivative of the chi-squared density, including finite zero limits."""
    if argument <= 1.0e-12:
      if degrees_freedom == 2:
        return -0.25
      if degrees_freedom == 4:
        return 1.0 / (4.0 * gamma(2.0))
      return 0.0
    return float(
      density
      * ((degrees_freedom / 2.0 - 1.0) / argument - 0.5)
    )

  def _transform(
    self,
    pvalue: float,
    density: float,
    density_derivative: float,
    argument_scale: float,
  ) -> tuple[float, float, float]:
    pvalue = float(np.clip(pvalue, np.finfo(float).tiny, 1.0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
      inverse_pvalue = float(np.divide(1.0, pvalue))
      density_over_pvalue = float(np.multiply(density, inverse_pvalue))
      density_derivative_over_pvalue = float(
        np.multiply(density_derivative, inverse_pvalue)
      )
    if self._single_test:
      return (
        -pvalue,
        argument_scale * density,
        argument_scale * argument_scale * density_derivative,
      )
    if self.method == "EHMP":
      value = inverse_pvalue
      derivative = argument_scale * density_over_pvalue * inverse_pvalue
      second = (
        argument_scale
        * argument_scale
        * inverse_pvalue
        * (
          density_derivative_over_pvalue
          + 2.0 * density_over_pvalue * density_over_pvalue
        )
      )
      return value, derivative, second
    if pvalue < 1.0e-7:
      value = 2.0 / np.pi * inverse_pvalue
      derivative = (
        (2.0 / np.pi)
        * argument_scale
        * density_over_pvalue
        * inverse_pvalue
      )
      second = (
        (2.0 / np.pi)
        * argument_scale
        * argument_scale
        * inverse_pvalue
        * (
          density_derivative_over_pvalue
          + 2.0 * density_over_pvalue * density_over_pvalue
        )
      )
      return value, derivative, second
    angle = 0.5 * np.pi * pvalue
    sine = np.sin(angle)
    cotangent = 1.0 / np.tan(angle)
    value = cotangent
    derivative = 0.5 * np.pi * argument_scale * density / (sine * sine)
    second = (
      0.5
      * np.pi
      * argument_scale
      * argument_scale
      / (sine * sine)
      * (density_derivative + np.pi * density * density * cotangent)
    )
    return value, derivative, second

  def _block_distribution(
    self,
    argument: float,
    block: _ProjectionBlock,
  ) -> tuple[float, float, float]:
    evaluation_argument = argument
    if block.dimension == 1 and argument <= 1.0e-14:
      evaluation_argument = 1.0e-14
    if block.degrees_freedom is None:
      pvalue = float(chi2.sf(argument, block.dimension))
      density = float(chi2.pdf(evaluation_argument, block.dimension))
      density_derivative = self._chi2_density_derivative(
        evaluation_argument, block.dimension, density
      )
      return pvalue, density, density_derivative
    denominator_df = block.degrees_freedom + 1.0 - block.dimension
    pvalue = float(f.sf(argument, block.dimension, denominator_df))
    density = float(f.pdf(evaluation_argument, block.dimension, denominator_df))
    density_derivative = self._f_density_derivative(
      evaluation_argument, block.dimension, denominator_df, density
    )
    return pvalue, density, density_derivative

  def _block_value(
    self,
    residual: FloatArray,
    block: _ProjectionBlock,
  ) -> float:
    solved = block.precision @ residual
    quadratic = max(0.0, float(residual @ solved))
    argument = block.argument_scale * quadratic
    pvalue, density, density_derivative = self._block_distribution(
      argument, block
    )
    value, _, _ = self._transform(
      pvalue, density, density_derivative, block.argument_scale
    )
    return float(value)

  def _scalar_radial_derivatives(
    self,
    residual: float,
    block: _ProjectionBlock,
  ) -> tuple[float, float, float]:
    """Evaluate in |r| units without the singular chi-square/F density.

    Using the quadratic argument near zero introduces an artificial flat
    spot and catastrophic cancellation in the Hessian for q=1. Normal/t
    densities give the exact one-sided radial derivatives instead.
    """
    radial = abs(residual) * np.sqrt(block.precision[0, 0])
    if block.degrees_freedom is None:
      pvalue = float(2.0 * norm.sf(radial))
      density = float(2.0 * norm.pdf(radial))
      density_derivative = -radial * density
    else:
      df = block.degrees_freedom
      pvalue = float(2.0 * t.sf(radial, df))
      density = float(2.0 * t.pdf(radial, df))
      density_derivative = -(df + 1.0) * radial / (df + radial**2) * density
    return self._transform(pvalue, density, density_derivative, 1.0)

  def _evaluate(
    self,
    z: FloatArray,
    blocks: tuple[_ProjectionBlock, ...],
    *,
    initial_score: float = 0.0,
    with_hessian: bool = True,
  ) -> tuple[float, FloatArray, FloatArray | None]:
    z = np.asarray(z, dtype=float)
    score = float(initial_score)
    gradient = np.zeros_like(z)
    hessian = (
      np.zeros((z.size, z.size), dtype=float) if with_hessian else None
    )
    for block in blocks:
      residual = block.residual_at_center + block.projection_scaled @ z
      if block.dimension == 1:
        scalar_residual = float(residual[0])
        value, derivative, second = self._scalar_radial_derivatives(
          scalar_residual, block
        )
        radial_direction = (
          np.sqrt(block.precision[0, 0]) * block.projection_scaled[0]
        )
        score += block.weight * value
        gradient += (
          block.weight * derivative * np.sign(scalar_residual) * radial_direction
        )
        if hessian is not None:
          if scalar_residual == 0.0:
            # The classical Hessian does not exist at this cusp. Returning
            # nonfinite curvature explicitly disables the Newton step.
            hessian[:] = np.nan
          else:
            hessian += block.weight * second * np.outer(
              radial_direction, radial_direction
            )
        continue
      solved = block.precision @ residual
      quadratic = max(0.0, float(residual @ solved))
      argument = block.argument_scale * quadratic
      pvalue, density, density_derivative = self._block_distribution(
        argument, block
      )
      value, derivative, second = self._transform(
        pvalue, density, density_derivative, block.argument_scale
      )
      projected_solved = block.projection_scaled.T @ solved
      score += block.weight * value
      gradient += block.weight * derivative * 2.0 * projected_solved
      if hessian is not None:
        hessian += block.weight * (
          2.0
          * derivative
          * block.projection_scaled.T
          @ block.precision
          @ block.projection_scaled
          + 4.0 * second * np.outer(projected_solved, projected_solved)
        )
    return float(score), gradient, hessian

  def _score_only(
    self,
    z: FloatArray,
    blocks: tuple[_ProjectionBlock, ...],
    *,
    initial_score: float = 0.0,
  ) -> float:
    score = float(initial_score)
    for block in blocks:
      residual = block.residual_at_center + block.projection_scaled @ z
      score += block.weight * self._block_value(residual, block)
    return float(score)

  def _score_gradient_hessian(
    self,
    z: FloatArray,
  ) -> tuple[float, FloatArray, FloatArray]:
    z = np.asarray(z, dtype=float)
    if z.shape != (self.dimension,):
      raise ValueError("standardized point has the wrong shape")
    score, gradient, hessian = self._evaluate(z, self.blocks)
    if hessian is None:
      raise RuntimeError("internal Hessian evaluation was unexpectedly disabled")
    return score, gradient, hessian

  def _score_gradient(self, z: FloatArray) -> tuple[float, FloatArray]:
    z = np.asarray(z, dtype=float)
    if z.shape != (self.dimension,):
      raise ValueError("standardized point has the wrong shape")
    score, gradient, _ = self._evaluate(
      z,
      self.blocks,
      with_hessian=False,
    )
    return score, gradient

  def score(self, z: FloatArray) -> float:
    z = np.asarray(z, dtype=float)
    if z.shape != (self.dimension,):
      raise ValueError("standardized point has the wrong shape")
    return self._score_only(z, self.blocks)


__all__ = ["ProjectedScoreComponent", "ProjectedScoreRegion"]
