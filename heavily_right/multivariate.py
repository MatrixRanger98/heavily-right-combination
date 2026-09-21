"""Multidimensional combined-test confidence regions."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import chi2, f

from .combination import CombinationTest
from .empty_sets import (
  EmptyRegionError,
  EmptySetDiagnostics,
  EmptySetNumericalError,
  EmptySetPolicy,
)
from .minimum import MinimumDiagnostics, minimize_score_region
from .multivariate_empty import MultivariateEmptySetMixin
from .multivariate_legacy import LegacySupportMixin
from .multivariate_slices import MultivariateSliceMixin
from .projected_region import ProjectedScoreRegion
from .support import (
  SupportDiagnostics,
  _candidate_is_certified,
  _kkt_residual,
  solve_support_point,
)

FloatArray = NDArray[np.float64]
FormattedEstimates = np.ndarray | list[np.ndarray]
FormattedCovariances = np.ndarray | list[np.ndarray]
FormattedSubdimensions = np.ndarray | list[np.ndarray]
FormattedMultivariateInputs = tuple[
  FormattedEstimates,
  FormattedCovariances,
  FormattedSubdimensions,
  np.ndarray | None,
  np.ndarray,
  bool,
]


@dataclass(frozen=True)
class SupportIntervalResult:
  """Support-function interval with endpoint certificates."""

  lower: float
  upper: float
  lower_point: FloatArray
  upper_point: FloatArray
  lower_diagnostics: SupportDiagnostics
  upper_diagnostics: SupportDiagnostics
  central_symmetry_used: bool
  active_dimension: int
  minimum_point: FloatArray
  minimum_score: float = np.nan
  empty_set_diagnostics: EmptySetDiagnostics | None = None
  minimum_diagnostics: MinimumDiagnostics | None = None

  @property
  def success(self) -> bool:
    return self.lower_diagnostics.success and self.upper_diagnostics.success


class MetaAnalysisMD(
  MultivariateEmptySetMixin,
  LegacySupportMixin,
  MultivariateSliceMixin,
  CombinationTest,
):
  """Combined-test confidence regions for a vector parameter.

  ``dim`` is at least two; studies may measure different projections.
  ``Sigma`` contains covariance matrices of estimates, not observations.
  Inspect both endpoint diagnostics before using a support result.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import MetaAnalysisMD
  >>> analysis = MetaAnalysisMD(1, dim=2)
  >>> result = analysis.support_interval(
  ...     [1., -1.], xi_hat=[[0.2, 0.4]], Sigma=[np.eye(2) * 0.04], sub_dim=2,
  ... )
  >>> assert result.success
  >>> assert np.isclose(result.lower, np.array([1., -1.]) @ result.lower_point)
  >>> assert result.lower < -0.2 < result.upper
  """

  def __init__(
    self,
    arg: np.int_ | float | np.float64 | Sequence[float | np.float64] | np.ndarray,
    dim: np.int_ | float | np.float64,
    method: str = "HCauchy",
    level: float = 0.05,
    **kwargs: Any,
  ) -> None:
    if method != "HCauchy" and method != "EHMP":
      warnings.warn(
        "confidence regions and simultaneous intervals are guaranteed only "
        "for HCauchy or EHMP",
        stacklevel=2,
      )
    super().__init__(arg, method, level, **kwargs)
    if isinstance(dim, (bool, np.bool_)) or not np.isfinite(dim) or dim <= 1 or int(dim) != dim:
      raise ValueError(
        "dim must be an integer of at least 2; use MetaAnalysis1D for one dimension"
      )
    self._dimension = int(np.int_(dim))

  @property
  def dim(self) -> int:
    return self._dimension

  # check if the dimensions of sub studies are the same
  def _check_equal_sub_dim(
    self, xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray
  ) -> bool:
    # format xi_hat and decide if the dimensions of estimates in all studies are the same
    equal_sub_dim = True
    try:
      xi_hat = np.array(xi_hat)
    except ValueError:
      equal_sub_dim = False
    return equal_sub_dim

  # check and format input variables. note that sub_dim could originally be either an integer, a 1d-array of integers, list of indexes or a 2d-array of indexes (indexing the row of projs). but the output sub_dim is either a list of length num_test, or a 2d-array of shape (num_test, len(sub_dim[0])
  def _format_input(
    self,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None,
    projs: Sequence[np.ndarray] | np.ndarray | None,
    equal_sub_dim: bool | None = None,
  ) -> FormattedMultivariateInputs:
    # check format of projections. it is of shape (num_proj, dim)
    if projs is None:
      projs = np.eye(self._dimension)
    else:
      projs = np.array(projs)
    if projs.shape[1] != self._dimension:
      raise ValueError("The dimenion in projs do not match that of the parameter.")
    # get the number of projections
    num_proj = projs.shape[0]

    # check if the dimensions in all studies are the same
    if equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)
    # case 1: the dimensions in all studies are the same
    if equal_sub_dim:
      # format xi_hat and Sigma into np.ndarray. xi_hat is of shape (num_test, sub_dim_value)
      xi_hat = np.array(xi_hat)
      # deal with the case where all subdimensions are 1
      if xi_hat.ndim == 1:
        xi_hat = np.expand_dims(xi_hat, axis=-1)
      Sigma = np.array(Sigma)
      # if all Sigma matrices are the same, insert a dimension at the beginning so that matrix operations are correctly broadcasted
      if Sigma.ndim == 2:
        Sigma = np.expand_dims(Sigma, axis=0)
      # deal with the case where subdimensions are all 1
      elif Sigma.ndim == 1:
        Sigma = np.expand_dims(Sigma, axis=(-2, -1))
      # if sub_dim is an integer, get a 2d-array of indexes indicating the rows of projections used in each study. the array is obtained in a cyclic manner.
      if isinstance(sub_dim, (int, float, np.int_, np.float64)):
        sub_dim = (
          np.arange(self.num_test * int(np.int_(sub_dim))).reshape(self.num_test, -1) % num_proj
        )
      # if sub_dim is already a 2d-array of indexes, everything is all set
      elif isinstance(sub_dim, Iterable):
        sub_dim = np.array(sub_dim)
        # dealing with the corner case where sub_dim is a 1d-array of the same integer
        if sub_dim.ndim == 1:
          sub_dim = (
            np.arange(self.num_test * int(np.int_(sub_dim[0]))).reshape(self.num_test, -1)
            % num_proj
          )
      # catch illegal input of sub_dim
      else:
        raise TypeError(
          "sub_dim should be either an int, a 1d-array or 2d-array of ints."
        )

    # case 2: the dimension of studies are not the same
    else:
      # xi_hat and Sigma are both list of length num_test
      xi_hat = [np.array(x) for x in xi_hat]
      Sigma = [np.array(x) for x in Sigma]
      # if sub_dim is a list of integers, get the list of projection indexes used in each study in a cyclic manner. the output is a list of 1d-arrays of integers.
      try:
        sub_dim = np.array(sub_dim)
        sub_dim = np.split(np.arange(sum(sub_dim)), np.cumsum(sub_dim)[:-1])
        sub_dim = [x % num_proj for x in sub_dim]
      # if sub_dim is already a list of projection indexes, everything is all set.
      except ValueError:
        sub_dim = [np.array(x) for x in sub_dim]

    # format the degrees of freedom. if df is None, do nothing.
    if df is None:
      pass
    # if df is a list of integers, check the length
    elif isinstance(df, Iterable):
      df = np.asarray(df)
      if df.shape[0] != self.num_test:
        raise ValueError("The length of df should be the same as num_test.")
    # if df is a single integer, repeat it to get a list (1d-array)
    elif isinstance(df, (float, int, np.float64, np.int_)):
      df = np.full(self.num_test, float(df))
    # catch illegal input of df
    else:
      raise TypeError("df should be either an int or an array of ints.")

    # return formatted variables along with the judgment
    return xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim

  # get p-values from estimates for each study
  def get_p_vector(
    self,
    point: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> np.ndarray:
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)

    # to distinguish sub_dim_index with sub_dim_value
    sub_dim_index = sub_dim

    # prepare the projections of points in each study and then get chi2 or hotelling scores. note that in calculating xi, we have taken into account the case where point could be an nd-array with the last dimension being the dimension of the parameter
    point_projs = projs @ np.expand_dims(point, axis=-1)
    if equal_sub_dim:
      # for each point, xi is a 2d-array of shape (num_test, sub_dim_value), where sub_dim_value is an integer. together xi is of shape (*point.shape[:-1], num_test, sub_dim_value). the ellipsis allows for the flexity of point.shape.
      xi = np.squeeze(point_projs, axis=-1)[..., sub_dim_index]
      sub_dim_value = len(sub_dim_index[0])
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.squeeze(
        np.expand_dims(xi - xi_hat, axis=-2)
        @ np.linalg.inv(Sigma)
        @ np.expand_dims(xi - xi_hat, axis=-1),
        (-1, -2),
      )
    else:
      # xi is a list of nd-arrays, and is of length num_test. xi[i] is of shape (*point.shape[:-1], sub_dim_value[i]). here sub_dim_value is a 1d-array of integers, and is of length num_test
      xi = [np.squeeze(point_projs, axis=-1)[..., x] for x in sub_dim_index]
      sub_dim_value = np.array([len(x) for x in sub_dim_index])
      # all elements in this list have the same shape point.shape[:-1]
      chi2_f_score_list = [
        np.squeeze(
          np.expand_dims(x - y, axis=-2)
          @ np.linalg.inv(z)
          @ np.expand_dims(x - y, axis=-1),
          (-1, -2),
        )
        for x, y, z in zip(xi, xi_hat, Sigma, strict=True)
      ]
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.moveaxis(np.array(chi2_f_score_list), 0, -1)

    # if df is not provided, we use chi-squared distribution for each individual study
    if df is None:
      # sub_dim_value is broadcasted in chi2.sf
      return np.maximum(chi2.sf(chi2_f_score, sub_dim_value), 1e-150)
    # if df is provided, we use hotelling T-squared for each individual study
    else:
      # the arguments in f.sf are broadcasted
      return np.maximum(
        f.sf(
          chi2_f_score * (df + 1 - sub_dim_value) / (sub_dim_value * df),
          sub_dim_value,
          df + 1 - sub_dim_value,
        ),
        1e-150,
      )
    
  # get p-values from estimates for each study
  def get_dif_vector(
    self,
    point: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> np.ndarray:
    """Deprecated historical density helper, not a certified score derivative.

    Its legacy Hotelling-density convention is retained for compatibility.
    Inference uses the tested analytic derivatives in ``projected_region``.
    """
    warnings.warn(
      "get_dif_vector is a deprecated legacy helper, not a validated score "
      "gradient; use the current inference methods instead",
      DeprecationWarning, stacklevel=2,
    )
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)

    # to distinguish sub_dim_index with sub_dim_value
    sub_dim_index = sub_dim

    # prepare the projections of points in each study and then get chi2 or hotelling scores. note that in calculating xi, we have taken into account the case where point could be an nd-array with the last dimension being the dimension of the parameter
    point_projs = projs @ np.expand_dims(point, axis=-1)
    if equal_sub_dim:
      # for each point, xi is a 2d-array of shape (num_test, sub_dim_value), where sub_dim_value is an integer. together xi is of shape (*point.shape[:-1], num_test, sub_dim_value). the ellipsis allows for the flexity of point.shape.
      xi = np.squeeze(point_projs, axis=-1)[..., sub_dim_index]
      sub_dim_value = len(sub_dim_index[0])
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.squeeze(
        np.expand_dims(xi - xi_hat, axis=-2)
        @ np.linalg.inv(Sigma)
        @ np.expand_dims(xi - xi_hat, axis=-1),
        (-1, -2),
      )
    else:
      # xi is a list of nd-arrays, and is of length num_test. xi[i] is of shape (*point.shape[:-1], sub_dim_value[i]). here sub_dim_value is a 1d-array of integers, and is of length num_test
      xi = [np.squeeze(point_projs, axis=-1)[..., x] for x in sub_dim_index]
      sub_dim_value = np.array([len(x) for x in sub_dim_index])
      # all elements in this list have the same shape point.shape[:-1]
      chi2_f_score_list = [
        np.squeeze(
          np.expand_dims(x - y, axis=-2)
          @ np.linalg.inv(z)
          @ np.expand_dims(x - y, axis=-1),
          (-1, -2),
        )
        for x, y, z in zip(xi, xi_hat, Sigma, strict=True)
      ]
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.moveaxis(np.array(chi2_f_score_list), 0, -1)

    # if df is not provided, we use chi-squared distribution for each individual study
    if df is None:
      # sub_dim_value is broadcasted in chi2.sf
      return np.maximum(chi2.pdf(chi2_f_score, sub_dim_value), 1e-150)
    # if df is provided, we use hotelling T-squared for each individual study
    else:
      # the arguments in f.sf are broadcasted
      return np.maximum(
        f.pdf(
          chi2_f_score * (df + 1 - sub_dim_value) * (df + 1) / (sub_dim_value * df),
          sub_dim_value,
          df + 1 - sub_dim_value,
        ),
        1e-150,
      )

  # check if a point or points are covered by the confidence region
  def check_cover(
    self,
    point: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> bool:
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)
    current = self.get_global_score(
        self.get_p_vector(
          point,
          xi_hat,
          Sigma,
          sub_dim,
          df,
          projs,
          format_input=False,
          equal_sub_dim=equal_sub_dim,
        )
      )
    print('current score', current)
    print('threshold', self.threshold)
    return (current <= self.threshold)

  # find the minimizer in the confidence region (as a global point estimate)
  def find_minimizer(
    self,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    x0: np.ndarray | None = None,
    method: str = "Powell",
    print_res: bool = False,
    require_grad: bool = False,
    **kwargs: Any,
  ) -> np.ndarray:
    """Find a checked score minimum; convex methods use standardized coordinates."""
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)
    options = dict(kwargs)
    options.setdefault("maxiter", 1000)
    if method == "Powell":
      options.setdefault("xtol", 1.0e-7)
    if self.method in {"HCauchy", "EHMP"} or self.num_test == 1:
      region = ProjectedScoreRegion(
        dimension=self.dim, weights=np.asarray(self.weights), method=self.method,
        xi_hat=xi_hat, sigma=Sigma, sub_dim=sub_dim, degrees_freedom=df,
        projections=np.asarray(projs, dtype=float), equal_sub_dim=bool(equal_sub_dim),
      )
      if self.num_test == 1:
        return region.center.copy()
      start = np.zeros(self.dim) if x0 is None else region.standardized_from_point(np.asarray(x0, dtype=float))
      if start.shape != (self.dim,) or not np.isfinite(start).all():
        raise ValueError("x0 must be a finite vector matching the parameter dimension")
      gradient = (lambda z: region._score_gradient(z)[1]) if require_grad else None
      preliminary = minimize(region.score, start, method=method, jac=gradient, options=options)
      result = minimize_score_region(
        region, np.asarray(preliminary.x), cutoff=float(self.threshold),
        maxiter=options["maxiter"],
      )
      if not result.success:
        raise EmptySetNumericalError("score minimum was not certified: " + result.diagnostics.message)
      point = region.point_from_standardized(result.point)
    else:
      if require_grad:
        raise NotImplementedError("analytic minimum gradients are supported only for HCauchy and EHMP")
      def objective(point: np.ndarray) -> float:
        return float(self.get_global_score(self.get_p_vector(
          point, xi_hat, Sigma, sub_dim, df, projs,
          format_input=False, equal_sub_dim=equal_sub_dim,
        )))
      if x0 is None:
        rows = np.concatenate([np.asarray(rows, dtype=int) for rows in sub_dim])
        estimates = np.concatenate([np.atleast_1d(estimate) for estimate in xi_hat])
        x0 = np.linalg.lstsq(np.asarray(projs)[rows], estimates, rcond=None)[0]
      result = minimize(objective, x0=x0, method=method, options=options)
      if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise EmptySetNumericalError(f"local score minimization failed: {result.message}")
      point = np.asarray(result.x, dtype=float)
    if print_res:
      print(f"The checked minimizer is {point}")
    return point


  def _single_block_support(
    self, region: ProjectedScoreRegion, direction: FloatArray,
  ) -> SupportIntervalResult:
    """Solve the one-study ellipsoid analytically, including oblique projections.

    Its score representation is bounded above, so an exterior quadratic
    penalty is not a suitable global solver. The quadratic form is sufficient
    even when the radial score itself is not convex.
    """
    block = region.blocks[0]
    dimension = block.dimension
    if block.degrees_freedom is None:
      cutoff_quadratic = float(chi2.isf(self.level, dimension))
    else:
      cutoff_quadratic = float(f.isf(
        self.level, dimension, block.degrees_freedom + 1.0 - dimension,
      ) / block.argument_scale)
    residual = block.residual_at_center
    minimum_quadratic = float(residual @ block.precision @ residual)
    minimum_score = float(region.score(np.zeros(self.dim)))
    budget = cutoff_quadratic - minimum_quadratic
    if budget < 0.0:
      raise EmptyRegionError(
        point=region.center.copy(), p_values=np.array([-minimum_score]),
        minimum_score=minimum_score, threshold=float(self.threshold),
      )
    if budget == 0.0:
      raise RuntimeError("the single-block region is degenerate; no regular support boundary exists")
    projection = block.projection_scaled
    information = projection.T @ block.precision @ projection
    b = direction * region.scale
    b_unit = b / np.linalg.norm(b)
    solved = np.linalg.solve(information, b_unit)
    length_squared = float(b_unit @ solved)
    if not np.isfinite(length_squared) or length_squared <= 0.0:
      raise RuntimeError("the single-block ellipsoid could not be resolved numerically")
    displacement = np.sqrt(budget / length_squared) * solved
    # A one-study score is -p. Small absolute probability error is not enough
    # at a very small alpha: certify relative to that tail probability too.
    score_scale = min(self.level, abs(float(self.threshold) - minimum_score))

    def endpoint(sign: float) -> tuple[FloatArray, SupportDiagnostics]:
      z = sign * displacement
      point = region.point_from_standardized(z)
      value, gradient, _ = region._evaluate(z, region.blocks, with_hessian=False)
      local_direction = sign * b
      eta = float(gradient @ local_direction) / float(local_direction @ local_direction)
      kkt = _kkt_residual(gradient, local_direction)
      boundary = float(value - self.threshold)
      success = _candidate_is_certified(
        optimizer_success=True, value=float(direction @ point), eta=eta,
        boundary_error=boundary / score_scale, cutoff=1.0, kkt_residual=kkt,
      )
      return point, SupportDiagnostics(
        success=success, solver="analytic-ellipsoid", score=value,
        cutoff=float(self.threshold), boundary_error=boundary, kkt_residual=kkt,
        eta=eta, iterations=0, evaluations=1, line_search_steps=0,
        message="analytic full-rank single-study ellipsoid with independent endpoint check",
        certificate_kind="analytic-ellipsoid", score_scale=score_scale,
      )

    lower_point, lower_diagnostics = endpoint(-1.0)
    upper_point, upper_diagnostics = endpoint(1.0)
    return SupportIntervalResult(
      lower=float(direction @ lower_point), upper=float(direction @ upper_point),
      lower_point=lower_point, upper_point=upper_point,
      lower_diagnostics=lower_diagnostics, upper_diagnostics=upper_diagnostics,
      central_symmetry_used=True, active_dimension=self.dim,
      minimum_point=region.center.copy(), minimum_score=minimum_score,
    )

  def _support_interval_once(
    self,
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
    **kwargs: Any,
  ) -> SupportIntervalResult:
    """Project the region using Newton KKT solves and certified fallbacks.

    The overlap graph limits dense Hessian formation to connected components
    touched by ``direction``. Components larger than
    ``dense_newton_max_dimension`` bypass dense Newton and use SLSQP.
    """
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (self._dimension,) or not np.isfinite(direction).all():
      raise ValueError("direction must be finite and match the parameter dimension")
    if not np.any(direction != 0.0):
      raise ValueError("direction must be nonzero")
    for name, value, lower_bound in (
      ("dense_newton_max_dimension", dense_newton_max_dimension, 1),
      ("maxiter", maxiter, 1), ("newton_maxiter", newton_maxiter, 0),
    ):
      if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower_bound:
        raise ValueError(f"{name} must be an integer of at least {lower_bound}")
    if not np.isfinite(ftol) or not 0.0 < ftol < 1.0:
      raise ValueError("ftol must lie strictly between zero and one")
    if x0 is not None:
      x0 = np.asarray(x0, dtype=float)
      if x0.shape != (self.dim,) or not np.isfinite(x0).all():
        raise ValueError("x0 must be a finite vector matching the parameter dimension")
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)

    region = ProjectedScoreRegion(
      dimension=self._dimension,
      weights=np.asarray(self.weights, dtype=float),
      method=self.method,
      xi_hat=xi_hat,
      sigma=Sigma,
      sub_dim=sub_dim,
      degrees_freedom=df,
      projections=np.asarray(projs, dtype=float),
      equal_sub_dim=bool(equal_sub_dim),
    )
    def check_returned_points(result: SupportIntervalResult) -> SupportIntervalResult:
      # Standardized calculations can be accurate even when adding a huge
      # location rounds away an endpoint displacement in the public float.
      # Report the score of the actual returned point, never silently certify
      # only its more accurate internal representation.
      def check(point: FloatArray, diagnostics: SupportDiagnostics) -> SupportDiagnostics:
        physical_score = float(self.get_global_score(self.get_p_vector(
          point, xi_hat, Sigma, sub_dim, df, projs,
          format_input=False, equal_sub_dim=equal_sub_dim,
        )))
        boundary_error = physical_score - float(self.threshold)
        reconstruction_error = abs(physical_score - diagnostics.score)
        reconstructed = bool(
          np.isfinite(physical_score)
          and abs(boundary_error) / diagnostics.score_scale <= 2.0e-8
        )
        return replace(
          diagnostics, success=diagnostics.success and reconstructed,
          score=physical_score, boundary_error=boundary_error,
          evaluations=diagnostics.evaluations + 1,
          reconstruction_error=reconstruction_error,
          message=diagnostics.message + (
            "" if reconstructed else
            "; returned-coordinate boundary check failed: recenter/rescale "
            "inputs to avoid floating-point reconstruction loss"
          ),
        )
      return replace(
        result,
        lower_diagnostics=check(result.lower_point, result.lower_diagnostics),
        upper_diagnostics=check(result.upper_point, result.upper_diagnostics),
      )

    if self.num_test == 1:
      if self.level <= 1.0e-150:
        raise EmptySetNumericalError(
          "the requested level reaches the public p-value floor; "
          "the clipped score cannot certify this confidence region"
        )
      return check_returned_points(self._single_block_support(region, direction))
    base_z = np.zeros(self._dimension)
    if not region.centrally_symmetric and find_min:
      if x0 is not None:
        base_z = region.standardized_from_point(np.asarray(x0, dtype=float))
      minimizer_options = dict(kwargs)
      minimizer_options.setdefault("maxiter", maxiter)
      if method == "Powell":
        minimizer_options.setdefault("xtol", 1.0e-7)
      minimum = minimize(
        region.score,
        x0=base_z,
        method=method,
        options=minimizer_options,
      )
      base_z = np.asarray(minimum.x, dtype=float)
    elif not region.centrally_symmetric and x0 is not None:
      base_z = region.standardized_from_point(np.asarray(x0, dtype=float))
    # A feasible but nonminimal base could understate support when inactive
    # connected components are held fixed. Independently validate/refine it,
    # including when the caller supplies a reusable minimum via find_min=False.
    minimum_result = minimize_score_region(
      region, base_z, cutoff=float(self.threshold), maxiter=maxiter, ftol=ftol,
    )
    base_z = minimum_result.point
    base_score = minimum_result.score
    if not np.isfinite(base_score):
      raise RuntimeError("the confidence-region minimum score was nonfinite")
    if base_score > self.threshold:
      if not (minimum_result.success and minimum_result.diagnostics.empty_certified):
        raise EmptySetNumericalError(
          "no feasible point was found, but emptiness was not established: "
          + minimum_result.diagnostics.message
        )
      base_point = region.point_from_standardized(base_z)
      p_values = np.asarray(
        self.get_p_vector(
          base_point,
          xi_hat,
          Sigma,
          sub_dim,
          df,
          projs,
          format_input=False,
          equal_sub_dim=equal_sub_dim,
        ),
        dtype=float,
      )
      raise EmptyRegionError(
        point=base_point,
        p_values=p_values,
        minimum_score=float(base_score),
        threshold=float(self.threshold),
        minimum_diagnostics=minimum_result.diagnostics,
      )

    if not minimum_result.success:
      raise EmptySetNumericalError(
        "a feasible point was found, but the minimum needed for component "
        "decomposition was not certified: " + minimum_result.diagnostics.message
      )

    component = region.component(direction, base_z=base_z)
    local_direction = (
      direction[component.global_indices]
      * region.scale[component.global_indices]
    )
    component_newton_maxiter = (
      newton_maxiter
      if component.dimension <= dense_newton_max_dimension
      else 0
    )
    zero = np.zeros(component.dimension)
    upper_support = solve_support_point(
      score=component.score,
      score_gradient=component._score_gradient,
      score_gradient_hessian=component._score_gradient_hessian,
      feasible_point=zero,
      direction=local_direction,
      cutoff=float(self.threshold),
      ftol=ftol,
      maxiter=maxiter,
      newton_maxiter=component_newton_maxiter,
      candidate_refiner=component.certify_support_candidate,
      nonsmooth_solver=component.solve_nonsmooth_support_candidate,
    )
    upper_z = component.full_standardized_point(upper_support.point)
    upper_point = region.point_from_standardized(upper_z)

    upper_diagnostics = upper_support.diagnostics
    lower_diagnostics = None
    symmetry_used = False
    if region.centrally_symmetric and upper_diagnostics.success:
      lower_z = -upper_z
      lower_point = region.point_from_standardized(lower_z)
      reflected_delta = lower_z[component.global_indices] - base_z[component.global_indices]
      reflected_score, reflected_gradient = component._score_gradient(reflected_delta)
      reflected_eta = float(reflected_gradient @ -local_direction) / float(local_direction @ local_direction)
      reflected_residual = _kkt_residual(reflected_gradient, -local_direction)
      reflected_error = float(reflected_score - self.threshold)
      reflected_success = _candidate_is_certified(
        optimizer_success=True, value=float(direction @ lower_point), eta=reflected_eta,
        boundary_error=reflected_error / upper_diagnostics.score_scale, cutoff=1.0,
        kkt_residual=reflected_residual,
      )
      reflection = SupportDiagnostics(
        success=reflected_success,
        solver="central-symmetry-reflection",
        score=reflected_score,
        cutoff=upper_diagnostics.cutoff,
        boundary_error=reflected_error,
        kkt_residual=reflected_residual,
        eta=reflected_eta,
        iterations=0,
        evaluations=1,
        line_search_steps=0,
        message="reflected endpoint independently checked against its own score and gradient",
        score_scale=upper_diagnostics.score_scale,
      )
      if reflected_success:
        lower_diagnostics = reflection
        symmetry_used = True
    if lower_diagnostics is None:
      lower_support = solve_support_point(
        score=component.score,
        score_gradient=component._score_gradient,
        score_gradient_hessian=component._score_gradient_hessian,
        feasible_point=zero,
        direction=-local_direction,
        cutoff=float(self.threshold),
        ftol=ftol,
        maxiter=maxiter,
        newton_maxiter=component_newton_maxiter,
        candidate_refiner=component.certify_support_candidate,
        nonsmooth_solver=component.solve_nonsmooth_support_candidate,
      )
      lower_z = component.full_standardized_point(lower_support.point)
      lower_point = region.point_from_standardized(lower_z)
      lower_diagnostics = lower_support.diagnostics

    lower = float(direction @ lower_point)
    upper = float(direction @ upper_point)
    return check_returned_points(SupportIntervalResult(
      lower=lower,
      upper=upper,
      lower_point=lower_point,
      upper_point=upper_point,
      lower_diagnostics=lower_diagnostics,
      upper_diagnostics=upper_diagnostics,
      central_symmetry_used=symmetry_used,
      active_dimension=component.dimension,
      minimum_point=region.point_from_standardized(base_z),
      minimum_score=float(base_score),
      minimum_diagnostics=minimum_result.diagnostics,
    ))

  # get simultaneous confidence interval
  def simultaneous_interval(
    self,
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
    print_res: bool = False,
    lambda_inv: float = np.exp(-20),
    solver: str = "kkt",
    return_diagnostics: bool = False,
    newton_maxiter: int = 80,
    dense_newton_max_dimension: int = 250,
    empty_set_policy: EmptySetPolicy | None = None,
    study_labels: Sequence[str] | None = None,
    **kwargs: Any,
  ) -> (
    tuple[float, float, FloatArray, FloatArray]
    | tuple[float, float, FloatArray, FloatArray, SupportIntervalResult]
  ):
    """Return the legacy interval tuple, optionally with support diagnostics."""
    if solver == "legacy":
      warnings.warn(
        "solver='legacy' returns historical penalty estimates without endpoint "
        "certificates; use the default solver for validated inference",
        RuntimeWarning, stacklevel=2,
      )
      if empty_set_policy is not None:
        raise ValueError(
          "empty-set handling is available only with the default KKT solver"
        )
      return self._legacy_simultaneous_interval(
        direction=direction,
        xi_hat=xi_hat,
        Sigma=Sigma,
        sub_dim=sub_dim,
        df=df,
        projs=projs,
        format_input=format_input,
        equal_sub_dim=equal_sub_dim,
        x0=x0,
        find_min=find_min,
        method=method,
        print_res=print_res,
        lambda_inv=lambda_inv,
        **kwargs,
      )
    if solver != "kkt":
      raise ValueError("solver must be 'kkt' or 'legacy'")
    result = self.support_interval(
      direction=direction,
      xi_hat=xi_hat,
      Sigma=Sigma,
      sub_dim=sub_dim,
      df=df,
      projs=projs,
      format_input=format_input,
      equal_sub_dim=equal_sub_dim,
      x0=x0,
      find_min=find_min,
      method=method,
      newton_maxiter=newton_maxiter,
      dense_newton_max_dimension=dense_newton_max_dimension,
      empty_set_policy=empty_set_policy,
      study_labels=study_labels,
      **kwargs,
    )
    if not result.success:
      raise RuntimeError(
        "support-function solve failed: "
        f"lower={result.lower_diagnostics.message}; "
        f"upper={result.upper_diagnostics.message}"
      )
    if print_res:
      print(
        "Simultaneous confidence interval for "
        f"<{np.asarray(direction)}, x> is ({result.lower}, {result.upper}); "
        f"active dimension={result.active_dimension}; "
        f"upper solver={result.upper_diagnostics.solver}, "
        f"boundary error={result.upper_diagnostics.boundary_error:.3g}, "
        f"KKT residual={result.upper_diagnostics.kkt_residual:.3g}."
      )
    legacy_result = (
      result.lower,
      result.upper,
      result.lower_point,
      result.upper_point,
    )
    if return_diagnostics:
      return (*legacy_result, result)
    return legacy_result



__all__ = ["MetaAnalysisMD", "SupportIntervalResult"]
