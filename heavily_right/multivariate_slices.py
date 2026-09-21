"""Line and plane slice helpers for multidimensional confidence regions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

if TYPE_CHECKING:
  from .multivariate import MetaAnalysisMD


class MultivariateSliceMixin:
  """Compatibility slice calculations separated from core region inference."""

  def interval_slice(
    self: MetaAnalysisMD,
    point: Sequence[float | np.float64] | np.ndarray,
    direction: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    new_point: np.float64 | float | None=None,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    method: str = "brentq",
    print_res: bool = False,
  ) -> tuple[float, float, float, np.ndarray]:
    # format point and direction
    point = np.array(point)
    direction = np.array(direction)
    # DONT normalize direction
    # direction = direction / np.sqrt((direction**2).sum())
    if new_point is None:
      new_point = point @ direction

    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)

    # print(xi_hat.shape, Sigma.shape, sub_dim.shape, df.shape, projs.shape)
    func = lambda x: self.get_global_score(
      self.get_p_vector(
        point + (x - new_point) * direction,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )
    # get estimates of lower and upper bounds
    if equal_sub_dim:
      rebuild = np.squeeze(
        np.linalg.pinv(projs[sub_dim, :]) @ np.expand_dims(xi_hat, axis=-1), -1
      )
    else:
      rebuild = np.array(
        [np.linalg.pinv(projs[x, :]) @ y for x, y in zip(sub_dim, xi_hat, strict=True)]
      )
    low = min((rebuild - point) @ direction)+new_point
    high = max((rebuild - point) @ direction)+new_point
    # print(low, high)
    # find minimizer of the func
    minimizer = minimize_scalar(func, method="Brent").x
    if low >= minimizer:
      low = minimizer - 1
    if high <= minimizer:
      high = minimizer + 1
    # print(minimizer, low, high)
    # deal with the case that the solution set is empty
    if func(minimizer) > self.threshold:
      print(f"{1-self.level} confidence interval is empty.")
      root_low = minimizer
      root_high = minimizer
    # when the solution is not empty
    else:
      if method == "brentq":
        # get correct lower and upper bounds
        low_1 = high_1 = minimizer
        while True:
          if func(low) < self.threshold:
            low_1 = low
            low = 2 * low - minimizer
          else:
            break
        while True:
          if func(high) < self.threshold:
            high_1 = high
            high = 2 * high - minimizer
          else:
            break
        # find the two roots
        root_low = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[low, low_1], method="brentq"
        ).root
        root_high = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[high_1, high], method="brentq"
        ).root
      else:
        raise NotImplementedError("The root finding method is not implemented.")
      if print_res:
        if root_low < root_high:
          print(
            f"{1-self.level} confidence interval is [{round(root_low,4)}, {round(root_high,4)}] along the direction {direction}. The point estimate {point+round(minimizer,4)*direction}."
          )
        else:
          print(
            f"Point estimate along the given line is {point+round(minimizer,4)*direction}."
          )
    return root_low, root_high, minimizer, direction

  def area_slice(
    self: MetaAnalysisMD,
    point: Sequence[float | np.float64] | np.ndarray,
    direction_1: Sequence[float | np.float64] | np.ndarray,
    direction_2: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    new_point_x: np.float64 | float | None=None,
    new_point_y: np.float64 | float | None=None,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    xrange: Sequence[float | np.float64] | np.ndarray | None = None,
    yrange: Sequence[float | np.float64] | np.ndarray | None = None,
    levels: Sequence[float] = (0.005, 0.01, 0.025, 0.05, 0.1),
    delta: float | None = None,
    ax: Any = None,
    clabel: bool = True,
    strs: Sequence[str] | None = None,
    **kwargs: Any,
  ) -> Any:
    # format point and directions
    point = np.array(point)
    direction_1 = np.array(direction_1)
    direction_2 = np.array(direction_2)
    # direction_1 = direction_1 / np.sqrt((direction_1**2).sum())
    # direction_2 = direction_2 - (direction_2 @ direction_1) * direction_1
    # direction_2 = direction_2 / np.sqrt((direction_2**2).sum())

    if new_point_x is None:
      new_point_x = point @ direction_1
    if new_point_y is None:
      new_point_y = point @ direction_2

    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self._format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self._check_equal_sub_dim(xi_hat)

    # if xrange or yrange is not given, we provide some suggestions of lower, upper and center of x and y based on observed data. but note that we do not automatically proceed with the plot after providing the information.
    if xrange is None or yrange is None:
      if equal_sub_dim:
        rebuild = np.squeeze(
          np.linalg.pinv(projs[sub_dim, :]) @ np.expand_dims(xi_hat, axis=-1), -1
        )
      else:
        rebuild = np.array(
          [np.linalg.pinv(projs[x, :]) @ y for x, y in zip(sub_dim, xi_hat, strict=True)]
        )
      xlow = min((rebuild - point) @ direction_1)+new_point_x
      x0 = self.weights @ ((rebuild - point) @ direction_1)+new_point_x
      xhigh = max((rebuild - point) @ direction_1)+new_point_x
      if xlow >= x0:
        xlow = x0 - 1
      if xhigh <= x0:
        xhigh = x0 + 1
      ylow = min((rebuild - point) @ direction_2)+new_point_y
      y0 = self.weights @ ((rebuild - point) @ direction_2)+new_point_y
      yhigh = max((rebuild - point) @ direction_2)+new_point_y
      if ylow >= y0:
        ylow = y0 - 1
      if yhigh <= y0:
        yhigh = y0 + 1
      print(
        f"The centers of x, y are approximately {x0}, {y0}. Possible xrange: [{xlow}, {xhigh}]. Possible yrange: [{ylow}, {yhigh}]"
      )
    # otherwise, use the xrange and yrange provided by user
    else:
      xlow, xhigh = xrange
      ylow, yhigh = yrange

    # create meshgrid
    if delta is None:
      delta = min(xhigh - xlow, yhigh - ylow) / 100
    x_seq = np.arange(xlow, xhigh, delta)
    y_seq = np.arange(ylow, yhigh, delta)
    x_grid, y_grid = np.meshgrid(x_seq, y_seq)
    # print(x_grid, y_grid)
    # calculate global p_scores for the meshgrid
    func = lambda x, y: self.get_global_score(
      self.get_p_vector(
        point
        + np.expand_dims(x-new_point_x, axis=-1) * direction_1
        + np.expand_dims(y-new_point_y, axis=-1) * direction_2,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )
    z_grid = func(x_grid, y_grid)
    # print(x_grid.shape, y_grid.shape, z_grid.shape)

    # if the figure ax is given, use it. otherwise, create a new figure and return it later. but we should record this distinction.
    ax_given = True
    if ax is None:
      ax_given = False
      import matplotlib.pyplot as plt

      fig, ax = plt.subplots()
    # calculate score levels from p-value levels
    score_levels = [self.get_threshold_from_p(level) for level in levels]
    zipped = list(zip(score_levels, levels, strict=True))
    zipped.sort(key=lambda x: x[0])
    score_levels, levels = list(zip(*zipped, strict=True))
    # create contour plot
    CS = ax.contour(x_grid, y_grid, z_grid, score_levels, **kwargs)
    # if clabel, put labels onto the curves. if not, you can still add curve labels outside this function since we will return the figure and ax anyway.
    if clabel:
      fmt = {}
      # if strs is not provided to indicate levels, use the p-value levels as the default
      if strs is None:
        strs = [str(1 - level) for level in levels]
      # create fmt, a dictionary storing the formatting of labels
      for level, string in zip(CS.levels, strs, strict=True):
        fmt[level] = string
      # create the curve labels
      ax.clabel(CS, fmt=fmt, inline=True)
    # if ax was given, returning CS is enough. otherwise, return the newly created fig and ax as well.
    if ax_given:
      return CS
    else:
      return fig, ax, CS


__all__ = ["MultivariateSliceMixin"]
