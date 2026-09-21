"""Deprecated penalty-based support calculation kept for migration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
  from .multivariate import MetaAnalysisMD


class LegacySupportMixin:
  """Compatibility implementation selected only by the legacy solver."""

  def _legacy_simultaneous_interval(
    self: MetaAnalysisMD,
    direction: Sequence[float | np.float64] | np.ndarray,
    xi_hat: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float64 | np.ndarray] | np.ndarray,
    sub_dim: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64 | np.ndarray] | np.ndarray,
    df: np.int_ | float | np.float64 | Sequence[int | np.int_ | float | np.float64] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    x0: np.ndarray | None = None,
    find_min: bool = True,
    method: str = "Powell",
    print_res: bool = False,
    lambda_inv: float = np.exp(-20),
    **kwargs: Any,
  ) -> tuple[float, float, np.ndarray, np.ndarray]:
    # format point and direction
    direction = np.array(direction)

    # DONT normalize direction
    # direction = direction / np.sqrt((direction**2).sum())

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
        x,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )-self.threshold

    if x0 is None:
      if equal_sub_dim:
        mat_tmp = projs[sub_dim.reshape(-1), :]
      else:
        mat_tmp = projs[np.concatenate(sub_dim), :]
      x0 = np.linalg.pinv((mat_tmp.T * self.weights) @ mat_tmp) @ (mat_tmp.T * self.weights) @ xi_hat.reshape(-1)
    
    if find_min:
      x1 = self.find_minimizer(xi_hat=xi_hat, Sigma=Sigma, sub_dim=sub_dim, df=df, projs=projs, format_input = False,equal_sub_dim=equal_sub_dim, x0=x0, method=method, print_res=print_res)
    else:
      x1 = x0

    # def dfunc(xi):
    #   p_vector = self.get_p_vector(
    #     xi,
    #     xi_hat,
    #     Sigma,
    #     sub_dim,
    #     df,
    #     projs,
    #     format_input=False,
    #     equal_sub_dim=equal_sub_dim,
    #   )
    #   dif_vector = self.get_dif_vector(
    #     xi,
    #     xi_hat,
    #     Sigma,
    #     sub_dim,
    #     df,
    #     projs,
    #     format_input=False,
    #     equal_sub_dim=equal_sub_dim,
    #   )
    #   new_weights = self.weights * dif_vector / np.sin(p_vector)**2 * np.pi
    #   res = np.concatenate([w * np.linalg.inv(z) @ (x - y) for w, z, x, y in zip(new_weights, Sigma, xi, xi_hat)])
    #   print(res)
    #   return res
    
    # constraint = {'type': 'ineq', 'fun': func, 'jac': dfunc}

    if "xtol" not in kwargs:
      kwargs["xtol"]=1E-7

    x_min = minimize(
      lambda x: lambda_inv*np.dot(direction, x)+np.maximum(func(x),0), x0=x1, method=method, options=kwargs
    ).x
    print(x_min @ direction)
    proj_min=x_min @ direction

    x_max = minimize(
      lambda x: -lambda_inv*np.dot(direction, x)+np.maximum(func(x),0), x0=x1, method=method, options=kwargs
    ).x
    print(x_max @ direction)
    proj_max = x_max @ direction

    # print(proj_min, func(x_min), proj_max, func(x_max))

    if print_res:
      print(f"Simultaneous confidence interval for <{direction}, x> is given by ({proj_min}, {proj_max})")
    return proj_min, proj_max, x_min, x_max


__all__ = ["LegacySupportMixin"]
