"""Cached SVD sampling for the fixed covariances in reproduction experiments.

The two NumPy RNG APIs use different SVD factors. Keep their respective
factor, draw shape, and multiplication order to preserve existing streams;
do not replace these draws by Cholesky or common-factor simulations.
"""

from __future__ import annotations

import numpy as np

RandomSource = np.random.RandomState | np.random.Generator | None
SampleSize = int | tuple[int, ...]


class CachedMultivariateNormal:
  """Like SciPy's frozen normal ``rvs``, caching its NumPy SVD factor.

  This reproduction-only helper accepts finite positive-definite covariance
  matrices. ``None`` uses NumPy's global RandomState, as the original scripts
  do. Explicit RandomState and Generator instances retain their own streams.
  """

  def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
    self.mean = np.array(mean, dtype=float, copy=True)
    covariance = np.array(cov, dtype=float, copy=True)
    if (
      self.mean.ndim != 1
      or covariance.shape != (self.mean.size, self.mean.size)
      or not np.isfinite(self.mean).all()
      or not np.isfinite(covariance).all()
      or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
    ):
      raise ValueError("mean and covariance must be finite and dimensionally compatible")
    # Validate before caching: the active experiments all have nonsingular
    # correlation matrices, and an SVD alone would conceal negative eigenvalues.
    np.linalg.cholesky(covariance)
    u, singular_values, vh = np.linalg.svd(covariance)
    roots = np.sqrt(singular_values)
    self._randomstate_factor = roots[:, None] * vh
    self._generator_factor = (u * roots).T

  def _centered(self, size: SampleSize, random_state: RandomSource) -> np.ndarray:
    shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
    rng = np.random if random_state is None else random_state
    normals = rng.standard_normal(shape + (self.mean.size,))
    factor = (
      self._generator_factor
      if isinstance(random_state, np.random.Generator)
      else self._randomstate_factor
    )
    return (normals.reshape(-1, self.mean.size) @ factor).reshape(
      shape + (self.mean.size,)
    )

  def rvs(
    self, size: SampleSize = 1, random_state: RandomSource = None,
  ) -> np.ndarray:
    """Draw in the original NumPy order, then apply SciPy's squeezing rule."""
    return np.squeeze(self._centered(size, random_state) + self.mean)


class CachedMultivariateT(CachedMultivariateNormal):
  """Cache the normal factor without changing SciPy's shared radial t draw.

  The chi-square variates are drawn *before* every corresponding normal batch;
  a single radial variate is shared by all coordinates of each observation.
  """

  def __init__(self, loc: np.ndarray, shape: np.ndarray, df: float) -> None:
    super().__init__(mean=loc, cov=shape)
    if not np.isfinite(df) or df <= 0:
      raise ValueError("df must be positive and finite")
    self.df = float(df)

  def rvs(
    self, size: SampleSize = 1, random_state: RandomSource = None,
  ) -> np.ndarray:
    rng = np.random if random_state is None else random_state
    radial = rng.chisquare(self.df, size=size) / self.df
    centered = self._centered(size, random_state)
    return np.squeeze(self.mean + centered / np.sqrt(radial)[..., None])
