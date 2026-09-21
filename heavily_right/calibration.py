"""Finite-sample covariance calibration for sample means."""

import numpy as np


def standard_error_of_mean(sample: np.ndarray, axis: int = 0) -> np.ndarray:
  """Estimate componentwise standard errors of a sample mean.

  The sample variance uses Bessel's correction, so the returned squared
  standard errors equal the diagonal of ``W / [n(n - 1)]``.

  Parameters
  ----------
  sample : array_like
      Observations; ``axis`` indexes independent sampling units.
  axis : int, default 0
      Axis removed by the mean calculation. At least two observations are
      required. Supply finite real data; this helper does not impute missingness.

  Returns
  -------
  ndarray or numpy scalar
      Standard errors in the same units as the sample values.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import standard_error_of_mean
  >>> sample = np.array([[1., 2.], [2., 4.], [3., 6.]])
  >>> se = standard_error_of_mean(sample)
  >>> assert np.allclose(se, [1 / np.sqrt(3), 2 / np.sqrt(3)])
  """
  sample = np.asarray(sample)
  num_sample = sample.shape[axis]
  if num_sample < 2:
    raise ValueError("At least two observations are required.")
  return sample.std(axis=axis, ddof=1) / np.sqrt(num_sample)


def covariance_of_mean(sample: np.ndarray, sample_axis: int = 0) -> np.ndarray:
  """Estimate covariance matrices for one or more sample means.

  ``sample`` may have shape ``(n, d)`` for one mean or ``(n, m, d0)`` for
  ``m`` block means. The observation axis is moved to the front and the last
  axis is treated as the component axis. The result is ``W / [n(n - 1)]``.

  Parameters
  ----------
  sample : array_like
      Finite real observations with a final component axis. Other axes index
      separate mean vectors. At least two observations are required.
  sample_axis : int, default 0
      Observation axis, distinct from the component axis.

  Returns
  -------
  ndarray
      Shape ``(d, d)`` or ``(..., d, d)`` in squared input units. This is
      covariance of the estimated mean, not of individual observations.
      Positive definiteness is checked by the inference constructors, not here.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right import covariance_of_mean, standard_error_of_mean
  >>> sample = np.array([[1., 2.], [2., 1.], [4., 5.]])
  >>> covariance = covariance_of_mean(sample)
  >>> assert np.allclose(covariance, np.cov(sample, rowvar=False, ddof=1) / 3)
  >>> assert np.allclose(np.diag(covariance), standard_error_of_mean(sample)**2)
  """
  sample = np.asarray(sample)
  if sample.ndim < 2:
    raise ValueError("Sample input must include observations and components.")

  sample = np.moveaxis(sample, sample_axis, 0)
  num_sample = sample.shape[0]
  if num_sample < 2:
    raise ValueError("At least two observations are required.")

  centered = sample - sample.mean(axis=0)
  scatter = np.einsum("n...i,n...j->...ij", centered, centered)
  return scatter / (num_sample * (num_sample - 1))
