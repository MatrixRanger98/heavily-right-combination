"""Self-contained Senn2013 data and common-effect WLS calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_PATH = Path(__file__).with_name("data") / "senn2013.csv"
REFERENCE_TREATMENT = "plac"
TREATMENTS = np.array(
  ["acar", "benf", "metf", "migl", "piog", "rosi", "sita", "sulf", "vild"]
)
REQUIRED_COLUMNS = {
  "TE",
  "seTE",
  "treat1.long",
  "treat2.long",
  "treat1",
  "treat2",
  "studlab",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
  """Load and validate the comparison-based Senn2013 data snapshot."""
  data = pd.read_csv(path)
  missing = REQUIRED_COLUMNS.difference(data.columns)
  if missing:
    raise ValueError(f"Senn2013 data is missing columns: {sorted(missing)}")
  if data[["TE", "seTE"]].isna().any().any():
    raise ValueError("Senn2013 estimates and standard errors must be complete")
  if (data["seTE"] <= 0).any():
    raise ValueError("Senn2013 standard errors must be positive")
  return data


def build_design_matrix(data: pd.DataFrame) -> np.ndarray:
  """Construct treatment-versus-placebo contrast rows in canonical order."""
  treatment_index = {name: index for index, name in enumerate(TREATMENTS)}
  design = np.zeros((len(data), len(TREATMENTS)))
  for row_index, (_, row) in enumerate(data.iterrows()):
    for treatment, sign in ((row["treat1"], 1), (row["treat2"], -1)):
      if treatment == REFERENCE_TREATMENT:
        continue
      try:
        design[row_index, treatment_index[treatment]] = sign
      except KeyError as error:
        raise ValueError(f"unknown treatment label: {treatment!r}") from error
  return design


def _adjust_multiarm_study(
  standard_errors: np.ndarray,
  treat1: np.ndarray,
  treat2: np.ndarray,
) -> np.ndarray:
  """Return contrast SEs with the usual multi-arm covariance adjustment."""
  arms = tuple(dict.fromkeys((*treat1, *treat2)))
  expected_comparisons = len(arms) * (len(arms) - 1) // 2
  if len(standard_errors) != expected_comparisons:
    raise ValueError(
      "multi-arm studies must contain every pairwise comparison; "
      f"found {len(standard_errors)} for {len(arms)} arms"
    )

  arm_index = {name: index for index, name in enumerate(arms)}
  incidence = np.zeros((len(standard_errors), len(arms)))
  for row_index, (first, second) in enumerate(zip(treat1, treat2, strict=True)):
    incidence[row_index, arm_index[first]] = 1
    incidence[row_index, arm_index[second]] = -1

  variances = standard_errors**2
  weighted_incidence = incidence.T @ np.diag(variances) @ incidence
  pairwise_variances = np.diag(np.diag(weighted_incidence)) - weighted_incidence
  arm_count = len(arms)
  transformed_laplacian = (
    -0.5
    * incidence.T
    @ incidence
    @ pairwise_variances
    @ incidence.T
    @ incidence
    / arm_count**2
  )
  information_laplacian = np.linalg.pinv(transformed_laplacian)
  pairwise_weights = (
    np.diag(np.diag(information_laplacian)) - information_laplacian
  )

  adjusted = np.empty_like(standard_errors)
  for row_index, (first, second) in enumerate(zip(treat1, treat2, strict=True)):
    weight = pairwise_weights[arm_index[first], arm_index[second]]
    if weight <= 0:
      raise ValueError("multi-arm adjustment produced a non-positive weight")
    adjusted[row_index] = np.sqrt(1 / weight)
  return adjusted


def adjust_multiarm_standard_errors(data: pd.DataFrame) -> np.ndarray:
  """Adjust only studies containing more than one reported contrast."""
  adjusted = data["seTE"].to_numpy(dtype=float, copy=True)
  for _, study in data.groupby("studlab", sort=False):
    if len(study) == 1:
      continue
    positions = data.index.get_indexer(study.index)
    adjusted[positions] = _adjust_multiarm_study(
      study["seTE"].to_numpy(dtype=float),
      study["treat1"].to_numpy(dtype=str),
      study["treat2"].to_numpy(dtype=str),
    )
  return adjusted


def wls_covariance(
  design: np.ndarray | None = None,
  standard_errors: np.ndarray | None = None,
) -> np.ndarray:
  """Covariance of the common-effect WLS treatment estimates."""
  if design is None:
    design = DESIGN_MATRIX
  if standard_errors is None:
    standard_errors = ADJUSTED_STANDARD_ERRORS
  information = design.T @ (design / standard_errors[:, np.newaxis] ** 2)
  return np.linalg.inv(information)


def wls_point_estimate(
  estimates: np.ndarray,
  standard_errors: np.ndarray,
  design: np.ndarray | None = None,
) -> np.ndarray:
  """Solve inverse-variance WLS without explicitly inverting its information."""
  if design is None:
    design = DESIGN_MATRIX
  observations = np.asarray(estimates, dtype=float)
  errors = np.asarray(standard_errors, dtype=float)
  projections = np.asarray(design, dtype=float)
  if (
    projections.ndim != 2 or observations.shape != (projections.shape[0],)
    or errors.shape != observations.shape or not np.isfinite(projections).all()
    or not np.isfinite(observations).all() or not np.isfinite(errors).all()
    or np.any(errors <= 0)
  ):
    raise ValueError("WLS requires finite aligned data and positive standard errors")
  solution, _, rank, _ = np.linalg.lstsq(
    projections / errors[:, None], observations / errors, rcond=None,
  )
  if rank != projections.shape[1]:
    raise ValueError("WLS design must identify every treatment")
  return solution


def wls_interval_width_matrix(level: float = 0.05) -> np.ndarray:
  """Widths for all treatment contrasts, with placebo widths on the diagonal."""
  if not 0 < level < 1:
    raise ValueError("level must lie strictly between zero and one")
  covariance = wls_covariance()
  critical_value = norm.ppf(1 - level / 2)
  widths = np.empty((len(TREATMENTS), len(TREATMENTS)))
  for first in range(len(TREATMENTS)):
    for second in range(len(TREATMENTS)):
      direction = np.zeros(len(TREATMENTS))
      direction[first] = 1
      if first != second:
        direction[second] = -1
      widths[first, second] = 2 * critical_value * np.sqrt(
        direction @ covariance @ direction
      )
  return widths


DATA = load_data()
ESTIMATES = DATA["TE"].to_numpy(dtype=float)
STANDARD_ERRORS = DATA["seTE"].to_numpy(dtype=float)
ADJUSTED_STANDARD_ERRORS = adjust_multiarm_standard_errors(DATA)
DESIGN_MATRIX = build_design_matrix(DATA)

for _array in (
  TREATMENTS,
  ESTIMATES,
  STANDARD_ERRORS,
  ADJUSTED_STANDARD_ERRORS,
  DESIGN_MATRIX,
):
  _array.setflags(write=False)
