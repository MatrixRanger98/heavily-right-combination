"""Real diabetes network meta-analysis and simultaneous-width figures."""

from __future__ import annotations

import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from heavily_right.empty_sets import EmptySetNumericalError, EmptySetPolicy, EmptySetResolutionError
from heavily_right.meta_analysis import MetaAnalysisMD, SupportIntervalResult
from reproduction.artifacts import ArtifactStore
from reproduction.network_meta_analysis.senn2013 import (
  DATA,
  DESIGN_MATRIX,
  ESTIMATES,
  STANDARD_ERRORS,
  TREATMENTS,
  wls_interval_width_matrix,
)
from reproduction.plotting import save_pdf
from reproduction.progress import Progress

EXPERIMENT_ID = "network-meta-analysis/real-data"
def network_score(
  analysis: MetaAnalysisMD,
  point: np.ndarray,
  estimates: np.ndarray,
  standard_errors: np.ndarray,
  projections: np.ndarray,
) -> float:
  p_values = analysis.get_p_vector(
    point,
    estimates,
    standard_errors**2,
    sub_dim=1,
    df=None,
    projs=projections,
  )
  return float(analysis.get_global_score(p_values))


def fit_with_empty_set_adjustment(
  *, artifacts: ArtifactStore | None = None,
) -> SupportIntervalResult:
  """Fit the existing real-data model and preserve its intervention trace."""
  original_analysis = MetaAnalysisMD(
    len(ESTIMATES),
    dim=DESIGN_MATRIX.shape[1],
    method="HCauchy",
    level=0.05,
  )
  labels = [
    f"{row.studlab}: {row.treat1}-{row.treat2}"
    for row in DATA.itertuples()
  ]
  try:
    probe = original_analysis.support_interval(
      direction=np.eye(DESIGN_MATRIX.shape[1])[0],
      xi_hat=ESTIMATES,
      Sigma=STANDARD_ERRORS**2,
      sub_dim=1,
      df=None,
      projs=DESIGN_MATRIX,
      empty_set_policy=EmptySetPolicy(max_removals=5),
      study_labels=labels,
    )
  except (EmptySetResolutionError, EmptySetNumericalError) as error:
    if artifacts is not None and error.diagnostics is not None:
      error.diagnostics.save_json(
        artifacts.diagnostic("nma-real-data-empty-set-adjustment.json")
      )
    raise
  diagnostics = probe.empty_set_diagnostics
  if diagnostics is None or not diagnostics.resolved:
    raise RuntimeError("failed to obtain a nonempty network confidence region")
  if artifacts is not None:
    diagnostics.save_json(
      artifacts.diagnostic("nma-real-data-empty-set-adjustment.json")
    )
  return probe


def exclude_inconsistent_studies(
  *, artifacts: ArtifactStore | None = None,
) -> tuple[MetaAnalysisMD, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  probe = fit_with_empty_set_adjustment(artifacts=artifacts)
  diagnostics = probe.empty_set_diagnostics
  assert diagnostics is not None
  retained = np.asarray(diagnostics.final_original_indices, dtype=int)
  estimates = ESTIMATES[retained]
  standard_errors = STANDARD_ERRORS[retained]
  projections = DESIGN_MATRIX[retained]
  analysis = MetaAnalysisMD(
    diagnostics.final_weights,
    dim=DESIGN_MATRIX.shape[1],
    method="HCauchy",
    level=0.05,
  )
  point_estimate = np.asarray(diagnostics.last_global_minimizer, dtype=float)
  print(
    "empty-set adjustment: "
    + "; ".join(removal.description for removal in diagnostics.removals),
    flush=True,
  )
  print(
    f"final score={network_score(analysis, point_estimate, estimates, standard_errors, projections):.6g}; "
    f"threshold={analysis.threshold:.6g}",
    flush=True,
  )
  return analysis, point_estimate, estimates, standard_errors, projections


def hcct_width_matrix(
  analysis: MetaAnalysisMD,
  estimates: np.ndarray,
  standard_errors: np.ndarray,
  projections: np.ndarray,
  *,
  artifacts: ArtifactStore | None = None,
) -> np.ndarray:
  dimension = projections.shape[1]
  widths = np.zeros((dimension, dimension))
  progress = Progress("real-data HCCT intervals", dimension * (dimension + 1) // 2)
  completed = 0
  audit: dict[str, object] = {
    "schema_version": 1,
    "experiment": EXPERIMENT_ID,
    "status": "running",
    "expected_directions": dimension * (dimension + 1) // 2,
    "completed_directions": 0,
    "directions": [],
  }
  direction_records: list[dict[str, object]] = []
  audit["directions"] = direction_records
  current_direction: dict[str, object] = {}
  try:
    for first in range(dimension):
      for second in range(first, dimension):
        direction = np.zeros(dimension)
        direction[first] = 1
        if first != second:
          direction[second] = -1
        current_direction = {
          "index": completed,
          "treatment_indices": [first, second],
          "treatment_labels": [str(TREATMENTS[first]), str(TREATMENTS[second])],
          "direction": direction.tolist(),
        }
        result = analysis.support_interval(
          direction=direction,
          xi_hat=estimates,
          Sigma=standard_errors**2,
          sub_dim=1,
          df=None,
          projs=projections,
        )
        direction_records.append({**current_direction, "result": asdict(result)})
        if not result.success:
          raise RuntimeError(
            f"uncertified NMA support direction {completed}: "
            f"{TREATMENTS[first]} / {TREATMENTS[second]}; "
            f"lower={result.lower_diagnostics.message}; "
            f"upper={result.upper_diagnostics.message}"
          )
        widths[first, second] = widths[second, first] = result.upper - result.lower
        completed += 1
        audit["completed_directions"] = completed
        progress.update(completed)
    audit["status"] = "completed"
  except BaseException as error:
    audit["status"] = "failed"
    audit["failure"] = {
      **current_direction,
      "exception_type": type(error).__name__,
      "message": str(error),
    }
    raise
  finally:
    if artifacts is not None:
      def diagnostic_value(value: object) -> object:
        if isinstance(value, dict):
          return {key: diagnostic_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
          return [diagnostic_value(item) for item in value]
        if isinstance(value, np.ndarray):
          return diagnostic_value(value.tolist())
        if isinstance(value, np.generic):
          return diagnostic_value(value.item())
        if isinstance(value, float) and not np.isfinite(value):
          return str(value)
        return value

      path = artifacts.diagnostic("nma-real-data-support-certificates.json")
      with path.open("x") as output:
        json.dump(diagnostic_value(audit), output, indent=2, allow_nan=False)
        output.write("\n")
  return widths


def wls_width_matrix() -> np.ndarray:
  print("computing WLS interval widths from the Senn2013 CSV snapshot", flush=True)
  return wls_interval_width_matrix()


def build_heatmap(
  values: np.ndarray,
  *,
  cmap: str,
  vmin: float,
  vmax: float,
  annotations: np.ndarray | bool = True,
) -> plt.Figure:
  figure, axis = plt.subplots()
  heatmap = sns.heatmap(
    pd.DataFrame(values, columns=TREATMENTS, index=TREATMENTS),
    annot=annotations,
    fmt="" if isinstance(annotations, np.ndarray) else ".2g",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    ax=axis,
  )
  colorbar = heatmap.collections[0].colorbar
  if colorbar is not None and colorbar.solids is not None:
    colorbar.solids.set_rasterized(False)
  return figure


def main() -> None:
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  analysis, _, estimates, standard_errors, projections = exclude_inconsistent_studies(artifacts=artifacts)
  hcct = hcct_width_matrix(
    analysis, estimates, standard_errors, projections, artifacts=artifacts
  )
  np.save(artifacts.data("nma-real-data-hcct-interval-widths.npy"), hcct)
  save_pdf(
    build_heatmap(hcct, cmap="viridis", vmin=0, vmax=1.5),
    artifacts,
    "nma-real-data-hcct-interval-width-heatmap.pdf",
    tight=True,
  )

  wls = wls_width_matrix()
  save_pdf(
    build_heatmap(wls, cmap="viridis", vmin=0, vmax=1.5),
    artifacts,
    "nma-real-data-wls-interval-width-heatmap.pdf",
    tight=True,
  )
  bonferroni_factor = norm.ppf(0.025 / 45) / norm.ppf(0.025)
  adjusted_wls = wls * bonferroni_factor
  save_pdf(
    build_heatmap(adjusted_wls, cmap="viridis", vmin=0, vmax=1.5),
    artifacts,
    "nma-real-data-bonferroni-wls-heatmap.pdf",
    tight=True,
  )

  difference = hcct - adjusted_wls
  annotations = np.vectorize(lambda value: f"{value:.2f}".rstrip("0").rstrip("."))(
    difference
  )
  save_pdf(
    build_heatmap(
      difference,
      cmap="coolwarm",
      vmin=-0.5,
      vmax=0.5,
      annotations=annotations,
    ),
    artifacts,
    "nma-real-data-hcct-minus-bonferroni-wls-heatmap.pdf",
    tight=True,
  )

  comparisons = np.arange(1, 46)
  for treatment_index in (1, 2, 6, 5):
    varying_wls = (
      norm.ppf(0.025 / comparisons) / norm.ppf(0.025) * wls[treatment_index, treatment_index]
    )
    figure, axis = plt.subplots()
    axis.plot(comparisons, varying_wls, label="WLS", linewidth=1.5)
    axis.axhline(
      hcct[treatment_index, treatment_index],
      label="HCCT",
      linewidth=1.5,
    )
    axis.legend()
    axis.set(xlabel="number of comparisons", ylabel="width of simultaneous CI")
    save_pdf(
      figure,
      artifacts,
      f"nma-real-data-width-vs-comparisons-{TREATMENTS[treatment_index]}.pdf",
      tight=True,
    )


if __name__ == "__main__":
  main()
