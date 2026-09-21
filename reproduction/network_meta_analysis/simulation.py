"""Synthetic network meta-analysis coverage and interval-width experiment."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import multivariate_normal, norm

from heavily_right.calibration import standard_error_of_mean
from heavily_right.correlation import cor_fixed
from heavily_right.empty_sets import EmptyRegionError
from heavily_right.meta_analysis import MetaAnalysisMD
from reproduction.artifacts import ArtifactStore
from reproduction.network_meta_analysis.manual_benchmark import WidthPolicy, select_wls_ma_widths
from reproduction.network_meta_analysis.senn2013 import DESIGN_MATRIX, wls_point_estimate
from reproduction.plotting import save_pdf
from reproduction.profiles import PROFILES, current_profile
from reproduction.progress import Progress

EXPERIMENT_ID = "network-meta-analysis/simulation"
SEED = 2025
PAPER_REPLICATES = 100
WORKLOAD_REVISION = "nma-100-replicates-2026-09-21"
THETA = np.array([0, -0.5, -1, 0, -0.5, -1, 0, -0.5, -1])
ORACLE_QUALIFICATION = (
  "WLS-MA is an in-sample oracle benchmark: for each rho, use the empirical 0.95 "
  "quantile (inverted empirical CDF) of the same replicates' maximum absolute "
  "standardized coordinate error. It uses the true theta and those realizations; "
  "its achieved calibration-sample coverage is not an out-of-sample guarantee "
  "and this procedure is unavailable for unknown real-world truth."
)
EMPTY_REGION_CONVENTION = (
  "An explicitly certified empty region counts as noncoverage "
  "and width zero in unconditional mean widths; keep the replicate in the denominator "
  "and save its inputs/certificate. Numerical failures are not empty-region outcomes "
  "and stop the run. No studies are removed and no replicate is resampled."
)
COVERAGE_EVENTS = {
  "hcct": {
    "event": "truth_in_joint_score_region",
    "definition": "G(THETA) <= HCCT cutoff; not coverage of a coordinate-interval box",
    "comparison": "less_than_or_equal",
    "coordinate_count": 9,
  },
  "wls": {
    "event": "all_coordinate_intervals_cover",
    "definition": "For all 9 coordinates j: estimate_j - q*SE_j < THETA_j < estimate_j + q*SE_j",
    "comparison": "strict_interval_containment",
    "coordinate_count": 9,
    "multiplier": "nominal_wls_multiplier",
  },
  "wls_ma": {
    "event": "all_coordinate_intervals_cover",
    "definition": "max_j abs((estimate_j - THETA_j)/SE_j) <= oracle multiplier at this rho",
    "comparison": "less_than_or_equal",
    "coordinate_count": 9,
    "multiplier": "oracle_multipliers",
    "qualification": "in-sample oracle; calibrated and evaluated on the same replicates",
  },
  "relation": {
    "hcct_region_implies_own_coordinate_support_box": True,
    "reverse_implication_valid_in_general": False,
    "hcct_coordinate_box_coverage_evaluated": False,
    "hcct_width_coordinates_evaluated": [1, 2],
    "like_for_like_coverage_events": False,
    "qualification": (
      "The containment relation concerns the HCCT region and its own coordinate-support "
      "box, not the WLS boxes. The manuscript describes simultaneous active-placebo "
      "interval coverage; this retained simulation instead measures HCCT joint-region "
      "coverage versus WLS simultaneous-box coverage. Its interpretation is unresolved."
    ),
  },
}


def _json_safe(value: Any) -> Any:
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  if isinstance(value, np.ndarray):
    return _json_safe(value.tolist())
  if isinstance(value, np.generic):
    return _json_safe(value.item())
  if isinstance(value, float) and not np.isfinite(value):
    return None
  return value


def _certified_empty_region(error: Exception) -> EmptyRegionError | None:
  """Accept only the canonical typed signal with an explicit finite lower bound."""
  candidate = error if isinstance(error, EmptyRegionError) else error.__cause__
  if not isinstance(candidate, EmptyRegionError):
    return None
  certificate = candidate.minimum_diagnostics
  if certificate is None or not certificate.empty_certified:
    return None
  bound = certificate.feasible_set_score_lower_bound
  if (
    bound is None or not np.isfinite(bound) or not np.isfinite(candidate.threshold)
    or not np.isfinite(candidate.minimum_score) or bound <= candidate.threshold
    or candidate.minimum_score <= candidate.threshold
  ):
    return None
  return candidate


def calibrate_wls_oracle(
  estimates: np.ndarray, standard_errors: np.ndarray, truth: np.ndarray, *, level: float = 0.05,
) -> dict[str, np.ndarray | float]:
  """Fit the empirical oracle multiplier; retain ties in the coverage count."""
  estimates = np.asarray(estimates, dtype=float)
  standard_errors = np.asarray(standard_errors, dtype=float)
  truth = np.asarray(truth, dtype=float)
  if (
    estimates.ndim != 2 or not all(estimates.shape)
    or standard_errors.shape != estimates.shape or truth.shape != estimates.shape[1:]
    or not np.isfinite(estimates).all() or not np.isfinite(standard_errors).all()
    or not np.isfinite(truth).all() or np.any(standard_errors <= 0)
    or not np.isfinite(level) or not 0 < level < 1
  ):
    raise ValueError("oracle calibration requires aligned finite estimates, positive SEs and level")
  maxima = np.max(np.abs((estimates - truth) / standard_errors), axis=1)
  multiplier = float(np.quantile(maxima, 1 - level, method="inverted_cdf"))
  covered = maxima <= multiplier
  return {
    "multiplier": multiplier, "maximum_standardized_errors": maxima,
    "covered": covered, "coverage": float(np.mean(covered)),
    "mean_widths": np.mean(2 * multiplier * standard_errors, axis=0),
  }


def _is_covered(
  analysis: MetaAnalysisMD,
  point: np.ndarray,
  mean: np.ndarray,
  variance: np.ndarray,
  degrees_freedom: int,
) -> bool:
  p_values = analysis.get_p_vector(
    point,
    mean,
    variance,
    sub_dim=1,
    df=degrees_freedom,
    projs=DESIGN_MATRIX,
  )
  return bool(analysis.get_global_score(p_values) <= analysis.threshold)


def simulation_configuration(profile: str) -> dict[str, Any]:
  """Record the user-approved workload independently of ambient profile state.

  The paper profile was reduced from 500 to 100 replicates per correlation on
  2026-09-21. Historical runs retain their original counts and are not relabeled.
  """
  if profile not in PROFILES:
    raise ValueError(f"unsupported NMA profile: {profile!r}")
  paper = profile == "paper"
  sample_size = 100 if paper else 30
  return {
    "schema_version": 1, "workload_revision": WORKLOAD_REVISION, "profile": profile,
    "seed": SEED, "replicates_per_rho": PAPER_REPLICATES if paper else 2,
    "sample_size": sample_size, "degrees_freedom": sample_size - 1,
    "rhos": (np.arange(0, 1, .1) if paper else np.array([0., .6])).tolist(),
    "level": .05, "parameter_dimension": 9, "study_count": 28,
  }


def simulate(artifacts: ArtifactStore) -> dict[str, np.ndarray]:
  configuration = simulation_configuration(current_profile())
  repeat = configuration["replicates_per_rho"]
  num_sample = configuration["sample_size"]
  rhos = np.asarray(configuration["rhos"])
  num_study, dimension = DESIGN_MATRIX.shape
  eta = DESIGN_MATRIX @ THETA
  directions = np.eye(dimension)[:2]
  results = {
    "rhos": np.asarray(rhos),
    "coverage_hcct": np.zeros(len(rhos)),
    "coverage_wls": np.zeros(len(rhos)),
    "width_hcct_1": np.zeros(len(rhos)),
    "width_hcct_2": np.zeros(len(rhos)),
    "width_wls_1": np.zeros(len(rhos)),
    "width_wls_2": np.zeros(len(rhos)),
  }
  progress = Progress("network meta-analysis replicates", len(rhos) * repeat)
  completed = 0
  raw = {
    "rhos": np.asarray(rhos), "true_theta": THETA, "design": DESIGN_MATRIX,
    "xi_hat": np.full((len(rhos), repeat, num_study), np.nan),
    "estimated_mean_variances": np.full((len(rhos), repeat, num_study), np.nan),
    "wls_estimates": np.full((len(rhos), repeat, dimension), np.nan),
    "wls_standard_errors": np.full((len(rhos), repeat, dimension), np.nan),
    "wls_maximum_standardized_errors": np.full((len(rhos), repeat), np.nan),
    "covered_wls": np.full((len(rhos), repeat), np.nan),
    "covered_hcct": np.full((len(rhos), repeat), np.nan),
    "empty_hcct": np.full((len(rhos), repeat), np.nan),
    "widths_hcct": np.full((len(rhos), repeat, 2), np.nan),
    "hcct_boundary_errors": np.full((len(rhos), repeat, 2, 2), np.nan),
    "hcct_kkt_residuals": np.full((len(rhos), repeat, 2, 2), np.nan),
    "hcct_minimum_scores": np.full((len(rhos), repeat, 2), np.nan),
    "hcct_minimum_points": np.full((len(rhos), repeat, 2, dimension), np.nan),
    "hcct_support_points": np.full((len(rhos), repeat, 2, 2, dimension), np.nan),
    "covered_wls_ma": np.full((len(rhos), repeat), np.nan),
  }
  nominal_multiplier = float(-norm.ppf(0.025 / dimension))

  for rho_index, rho in enumerate(rhos):
    distribution = multivariate_normal(
      mean=eta,
      cov=cor_fixed(num_study, float(rho)),
    )
    analysis = MetaAnalysisMD(num_study, dim=dimension, method="HCauchy", level=0.05)
    for replicate in range(repeat):
      sample = distribution.rvs(size=num_sample)
      mean = sample.mean(axis=0)
      standard_error = standard_error_of_mean(sample)
      variance = standard_error**2
      raw["xi_hat"][rho_index, replicate] = mean
      raw["estimated_mean_variances"][rho_index, replicate] = variance
      empty = None
      current_support = None
      try:
        intervals = []
        for direction in directions:
          # The second direction independently validates/refines this minimum,
          # but need not repeat the preliminary derivative-free Powell search.
          options = {} if not intervals else {
            "x0": intervals[0].minimum_point, "find_min": False,
          }
          current_support = None
          current_support = analysis.support_interval(
            direction=direction,
            xi_hat=mean,
            Sigma=variance,
            df=num_sample - 1,
            sub_dim=1,
            projs=DESIGN_MATRIX,
            method="Powell",
            **options,
          )
          if not current_support.success:
            raise RuntimeError(
              "uncertified HCCT support: "
              f"lower={current_support.lower_diagnostics.message}; "
              f"upper={current_support.upper_diagnostics.message}"
            )
          intervals.append(current_support)
      except Exception as error:
        # A prior certified feasible endpoint rules out an empty set: do not
        # accept a contradictory second-direction empty certificate as data.
        empty = _certified_empty_region(error) if not intervals else None
        if empty is not None:
          payload = {
            "rho": float(rho), "replicate": replicate + 1,
            "xi_hat": mean, "estimated_mean_variances": variance,
            "design": DESIGN_MATRIX, "degrees_freedom": num_sample - 1,
            "minimum_point": empty.point, "individual_pvalues": empty.p_values,
            "minimum_score": empty.minimum_score, "threshold": empty.threshold,
            "minimum_diagnostics": asdict(empty.minimum_diagnostics),
            "convention": EMPTY_REGION_CONVENTION,
          }
          filename = f"empty-region-rho-{rho:.1f}-replicate-{replicate + 1}.json"
          with artifacts.diagnostic(filename).open("x") as output:
            json.dump(_json_safe(payload), output, indent=2, allow_nan=False)
            output.write("\n")
        else:
          np.savez(artifacts.diagnostic("nma-simulation-partial-replicates.npz"), **raw)
          failure = {
            "rho": float(rho), "replicate": replicate + 1,
            "completed_replicates": completed, "exception_type": type(error).__name__,
            "message": str(error), "degrees_freedom": num_sample - 1,
            "direction_index": len(intervals),
            "threshold": float(analysis.threshold),
            "action": "failed run; no replicate skipped, deleted, or resampled",
          }
          if error.__cause__ is not None:
            failure["cause_type"] = type(error.__cause__).__name__
            failure["cause_message"] = str(error.__cause__)
          diagnostics = getattr(error, "diagnostics", None)
          if diagnostics is not None:
            failure["empty_set_diagnostics"] = diagnostics.to_dict()
          if current_support is not None:
            failure["failed_support_result"] = asdict(current_support)
          failure["completed_direction_results"] = [asdict(result) for result in intervals]
          with artifacts.diagnostic("nma-simulation-failure.json").open("x") as output:
            json.dump(_json_safe(failure), output, indent=2, allow_nan=False)
            output.write("\n")
          raise
      widths = np.zeros(2) if empty is not None else np.array([
        interval.upper - interval.lower for interval in intervals
      ])
      results["width_hcct_1"][rho_index] += widths[0]
      results["width_hcct_2"][rho_index] += widths[1]
      raw["widths_hcct"][rho_index, replicate] = widths
      raw["empty_hcct"][rho_index, replicate] = empty is not None
      for direction_index, result in enumerate(intervals):
        slot = (rho_index, replicate, direction_index)
        endpoints = (result.lower_diagnostics, result.upper_diagnostics)
        raw["hcct_boundary_errors"][slot] = [endpoint.boundary_error for endpoint in endpoints]
        raw["hcct_kkt_residuals"][slot] = [endpoint.kkt_residual for endpoint in endpoints]
        raw["hcct_minimum_scores"][slot] = result.minimum_score
        raw["hcct_minimum_points"][slot] = result.minimum_point
        raw["hcct_support_points"][slot] = (result.lower_point, result.upper_point)

      covered = empty is None and _is_covered(analysis, THETA, mean, variance, num_sample - 1)
      results["coverage_hcct"][rho_index] += covered
      raw["covered_hcct"][rho_index, replicate] = covered
      if not covered:
        np.savez(
          artifacts.diagnostic(
            f"failed-coverage-rho-{rho:.1f}-replicate-{replicate + 1}.npz"
          ),
          point=THETA,
          xi_hat=mean,
          Sigma=variance,
          projs=DESIGN_MATRIX,
        )

      wls_standard_error = np.sqrt(
        np.diag(np.linalg.inv((DESIGN_MATRIX.T / variance) @ DESIGN_MATRIX))
      )
      wls_half_width = wls_standard_error * nominal_multiplier
      results["width_wls_1"][rho_index] += 2 * wls_half_width[0]
      results["width_wls_2"][rho_index] += 2 * wls_half_width[1]
      wls_estimate = wls_point_estimate(mean, standard_error)
      wls_covered = np.all(
        (wls_estimate - wls_half_width < THETA)
        & (THETA < wls_estimate + wls_half_width)
      )
      results["coverage_wls"][rho_index] += wls_covered
      raw["wls_estimates"][rho_index, replicate] = wls_estimate
      raw["wls_standard_errors"][rho_index, replicate] = wls_standard_error
      raw["covered_wls"][rho_index, replicate] = wls_covered
      completed += 1
      if repeat <= 20 or (replicate + 1) % max(1, repeat // 20) == 0:
        progress.update(completed)

  for name in results:
    if name != "rhos":
      results[name] /= repeat
  results["empty_hcct_count"] = np.sum(raw["empty_hcct"] == 1, axis=1)
  results["replicates_per_rho"] = np.full(len(rhos), repeat, dtype=int)
  results["wls_ma_multiplier"] = np.zeros(len(rhos))
  results["coverage_wls_ma"] = np.zeros(len(rhos))
  results["width_wls_ma_1"] = np.zeros(len(rhos))
  results["width_wls_ma_2"] = np.zeros(len(rhos))
  for index in range(len(rhos)):
    oracle = calibrate_wls_oracle(
      raw["wls_estimates"][index], raw["wls_standard_errors"][index], THETA,
    )
    results["wls_ma_multiplier"][index] = oracle["multiplier"]
    results["coverage_wls_ma"][index] = oracle["coverage"]
    results["width_wls_ma_1"][index] = oracle["mean_widths"][0]
    results["width_wls_ma_2"][index] = oracle["mean_widths"][1]
    raw["covered_wls_ma"][index] = oracle["covered"]
    raw["wls_maximum_standardized_errors"][index] = oracle["maximum_standardized_errors"]
  np.savez(artifacts.data("nma-simulation-replicates.npz"), **raw)
  calibration = {
    "schema_version": 1, "seed": SEED, "replicates_per_rho": repeat,
    "configuration": configuration,
    "sample_size": num_sample, "degrees_freedom": num_sample - 1,
    "rhos": np.asarray(rhos).tolist(), "nominal_wls_multiplier": nominal_multiplier,
    "wls_model": "inverse-estimated-variance WLS; independent-study covariance formula",
    "mean_variance_calibration": "W/[n(n-1)]",
    "oracle_quantile_method": "inverted_cdf", "oracle_qualification": ORACLE_QUALIFICATION,
    "oracle_multipliers": results["wls_ma_multiplier"].tolist(),
    "oracle_empirical_coverage": results["coverage_wls_ma"].tolist(),
    "empty_region_convention": EMPTY_REGION_CONVENTION,
    "coverage_events": COVERAGE_EVENTS,
    "empty_hcct_counts": results["empty_hcct_count"].tolist(),
    "hcct_minimum_reuse": (
      "First direction uses the unchanged Powell start; second reuses its validated "
      "minimum with find_min=False. Canonical minimum and endpoint certification "
      "still run independently for both directions; no tolerances are changed."
    ),
  }
  with artifacts.diagnostic("nma-simulation-calibration.json").open("x") as output:
    json.dump(calibration, output, indent=2, allow_nan=False)
    output.write("\n")
  return results


def _smooth(rhos: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  x_new = np.arange(rhos[0], rhos[-1] + 0.01, 0.02)
  degree = min(3, len(rhos) - 1)
  return x_new, make_interp_spline(rhos, values, k=degree)(x_new)


def build_comparison_figure(
  rhos: np.ndarray,
  series: tuple[np.ndarray, np.ndarray, np.ndarray],
  *,
  ylabel: str,
  legend_outside: bool = False,
  coverage_events: bool = False,
  wls_ma_label: str = "WLS-MA (in-sample oracle)",
) -> plt.Figure:
  figure, axis = plt.subplots()
  colors = ("steelblue", "coral", "orchid")
  labels = ("HCCT", "WLS", wls_ma_label)
  if coverage_events:
    labels = (
      "HCCT joint region", "WLS simultaneous box", "WLS-MA box (in-sample oracle)",
    )
    figure.subplots_adjust(bottom=0.22)
    figure.text(
      0.5, 0.025,
      "HCCT: true vector in joint score region; WLS: all 9 intervals cover.\n"
      "Coverage events differ; HCCT's 9-interval box coverage is not evaluated.",
      ha="center", va="bottom", fontsize=8,
    )
  lines = []
  for values, color in zip(series, colors, strict=True):
    x_new, curve = _smooth(rhos, values)
    (line,) = axis.plot(x_new, curve, color=color)
    axis.scatter(rhos, values, color=color)
    lines.append(line)
  axis.set(xlabel=r"$\rho$", ylabel=ylabel)
  if legend_outside:
    # Keep the event labels clear of the data; tight PDF export includes them.
    axis.legend(lines, labels, loc="lower left", bbox_to_anchor=(0, 1.02),
                borderaxespad=0, fontsize=8)
  else:
    axis.legend(lines, labels, fontsize=8)
  return figure


def render(
  results: dict[str, np.ndarray], artifacts: ArtifactStore, *, profile: str,
  width_policy: WidthPolicy = "historical-manual",
) -> None:
  """Plot historical manual widths by default, retaining empirical provenance."""
  rhos = results["rhos"]
  widths, policy = select_wls_ma_widths(results, profile=profile, policy=width_policy)
  plotted = {name: values for name, values in results.items() if name != "wls_ma_multiplier"}
  plotted["empirical_wls_ma_multiplier"] = results["wls_ma_multiplier"]
  plotted.update({f"width_wls_ma_{index}": values
                  for index, values in enumerate(widths, start=1)})
  np.savez_compressed(artifacts.data("nma-simulation-plotted-summary.npz"), **plotted)
  with artifacts.diagnostic("nma-simulation-plot-policy.json").open("x") as output:
    json.dump(policy, output, indent=2, allow_nan=False)
    output.write("\n")
  label = ("WLS-MA (historical manual)"
           if policy["selected_width_policy"] == "historical-manual"
           else "WLS-MA (in-sample oracle)")
  save_pdf(
    build_comparison_figure(
      rhos,
      (results["coverage_hcct"], results["coverage_wls"], results["coverage_wls_ma"]),
      ylabel="coverage of stated event",
      legend_outside=True,
      coverage_events=True,
    ),
    artifacts,
    "nma-simulation-coverage.pdf",
    tight=True,
  )
  for index in (1, 2):
    save_pdf(
      build_comparison_figure(
        rhos,
        (
          results[f"width_hcct_{index}"],
          results[f"width_wls_{index}"],
          widths[index - 1],
        ),
        ylabel="widths of simultaneous interval",
        wls_ma_label=label,
      ),
      artifacts,
      f"nma-simulation-interval-width-theta-{index}.pdf",
      tight=True,
    )


def main() -> None:
  np.random.seed(SEED)
  artifacts = ArtifactStore.for_experiment(EXPERIMENT_ID)
  results = simulate(artifacts)
  np.savez(artifacts.data("nma-simulation-summary.npz"), **results)
  render(results, artifacts, profile=current_profile())


if __name__ == "__main__":
  main()
