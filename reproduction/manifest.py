"""Inventory linking paper results to their reproduction entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
  id: str
  script: str
  description: str
  paper_artifacts: tuple[str, ...] = ()
  alternate_scripts: tuple[str, ...] = ()
  language: str = "python"
  seed: int | None = None
  runner_enabled: bool = True
  notes: str = ""
  outputs: tuple[str, ...] = ()
  paper_runtime: str = "unknown"

  @property
  def script_path(self) -> Path:
    return CODE_ROOT / self.script

  @property
  def alternate_script_paths(self) -> tuple[Path, ...]:
    return tuple(CODE_ROOT / script for script in self.alternate_scripts)


EXPERIMENTS = (
  Experiment(
    "illustrations/hcct-cct-geometry",
    "reproduction/illustrations/hcct_cct_geometry.py",
    "Blockwise HCCT region and CCT horn geometry, including disjoint equal-area tubes.",
    (
      "hcct_region_no_caption.pdf",
      "cct_region_no_caption.pdf",
      "cct_horn_magnified_no_caption.pdf",
      "cct_infinite_area_tubes_no_caption.pdf",
    ),
    notes="Recovered from the manuscript figure-folder generator (identical to the Note/ copy); preserves its rounded HCCT cutoff 13.68751. Both profiles use the same deterministic panels.",
    outputs=(
      "data/hcct-cct-geometry-config.json",
      "figures/hcct-cct-geometry-hcct-region.pdf",
      "figures/hcct-cct-geometry-cct-region.pdf",
      "figures/hcct-cct-geometry-cct-horn-magnified.pdf",
      "figures/hcct-cct-geometry-cct-infinite-area-tubes.pdf",
    ),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "illustrations/connectivity-scores",
    "reproduction/illustrations/connectivity_scores.py",
    "One-dimensional CCT and HCCT score curves with 95% thresholds.",
    ("cct_example.pdf",),
    ("wolfram/cct_connectivity.wl",),
    outputs=("figures/connectivity-scores.pdf",),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "illustrations/connectivity-regions",
    "reproduction/illustrations/connectivity_regions.py",
    "Two-dimensional CCT and HCCT confidence-region contours.",
    ("cct_example_2.pdf",),
    outputs=("figures/connectivity-regions.pdf",),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "illustrations/two-pvalue-regions",
    "reproduction/illustrations/pvalue_regions.py",
    "Acceptance regions for six two-p-value combination rules.",
    tuple(f"{name}_2p.pdf" for name in ("fisher", "stouffer", "bonferroni", "cauchy", "hcauchy", "harmonic")),
    outputs=tuple(
      f"figures/pvalue-regions-{name}.pdf"
      for name in ("fisher", "stouffer", "bonferroni", "cauchy", "hcauchy", "harmonic")
    ),
    paper_runtime="under 5 minutes",
  ),
  Experiment(
    "illustrations/convexity",
    "geogebra/convexity-half-cauchy-student-t.ggb",
    "Convexity score-function illustrations for normal and Student t inputs.",
    ("convexity_hc_t.pdf", "convexity_c_t.pdf", "convexity_f_n.pdf", "convexity_f_t.pdf"),
    (
      "geogebra/convexity-cauchy-student-t.ggb",
      "geogebra/convexity-fisher-normal.ggb",
      "geogebra/convexity-fisher-student-t.ggb",
    ),
    language="geogebra",
    runner_enabled=False,
    notes="GeoGebra Classic 6 has no supported CLI PDF export; export the four sources manually as vector PDFs.",
    outputs=tuple(
      f"figures/{name}.pdf"
      for name in (
        "convexity-half-cauchy-student-t",
        "convexity-cauchy-student-t",
        "convexity-fisher-normal",
        "convexity-fisher-student-t",
      )
    ),
  ),
  Experiment(
    "illustrations/distribution-densities",
    "geogebra/density-cauchy-score-transform.ggb",
    "Cauchy and half-Cauchy density and score-transform illustrations.",
    ("cauchy_density.pdf", "hcauchy_density.pdf"),
    ("geogebra/density-half-cauchy-score-transform.ggb",),
    language="geogebra",
    runner_enabled=False,
    notes="GeoGebra Classic 6 has no supported CLI PDF export; export both sources manually as vector PDFs.",
    outputs=(
      "figures/density-cauchy-score-transform.pdf",
      "figures/density-half-cauchy-score-transform.pdf",
    ),
  ),
  Experiment(
    "illustrations/simultaneous-projection",
    "geogebra/simultaneous-interval-projection.ggb",
    "Projection of confidence regions into simultaneous intervals.",
    ("simultaneous-1.pdf",),
    language="geogebra",
    runner_enabled=False,
    notes="GeoGebra Classic 6 has no supported CLI PDF export; export the source manually as a vector PDF.",
    outputs=("figures/simultaneous-interval-projection.pdf",),
  ),
  Experiment(
    "illustrations/contour-integration",
    "geogebra/contour-integration.ggb",
    "Contour-integration illustration in the supplement.",
    ("contour2.pdf",),
    language="geogebra",
    runner_enabled=False,
    notes="GeoGebra Classic 6 has no supported CLI PDF export; export the source manually as a vector PDF.",
    outputs=("figures/contour-integration.pdf",),
  ),
  Experiment(
    "numerical/distribution-precision",
    "reproduction/numerical/distribution_precision.py",
    "Distribution accuracy and runtime tables printed by the script.",
    notes="Complete HCCT/EHMP table grids, corrected Landau locations, numerical error estimates and machine-dependent runtimes; historical and new outputs remain separate.",
    outputs=("data/distribution-precision.csv", "data/distribution-precision-hcct.csv",
             "data/distribution-precision-ehmp.csv"),
    paper_runtime="under 5 minutes",
  ),
  Experiment(
    "numerical/support-solver-benchmark",
    "reproduction/numerical/support_solver_benchmark.py",
    "Dense-KKT and constrained-fallback timings across the component-size cap.",
    seed=20260901,
    notes="A scalability benchmark for the reusable solver, not a paper result.",
    outputs=("data/support-solver-benchmark.csv",),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "numerical/pvalue-examples",
    "reproduction/numerical/pvalue_examples.py",
    "Examples comparing combined p-values across methods.",
    notes="All non-CAtr columns of supplement table tab:expcombine, including Bonferroni; CAtr intentionally omitted at the user's request.",
    outputs=("data/pvalue-examples.csv",),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "numerical/independence-thresholds",
    "reproduction/numerical/independence_thresholds.py",
    "All non-CAtr cells of main table tab:wilson, with canonical finite-m thresholds and discrepancy diagnostics.",
    notes="Fang/CAtr columns intentionally omitted; the historical cardinality-based Wilson comparator is distinguished from the weight-aware HMP extension.",
    outputs=("data/independence-thresholds.csv", "diagnostics/independence-thresholds.json"),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "numerical/legacy-thresholds",
    "reproduction/numerical/legacy_thresholds.py",
    "Legacy Monte Carlo threshold calculations related to main table tab:wilson.",
    seed=2025,
    notes="Bounded deterministic audit; not an exact reproduction of the recovered 100-million-draw exploration.",
    outputs=("data/legacy-thresholds.csv",),
    paper_runtime="under 1 minute at the bounded setting",
  ),
  Experiment(
    "numerical/landau-approximation",
    "reproduction/numerical/landau_approximation.py",
    "Landau approximations for half-Cauchy sums.",
    tuple(f"landau_{size}.pdf" for size in (1, 10, 100, 1000)),
    outputs=(
      *(f"figures/landau-approximation-density-sum-{size}.pdf" for size in (1, 10, 100, 1000)),
      "figures/landau-approximation-cdf-sum-1000.pdf",
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "exploratory/bayes-factor-comparisons",
    "reproduction/exploratory/bayes_factor_comparisons.py",
    "Recovered Bayes-factor comparison curves; not referenced by the current paper.",
    notes="Migrated from the former top-level Code/test.py scratch script.",
    outputs=tuple(
      f"figures/bayes-factor-comparisons-{label}.pdf"
      for label in ("beta-one", "goodman", "edwards", "edwards-two")
    ),
    paper_runtime="under 1 minute",
  ),
  Experiment(
    "one-dimensional/normal-coverage",
    "reproduction/one_dimensional/normal_coverage.py",
    "Coverage under AR(1) and equicorrelated normal statistics.",
    (
      "run_101_102.pdf",
      "run_103_104.pdf",
      "coverage_1d.pdf",
      "coverage_1d_corrected.pdf",
    ),
    seed=2025,
    notes="The coverage_1d panels are historical commented manuscript panels; the active script retains the uncorrected experiment but not its former 0.7-level correction branch.",
    outputs=(
      "data/normal-coverage-ar1.npy",
      "data/normal-coverage-equicorrelated.npy",
      "figures/normal-coverage-ar1.pdf",
      "figures/normal-coverage-equicorrelated.pdf",
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "one-dimensional/false-positive-rate",
    "reproduction/one_dimensional/false_positive_rate.py",
    "False-positive rates for competing combination methods.",
    ("run_105.pdf", "run_106.pdf"),
    seed=2025,
    outputs=(
      "data/false-positive-rate.npy",
      "data/false-positive-rate-paired.npz",
      "figures/false-positive-rate-ar1.pdf",
      "figures/false-positive-rate-equicorrelated.pdf",
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "one-dimensional/interval-width",
    "reproduction/one_dimensional/interval_width.py",
    "Confidence-interval widths under two correlation structures.",
    ("run_107.pdf", "run_108.pdf"),
    seed=2025,
    outputs=(
      "data/interval-width.npy",
      "figures/interval-width-ar1.pdf",
      "figures/interval-width-equicorrelated.pdf",
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "one-dimensional/power",
    "reproduction/one_dimensional/power_comparison.py",
    "Power comparison under dense and sparse signals.",
    ("run_109.pdf", "run_110.pdf"),
    seed=2025,
    outputs=(
      "data/power-comparison.npy",
      "data/power-comparison-paired.npz",
      "figures/power-comparison-dense.pdf",
      "figures/power-comparison-sparse.pdf",
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "one-dimensional/copulas",
    "reproduction/one_dimensional/copula_coverage.py",
    "Coverage under multivariate-t and Archimedean copula dependence.",
    (
      "run_201_202.pdf",
      "run_203_204.pdf",
      "run_205_206_HCauchy.pdf",
      "run_205_206_Fisher.pdf",
      "run_207_208_HCauchy.pdf",
      "run_207_208_Fisher.pdf",
    ),
    seed=2025,
    outputs=tuple(
      f"{kind}/copula-coverage-{label}.{extension}"
      for kind, extension in (("data", "npy"), ("figures", "pdf"))
      for label in (
        "student-t-ar1",
        "student-t-equicorrelated",
        "ali-mikhail-haq-hcauchy",
        "ali-mikhail-haq-fisher",
        "farlie-gumbel-morgenstern-hcauchy",
        "farlie-gumbel-morgenstern-fisher",
      )
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "multidimensional/coverage",
    "reproduction/multidimensional/coverage.py",
    "Multidimensional coverage for 10 and 500 studies.",
    (
      "run_301_302_m_10.pdf",
      "run_301_302.pdf",
      "coverage_md.pdf",
      "coverage_md_corrected.pdf",
    ),
    seed=2025,
    notes="The coverage_md panels are historical commented manuscript panels; the active script retains the uncorrected experiment but not its former 0.7-level correction branch.",
    outputs=tuple(
      f"{kind}/multivariate-coverage-{study_count}-studies.{extension}"
      for kind, extension in (("data", "npy"), ("figures", "pdf"))
      for study_count in (500, 10)
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "multidimensional/contours",
    "reproduction/multidimensional/contours.py",
    "Two-dimensional confidence-region contours across correlations.",
    tuple(f"run_303_rho_{rho}.pdf" for rho in (0, 0.3, 0.6, 0.9)),
    seed=2024,
    outputs=tuple(
      f"figures/multivariate-contours-dimension-{dimension}-rho-{rho:g}.pdf"
      for dimension in (2, 10)
      for rho in (0.0, 0.3, 0.6, 0.9)
    ),
    paper_runtime="under 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-normal",
    "reproduction/multidimensional/dac_normal.py",
    "Divide-and-combine slices for multivariate normal samples.",
    tuple(f"run_{run}_norm_{block}_{coords}.pdf" for run, block, coords in ((305, 1, "1_51"), (306, 5, "1_51"), (307, 25, "1_51"), (311, 100, "1_51"), (312, 1, "1_2"), (308, 5, "1_2"), (309, 25, "1_2"), (310, 100, "1_2"))),
    seed=2025,
    outputs=(
      *(f"figures/dac-normal-slice-block-{block}-coordinates-{coordinates}.pdf" for block in (1, 5, 25, 100) for coordinates in ("1-51", "1-2")),
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-lognormal-estimate",
    "reproduction/multidimensional/dac_lognormal_estimate.py",
    "Lognormal divide-and-combine slices centered at the projected estimate.",
    tuple(f"run_{run}_lognorm_{block}_{coords}.pdf" for run, block, coords in ((305, 1, "1_51"), (306, 5, "1_51"), (307, 25, "1_51"), (311, 100, "1_51"), (312, 1, "1_2"), (308, 5, "1_2"), (309, 25, "1_2"), (310, 100, "1_2"))),
    seed=2024,
    outputs=(
      *(f"figures/dac-lognormal-estimate-slice-block-{block}-coordinates-{coordinates}.pdf" for block in (1, 5, 25, 100) for coordinates in ("1-51", "1-2")),
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-lognormal-truth",
    "reproduction/multidimensional/dac_lognormal_truth.py",
    "Lognormal divide-and-combine slices centered at the true mean.",
    tuple(f"run_{run}_lognorm_true_{block}_{coords}.pdf" for run, block, coords in ((305, 1, "1_51"), (306, 5, "1_51"), (307, 25, "1_51"), (311, 100, "1_51"), (312, 1, "1_2"), (308, 5, "1_2"), (309, 25, "1_2"), (310, 100, "1_2"))),
    seed=2024,
    outputs=(
      *(f"figures/dac-lognormal-truth-slice-block-{block}-coordinates-{coordinates}.pdf" for block in (1, 5, 25, 100) for coordinates in ("1-51", "1-2")),
    ),
    paper_runtime="possibly over 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-normal-diagnostics",
    "reproduction/multidimensional/dac_normal_diagnostics.py",
    "Normal DAC direction widths and coverage by block size.",
    seed=2025,
    outputs=(
      "data/dac-normal-direction-widths.npz",
      "data/dac-normal-coverage.npz",
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-lognormal-estimate-diagnostics",
    "reproduction/multidimensional/dac_lognormal_estimate_diagnostics.py",
    "Lognormal DAC direction widths and coverage for the estimate-centered setup.",
    seed=2024,
    outputs=(
      "data/dac-lognormal-estimate-direction-widths.npz",
      "data/dac-lognormal-estimate-coverage.npz",
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "multidimensional/dac-lognormal-truth-diagnostics",
    "reproduction/multidimensional/dac_lognormal_truth_diagnostics.py",
    "Lognormal DAC direction widths and 95%/99% coverage for the truth-centered setup.",
    seed=2024,
    outputs=(
      "data/dac-lognormal-truth-direction-widths.npz",
      "data/dac-lognormal-truth-coverage.npz",
    ),
    paper_runtime="likely over 5 minutes",
  ),
  Experiment(
    "network-meta-analysis/treatment-estimates",
    "reproduction/network_meta_analysis/treatment_estimates.py",
    "Complete numerical support for simulated and real-data treatment-effect tables, with discrepancy records.",
    seed=2025,
    notes="A fresh reproducible single sample per simulated correlation; original historical samples were not recovered. Both profiles export all table rows. Corrected estimated-SE calibration and adjusted real-data WLS are explicit.",
    outputs=(
      "data/nma-simulation-treatment-estimates.csv",
      "data/nma-real-data-treatment-estimates.csv",
      "data/nma-simulation-treatment-inputs.npz",
      "diagnostics/nma-treatment-estimates.json",
      "diagnostics/nma-real-data-empty-set-adjustment.json",
    ),
    paper_runtime="under 5 minutes",
  ),
  Experiment(
    "network-meta-analysis/simulation",
    "reproduction/network_meta_analysis/simulation.py",
    "Network meta-analysis coverage and simultaneous-interval widths.",
    ("run_401_coverage.pdf", "run_401_theta_1.pdf", "run_401_theta_2.pdf"),
    seed=2025,
    notes="Paper profile uses the user-approved 100 replicates per rho since 2026-09-21 (manuscript: 500). Saved simulation summaries retain empirical WLS-MA. Plots default to stored historical manual WLS-MA widths on the matching paper grid; otherwise they use the empirical oracle fallback. Plot policy and plotted summaries record the distinction; coverage always uses saved simulation events.",
    outputs=(
      "data/nma-simulation-summary.npz",
      "data/nma-simulation-plotted-summary.npz",
      "data/nma-simulation-replicates.npz",
      "diagnostics/nma-simulation-calibration.json",
      "diagnostics/nma-simulation-plot-policy.json",
      "figures/nma-simulation-coverage.pdf",
      "figures/nma-simulation-interval-width-theta-1.pdf",
      "figures/nma-simulation-interval-width-theta-2.pdf",
    ),
    paper_runtime="about 40 minutes for 100 replicates per rho; solver-dependent estimate",
  ),
  Experiment(
    "network-meta-analysis/real-data",
    "reproduction/network_meta_analysis/real_data.py",
    "Real-data heatmaps and treatment-specific confidence regions.",
    ("real_heatmap_1.pdf", "real_heatmap_3.pdf", "real_tr_2.pdf", "real_tr_3.pdf", "real_tr_6.pdf"),
    notes="Loads the versioned Senn2013 CSV snapshot and reproduces the netmeta multi-arm WLS adjustment in Python; no R runtime is required.",
    outputs=(
      "data/nma-real-data-hcct-interval-widths.npy",
      "diagnostics/nma-real-data-empty-set-adjustment.json",
      "diagnostics/nma-real-data-support-certificates.json",
      "figures/nma-real-data-hcct-interval-width-heatmap.pdf",
      "figures/nma-real-data-wls-interval-width-heatmap.pdf",
      "figures/nma-real-data-bonferroni-wls-heatmap.pdf",
      "figures/nma-real-data-hcct-minus-bonferroni-wls-heatmap.pdf",
      *(f"figures/nma-real-data-width-vs-comparisons-{treatment}.pdf" for treatment in ("benf", "metf", "sita", "rosi")),
    ),
    paper_runtime="under 5 minutes",
  ),
)

_BY_ID = {experiment.id: experiment for experiment in EXPERIMENTS}


def get_experiment(experiment_id: str) -> Experiment:
  try:
    return _BY_ID[experiment_id]
  except KeyError as error:
    choices = ", ".join(sorted(_BY_ID))
    raise KeyError(f"unknown experiment {experiment_id!r}; choose one of: {choices}") from error
