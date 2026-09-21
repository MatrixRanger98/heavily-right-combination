# Experiment catalog

Run these commands from the release repository root after
`python -m pip install -e ".[reproduction]"`.
Every command uses a new immutable run directory. Replace the example run ID
if it already exists. The [results index](../results/README.md) maps actual
included outputs in paper order; declared filenames below describe what a
new run generates, not a promise that every full-profile experiment is bundled.

```bash
python -m reproduction.run --list
python -m reproduction.run network-meta-analysis/simulation --describe
```

Use `--profile smoke` for a small end-to-end check. An optional
`--max-seconds 300` limits wall time; omit a cap for a complete long run.
Seed/profile settings and uncertainty qualifications are in
[DATA_AND_PROFILES.md](DATA_AND_PROFILES.md); all seven simulation figure
families support [plot-only reuse](SAVED_DATA.md).

Main Figures 10 and 14 are fixed manuscript selections; included reruns are
comparison-only. NMA's executable paper profile uses 100 rather than the
manuscript's 500 replications per correlation. CAtr numerical table columns
are outside the supplied table reproduction scope.

## Contents

| Experiment ID | Language | Seed |
|---|---|---|
| [illustrations/hcct-cct-geometry](#illustrationshcct-cct-geometry) | python | NA |
| [illustrations/connectivity-scores](#illustrationsconnectivity-scores) | python | NA |
| [illustrations/connectivity-regions](#illustrationsconnectivity-regions) | python | NA |
| [illustrations/two-pvalue-regions](#illustrationstwo-pvalue-regions) | python | NA |
| [illustrations/convexity](#illustrationsconvexity) | geogebra | NA |
| [illustrations/distribution-densities](#illustrationsdistribution-densities) | geogebra | NA |
| [illustrations/simultaneous-projection](#illustrationssimultaneous-projection) | geogebra | NA |
| [illustrations/contour-integration](#illustrationscontour-integration) | geogebra | NA |
| [numerical/distribution-precision](#numericaldistribution-precision) | python | NA |
| [numerical/support-solver-benchmark](#numericalsupport-solver-benchmark) | python | 20260901 |
| [numerical/pvalue-examples](#numericalpvalue-examples) | python | NA |
| [numerical/independence-thresholds](#numericalindependence-thresholds) | python | NA |
| [numerical/legacy-thresholds](#numericallegacy-thresholds) | python | 2025 |
| [numerical/landau-approximation](#numericallandau-approximation) | python | NA |
| [exploratory/bayes-factor-comparisons](#exploratorybayes-factor-comparisons) | python | NA |
| [one-dimensional/normal-coverage](#one-dimensionalnormal-coverage) | python | 2025 |
| [one-dimensional/false-positive-rate](#one-dimensionalfalse-positive-rate) | python | 2025 |
| [one-dimensional/interval-width](#one-dimensionalinterval-width) | python | 2025 |
| [one-dimensional/power](#one-dimensionalpower) | python | 2025 |
| [one-dimensional/copulas](#one-dimensionalcopulas) | python | 2025 |
| [multidimensional/coverage](#multidimensionalcoverage) | python | 2025 |
| [multidimensional/contours](#multidimensionalcontours) | python | 2024 |
| [multidimensional/dac-normal](#multidimensionaldac-normal) | python | 2025 |
| [multidimensional/dac-lognormal-estimate](#multidimensionaldac-lognormal-estimate) | python | 2024 |
| [multidimensional/dac-lognormal-truth](#multidimensionaldac-lognormal-truth) | python | 2024 |
| [multidimensional/dac-normal-diagnostics](#multidimensionaldac-normal-diagnostics) | python | 2025 |
| [multidimensional/dac-lognormal-estimate-diagnostics](#multidimensionaldac-lognormal-estimate-diagnostics) | python | 2024 |
| [multidimensional/dac-lognormal-truth-diagnostics](#multidimensionaldac-lognormal-truth-diagnostics) | python | 2024 |
| [network-meta-analysis/treatment-estimates](#network-meta-analysistreatment-estimates) | python | 2025 |
| [network-meta-analysis/simulation](#network-meta-analysissimulation) | python | 2025 |
| [network-meta-analysis/real-data](#network-meta-analysisreal-data) | python | NA |

## illustrations/hcct-cct-geometry

Blockwise HCCT region and CCT horn geometry, including disjoint equal-area tubes.

Source: [reproduction/illustrations/hcct_cct_geometry.py](../reproduction/illustrations/hcct_cct_geometry.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run illustrations/hcct-cct-geometry \
  --profile paper --run-id my-hcct-cct-geometry \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/hcct-cct-geometry-config.json`
- `figures/hcct-cct-geometry-hcct-region.pdf`
- `figures/hcct-cct-geometry-cct-region.pdf`
- `figures/hcct-cct-geometry-cct-horn-magnified.pdf`
- `figures/hcct-cct-geometry-cct-infinite-area-tubes.pdf`

Deterministic in both profiles. The rounded HCCT threshold is 13.68751; the configuration JSON records the mathematical settings.

## illustrations/connectivity-scores

One-dimensional CCT and HCCT score curves with 95% thresholds.

Source: [reproduction/illustrations/connectivity_scores.py](../reproduction/illustrations/connectivity_scores.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run illustrations/connectivity-scores \
  --profile paper --run-id my-connectivity-scores \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/connectivity-scores.pdf`

Deterministic score curves. The alternative Wolfram script is documented in [the Wolfram guide](../wolfram/README.md).

## illustrations/connectivity-regions

Two-dimensional CCT and HCCT confidence-region contours.

Source: [reproduction/illustrations/connectivity_regions.py](../reproduction/illustrations/connectivity_regions.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run illustrations/connectivity-regions \
  --profile paper --run-id my-connectivity-regions \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/connectivity-regions.pdf`

Deterministic two-dimensional contours in both profiles.

## illustrations/two-pvalue-regions

Acceptance regions for six two-p-value combination rules.

Source: [reproduction/illustrations/pvalue_regions.py](../reproduction/illustrations/pvalue_regions.py). Deterministic; seed NA. Runtime guidance: under 5 minutes.

```bash
python -m reproduction.run illustrations/two-pvalue-regions \
  --profile paper --run-id my-two-pvalue-regions \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/pvalue-regions-fisher.pdf`
- `figures/pvalue-regions-stouffer.pdf`
- `figures/pvalue-regions-bonferroni.pdf`
- `figures/pvalue-regions-cauchy.pdf`
- `figures/pvalue-regions-hcauchy.pdf`
- `figures/pvalue-regions-harmonic.pdf`

Six deterministic two-p-value acceptance-region panels.

## illustrations/convexity

Convexity score-function illustrations for normal and Student t inputs.

Source: [geogebra/convexity-half-cauchy-student-t.ggb](../geogebra/convexity-half-cauchy-student-t.ggb). Deterministic; seed NA. Runtime guidance: depends on external export.

This external GeoGebra construction is listed by the Python runner but is not
executed by it. Follow [the GeoGebra export guide](../geogebra/README.md).
Source and PDF stems match; export into a new result directory.

Additional sources: [convexity-cauchy-student-t.ggb](../geogebra/convexity-cauchy-student-t.ggb), [convexity-fisher-normal.ggb](../geogebra/convexity-fisher-normal.ggb), [convexity-fisher-student-t.ggb](../geogebra/convexity-fisher-student-t.ggb).

Declared outputs, relative to the run directory:

- `figures/convexity-half-cauchy-student-t.pdf`
- `figures/convexity-cauchy-student-t.pdf`
- `figures/convexity-fisher-normal.pdf`
- `figures/convexity-fisher-student-t.pdf`

## illustrations/distribution-densities

Cauchy and half-Cauchy density and score-transform illustrations.

Source: [geogebra/density-cauchy-score-transform.ggb](../geogebra/density-cauchy-score-transform.ggb). Deterministic; seed NA. Runtime guidance: depends on external export.

This external GeoGebra construction is listed by the Python runner but is not
executed by it. Follow [the GeoGebra export guide](../geogebra/README.md).
Source and PDF stems match; export into a new result directory.

Additional sources: [density-half-cauchy-score-transform.ggb](../geogebra/density-half-cauchy-score-transform.ggb).

Declared outputs, relative to the run directory:

- `figures/density-cauchy-score-transform.pdf`
- `figures/density-half-cauchy-score-transform.pdf`

## illustrations/simultaneous-projection

Projection of confidence regions into simultaneous intervals.

Source: [geogebra/simultaneous-interval-projection.ggb](../geogebra/simultaneous-interval-projection.ggb). Deterministic; seed NA. Runtime guidance: depends on external export.

This external GeoGebra construction is listed by the Python runner but is not
executed by it. Follow [the GeoGebra export guide](../geogebra/README.md).
Source and PDF stems match; export into a new result directory.

Declared outputs, relative to the run directory:

- `figures/simultaneous-interval-projection.pdf`

## illustrations/contour-integration

Contour-integration illustration in the supplement.

Source: [geogebra/contour-integration.ggb](../geogebra/contour-integration.ggb). Deterministic; seed NA. Runtime guidance: depends on external export.

This external GeoGebra construction is listed by the Python runner but is not
executed by it. Follow [the GeoGebra export guide](../geogebra/README.md).
Source and PDF stems match; export into a new result directory.

Declared outputs, relative to the run directory:

- `figures/contour-integration.pdf`

## numerical/distribution-precision

Distribution accuracy and runtime tables exported as CSV.

Source: [reproduction/numerical/distribution_precision.py](../reproduction/numerical/distribution_precision.py). Deterministic; seed NA. Runtime guidance: under 5 minutes.

```bash
python -m reproduction.run numerical/distribution-precision \
  --profile paper --run-id my-distribution-precision \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/distribution-precision.csv`
- `data/distribution-precision-hcct.csv`
- `data/distribution-precision-ehmp.csv`

Paper profile: all 60 PDF/CDF measurements (16 HCCT points and 14 EHMP points); smoke: two points per law. Error estimates are not rigorous bounds, and measured milliseconds depend on hardware and dependencies. EHMP m=1000,x=4 PDF/CDF values differ from displayed manuscript digits within its stated numerical errors. Signed differences are Landau minus finite-law values.

## numerical/support-solver-benchmark

Dense-KKT and constrained-fallback timings across the component-size cap.

Source: [reproduction/numerical/support_solver_benchmark.py](../reproduction/numerical/support_solver_benchmark.py). Fixed seed: **20260901**. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run numerical/support-solver-benchmark \
  --profile paper --run-id my-support-solver-benchmark \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/support-solver-benchmark.csv`

Paper scenarios use (total, active-component) dimensions (50,50), (170,170), (260,260), (2000,50). Smoke uses (20,20), (260,260), (400,20). The dense-Newton cap is 250; component decomposition avoids solving a 2,000-variable dense system when only one 50-variable component is active. This is an additional benchmark, not a paper table.

## numerical/pvalue-examples

Examples comparing combined p-values across methods.

Source: [reproduction/numerical/pvalue_examples.py](../reproduction/numerical/pvalue_examples.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run numerical/pvalue-examples \
  --profile paper --run-id my-pvalue-examples \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/pvalue-examples.csv`

Deterministic support for 36 non-CAtr cells of Supplement Table 4, including Bonferroni. CAtr is outside this table reproduction scope.

## numerical/independence-thresholds

All non-CAtr cells of main table tab:wilson, with canonical finite-m thresholds and discrepancy diagnostics.

Source: [reproduction/numerical/independence_thresholds.py](../reproduction/numerical/independence_thresholds.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run numerical/independence-thresholds \
  --profile paper --run-id my-independence-thresholds \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/independence-thresholds.csv`
- `diagnostics/independence-thresholds.json`

Deterministic support for 20 non-CAtr cells of Main Table 3, with probability round-trip checks. The cardinality-based Wilson comparator is distinguished from the weight-aware HMP extension; Fang/CAtr columns are not included.

## numerical/legacy-thresholds

Bounded Monte Carlo threshold calculations related to main table tab:wilson.

Source: [reproduction/numerical/legacy_thresholds.py](../reproduction/numerical/legacy_thresholds.py). Fixed seed: **2025**. Runtime guidance: under 1 minute at the bounded setting.

```bash
python -m reproduction.run numerical/legacy-thresholds \
  --profile paper --run-id my-legacy-thresholds \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/legacy-thresholds.csv`

Optional bounded Monte Carlo calculation: 1,000,000 draws per scenario for paper and 20,000 for smoke, processed in chunks. It is separate from the deterministic Main Table 3 generator. This CAtr-related exploration is outside the selected table reruns; no output for it is bundled.

## numerical/landau-approximation

Landau approximations for half-Cauchy sums.

Source: [reproduction/numerical/landau_approximation.py](../reproduction/numerical/landau_approximation.py). Deterministic; seed NA. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run numerical/landau-approximation \
  --profile paper --run-id my-landau-approximation \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/landau-approximation-density-sum-1.pdf`
- `figures/landau-approximation-density-sum-10.pdf`
- `figures/landau-approximation-density-sum-100.pdf`
- `figures/landau-approximation-density-sum-1000.pdf`
- `figures/landau-approximation-cdf-sum-1000.pdf`

The paper profile compares half-Cauchy sum densities at m=1,10,100,1000 and a CDF at m=1000. Smoke produces only the m=1,10 density panels, on a coarser grid. The included full run took approximately 50 seconds. Guarded finite-law kernels report numerical failures rather than substituting endpoint probabilities.

## exploratory/bayes-factor-comparisons

Bayes-factor comparison curves; not referenced by the current paper.

Source: [reproduction/exploratory/bayes_factor_comparisons.py](../reproduction/exploratory/bayes_factor_comparisons.py). Deterministic; seed NA. Runtime guidance: under 1 minute.

```bash
python -m reproduction.run exploratory/bayes-factor-comparisons \
  --profile paper --run-id my-bayes-factor-comparisons \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/bayes-factor-comparisons-beta-one.pdf`
- `figures/bayes-factor-comparisons-goodman.pdf`
- `figures/bayes-factor-comparisons-edwards.pdf`
- `figures/bayes-factor-comparisons-edwards-two.pdf`

Four deterministic additional curves; marked Unused/NA in the results map.

## one-dimensional/normal-coverage

Coverage under AR(1) and equicorrelated normal statistics.

Source: [reproduction/one_dimensional/normal_coverage.py](../reproduction/one_dimensional/normal_coverage.py). Fixed seed: **2025**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run one-dimensional/normal-coverage \
  --profile paper --run-id my-normal-coverage \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/normal-coverage-ar1.npy`
- `data/normal-coverage-equicorrelated.npy`
- `figures/normal-coverage-ar1.pdf`
- `figures/normal-coverage-equicorrelated.pdf`

Paper: 500 studies, 50 batches of 10,000 draws per rho/family; rho=0,.1,...,.9; levels .05/.01. Smoke: 30 studies, 2 batches of 200, rho=0,.6. Bands show batch quantiles, not confidence intervals for the pooled mean.

## one-dimensional/false-positive-rate

False-positive rates for competing combination methods.

Source: [reproduction/one_dimensional/false_positive_rate.py](../reproduction/one_dimensional/false_positive_rate.py). Fixed seed: **2025**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run one-dimensional/false-positive-rate \
  --profile paper --run-id my-false-positive-rate \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/false-positive-rate.npy`
- `data/false-positive-rate-paired.npz`
- `figures/false-positive-rate-ar1.pdf`
- `figures/false-positive-rate-equicorrelated.pdf`

Paper: 500 studies and 10,000 shared outcomes for all nine methods per rho/family, ten rho values, level .05. Smoke: 30 studies, 200 outcomes, rho=0,.3,.6,.9. The included paired full run took about 13 seconds. All outcomes remain; empty inverted sets contribute noncoverage without adjustment. Main Figure 14 is fixed; these outputs are comparison-only.

## one-dimensional/interval-width

Confidence-interval widths under two correlation structures.

Source: [reproduction/one_dimensional/interval_width.py](../reproduction/one_dimensional/interval_width.py). Fixed seed: **2025**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run one-dimensional/interval-width \
  --profile paper --run-id my-interval-width \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/interval-width.npy`
- `figures/interval-width-ar1.pdf`
- `figures/interval-width-equicorrelated.pdf`

Paper: 500 studies, 10,000 interval inversions at each of rho=0,.3,.6,.9 for two families (80,000 intervals). Smoke: 30 studies, 40 intervals per cell. Full lengths, including zero for confirmed empty sets, are saved before plotting.

## one-dimensional/power

Power comparison under dense and sparse signals.

Source: [reproduction/one_dimensional/power_comparison.py](../reproduction/one_dimensional/power_comparison.py). Fixed seed: **2025**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run one-dimensional/power \
  --profile paper --run-id my-power \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/power-comparison.npy`
- `data/power-comparison-paired.npz`
- `figures/power-comparison-dense.pdf`
- `figures/power-comparison-sparse.pdf`

Paper: 500 studies, 10,000 shared outcomes across nine methods per rho/alternative, ten rho values. Smoke: 30 studies, 200 outcomes, four rho values. Dense s=0,r=.1; sparse s=r=.3; signal count floor(m**(1-s)), amplitude sqrt(2*r*log(m)). Power is nominal-.05 rejection, not equal-actual-size power. Main Figure 14 outputs remain comparison-only.

## one-dimensional/copulas

Coverage under multivariate-t and Archimedean copula dependence.

Source: [reproduction/one_dimensional/copula_coverage.py](../reproduction/one_dimensional/copula_coverage.py). Fixed seed: **2025**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run one-dimensional/copulas \
  --profile paper --run-id my-copulas \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/copula-coverage-student-t-ar1.npy`
- `data/copula-coverage-student-t-equicorrelated.npy`
- `data/copula-coverage-ali-mikhail-haq-hcauchy.npy`
- `data/copula-coverage-ali-mikhail-haq-fisher.npy`
- `data/copula-coverage-farlie-gumbel-morgenstern-hcauchy.npy`
- `data/copula-coverage-farlie-gumbel-morgenstern-fisher.npy`
- `figures/copula-coverage-student-t-ar1.pdf`
- `figures/copula-coverage-student-t-equicorrelated.pdf`
- `figures/copula-coverage-ali-mikhail-haq-hcauchy.pdf`
- `figures/copula-coverage-ali-mikhail-haq-fisher.pdf`
- `figures/copula-coverage-farlie-gumbel-morgenstern-hcauchy.pdf`
- `figures/copula-coverage-farlie-gumbel-morgenstern-fisher.pdf`

Paper: 500 studies, 50 batches of 10,000 draws per parameter, levels .05/.01. Standard-t df=10 uses a common radial draw; AMH/FGM use independent dependent pairs. Ten rho values for t and theta=-.9,-.7,...,.9 for AMH/FGM. Smoke: 20 studies, 2 batches of 200, rho=0,.6 and theta=-.6,0,.6. Arrays save nonrejection while figures plot 1-coverage.

## multidimensional/coverage

Multidimensional coverage for 10 and 500 studies.

Source: [reproduction/multidimensional/coverage.py](../reproduction/multidimensional/coverage.py). Fixed seed: **2025**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run multidimensional/coverage \
  --profile paper --run-id my-coverage \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/multivariate-coverage-500-studies.npy`
- `data/multivariate-coverage-10-studies.npy`
- `figures/multivariate-coverage-500-studies.pdf`
- `figures/multivariate-coverage-10-studies.pdf`

Paper: studies=500/10, dimensions=2/5/10/25, ten rho values, levels .05/.01; 10 batches × 10,000 outcomes are averaged per cell. This is 100,000 outcomes, versus the manuscript's stated 1,000. Smoke: studies=30/10, dimensions=2/5, rho=0,.6, 100 outcomes/cell. Smoke filenames use 30 rather than 500 studies.

## multidimensional/contours

Two-dimensional confidence-region contours across correlations.

Source: [reproduction/multidimensional/contours.py](../reproduction/multidimensional/contours.py). Fixed seed: **2024**. Runtime guidance: under 5 minutes.

```bash
python -m reproduction.run multidimensional/contours \
  --profile paper --run-id my-contours \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/multivariate-contours-dimension-2-rho-0.pdf`
- `figures/multivariate-contours-dimension-2-rho-0.3.pdf`
- `figures/multivariate-contours-dimension-2-rho-0.6.pdf`
- `figures/multivariate-contours-dimension-2-rho-0.9.pdf`
- `figures/multivariate-contours-dimension-10-rho-0.pdf`
- `figures/multivariate-contours-dimension-10-rho-0.3.pdf`
- `figures/multivariate-contours-dimension-10-rho-0.6.pdf`
- `figures/multivariate-contours-dimension-10-rho-0.9.pdf`

Paper outputs cover dimensions 2 and 10, 500 studies, and rho=0,.3,.6,.9. Smoke uses the same dimensions, 30 studies, and rho=0,.6, producing four panels.

## multidimensional/dac-normal

Divide-and-combine slices for multivariate normal samples.

Source: [reproduction/multidimensional/dac_normal.py](../reproduction/multidimensional/dac_normal.py). Fixed seed: **2025**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-normal \
  --profile paper --run-id my-dac-normal \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/dac-normal-slice-block-1-coordinates-1-51.pdf`
- `figures/dac-normal-slice-block-1-coordinates-1-2.pdf`
- `figures/dac-normal-slice-block-5-coordinates-1-51.pdf`
- `figures/dac-normal-slice-block-5-coordinates-1-2.pdf`
- `figures/dac-normal-slice-block-25-coordinates-1-51.pdf`
- `figures/dac-normal-slice-block-25-coordinates-1-2.pdf`
- `figures/dac-normal-slice-block-100-coordinates-1-51.pdf`
- `figures/dac-normal-slice-block-100-coordinates-1-2.pdf`

Paper: dimension 100, n=1,000, rho=.6, block sizes 1/5/25/100, coordinate pairs 1–51 and 1–2. Smoke: dimension 10, n=50, blocks 1/5/10, pairs 1–6 and 1–2; output stems reflect those coordinates. Normal samples. Coverage diagnostics are separate experiments. Included full slice workflows take roughly half a minute.

## multidimensional/dac-lognormal-estimate

Lognormal divide-and-combine slices centered at the projected estimate.

Source: [reproduction/multidimensional/dac_lognormal_estimate.py](../reproduction/multidimensional/dac_lognormal_estimate.py). Fixed seed: **2024**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-lognormal-estimate \
  --profile paper --run-id my-dac-lognormal-estimate \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/dac-lognormal-estimate-slice-block-1-coordinates-1-51.pdf`
- `figures/dac-lognormal-estimate-slice-block-1-coordinates-1-2.pdf`
- `figures/dac-lognormal-estimate-slice-block-5-coordinates-1-51.pdf`
- `figures/dac-lognormal-estimate-slice-block-5-coordinates-1-2.pdf`
- `figures/dac-lognormal-estimate-slice-block-25-coordinates-1-51.pdf`
- `figures/dac-lognormal-estimate-slice-block-25-coordinates-1-2.pdf`
- `figures/dac-lognormal-estimate-slice-block-100-coordinates-1-51.pdf`
- `figures/dac-lognormal-estimate-slice-block-100-coordinates-1-2.pdf`

Paper: dimension 100, n=1,000, rho=.6, block sizes 1/5/25/100, coordinate pairs 1–51 and 1–2. Smoke: dimension 10, n=50, blocks 1/5/10, pairs 1–6 and 1–2; output stems reflect those coordinates. Lognormal samples; slices through the projected estimate. Coverage diagnostics are separate experiments. Included full slice workflows take roughly half a minute.

## multidimensional/dac-lognormal-truth

Lognormal divide-and-combine slices centered at the true mean.

Source: [reproduction/multidimensional/dac_lognormal_truth.py](../reproduction/multidimensional/dac_lognormal_truth.py). Fixed seed: **2024**. Runtime guidance: possibly over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-lognormal-truth \
  --profile paper --run-id my-dac-lognormal-truth \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `figures/dac-lognormal-truth-slice-block-1-coordinates-1-51.pdf`
- `figures/dac-lognormal-truth-slice-block-1-coordinates-1-2.pdf`
- `figures/dac-lognormal-truth-slice-block-5-coordinates-1-51.pdf`
- `figures/dac-lognormal-truth-slice-block-5-coordinates-1-2.pdf`
- `figures/dac-lognormal-truth-slice-block-25-coordinates-1-51.pdf`
- `figures/dac-lognormal-truth-slice-block-25-coordinates-1-2.pdf`
- `figures/dac-lognormal-truth-slice-block-100-coordinates-1-51.pdf`
- `figures/dac-lognormal-truth-slice-block-100-coordinates-1-2.pdf`

Paper: dimension 100, n=1,000, rho=.6, block sizes 1/5/25/100, coordinate pairs 1–51 and 1–2. Smoke: dimension 10, n=50, blocks 1/5/10, pairs 1–6 and 1–2; output stems reflect those coordinates. Lognormal samples; slices through the true mean. Coverage diagnostics are separate experiments. Included full slice workflows take roughly half a minute.

## multidimensional/dac-normal-diagnostics

Normal DAC direction widths and coverage by block size.

Source: [reproduction/multidimensional/dac_normal_diagnostics.py](../reproduction/multidimensional/dac_normal_diagnostics.py). Fixed seed: **2025**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-normal-diagnostics \
  --profile paper --run-id my-dac-normal-diagnostics \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/dac-normal-direction-widths.npz`
- `data/dac-normal-coverage.npz`

Paper: 2,000 coverage replications with direction-width block sizes 2 through 25; smoke: 5 replications and blocks 2 through 5. These data-only diagnostic outputs are distinct from slice figures and full diagnostic runs are not included. Seed 2025; numerical calibration uses estimated covariance of the mean.

## multidimensional/dac-lognormal-estimate-diagnostics

Lognormal DAC direction widths and coverage for the estimate-centered setup.

Source: [reproduction/multidimensional/dac_lognormal_estimate_diagnostics.py](../reproduction/multidimensional/dac_lognormal_estimate_diagnostics.py). Fixed seed: **2024**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-lognormal-estimate-diagnostics \
  --profile paper --run-id my-dac-lognormal-estimate-diagnostics \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/dac-lognormal-estimate-direction-widths.npz`
- `data/dac-lognormal-estimate-coverage.npz`

Paper: 2,000 coverage replications with direction-width block sizes 2 through 25; smoke: 5 replications and blocks 2 through 5. These data-only diagnostic outputs are distinct from slice figures and full diagnostic runs are not included. Seed 2024; numerical calibration uses estimated covariance of the mean.

## multidimensional/dac-lognormal-truth-diagnostics

Lognormal DAC direction widths and 95%/99% coverage for the truth-centered setup.

Source: [reproduction/multidimensional/dac_lognormal_truth_diagnostics.py](../reproduction/multidimensional/dac_lognormal_truth_diagnostics.py). Fixed seed: **2024**. Runtime guidance: likely over 5 minutes.

```bash
python -m reproduction.run multidimensional/dac-lognormal-truth-diagnostics \
  --profile paper --run-id my-dac-lognormal-truth-diagnostics \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/dac-lognormal-truth-direction-widths.npz`
- `data/dac-lognormal-truth-coverage.npz`

Paper: 2,000 coverage replications with direction-width block sizes 2 through 25; smoke: 5 replications and blocks 2 through 5. These data-only diagnostic outputs are distinct from slice figures and full diagnostic runs are not included. Seed 2024; numerical calibration uses estimated covariance of the mean.

## network-meta-analysis/treatment-estimates

Complete numerical support for simulated and real-data treatment-effect tables, with discrepancy records.

Source: [reproduction/network_meta_analysis/treatment_estimates.py](../reproduction/network_meta_analysis/treatment_estimates.py). Fixed seed: **2025**. Runtime guidance: under 5 minutes.

```bash
python -m reproduction.run network-meta-analysis/treatment-estimates \
  --profile paper --run-id my-treatment-estimates \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/nma-simulation-treatment-estimates.csv`
- `data/nma-real-data-treatment-estimates.csv`
- `data/nma-simulation-treatment-inputs.npz`
- `diagnostics/nma-treatment-estimates.json`
- `diagnostics/nma-real-data-empty-set-adjustment.json`

Both profiles export complete Tables 1 and 2. Table 1 uses private PCG64 seed 2025, one n=100 sample per rho=0,.3,.6,.9, with corrected estimated-SE calibration and no study exclusion. Table 2 is deterministic: WLS uses 28 comparisons and HCCT uses 26, with the two documented exclusions and full diagnostic trace. Primary estimates, raw inputs and certificates are retained.

## network-meta-analysis/simulation

Network meta-analysis coverage and simultaneous-interval widths.

Source: [reproduction/network_meta_analysis/simulation.py](../reproduction/network_meta_analysis/simulation.py). Fixed seed: **2025**. Runtime guidance: about 40 minutes for 100 replicates per rho; solver-dependent estimate.

```bash
python -m reproduction.run network-meta-analysis/simulation \
  --profile paper --run-id my-simulation \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/nma-simulation-summary.npz`
- `data/nma-simulation-plotted-summary.npz`
- `data/nma-simulation-replicates.npz`
- `diagnostics/nma-simulation-calibration.json`
- `diagnostics/nma-simulation-plot-policy.json`
- `figures/nma-simulation-coverage.pdf`
- `figures/nma-simulation-interval-width-theta-1.pdf`
- `figures/nma-simulation-interval-width-theta-2.pdf`

Paper executable: 100 replicates per rho, ten rho values, n=100, df=99, 28 comparisons, nine parameters. Manuscript Figure 10 describes 500; the included 100-repeat comparison took 37.6 minutes. Smoke: 2 replicates/rho, n=30, rho=0,.6. Empty sets count as noncoverage/zero width without study exclusion. Matching-grid plots default to fixed manual WLS-MA width benchmarks, not estimates from these replicates; other settings use empirical-oracle widths. HCCT joint-region and WLS interval-box coverage events differ. Main Figure 10 is fixed; reruns are comparison-only.

## network-meta-analysis/real-data

Real-data heatmaps and treatment-specific confidence regions.

Source: [reproduction/network_meta_analysis/real_data.py](../reproduction/network_meta_analysis/real_data.py). Deterministic; seed NA. Runtime guidance: under 5 minutes.

```bash
python -m reproduction.run network-meta-analysis/real-data \
  --profile paper --run-id my-real-data \
  --results-root results/runs
```

Declared outputs, relative to the run directory:

- `data/nma-real-data-hcct-interval-widths.npy`
- `diagnostics/nma-real-data-empty-set-adjustment.json`
- `diagnostics/nma-real-data-support-certificates.json`
- `figures/nma-real-data-hcct-interval-width-heatmap.pdf`
- `figures/nma-real-data-wls-interval-width-heatmap.pdf`
- `figures/nma-real-data-bonferroni-wls-heatmap.pdf`
- `figures/nma-real-data-hcct-minus-bonferroni-wls-heatmap.pdf`
- `figures/nma-real-data-width-vs-comparisons-benf.pdf`
- `figures/nma-real-data-width-vs-comparisons-metf.pdf`
- `figures/nma-real-data-width-vs-comparisons-sita.pdf`
- `figures/nma-real-data-width-vs-comparisons-rosi.pdf`

Uses the bundled Senn2013 CSV and Python multi-arm WLS adjustment; no R runtime is needed. Explicit empty-set treatment excludes comparisons 0 (DeFronzo1995: metf-plac) and 6 (Kerenyi2004: rosi-plac). It saves the complete intervention trace and 90 endpoint certificates. The certified HCCT width matrix differs from the manuscript heatmap by up to 0.367830; see [the NMA guide](network_meta_analysis/README.md). Typical included runtime is about a minute.
