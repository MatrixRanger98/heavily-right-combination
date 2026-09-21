# Saved data, seeds, and plot-only reproduction

The seven simulation figure workflows save numerical inputs before rendering.
Replotting does not draw new observations, refit confidence regions, or
recalibrate test thresholds. A PDF alone is not the numerical experiment: retain
its source data, `run.json`, and diagnostic metadata together.

The [NPZ catalog](NPZ_DATA.md) documents each distributed archive and checksum.
[Profile settings](DATA_AND_PROFILES.md) specify fixed seeds, workload sizes,
RNG semantics, and uncertainty qualifications.

## Plot again without simulating

From the release repository root:

```bash
python -m reproduction.replot \
  results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921 \
  --run-id my-fpr-replot --results-root results/runs
```

The input is a run directory containing `run.json`, not its `data/`
directory or a single array. The installed equivalent is
`heavily-right-replot`. Specify a new run ID; existing runs cannot be
overwritten. The source's experiment/profile/seed are preserved, with no
override to upgrade a smoke result into a paper-profile result.

Default output is `results/runs` in a source checkout or
`heavily-right-results/runs` in the current directory for an installed copy.
An explicit `--results-root` makes the destination unambiguous.

Completed, structurally valid source runs are accepted by default.
`--allow-failed-source` is an explicit recovery option only when **all**
required plotting inputs are complete and valid; it cannot resume missing
replicates, accept an active run, invent an omitted series, or restore a random
stream. Configuration, seed, rho grids, values, and array shapes are checked.
NMA additionally checks its stored n/df/level/study-count/dimension settings and
requires a supported completed replicate count: 100 or 500 for paper, 2 for smoke.

A replot creates:

- `data/replot-cache.npz`: versioned named plotting values and coordinates,
  written before plotting;
- `data/replot-cache-schema.json`: shapes, ordered axes, units and provenance;
- vector PDFs under `figures/`;
- `run.json`: source/input hashes, renderer hashes, mode and completion status.

NMA also saves `nma-simulation-plotted-summary.npz` and a plot-policy JSON,
because displayed manual-benchmark widths differ from raw empirical widths.
A complete replot cache can itself be used as another replot's source.

## Cache fields and provenance

Each cache has scalar `schema_version`, `experiment_id`, `profile`,
and `seed`; coordinates have an `axis__` prefix.

| Experiment | Numerical fields | Coordinate names (with `axis__` prefix) |
|---|---|---|
| False-positive rate | `rates` | methods, rhos, series |
| Power | `power` | methods, rhos, series |
| Interval widths | `widths` | dependence, rhos, replicates |
| Normal coverage | ar1, equicorrelated | replicates, rhos, levels |
| Copulas | Six case names below | replicates, rhos, levels, parameters |
| Multivariate coverage | studies_500, studies_10 (smoke: studies_30, studies_10) | dimensions, rhos, levels, study_counts |
| NMA | Twelve numerical summary members below | rhos |

For coverage caches, `replicates` indexes batches; for interval-width caches
it indexes individual intervals. Their meanings are not interchangeable.
NPY axes are reconstructed from the documented schema where no coordinates
were embedded. `configuration_interpretation` is therefore a documented
interpretation, not proof that all original simulation settings were embedded
in an array. Source-recorded configuration, when available, stays separate.

Keep `script_sha256`, `source_sha256`, `bundled_input_sha256`, package
versions, profile, seed and status. Replots separately record
`source_files_sha256`, `source_generator_sha256`, renderer
`source_sha256`, and `cache_sha256`. A source-code hash records the
generating revision; it need not equal a later renderer revision.
Shape and checksum consistency are not an independent mathematical proof.

Read arrays with `np.load(path, allow_pickle=False)`. The included archives
do not require pickle.

## Dataset-to-figure schemas

All paths below are relative to a run. Shapes describe the larger `paper`
profile; smoke outputs have smaller grids. Rates lie in [0,1], not percentages.
All seven figure workflows use seed **2025**; their RNG APIs and draw order
are documented in [DATA_AND_PROFILES.md](DATA_AND_PROFILES.md).

### 1. Normal coverage

`data/normal-coverage-{ar1,equicorrelated}.npy` each have shape
**(50,10,2)**, ordered `(batch,rho,level)`, levels [.05,.01].
Each cell is the nonrejection proportion among 10,000 outcomes. The matching
PDFs use batch means and 2.5%/97.5% batch quantiles, **not confidence intervals
for the pooled 500,000-outcome mean**. Individual draws and decisions are not
retained.

### 2. False-positive rates

`data/false-positive-rate.npy` has shape **(9,10,2)**, ordered
`(method,rho,dependence)`, with dependence [AR1,equicorrelation].
It generates `false-positive-rate-{ar1,equicorrelated}.pdf`.

Method order is HCauchy, EHMP, HMP, Cauchy, Levy, Fisher, Stouffer, Bonferroni,
Simes. Each rate is nominal-.05 rejection among 10,000 shared outcomes.
All methods are saved; the figure omits HMP. The paired archive described
below stores individual decisions, not raw p-values or normal observations.

### 3. Scalar interval widths

`data/interval-width.npy` has shape **(2,4,10000)**, ordered
`(dependence,rho,replicate)`; dependence is [AR1,equicorrelation],
rho is [0,.3,.6,.9]. It generates the two matching
`interval-width-{ar1,equicorrelated}.pdf` files.

All 80,000 full interval lengths are retained, in scalar parameter units.
Endpoints, study estimates and individual solver diagnostics are not stored.
A confirmed empty region contributes zero width; this is not a singleton
interval. KDE plots smooth and crop the distribution, so inspect saved zeros
rather than inferring their absence visually. The included
`interval-width-layout-replot-20260921` cache preserves these values.

### 4. Power

`data/power-comparison.npy` has shape **(9,10,2)**, ordered
`(method,rho,alternative)`, with alternatives [dense,sparse] and the
same method order as FPR. It generates `power-comparison-{dense,sparse}.pdf`.

Dense means s=0,r=.1; sparse means s=r=.3, with `floor(m**(1-s))` signals
of size `sqrt(2*r*log(m))`. Values are rejection probabilities at nominal
level .05, **not equal-actual-size power under dependence**.

### Paired FPR/power archives and method comparisons

The corresponding `*-paired.npz` archives have:

| Member | Meaning / shape |
|---|---|
| `rejections` | Boolean (9,10,2,10000), ordered (method,rho,case,draw) |
| `axes`, `methods`, `rhos`, `cases` | Axis labels and coordinates |
| `thresholds` | Nine method-specific rejection thresholds |
| `study_count`, `simulations_per_case`, `level`, `seed` | 500, 10000, .05, 2025 |
| `rng`, `sampling_scheme` | RandomState/MT19937; rho-major/case-minor; shared within cell |
| `alternative_means` | Power only: (2,500), dense/sparse means |

Smoke shapes are (9,4,2,200), with 30 studies.
The NPY summaries equal `rejections.mean(axis=-1)`. An empty inverted
confidence set contributes noncoverage directly; no adjustment or study
exclusion is applied. Different rho/case cells use distinct batches.

For methods A and B, paired differences use
`D = rejection_A - rejection_B` after casting Boolean values to a signed
integer type. Report `D.mean()` and `D.std(ddof=1)/sqrt(N)`.
Zero observed discordances give empirical SE zero but do not establish zero
population uncertainty or identical methods.

```python
from pathlib import Path
import numpy as np

run = Path("results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921")
with np.load(run / "data/false-positive-rate-paired.npz", allow_pickle=False) as saved:
    rejections = saved["rejections"]
    methods = saved["methods"].tolist()
    hc = rejections[methods.index("HCauchy")]
    eh = rejections[methods.index("EHMP")]
    difference = hc.astype(np.int8) - eh.astype(np.int8)
    paired_gap = difference.mean(axis=-1)
    paired_se = difference.std(axis=-1, ddof=1) / np.sqrt(difference.shape[-1])
    summary = np.load(run / "data/false-positive-rate.npy", allow_pickle=False)
    assert np.array_equal(summary, rejections.mean(axis=-1))
```

Replotting uses summary rates; keep the paired archives for uncertainty
calculations. Thicker HCauchy beneath dashed EHMP makes overlapping curves
visible without changing rates. All included Main Figure 14 reruns are
comparison-only, not replacements for its fixed manuscript PDFs.

### 5. Copula coverage

Six `data/copula-coverage-<case>.npy` files map to matching PDFs:

- `student-t-ar1`, `student-t-equicorrelated`;
- `ali-mikhail-haq-hcauchy`, `ali-mikhail-haq-fisher`;
- `farlie-gumbel-morgenstern-hcauchy`, `farlie-gumbel-morgenstern-fisher`.

Each has shape **(50,10,2)**, ordered `(batch,parameter,level)`.
Parameter is rho for t or theta for AMH/FGM; levels are [.05,.01].
Data are coverage/nonrejection; plots show the exact complement **1-coverage**.
The t copula has df=10 and one common chi-square radial draw per observation.
AMH/FGM use 250 independent dependent pairs per 500-study observation.
Raw uniforms, observations and p-values are not stored. Bands have the same
batch-quantile interpretation as normal coverage.

### 6. Multivariate coverage

`data/multivariate-coverage-{500,10}-studies.npy` each have shape
**(4,10,2)**, ordered `(dimension,rho,level)`, dimensions [2,5,10,25]
and levels [.05,.01]. They generate correspondingly named PDFs.

Ten batches of 10,000 outcomes are averaged before saving: **100,000 outcomes
per cell**, versus the manuscript's stated 1,000. Individual batches,
indicators and normal draws cannot be recovered. Smoke uses study counts 30/10,
dimensions [2,5], and arrays (2,2,2).

### 7. Network simulation: summary, replicates, and diagnostics

The included `nma-100-reproduction-20260921` has n=100, df=99,
28 comparisons, nine parameters, seed 2025, and **R=100** replicates per rho.
It took about 37.6 minutes. The manuscript describes 500 replications;
near 95% coverage, Monte Carlo SE is about .022 rather than .010.
Changing the repeat count affects subsequent random draws, not the statistical
calibration. Replot supports complete documented 100/500 counts without
relabelling either count.

`data/nma-simulation-summary.npz` contains 13 length-10 members:

- `rhos`;
- `coverage_hcct`, `coverage_wls`, `coverage_wls_ma`;
- `width_hcct_1`, `width_hcct_2`, `width_wls_1`, `width_wls_2`,
  `width_wls_ma_1`, `width_wls_ma_2`;
- `wls_ma_multiplier`, `empty_hcct_count`, `replicates_per_rho`.

Widths are unconditional mean **full** widths in treatment-effect units.
HCCT coverage means joint score-region membership; WLS coverage means all nine
coordinate intervals cover. WLS-MA is an **in-sample oracle** calibrated and
assessed on these same replicates using known truth, not held-out validation.

The default matching-grid plot uses fixed manual WLS-MA width benchmarks;
these are **not estimated from the included R=100 sample**. Other grids/profile
settings use empirical widths, or Python callers can request
`width_policy="empirical"`. The benchmark's exact calibration/sample
provenance is not independently established.

The manual-width replot preserves raw values in `replot-cache.npz`.
`nma-simulation-plotted-summary.npz` records actual displayed series:
only the two WLS-MA width series use the fixed benchmark. The raw multiplier
is explicitly named `empirical_wls_ma_multiplier`. The policy JSON reports
the selected method and fallback. Empirical-oracle coverage in the coverage
panel **does not validate the manual widths** in the width panels.
Main Figure 10 remains fixed to its manuscript PDF selections.

`data/nma-simulation-replicates.npz` contains:

| Members | Shape | Meaning / units |
|---|---|---|
| `rhos`, `true_theta`, `design` | (10,), (9,), (28,9) | Grid, true effects, study design |
| `xi_hat`, `estimated_mean_variances` | (10,R,28) | Means / corrected estimated variances of means |
| `wls_estimates`, `wls_standard_errors` | (10,R,9) | Effect estimates / standard errors |
| `wls_maximum_standardized_errors` | (10,R) | Maximum absolute standardized error over nine coordinates |
| `covered_wls`, `covered_hcct`, `covered_wls_ma`, `empty_hcct` | (10,R) | Numeric 0/1 indicators |
| `widths_hcct` | (10,R,2) | Full widths for theta1/theta2 |
| `hcct_minimum_scores` | (10,R,2) | Minimum score in each direction's solve |
| `hcct_minimum_points` | (10,R,2,9) | Minimum parameter vectors |
| `hcct_boundary_errors`, `hcct_kkt_residuals` | (10,R,2,2) | Direction × [lower,upper]; score-unit error / normalized stationarity |
| `hcct_support_points` | (10,R,2,2,9) | Direction × [lower,upper] × parameter |

Treatment order is acar, benf, metf, migl, piog, rosi, sita, sulf, vild, each
versus placebo. All raw n×28 observations are not retained; their means and
estimated variances are. These permit a diagnostic refit without new random
draws, but refitting is not part of replotting.

Calibration diagnostics state `W/[n(n-1)]`, seed, n/df, counts, multiplier,
event definitions and empty-set conventions. Certified empty outcomes remain
in the denominator with coverage=0, width=0, and `empty_hcct=1`; endpoint slots
may be NaN because no endpoint exists. Empty-region certificates and inputs
are saved explicitly. No study is excluded and no replicate is resampled.

The included run contains zero empty regions and **51 statistical noncoverage**
observations, whose `failed-coverage-*.npz` files retain point, study
estimates, variances and projection matrix. These files are not optimizer
failures. Genuine numerical failures instead produce separately labelled
partial arrays; do not treat unattempted NaNs as Boolean outcomes or average an
incomplete run.

Maximum included scaled boundary error is approximately 9.64e-10 and normalized
KKT residual 1.9972e-6, within 2e-8/2e-6 limits. These stored diagnostics allow
score/summary checks; successful-case multipliers, full active-set
subgradients, and full minimum certificates were not all saved, so independent
recertification of every numerical certificate is not possible from this
archive alone.

## Write timing and limits

Normal/copula/multivariate arrays are written after each complete
family/study-count section. FPR/power save the whole case grid and paired
indicators before figures. Interval width saves its whole width array after
all solves. NMA saves final replicates and calibration, then summary, before
rendering. These are not automatic checkpoints after every replicate.

An output's existence does not make an incomplete run complete. A saved
summary cannot recover omitted observations, restart an intermediate RNG state,
or continue a half-finished optimizer. Always use a new run ID for a new
execution and preserve the distinction between simulated inputs, plotted
inputs, and derived figures.
