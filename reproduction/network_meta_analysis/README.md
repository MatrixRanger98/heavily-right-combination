# Network meta-analysis reproduction

These experiments use the [general package interface](../../docs/API.md)
with a fixed Senn2013 study design. They are separate from the package's
general-purpose NMA frontend.

## Data and table estimates

[data/senn2013.csv](data/senn2013.csv) contains all 28 comparison-based
contrasts from `netmeta` 3.6-1; see [data provenance](data/README.md).
The Python loader builds the design matrix and applies the multi-arm variance
adjustment for the three Willms1999 contrasts before common-effect WLS fitting.
Neither R nor PyArrow is needed. Regression tests compare its WLS width matrix
with the portable [reference fixture](../../tests/fixtures/nma_wls_widths.json).

```bash
python -m reproduction.run network-meta-analysis/treatment-estimates \
  --profile paper --run-id my-treatment-tables --results-root results/runs
```

Main Table 1 uses private PCG64 seed **2025**, one n=100 sample at each of
rho=0,0.3,0.6,0.9. It is a table of single-sample estimates, not Monte Carlo
averages; all four HCCT minima are certified and no study is excluded. The
run saves 72 full-precision estimate rows, all four raw samples and estimated
variances, and minimum diagnostics. Both profiles produce the complete tables.

Main Table 2 is deterministic. WLS uses all 28 comparisons; HCCT uses 26
after the two explicit exclusions described below. The output has 18 estimates
plus the full intervention trace. Use the primary estimate columns for the
adopted table values; fixed-reference comparison fields, if present, are not
independent verification of the current manuscript.

## Synthetic coverage and interval widths

```bash
python -m reproduction.run network-meta-analysis/simulation \
  --profile paper --run-id my-nma-simulation --results-root results/runs
```

The executable `paper` profile uses **100 replicates at each of ten
correlations**, n=100 observations, 28 comparisons, nine parameters, df=99,
and seed 2025. The manuscript describes **500** replications; the included
100-replication counterpart is a reduced-workload comparison. Near coverage
0.95, binomial Monte Carlo SE is about **0.022**, versus **0.010** for 500.
The completed workload took about **37.6 minutes**, but optimization difficulty
and hardware affect elapsed time. `smoke` uses 2 replicates, n=30, and two
correlations. Changing an earlier repeat count shifts the continuous random
stream used at later correlations.

Data and diagnostic outputs include:

- `data/nma-simulation-summary.npz`: raw coverage, mean widths, oracle
  multipliers, empty-region counts, replicate counts, and rho grid.
- `data/nma-simulation-replicates.npz`: study means/estimated mean variances,
  WLS estimates, indicators, HCCT minima/endpoints, and numerical diagnostics.
- `data/nma-simulation-plotted-summary.npz`: actual displayed series,
  distinguished from raw summaries.
- `diagnostics/nma-simulation-calibration.json`: configuration, event
  definitions, empirical-oracle qualification, and empty-set conventions.
- `diagnostics/nma-simulation-plot-policy.json`: width policy and fallback.
- `failed-coverage-rho-<rho>-replicate-<r>.npz`: input snapshots for
  statistical noncoverage; these filenames do not indicate solver failure.

The three figures are `nma-simulation-coverage.pdf` and
`nma-simulation-interval-width-theta-{1,2}.pdf`. The detailed
[array schema](../SAVED_DATA.md#7-network-simulation-summary-replicates-and-diagnostics)
defines their units and retained evidence.

### Coverage events and empty sets

HCCT coverage is truth membership in the joint score region. WLS coverage
requires all nine coordinate intervals to cover simultaneously. These are
different geometric events. WLS-MA is an **in-sample empirical oracle**:
its multiplier is calibrated and evaluated using the same replicates and
known truth, not held-out data.

Certified empty HCCT regions contribute **noncoverage and zero width**, remain
in the denominator, and are saved with their inputs/certificates. No study is
excluded and no replicate is resampled in this simulation. An uncertified
minimum or endpoint stops the run and saves partial arrays separately; it must
not be classified as an empty region. The included 1,000-outcome run has zero
empty regions and 51 statistical noncoverage observations.

### Raw widths versus manual comparison benchmark

For the matching n=100 paper-profile rho grid, the default plotting policy
`historical-manual` selects the fixed WLS-MA width benchmarks in
[manual_benchmark.py](manual_benchmark.py). Those widths are **not estimated
from the included 100-replication sample**, and their exact original
calibration/sample provenance is not independently established.

Other profiles or nonmatching grids fall back to saved empirical-oracle widths.
Python callers may explicitly choose
`render(..., profile="paper", width_policy="empirical")`.
The policy record states the selected method and fallback reason.

The included `nma-100-manual-width-replot-20260921` preserves all 13 raw
summary members in `replot-cache.npz`. Only the two WLS-MA width series change
in the separately saved plotted summary. Its multiplier is named
`empirical_wls_ma_multiplier`, not attributed to the manual widths.
Coverage and HCCT/WLS widths remain unchanged. Empirical-oracle coverage
therefore **does not validate the manual widths** in the adjacent panels.

**Main Figure 10 is fixed to its three manuscript PDFs.** These simulation and
replot outputs are comparison-only. Main Figure 14 has the same fixed-selection
policy; consult the [results index](../../results/README.md).

## Real-data figures and opt-in study exclusions

```bash
python -m reproduction.run network-meta-analysis/real-data \
  --profile paper --run-id my-real-network --results-root results/runs
```

The program writes `nma-real-data-hcct-interval-widths.npy`, four heatmaps
(HCCT, WLS, Bonferroni-WLS, HCCT-minus-Bonferroni-WLS), and four
`nma-real-data-width-vs-comparisons-{benf,metf,sita,rosi}.pdf` curves.

The unadjusted HCCT network region is empty. This analysis explicitly enables
the package's [empty-set policy](../../docs/EMPTY_SETS.md), which is off by
default. Original zero-based comparisons **0 (DeFronzo1995: metf-plac)** and
**6 (Kerenyi2004: rosi-plac)** are excluded. This changes the retained evidence
and is not a claim of unchanged unconditional inference.

`nma-real-data-empty-set-adjustment.json` records stage fits, candidate
rankings, eligibility, labels/original indices, individual p-values, score
minima/cutoffs, weights, exclusions, and final status. The record is saved even
when an intervention cannot resolve a region. The separate
`nma-real-data-support-certificates.json` records all direction vectors,
endpoint values, boundary/KKT residuals, iterations/evaluations, certificate
types, and final status; partial calculations cannot appear complete.

The included run certifies 45 directions/90 endpoints and takes about a minute.
Its maximum normalized KKT residual is about 1.85e-6, within the 2e-6 limit.

### Difference from the manuscript heatmap

The certified HCCT width matrix differs from the manuscript heatmap:
maximum absolute width difference **0.367830**; mean difference **0.1100704**
over unique comparisons (**0.1220613** over all 81 cells). For acar–benf,
the manuscript width is about **1.033191**, compared with **1.401021**
from the certified computation (35.6% larger).

The exclusions match and point estimates differ by at most about 0.003485.
This real-data analysis uses reported standard errors; the sample-covariance
denominator convention does not explain the width difference. Differences in
numerical optimization are a plausible explanation, but the exact cause is
not established. A valid new certificate is not evidence that the manuscript
figure has identical values; inspect this discrepancy before adopting any
replacement figure.
