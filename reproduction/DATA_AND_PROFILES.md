# Experiment profiles, seeds, and uncertainty

A seed is only one part of a reproducible experiment. Keep the profile, RNG API,
draw order, dependency versions, source hashes, and saved configuration as well.
Seeds are fixed in the entry points; the runner records them without a seed
override. Runtime examples describe the included execution environment, not
portable performance guarantees.

## Workload sizes

| Experiment | Paper profile | Smoke profile | Seed |
|---|---|---|---:|
| Normal coverage | 50 batches × 10,000 draws; 500 studies; 10 correlations; two families | 2 × 200 draws; 30 studies; rho=0,0.6 | 2025 |
| False-positive rate | 10,000 shared draws; 500 studies; 9 methods; 10 correlations; two families | 200 draws; 30 studies; rho=0,0.3,0.6,0.9 | 2025 |
| Interval width | 10,000 intervals; 500 studies; 4 correlations × two families; 80,000 inversions | 40 intervals; 30 studies; same correlations/families | 2025 |
| Power | 10,000 shared draws; 500 studies; 9 methods; 10 correlations; dense/sparse signals | 200 draws; 30 studies; 4 correlations | 2025 |
| Copula coverage | 50 × 10,000 draws; 500 studies; t and bivariate AMH/FGM cases | 2 × 200 draws; 20 studies; reduced grids | 2025 |
| Multivariate coverage | 10 × 10,000 draws/cell; studies=500/10; dimensions=2/5/10/25; 10 correlations | 1 × 100 draws; studies=30/10; dimensions=2/5; rho=0,0.6 | 2025 |
| DAC normal slices | Dimension 100; 1,000 observations; block sizes 1/5/25/100 | Dimension 10; 50 observations; blocks 1/5/10 | 2025 |
| DAC lognormal slices (estimate/truth) | Same dimension/sample/block grids | Same smoke grids | 2024 |
| DAC diagnostics | 2,000 coverage replicates; direction-width blocks 2 through 25 | 5 replicates; width blocks 2 through 5 | 2025 normal; 2024 lognormal |
| NMA simulation | 100 replicates × 10 correlations; 100 observations; 28 comparisons; 9 parameters | 2 × 2 correlations; 30 observations | 2025 |
| NMA treatment estimates | One n=100 sample at each of four correlations; complete real-data table | Same complete tables | 2025 |
| Bounded threshold simulation | 1,000,000 draws per scenario, chunked | 20,000 draws per scenario | 2025 |
| Support benchmark | (total, active component) dimensions (50,50), (170,170), (260,260), (2000,50) | (20,20), (260,260), (400,20) | 20260901 |

Deterministic illustrations and quadrature-based table calculations have no
random seed. Contour figures use seed 2024. The full DAC diagnostic outputs are
not included in this release; the separate slice outputs are included.

## Grids and random streams

The ten-value correlation grid is generally `0,0.1,...,0.9`; interval widths
and treatment tables use `0,0.3,0.6,0.9`. Normal/t/copula and multivariate
coverage assess levels 0.05 and 0.01; FPR, power, scalar widths, and NMA use 0.05.
AMH/FGM theta values are `-0.9,-0.7,...,0.9`.

Normal coverage, FPR, interval width, power, multivariate coverage, and NMA
simulation use a seeded continuous NumPy RandomState stream; loop order
therefore matters. FPR/power share one sample across methods within each
rho/case, not across different cells. Copula functions initialize private
`default_rng(2025)` streams; HCCT/Fisher comparisons for the same bivariate
copula share latent samples. Treatment Table 1 uses a separate PCG64 generator.
No equality across different NumPy versions is guaranteed.

The standard multivariate-t copula has df=10 and a **common chi-square radial
variable across coordinates** in each observation. AMH/FGM observations contain
independent bivariate dependent pairs; they are not arbitrary-dimensional
exchangeable copulas.

## Differences in simulation effort

The executable NMA `paper` profile and included comparison use **100**
replications per rho, whereas the manuscript figure description uses **500**.
The 100-replicate workload took 37.6 minutes in its recorded environment.
Near 95% coverage, binomial Monte Carlo SE is about 0.022 rather than 0.010.
Neither solver acceptance tolerances nor statistical calibration change.
Fewer replicates at earlier rho values shift the random stream at later values;
this is not a per-rho prefix of a 500-replicate execution.

Multivariate coverage saves the mean of 10 batches of 10,000 outcomes:
**100,000 per cell**, compared with the manuscript's stated 1,000.
The included FPR/power design uses **10,000** outcomes per case. The reference
FPR curves have finer rate granularity consistent with a larger sample, but
their exact effective workload is not established; visual smoothness is not
evidence that the new calibration is inaccurate.

## Data conventions

See [SAVED_DATA.md](SAVED_DATA.md) for complete axes and
[NPZ_DATA.md](NPZ_DATA.md) for the per-file catalog.

| Data | Content |
|---|---|
| Normal/copula NPY | Coverage proportions by batch, dependence parameter, level |
| FPR/power NPY | Rejection rates by method, correlation, case |
| FPR/power paired NPZ | Boolean method decisions and explicit axes/settings |
| Interval-width NPY | Full interval lengths by family, correlation, replicate |
| Multivariate-coverage NPY | Batch-averaged coverage by dimension, correlation, level |
| DAC diagnostic NPZ | Block-size grids, direction widths, coverage |
| NMA summary NPZ | Coverages, mean full widths, oracle multipliers, counts, rho |
| NMA replicate NPZ | Saved means/variances, estimates, indicators, minima/endpoints and diagnostics |
| Replot-cache NPZ | Versioned named plotting values, axis labels, source profile/seed |
| Real-data HCCT NPY | Symmetric pairwise width matrix; diagonal compares each treatment with placebo |

FPR/power method order is HCauchy, EHMP, HMP, Cauchy, Levy, Fisher, Stouffer,
Bonferroni, Simes. All nine are saved even where a figure omits HMP.
Widths are **upper minus lower**, not half-widths or standard errors.
The NMA width-matrix diagonal is not zero by definition.

Precision CSVs retain finite-law values, error estimates, Landau approximations,
signed differences, and machine-dependent elapsed milliseconds. Error estimates
are not rigorous bounds. Solver benchmarks retain active dimension, solver,
elapsed time, iterations/evaluations, boundary/KKT residuals, and success.

## Empty sets, diagnostics, and uncertainty

Coverage calculations retain all outcomes in their denominators. Empty sets
count as noncoverage; certified NMA/scalar empty sets have zero width rather
than being treated as valid singleton intervals. Genuine numerical failures
are distinct and stop an experiment with incomplete status.

The separate NMA real-data experiment enables study exclusions explicitly;
its full trace is saved. This is not applied to simulation coverage or FPR/power.
Consult the [NMA guide](network_meta_analysis/README.md) and the package
[empty-set documentation](../docs/EMPTY_SETS.md).

HCCT NMA coverage measures the joint score region, whereas WLS covers a
nine-coordinate interval box. WLS-MA is calibrated and evaluated in sample.
Its coverage is not held-out validation of either its own widths or the fixed
manual benchmark widths used by matching NMA plots.

Normal/copula plot bands are empirical quantiles of **batch proportions**,
not confidence intervals for the pooled mean. Multivariate arrays average
batches before saving. FPR/power paired indicators support paired Monte Carlo
standard errors. Not every workflow retains individual observations or enough
information for every alternative uncertainty summary.
