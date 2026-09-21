# Numerical tables and approximations

Use the [common runner](../README.md); the [catalog](../EXPERIMENTS.md) lists
commands and exact output filenames.

| Experiment | Module | Output |
|---|---|---|
| `numerical/distribution-precision` | [distribution_precision.py](distribution_precision.py) | Long `distribution-precision.csv`, wide `distribution-precision-{hcct,ehmp}.csv` |
| `numerical/support-solver-benchmark` | [support_solver_benchmark.py](support_solver_benchmark.py) | `support-solver-benchmark.csv` |
| `numerical/pvalue-examples` | [pvalue_examples.py](pvalue_examples.py) | `pvalue-examples.csv` |
| `numerical/independence-thresholds` | [independence_thresholds.py](independence_thresholds.py) | `independence-thresholds.csv` and numerical diagnostic JSON |
| `numerical/legacy-thresholds` | [legacy_thresholds.py](legacy_thresholds.py) | `legacy-thresholds.csv` |
| `numerical/landau-approximation` | [landau_approximation.py](landau_approximation.py) | Four density PDFs at m=1/10/100/1000 and a CDF PDF at m=1000 |

The deterministic independence-threshold generator supplies the 20 non-CAtr
cells of Main Table 3, including probability round-trip checks. The p-value
example generator supplies 36 non-CAtr cells of Supplement Table 4, including
Bonferroni. CAtr/Fang columns are intentionally outside the numerical table
reproduction scope. The separately available `legacy-thresholds` command
performs a bounded, chunked Monte Carlo calculation (one million draws per
paper-profile scenario, seed 2025); it is not the generator of Main Table 3.

The precision workflow exports 60 PDF/CDF measurements: 16 HCCT point rows and
14 EHMP point rows. Both error estimates and measured runtimes are
environment-dependent; numerical error estimates are not rigorous error bounds.
At EHMP m=1000,x=4, the included PDF/CDF values differ from the manuscript's
displayed digits by about 3.36e-9 and -6.66e-10, respectively, within its reported
errors. Signed Landau differences are computed as approximation minus finite-law
value. Full precision is retained in CSV rather than inferred from rounded cells.

The complete Landau illustration workflow took about 50 seconds in its
recorded environment. The reusable kernels implement guarded finite-law
inversions and support-boundary handling; see
[implementation methods](../../docs/NUMERICS.md), not experiment settings, for
their numerical design.

The support benchmark tests connected overlapping blocks with 50/170/260
active variables. Dense KKT/Newton is capped at 250 active variables; the
260-variable case exercises constrained fallback. A fourth scenario has 2,000
total variables but a direction active in only one 50-variable component.
That benchmark does not establish dense-Newton scalability for a connected
2,000-variable problem.
