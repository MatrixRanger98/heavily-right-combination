# One-dimensional simulations

Use the [common runner](../README.md). The [catalog](../EXPERIMENTS.md) lists
commands, while the [saved-data guide](../SAVED_DATA.md) documents all axes.

| Experiment | Module | Data | Figures |
|---|---|---|---|
| `one-dimensional/normal-coverage` | [normal_coverage.py](normal_coverage.py) | `normal-coverage-{ar1,equicorrelated}.npy` | Matching PDFs |
| `one-dimensional/false-positive-rate` | [false_positive_rate.py](false_positive_rate.py) | `false-positive-rate.npy`, `false-positive-rate-paired.npz` | `false-positive-rate-{ar1,equicorrelated}.pdf` |
| `one-dimensional/interval-width` | [interval_width.py](interval_width.py) | `interval-width.npy` | `interval-width-{ar1,equicorrelated}.pdf` |
| `one-dimensional/power` | [power_comparison.py](power_comparison.py) | `power-comparison.npy`, `power-comparison-paired.npz` | `power-comparison-{dense,sparse}.pdf` |
| `one-dimensional/copulas` | [copula_coverage.py](copula_coverage.py) | Six `copula-coverage-<case>.npy` files | Matching PDFs |

The six copula cases are `student-t-ar1`, `student-t-equicorrelated`,
`ali-mikhail-haq-{hcauchy,fisher}`, and
`farlie-gumbel-morgenstern-{hcauchy,fisher}`.

## Seeds, paired comparisons, and coverage

All five entry points use seed **2025**. False-positive-rate and power profiles
use 500 studies and **10,000 shared draws per rho/case**, with nine methods and
ten correlations. All methods evaluate the same p-value matrix within a cell;
different cells use different samples from a continuous
RandomState/MT19937 stream in rho-major, case-minor order.

No study or outcome is discarded. Empty inverted confidence sets contribute
noncoverage directly; there is no empty-set adjustment in these comparisons.
Coverage is evaluated by nonrejection at the true parameter. Power uses
nominal-.05 rejection, not equal-actual-size calibration under dependence.

The paired NPZ files retain Boolean rejection indicators
`(method,rho,case,draw) = (9,10,2,10000)`, coordinates, thresholds, sample
sizes, seed/RNG metadata, and (for power) the two mean vectors. The NPY rate
arrays equal indicator means exactly. Raw normal observations and p-values
are not retained. For a method contrast use signed indicator differences and
paired Monte Carlo SE `D.std(ddof=1)/sqrt(N)`; independent-binomial SEs ignore
this pairing. Zero observed discordances do not prove identical methods.

The included HCCT–EHMP maximum rate differences are 0.0001/0.0005 for
AR1/equicorrelated FPR and 0.0023/0.0005 for dense/sparse power. Pairing reduces
comparison noise; 10,000 draws do not guarantee smooth individual curves.

## Plotting and paper selections

Main Figure 14 remains fixed to its manuscript PDFs. The included paired
results and `paired-visibility-replot-20260921` PDFs are comparison-only.
HCauchy is drawn thicker beneath dashed EHMP so near-coincident lines are both
visible without changing any values. The preferred interval-width rendering
uses `interval-width-layout-replot-20260921`; its two panels share the same
grid style and its cache retains all 80,000 saved widths.

```bash
python -m reproduction.run one-dimensional/false-positive-rate \
  --profile paper --run-id my-paired-fpr --results-root results/runs
python -m reproduction.run one-dimensional/power \
  --profile paper --run-id my-paired-power --results-root results/runs
python -m reproduction.replot \
  results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921 \
  --run-id my-paired-fpr-replot --results-root results/runs
```

The full paired simulations each took about 13 seconds in the recorded
environment; normal/copula coverage and interval inversions take longer.
Both `paper` and `smoke` configurations are described in
[DATA_AND_PROFILES.md](../DATA_AND_PROFILES.md).
