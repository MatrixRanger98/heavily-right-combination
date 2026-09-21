# Reproducing figures and tables

The reproduction modules configure experiments; the reusable statistical
implementation is in [heavily_right](../heavily_right/). Run commands below
from the release repository root.

Paper: [A Heavily Right Strategy for Statistical Inference with Dependent Studies in Arbitrary Dimensions](https://arxiv.org/abs/2501.01065)
([PDF](https://arxiv.org/pdf/2501.01065)).

## Install and inspect

```bash
python -m pip install -e ".[reproduction]"
python -m reproduction.run --list
python -m reproduction.run network-meta-analysis/simulation --describe
```

Python 3.11 or newer is required. Numerical results depend on the recorded
NumPy/SciPy versions and numerical libraries as well as the seed. The
[package guide](../USAGE.md) covers the general NMA/DAC interfaces; the
[R package guide](../heavilyrightR/README.md) links its runnable examples and
mathematical-methods vignettes.

## Run an experiment

```bash
python -m reproduction.run illustrations/connectivity-scores \
  --profile smoke --max-seconds 300 --run-id first-check \
  --results-root results/runs
python -m reproduction.run illustrations/connectivity-scores \
  --profile paper --run-id my-full-run --results-root results/runs
```

Use a fresh lowercase run ID. Output directories and existing artifacts are
never overwritten. A run is grouped as
`results/runs/<experiment-id>/<run-id>/`, with `run.json`, `data/`,
`figures/`, `diagnostics/`, and runtime logs as applicable. Figures are
vector PDFs; filenames are declared by their generating experiment.

`smoke` is a small end-to-end check, not paper-level Monte Carlo precision.
`paper` selects the documented larger workload. The optional
`--max-seconds` is a wall-time cap; omit it when allowing a full simulation
to finish. Large DAC diagnostics and confidence-region simulations can be
expensive. Inspect the [catalog](EXPERIMENTS.md) and
[workloads](DATA_AND_PROFILES.md) before running them.

The runner records the profile, fixed seed, package versions, generator hashes,
configuration where supplied, output inventory, and final status. It records
and propagates numerical failures: a partly produced figure set is not a
successful experiment.

## Find the result or reuse saved data

- [Results index](../results/README.md): paper-order mapping of every included
  figure and table to its saved outputs and generating code; additional
  non-paper outputs are marked Unused/NA.
- [Experiment catalog](EXPERIMENTS.md): commands and all declared output names.
- [Saved-data guide](SAVED_DATA.md): array schemas, units, seeds, and plot-only use.
- [NPZ catalog](NPZ_DATA.md): per-file descriptions and checksums.
- [Paper artifact guide](PAPER_ARTIFACTS.md): how to interpret associations.
- [GeoGebra sources and export](../geogebra/README.md) and
  [Wolfram cross-check](../wolfram/README.md): external illustration workflows.

```bash
python -m reproduction.replot \
  results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921 \
  --run-id my-fpr-replot --results-root results/runs
```

Replotting draws no new random samples. It writes a self-describing NPZ cache
before rendering, plus new PDFs and provenance in a new run. Keep the source
data and its metadata together.

## Scientific choices to retain

Main Figures **10 and 14** use the fixed manuscript selections listed in the
results index. Included simulation/replot counterparts are **comparison-only**
and must not replace those selections without explicit author approval.

The NMA comparison run uses **100 replications per correlation**, seed 2025,
and 100 observations per replicate; the manuscript figure description uses
500 replications. Near 95% coverage, its Monte Carlo standard error is about
0.022 rather than 0.010. The reduced count saves time and is not an identical
reproduction of the manuscript effort. The completed run took about
**37.6 minutes** in its recorded environment.

FPR and power use **10,000 shared observations per setting across methods**.
An empty confidence set contributes noncoverage directly; no observation or
study is discarded for these comparisons. NMA simulation likewise retains
certified-empty outcomes as noncoverage and zero width. Its separate real-data
analysis explicitly enables documented study exclusions. Details, including
the NMA manual-width benchmark and differing coverage events, are in the
[network guide](network_meta_analysis/README.md).

Main Table 1 uses seed 2025, one n=100 sample per correlation, not a Monte
Carlo average. Main Table 2 uses deterministic real data. CAtr-related table
columns are outside the supplied numerical table reproduction scope.
The three full-profile DAC diagnostic experiments are available but their
2,000-replication outputs are not included; slice plots are different experiments.

See [third-party provenance](../THIRD_PARTY.md) for the data and numerical
coefficient attribution, and [numerical implementation](../docs/NUMERICS.md)
for calibration and solver safeguards.
