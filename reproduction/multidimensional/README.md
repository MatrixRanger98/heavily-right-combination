# Multidimensional experiments

Use the [common runner](../README.md); the [catalog](../EXPERIMENTS.md) gives
every command and declared output.

| Experiment | Module | Output |
|---|---|---|
| `multidimensional/coverage` | [coverage.py](coverage.py) | `multivariate-coverage-{500,10}-studies.npy` and matching PDFs |
| `multidimensional/contours` | [contours.py](contours.py) | Eight `multivariate-contours-dimension-{2,10}-rho-{0,0.3,0.6,0.9}.pdf` |
| `multidimensional/dac-normal` | [dac_normal.py](dac_normal.py) | Eight `dac-normal-slice-block-<block>-coordinates-<pair>.pdf` |
| `multidimensional/dac-lognormal-estimate` | [dac_lognormal_estimate.py](dac_lognormal_estimate.py) | Eight `dac-lognormal-estimate-slice-...pdf` |
| `multidimensional/dac-lognormal-truth` | [dac_lognormal_truth.py](dac_lognormal_truth.py) | Eight `dac-lognormal-truth-slice-...pdf` |
| `multidimensional/dac-normal-diagnostics` | [dac_normal_diagnostics.py](dac_normal_diagnostics.py) | `dac-normal-{direction-widths,coverage}.npz` |
| `multidimensional/dac-lognormal-estimate-diagnostics` | [dac_lognormal_estimate_diagnostics.py](dac_lognormal_estimate_diagnostics.py) | `dac-lognormal-estimate-{direction-widths,coverage}.npz` |
| `multidimensional/dac-lognormal-truth-diagnostics` | [dac_lognormal_truth_diagnostics.py](dac_lognormal_truth_diagnostics.py) | `dac-lognormal-truth-{direction-widths,coverage}.npz` |

Paper-profile DAC slices use dimension 100, 1,000 observations, rho=0.6,
block sizes 1/5/25/100, and coordinate pairs 1–51 or 1–2. Seeds are 2025
(normal) and 2024 (lognormal). The shared [dac.py](dac.py) module handles
sampling, block estimation, slices, direction widths, and coverage.

Slice plots and repeated coverage diagnostics are distinct experiments.
The full diagnostics use 2,000 coverage replications and may run much longer;
these full diagnostic outputs are not included in the release. Small smoke
profiles test their complete data-generation paths.

Multivariate coverage summarizes 10 batches of 10,000 outcomes per cell:
**100,000 outcomes**, whereas the manuscript states 1,000. Saved arrays contain
batch-averaged coverage, not individual indicators or batch proportions. The
[data guide](../SAVED_DATA.md) records axes and interpretive limits.

All normal/lognormal block paths estimate covariance of the mean with
`W/[n(n-1)]`; scalar standard errors use sample standard deviation
(`ddof=1`) divided by `sqrt(n)`.
