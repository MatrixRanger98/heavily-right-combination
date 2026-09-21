# Using heavily-right

Install the core from the repository root with `python -m pip install .`.
Use `python -m pip install '.[reproduction]'` for experiments or
`python -m pip install -e '.[reproduction,dev]'` for development. The
package requires Python 3.11+; dependency requirements are in `pyproject.toml`.
The examples below run after installation without setting `PYTHONPATH`.
For self-contained help beside each main API, see the
[reference examples](docs/reference/README.md). The
[troubleshooting guide](docs/TROUBLESHOOTING.md) and
[mathematical-methods guide](docs/METHODS.md) explain failures and assumptions.
For algorithm-level details, read the [finite-law implementation](docs/NULL_LAW_IMPLEMENTATION.md)
and [optimization implementation](docs/OPTIMIZATION_IMPLEMENTATION.md) guides.

## Combine p-values

```python
import numpy as np
from heavily_right.combination import CombinationTest

test = CombinationTest([0.2, 0.3, 0.5], method="HCauchy", level=0.05)
p = np.array([[0.01, 0.20, 0.40], [0.20, 0.30, 0.40]])
print(test.get_global_p(p))   # shape (2,)
print(test.make_decision(p))  # True means reject the global null
```

Pass an integer study count instead of weights for equal weights. Study
p-values occupy the last axis; preceding axes index independent batches.
Explicit weights are positive and sum to one. `level=0.05` means a 5%
significance level, corresponding to a 95% confidence region. HCCT and EHMP
use finite-m independence calibration through 1,000 studies and a Landau
approximation above that. See the [API reference](docs/API.md) for other methods
and the [numerical guide](docs/NUMERICS.md) for the interpretation of exactness.

## Evaluate a null law

```python
from heavily_right.null_laws import HalfCauchyMean, NumericalCalibrationError

try:
    cutoff = HalfCauchyMean.ppf(0.95, [0.2, 0.3, 0.5])
    probability, error_estimate = HalfCauchyMean.sf(
        cutoff, [0.2, 0.3, 0.5], precision=True,
    )
    print(cutoff, probability, error_estimate)
except NumericalCalibrationError as error:
    print(f"Null-law evaluation failed: {error}")
```

`ppf` takes a CDF probability, so a 5% upper-tail cutoff uses `0.95`.
`precision=True` returns a numerical error estimate for `pdf`, `cdf`, or `sf`;
it is not a mathematical error bound.

## Calibrate sample means

```python
import numpy as np
from heavily_right.calibration import covariance_of_mean, standard_error_of_mean

sample = np.array([[0.8, 1.1], [1.0, 0.9], [1.2, 1.3], [0.9, 1.0]])
estimate = sample.mean(axis=0)
standard_errors = standard_error_of_mean(sample)
mean_covariance = covariance_of_mean(sample)
```

The helpers use sample standard deviation with `ddof=1`, divided by `sqrt(n)`,
and centered scatter `W/[n(n-1)]`, respectively. Supply covariance of the mean,
not covariance of individual observations, to meta-analysis. Hotelling
calibration uses `df=n-1` with this covariance. These helpers preserve the
corrected finite-sample behavior.

## A scalar confidence interval

```python
from heavily_right.meta_analysis import MetaAnalysis1D

analysis = MetaAnalysis1D(3, method="HCauchy", level=0.05)
result = analysis.confidence_interval_result(
    theta_hat=[0.10, 0.14, 0.08],
    sigma=[0.04, 0.05, 0.03],
)
print(result.lower, result.upper, result.estimate)
print(result.empty_set_diagnostics.encountered)
```

`sigma` contains study standard errors. `df=None` uses two-sided normal
p-values; positive `df` uses Student t. For the compatibility tuple API use
`confidence_interval(...)`, which returns `(lower, upper, estimate)`.
Fractional df values are retained; multi-study HCCT/EHMP interval inference
requires `df>=1`, while a single-study t interval accepts any positive df.

## Project a vector confidence region

```python
import numpy as np
from heavily_right.meta_analysis import MetaAnalysisMD

analysis = MetaAnalysisMD(1, dim=2, method="HCauchy", level=0.05)
result = analysis.support_interval(
    direction=[1.0, 0.0],
    xi_hat=[np.array([0.2, -0.1])],
    Sigma=[np.array([[0.04, 0.01], [0.01, 0.09]])],
    sub_dim=2,
    df=[29],
)
if not result.success:
    raise RuntimeError(result.upper_diagnostics.message)
print(result.lower, result.upper)
print(result.upper_diagnostics.boundary_error)
print(result.upper_diagnostics.kkt_residual)
```

The bounds refer to `direction @ theta`; the full endpoint vectors are in
`lower_point` and `upper_point`. For partial or overlapping observations,
provide a projection matrix `projs` and explicit row-index arrays in `sub_dim`.
The [API reference](docs/API.md) gives an example and the precise shape contract.

Small active components use safeguarded Newton iterations to solve the KKT
equations. Automatically scaled SLSQP and quadratic penalty continuation are
fallbacks, with Newton polishing where appropriate; scalar-study cusps have
additional active-set/epigraph fallbacks and subgradient checks. The one-study
example above uses the analytic ellipsoid path. A reported endpoint must
pass boundary and stationarity checks; inspect `result.success` and retain the
diagnostics. See [Numerics](docs/NUMERICS.md).

## Enable automatic treatment of empty sets

Study removal is off by default. To enable it and retain a record:

```python
from pathlib import Path
from heavily_right.empty_sets import (
    EmptySetNumericalError, EmptySetPolicy, EmptySetResolutionError,
)
from heavily_right.meta_analysis import MetaAnalysis1D

analysis = MetaAnalysis1D(2, method="HCauchy", level=0.05)
try:
    result = analysis.confidence_interval_result(
        theta_hat=[-3.0, 3.0],
        sigma=[0.5, 0.5],
        study_labels=["study A", "study B"],
        empty_set_policy=EmptySetPolicy(max_removals=1, min_remaining=1),
    )
    diagnostics = result.empty_set_diagnostics
except (EmptySetResolutionError, EmptySetNumericalError) as error:
    diagnostics = error.diagnostics
    print(f"Intervention incomplete: {error}")

if diagnostics is not None:
    diagnostics.save_json(Path("empty-set-audit.json"))
    print(diagnostics.message)
```

The JSON writer refuses to overwrite a file. Use a new destination for every
analysis. The same policy and study labels work with `support_interval`.
Records retain fitted stages, study p-values and weights, candidate rankings
and exclusion reasons, selected removals, original indices, and final status.
Warnings identify each removal. Numerical optimization failure does not justify
study removal. See the [empty-set guide](docs/EMPTY_SETS.md) for failure handling
and the statistical interpretation of this intervention.

## Reproduce a figure

```bash
heavily-right-reproduce --describe one-dimensional/normal-coverage
heavily-right-reproduce one-dimensional/normal-coverage \
  --profile smoke --max-seconds 300 --run-id normal-coverage-check \
  --results-root ./reproduction-results
```

Use [the reproduction guide](reproduction/README.md) for the full catalog,
runtime policy, saved data, figure names, and unresolved paper comparisons.
Smoke runs validate the workflow; they are not paper estimates.
