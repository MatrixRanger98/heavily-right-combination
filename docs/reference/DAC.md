# Divide-and-combine reference

`heavily_right.dac` partitions a sample's **coordinates** into blocks and combines inference about one common mean vector. Every block uses all observations. It does not divide observations into independent datasets, fit random effects, or estimate between-study heterogeneity.

Each example below is independent, includes imports and data, uses only NumPy/SciPy and the package, and creates no files. Run examples after installing the package or with the repository root on the Python import path.

## `dac_studies`

```text
dac_studies(sample, *, blocks=None, block_size=None,
            parameter_names=None) -> StudyCollection
```

| Parameter | Meaning |
| --- | --- |
| `sample` | Finite real observation-by-coordinate array with shape `(n, d)`, `n >= 2`, and `d >= 1`. One coordinate still requires shape `(n, 1)`. |
| `blocks` | An exact partition of all coordinates, using groups of zero-based indices or parameter names. Every coordinate occurs exactly once. Supplied block and coordinate order are retained. |
| `block_size` | Positive integer generating consecutive coordinate blocks. A final smaller block is retained. Cannot be combined with `blocks`. |
| `parameter_names` | Unique nonempty names, one per sample column. Defaults to available column labels or `theta1`, `theta2`, and so on. No dataframe dependency is required. |

Omitting both partition arguments creates one block per coordinate. If a block has `q` coordinates, it needs `n > q` and a positive-definite sample covariance. The returned `StudyCollection` contains blocks labeled `block1`, `block2`, and so on. Each block contains its sample mean, the projection selecting its coordinates, covariance of the mean, and `df=n-1`.

For a block with observations `X_i`, mean `X_bar`, and scatter

\[
W = \sum_{i=1}^{n}(X_i-\bar X)(X_i-\bar X)^\mathsf{T},
\]

the covariance passed to inference is

\[
\widehat{\operatorname{Cov}}(\bar X) = \frac{W}{n(n-1)}.
\]

It is the covariance of the estimated mean, not the raw-observation covariance `W/(n-1)`. It has squared coordinate units; estimates have the original coordinate units. Hotelling calibration uses `df=n-1` and denominator degrees of freedom `n-q`. Its finite-sample interpretation assumes the relevant normal-sample model. Fitting multiple blocks additionally enforces the engine's supported convexity regime; merely satisfying `n > q` does not guarantee every multiblock configuration is supported.

This example retains within-block covariance and verifies its normalization directly:

```python
import numpy as np
from heavily_right.dac import dac_studies

sample = np.array([
    [0., 1.], [1., 0.], [2., 1.],
    [1., 2.], [0., 0.], [2., 2.],
])
inputs = dac_studies(sample, blocks=[["y", "x"]], parameter_names=["x", "y"])
block = inputs.studies[0]
ordered = sample[:, [1, 0]]
centered = ordered - ordered.mean(axis=0)
np.testing.assert_allclose(block.covariance, centered.T @ centered / (6 * 5))
np.testing.assert_allclose(block.projection, [[0., 1.], [1., 0.]])
assert block.df == 5 and inputs.parameter_names == ("x", "y")
```

Partitioning correlated coordinates does not make their p-values independent. Finite combination calibration assumes independent block p-values. Correlation between columns assigned to different blocks is not represented in their separate covariance matrices. Choose blocks and the interpretation of calibration according to the scientific model; keeping correlated coordinates together retains their joint covariance. As with multiple contrasts from one trial, sharing observations is not a reason to silently replace a required covariance by independent marginal summaries.

## `fit_dac` and `DacFit`

```text
fit_dac(sample, *, blocks=None, block_size=None, parameter_names=None,
        weights=None, method="HCauchy", level=0.05,
        empty_set_policy=None, **solver_options) -> DacFit
```

Partition arguments have the same meaning as above. `weights` follows block order and must be positive and sum to one; omitted weights are equal. `method` is `"HCauchy"` or `"EHMP"`. `level` is the significance level, so `0.05` means a nominal 95% region under the calibration assumptions.

For two or more coordinates, `solver_options` may contain `ftol`, `maxiter`, `newton_maxiter`, `dense_newton_max_dimension`, and `x0`. Options are fixed at fitting and reused for later directions. One-coordinate fits use the scalar engine and reject multidimensional solver controls.

The result keeps `.sample_size` and `.blocks` as well as the shared fitted-region interface:

| Member | Meaning |
| --- | --- |
| `.estimate`, `.parameter_names` | Fitted score minimizer and original coordinate ordering. |
| `.support_interval(direction)` | Bounds for `direction @ theta`. Direction must be finite, nonzero, and match the coordinate count; one-dimensional fits also accept a scalar. |
| `.score(point)`, `.contains(point)` | Retained-block score and region-membership test at a candidate mean vector. |
| `.inputs`, `.active_inputs` | Original and retained `StudyCollection` objects. |
| `.original_indices`, `.weights`, `.threshold` | Retained blocks' original zero-based indices, normalized weights, and fitted score cutoff. |
| `.empty_set_diagnostics` | Classification and any empty-set intervention record. |

The projected result is a `WorkflowInterval` with `.lower`, `.upper`, `.lower_point`, `.upper_point`, `.success`, and `.native_result`. Bounds are in the units of the requested linear combination; endpoint points are complete mean vectors in original coordinate units. A direction such as `[1, -1]` computes the first coordinate minus the second. Check `.success` for each requested projection. The original engine's detailed diagnostics remain available through `.native_result`.

For one block, an independent Hotelling calculation verifies the support interval:

```python
import numpy as np
from scipy.stats import f
from heavily_right.dac import fit_dac

sample = np.array([
    [0., 1.], [1., 0.], [2., 1.],
    [1., 2.], [0., 0.], [2., 2.],
])
fitted = fit_dac(sample, blocks=[[0, 1]], parameter_names=["x", "y"])
direction = np.array([1., -1.])
interval = fitted.support_interval(direction)
n, q = sample.shape
covariance = np.cov(sample, rowvar=False) / n
critical_quadratic = q * (n - 1) / (n - q) * f.isf(0.05, q, n - q)
center = direction @ sample.mean(axis=0)
half_width = np.sqrt(critical_quadratic * direction @ covariance @ direction)
assert interval.success and fitted.contains(fitted.estimate)
np.testing.assert_allclose(
    [interval.lower, interval.upper],
    [center - half_width, center + half_width], atol=2e-9,
)
```

Dimension one is supported directly. Negative directions reverse the signed interval while endpoint points stay in mean coordinates:

```python
import numpy as np
from scipy.stats import t
from heavily_right.dac import fit_dac

sample = np.array([[0.], [1.], [2.], [1.]])
fitted = fit_dac(sample, parameter_names=["mean"])
interval = fitted.support_interval(1.0)
reverse = fitted.support_interval(-1.0)
half_width = t.isf(0.025, 3) * sample[:, 0].std(ddof=1) / np.sqrt(4)
assert interval.success and reverse.success
np.testing.assert_allclose([interval.lower, interval.upper], [1.0 - half_width, 1.0 + half_width])
np.testing.assert_allclose([reverse.lower, reverse.upper], [-interval.upper, -interval.lower])
assert fitted.sample_size == 4 and fitted.blocks == ((0,),)
```

## Errors, identification, and empty regions

Malformed samples, missing or duplicated partition coordinates, unknown coordinate names, incompatible partition options, `n <= q`, and singular block covariance raise `ValueError`. Fitting also checks methods, weights, supported degrees of freedom, and solver controls. A failed fit is not evidence that the statistical confidence set is empty.

No block is removed by default. `EmptySetPolicy` is opt-in, and the fitted retained set stays fixed for all subsequent projections. In an exact coordinate partition, each block uniquely identifies its coordinates; removing one would lose identification. Such a deletion is therefore ineligible. The policy is not a general way to drop inconvenient columns or bypass singular covariance.

With valid nonsingular blocks, the assembled sample-mean vector simultaneously matches every block's estimate. Under finite-number calibration, this supplies a natural minimum and a feasible center. Numerical failure or unsuitable approximate calibration still requires explicit handling.

Relevant exceptions include `EmptySetResolutionError`, `EmptySetNumericalError`, and its `WorkflowFitError` subclass. Inspect `.diagnostics` when available; `WorkflowFitError` also retains `.result`. If a later support calculation returns `.success == False`, inspect `.native_result` rather than treating its numerical bounds as certified. The workflow does not promote unresolved emptiness to a successful scalar interval. Data-dependent removal, where applicable in a more general study design, provides no post-selection coverage correction.

See [general workflows](../WORKFLOWS.md) for the shared result types and [empty-set diagnostics](../EMPTY_SETS.md) for the policy record.
