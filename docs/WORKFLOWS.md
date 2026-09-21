# General NMA and divide-and-combine workflows

These front ends accept your own data. They are separate from the paper's
reproduction runner and do not read datasets, create figures, or write files.
Install the package as described in the [README](../README.md); the examples
below need only the core NumPy/SciPy dependencies, not pandas or plotting tools.

The workflow is: construct validated study summaries, fit one joint confidence
region, then request named contrasts or directional intervals. The reusable
numerical engines and corrected calibration are shared with the paper experiments.
These are common-parameter confidence-region workflows, not a general-purpose
random-effects NMA, meta-regression, or outcome-model fitting framework. A
general conventional WLS comparator is not exposed by these front ends.

## Choose the input interface

| Available input | Constructor and fit |
|---|---|
| NMA contrast estimates and their standard errors/covariances | `network_studies(...)`, then `fit_nma(data)` |
| Independent-arm estimated means and their standard errors | `network_arm_studies(...)`, then `fit_nma(data)` |
| Observation-by-coordinate sample for divide-and-combine (DAC) | `fit_dac(sample, ...)`; `dac_studies(...)` exposes its summaries |
| Other linear study summaries or a custom projection design | `Study`, `StudyCollection`, then `fit_studies(inputs)` |

All fits support `method="HCauchy"` (the default) or `method="EHMP"`, and
`level=0.05` means a significance level of 0.05. An optional weight vector has
one strictly positive entry per **study block**, must sum to one, and follows
constructor order. Equal block weights are the default. A multi-arm trial
represented by one correlated block receives one weight, not one per contrast.

Finite-m calibration assumes independent study/block p-values. Supplying
within-study covariance handles dependence *inside* that block; it does not
adjust dependence between blocks. The constructors cannot establish these
assumptions from the data.

## NMA from contrast summaries

Every input estimate is signed **`treat1 - treat2`** on the scale you supply.
For example, log odds ratios must already be computed on the log scale; the
package does not transform binary outcomes or exponentiate returned bounds.

```python
from heavily_right.network import fit_nma, network_studies

network = network_studies(
    estimate=[0.20, 0.50, 0.30],
    se=[0.15, 0.18, 0.16],
    treat1=["A", "B", "B"],
    treat2=["control", "control", "A"],
    study_id=["trial-1", "trial-2", "trial-3"],
    reference="control",
    treatments=["control", "A", "B"],
)
fit = fit_nma(network)
print(fit.parameter_names)  # ("A", "B"): effects relative to control
print(fit.estimate)

interval = fit.contrast("B", "A")  # B minus A
assert interval.success, interval.native_result
print(interval.lower, interval.upper)

for (first, second), result in fit.pairwise_intervals().items():
    assert result.success, result.native_result
    print(f"{first} minus {second}: {result.lower}, {result.upper}")
```

`treatments` specifies the complete treatment order. Without it, first
appearance in the contrast pairs determines the order; without `reference`,
the first treatment is the reference. Parameters follow the treatment order
with the reference removed. Always specify both for reproducible reporting.
The reference effect is zero by definition, not an estimated extra parameter.
A disconnected network is rejected rather than silently choosing separate
reference treatments.

`pairwise_intervals()` returns one interval per unordered treatment pair. Its
dictionary key `(a, b)` means `a - b`, in treatment-list order. Reversing a
contrast reverses the interval to `[-upper, -lower]`. These are projections of
one joint region, not separately fitted pairwise analyses. Every returned
interval still has its own numerical success checks.

If `study_id` is omitted, each contrast row is treated as a separate study.
Supply study IDs whenever several rows originate from the same trial.

### Correlated multi-arm trials

Repeated study IDs define a single correlated block. By default, such a block
requires its covariance matrix. Here both contrasts share a control arm:

```python
import numpy as np
from heavily_right.network import network_studies

multiarm = network_studies(
    estimate=[0.20, 0.50],
    se=np.sqrt([0.05, 0.08]),
    treat1=["A", "B"],
    treat2=["control", "control"],
    study_id=["three-arm-trial", "three-arm-trial"],
    reference="control",
    treatments=["control", "A", "B"],
    covariance={"three-arm-trial": np.array([[0.05, 0.01], [0.01, 0.08]])},
)
```

Covariance may be a dictionary keyed by study ID, a sequence of matrices in
first-seen study order, or a full contrast-row covariance matrix. A full matrix
must have zero covariance between different study IDs. Within each matrix,
rows follow that study's input contrast order. A supplied covariance is
authoritative, including its diagonal; it can therefore contain corrected
variances that differ from the supplied `se**2`. Matrices must be finite,
symmetric, and positive definite. When covariance is supplied, you may pass
`se=None` instead of repeating its diagonal as standard errors.

Use only a linearly independent contrast set within each trial. For a
three-arm trial, providing A-control, B-control, and B-A is redundant and is
rejected. Do not give the extra contrast a different ID to bypass that check.

`multiarm="independent"` is an explicit alternative assumption that splits
repeated-ID rows into independent scalar studies. It is not a multi-arm
correlation correction and must not be used merely because the covariance is
unavailable. It cannot discard supplied covariance. In that mode, weights and
possible removals apply to the resulting separate contrast blocks.

### Independent-arm summaries

If independent treatment-arm means and their standard errors are available,
the arm constructor derives a nonredundant contrast set and its shared-baseline
covariance:

```python
from heavily_right.network import fit_nma, network_arm_studies

arms = network_arm_studies(
    mean=[0.0, 0.20, 0.50],
    se=[0.10, 0.20, 0.15],
    treatment=["control", "A", "B"],
    study_id=["three-arm-trial"] * 3,
    reference="control",
    treatments=["control", "A", "B"],
)
arm_fit = fit_nma(arms)
arm_interval = arm_fit.contrast("B", "A")
assert arm_interval.success, arm_interval.native_result
```

`se` is the standard error of each estimated arm mean, **not** the standard
deviation of individual observations. The first input arm within each study is
its local baseline; it need not be the network reference. For independent
arms, the off-diagonal covariance of two contrasts sharing that baseline is
the baseline mean's variance. Paired, crossover, clustered, or otherwise
dependent arms require appropriately derived contrast covariances instead;
this constructor does not infer them.

### Known covariance, finite degrees of freedom, and two treatments

With `df=None`, NMA uses known/asymptotic covariance calibration. To request
finite-sample calibration, supply `df` as a scalar, one value per study block
in first-seen order, or a dictionary keyed by study ID. Degrees of freedom
are not inferred from arm standard errors. All blocks must use known
covariance or all must use finite degrees of freedom; mixed calibration is
not yet supported. The convexity restrictions in the
[API reference](API.md#vector-meta-analysis-and-projections) also apply.

Two-treatment networks use the scalar numerical engine automatically:

```python
from heavily_right.network import fit_nma, network_studies

two_treatments = network_studies(
    [0.20, 0.25], [0.15, 0.18], ["A", "A"], ["control", "control"],
    reference="control", treatments=["control", "A"],
)
two_fit = fit_nma(two_treatments)
two_interval = two_fit.contrast("A", "control")
assert two_interval.success, two_interval.native_result
```

## DAC from a multivariate sample

Here DAC partitions **columns/coordinates**, not rows/observation shards.
Every block uses all observations. A sample has shape `(n, d)`; partition
indices are zero-based, or may be names supplied through `parameter_names`.

```python
import numpy as np
from heavily_right.dac import fit_dac

rng = np.random.default_rng(20260920)
sample = rng.normal(size=(80, 5)) + np.array([0.2, 0.0, 0.1, -0.1, 0.3])
dac = fit_dac(
    sample,
    block_size=2,
    parameter_names=["x1", "x2", "x3", "x4", "x5"],
)
assert dac.blocks == ((0, 1), (2, 3), (4,))
assert dac.sample_size == 80

projection = dac.support_interval([1, 0, 0, 0, -1])  # mean(x1) - mean(x5)
assert projection.success, projection.native_result
print(projection.lower, projection.upper)
```

The final block is smaller when `d` is not divisible by `block_size`; no
coordinate is duplicated to fill it. Alternatively, use
`blocks=[["x1", "x3"], ["x2", "x4", "x5"]]` with the same parameter names.
Explicit blocks must partition all coordinates exactly once. Supply `blocks`
or `block_size`, not both. Omitting both makes one block per coordinate.

For each block, `dac_studies(...)` computes its sample mean, covariance of the
mean `W / [n * (n - 1)]`, and degrees of freedom `n - 1`, using the shared
`covariance_of_mean` helper. Here `W` is the unnormalized centered
cross-product matrix, not the sample covariance. A block must contain fewer
coordinates than observations, and singular sample covariance is rejected.
The finite-df convexity restrictions can impose additional limits on small
samples.

Partitioning correlated columns does **not** make the resulting blocks
independent. Exact finite-m calibration requires independent block p-values;
exact Hotelling calibration additionally relies on its sampling assumptions.
The example uses independent normal columns deliberately. A proposed partition
for real data needs a scientific/statistical justification; the front end
does not manufacture independence or supply a general dependence adjustment.
A one-column sample is supported through the scalar engine.

## Custom summaries and projections

Use `StudyCollection` when your design is not directly represented by the NMA
or DAC constructors. Each study estimates `projection @ theta`, and its
covariance is the covariance of that estimate, not raw-observation covariance.

```python
import numpy as np
from heavily_right.workflows import Study, StudyCollection, fit_studies

inputs = StudyCollection(
    studies=(Study(
        estimate=[0.2, 0.4],
        covariance=[[0.04, 0.01], [0.01, 0.09]],
        projection=np.eye(2),
        label="joint-estimate",
    ),),
    parameter_names=("theta1", "theta2"),
)
custom_fit = fit_studies(inputs)
custom_interval = custom_fit.support_interval([1, -1])
assert custom_interval.success, custom_interval.native_result
```

All study projections must have the same number of columns and jointly
identify every parameter. Study labels and parameter names must be unique.
One-parameter collections currently require scalar study summaries, including
nonzero scalar projection factors. Those factors are converted to common
parameter units automatically. The constructors copy inputs; later edits to
your original arrays do not alter the fitted data.

## Empty sets: opt in, then keep the record

No automatic study removal occurs by default. Enable it explicitly at fitting:

```python
from heavily_right.empty_sets import EmptySetPolicy
from heavily_right.network import fit_nma

# Uses `network` from the first example. No removal is needed for those data.
adjusted = fit_nma(
    network,
    empty_set_policy=EmptySetPolicy(max_removals=2, min_remaining=1),
)
trace = adjusted.empty_set_diagnostics
if trace is not None:
    print(trace.to_json())
    for removal in trace.removals:
        print(removal.description)
```

The current strategy ranks studies by their individual p-values at the
validated combined-score minimizer and removes an eligible smallest-p-value
study when emptiness is numerically established. Rank-losing deletions are
rejected. NMA removals operate on an entire correlated study block, not a
selected row within it. In ordinary DAC partitions, removing a block loses
identifiability of its coordinates, so this policy generally cannot remove
such a block and preserve the same parameter space.

Fitting resolves the retained set **once**. Every later contrast uses that
same set, normalized retained weights, and recalibrated threshold; it cannot
silently remove further studies for a difficult direction. The common fit
is available as `nma_fit.fit` or `dac_fit.fit`, with `original_indices`
(zero-based), `inputs`, `active_inputs`, `weights`, and `threshold`.
`NetworkFit` also exposes these retained-input properties directly.

The trace retains study labels, original indices, candidate rankings,
ineligible-deletion reasons, minimum diagnostics, thresholds, and every
removal. `trace.to_dict()` and `trace.to_json()` serialize it; explicit
`trace.save_json(path)` refuses to overwrite an existing file. Preserve the
record with the analysis. See [EMPTY_SETS.md](EMPTY_SETS.md) for failures and
serialization details.

Data-dependent removal changes the procedure: the original unchanged nominal
coverage guarantee does not automatically apply to the selected-study region.
Treat it as an explicitly reported intervention/sensitivity analysis unless
a separate inferential justification is established. A numerical failure is
not proof of emptiness and must not trigger study removal.

## Results, diagnostics, and computational limits

`WorkflowInterval` provides `lower`, `upper`, `lower_point`, `upper_point`,
`success`, `empty_set_diagnostics`, and `native_result`. Bounds are scalar
directional effects; endpoint points stay in the original parameter
coordinates. For example, an endpoint for B-A remains the complete vector of
nonreference treatment effects, not a one-element B-A vector.

Always inspect `result.success` before reporting bounds. For multivariate
results, `result.native_result` retains endpoint boundary errors, normalized
KKT residuals, solver attempts, reconstruction checks, and minimum diagnostics.
For scalar results it is a `ConfidenceIntervalResult`; it does not have the
multivariate endpoint-diagnostic fields. A successful initial fit certifies
its first coordinate interval, not every possible later direction. A failed
initial numerical certificate raises `WorkflowFitError` with the native
result in `.result`; other typed empty-set errors may retain `.diagnostics`.
If a later direction raises `WorkflowQueryError`, `.diagnostics` preserves
the original study-removal history and `.query_diagnostics` describes the
failed solve on the retained studies. The latter uses local retained-study
indices; the fit's `original_indices` maps them back to the original input.
No additional removal occurs on this path.

The front ends reuse the audited numerical safeguards: validated minima,
component decomposition, analytic single-study regions where applicable,
Newton/KKT checks, constrained fallbacks, cusp handling, and reconstructed
endpoint validation. They do not eliminate all numerical limitations.
Extremely shifted or ill-conditioned data may need meaningful recentering or
rescaling. Unsupported tail-floor regimes fail explicitly. See
[NUMERICS.md](NUMERICS.md) for the actual algorithms and limitations.

For multivariate fits, `ftol`, `maxiter`, `newton_maxiter`,
`dense_newton_max_dimension`, and `x0` can be supplied to `fit_nma`, `fit_dac`,
or `fit_studies`; later directional queries reuse those settings. These
options do not apply to scalar fits. `maxiter` is a **per-stage** budget, not
a total-work or wall-clock cap. Bounded fallback stages can add work, and the
general front ends have no five-minute hard timer. For unattended long runs,
set an external process timeout and retain the configuration and any failure
record. The paper reproduction runner's runtime controls are separate.

For provenance, retain your input summaries, study/coordinate order,
covariances, degrees of freedom, weights, reference, method, significance
level, package version, solver options, and diagnostics alongside the results.
The general front ends intentionally leave output naming and storage to you;
the [reproduction guide](../reproduction/README.md) covers immutable paper-run
directories and figure generation.
