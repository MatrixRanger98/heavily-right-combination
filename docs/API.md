# API reference

For self-contained runnable examples beside each main entry point, use the
[individual API help pages](reference/README.md). These examples are also
available in installed Python docstrings for the principal classes and
functions. Read [Mathematical methods](METHODS.md) for derivations and
[Troubleshooting](TROUBLESHOOTING.md) for failed validations or numerical checks.

Use the `heavily_right` namespace. Underscore-prefixed modules are implementation
details rather than supported public entry points. Prefer the documented
capitalized classes to compatibility aliases.

## General fitted workflows

For arbitrary-data NMA and DAC, start with the
[general workflow guide](WORKFLOWS.md), rather than the paper-specific runner.

| API | Purpose / return |
|---|---|
| `network.network_studies(estimate, se, treat1, treat2, ...)` | Contrast summaries signed `treat1 - treat2`; returns `NetworkData` |
| `network.network_arm_studies(mean, se, treatment, study_id, ...)` | Independent-arm mean summaries; retains shared-baseline covariance in `NetworkData` |
| `network.fit_nma(data, ...)` | `NetworkFit` with `contrast(a, b)`, `pairwise_intervals()`, and directional support |
| `dac.dac_studies(sample, ...)` | Partitions sample **columns**, returning calibrated `StudyCollection` summaries |
| `dac.fit_dac(sample, ...)` | `DacFit` with partition/sample-size metadata and directional support |
| `workflows.Study(estimate, covariance, projection, df=None, label=None)` | One estimate of `projection @ theta` and covariance of that estimate |
| `workflows.StudyCollection(studies, parameter_names)` | Validated, identifiable collection with named parameters |
| `workflows.fit_studies(inputs, ...)` | General `WorkflowFit` over the scalar or multivariate numerical engine |

Fits support `HCauchy`/`EHMP`, positive normalized weights per study block,
`level`, and an optional `EmptySetPolicy`. Known/asymptotic-covariance and
finite-df blocks cannot yet be mixed. A fit fixes its retained study set once;
all subsequent contrasts use that same region. `NetworkFit.fit` and
`DacFit.fit` expose the common `WorkflowFit` and its retained-input metadata.

Directional queries return `WorkflowInterval`: scalar `lower`/`upper`, full
parameter-coordinate endpoint points, `success`, `empty_set_diagnostics`, and
the underlying engine result as `native_result`. Check `success` for every
direction; a successful fit does not pre-certify all later support solves.
`WorkflowFitError.result` retains a failed initial numerical certificate.
The workflow guide documents covariance assumptions, two-treatment/scalar
support, input ordering, empty-set selection caveats, and solver limits.

## Scores and null laws

| API | Input | Return |
|---|---|---|
| `scores.half_cauchy_score(p)` | Scalar or array of finite p-values in `[0,1]` | `cot(pi*p/2)`, same shape |
| `scores.reciprocal_score(p)` | Same | `1/p`, same shape |
| `null_laws.HalfCauchyMean.pdf/cdf/sf(x, w, precision=False)` | Scalar score `x`; equal-weight count or weight vector `w` | Scalar, or `(value, estimated_error)` |
| `null_laws.HarmonicMean.pdf/cdf/sf(x, w, precision=False)` | Same | Law of the weighted **reciprocal score**, not the reciprocal of that score |
| `null_laws.HalfCauchyMean.ppf(alpha, w)` | CDF probability in `(0,1)` | Scalar score quantile |
| `null_laws.HarmonicMean.ppf(alpha, w)` | Same | Scalar reciprocal-score quantile |
| `null_laws.Landau.pdf/cdf/sf/ppf(...)` | Scalar or array argument, location, scale | Legacy parameterization of the Landau approximation |

Zero p-values produce positive infinity in the two score transforms. Meta-analysis
has separate numerical tail floors when constructing study p-values. Do not
assume that input zero is silently replaced by an epsilon in `CombinationTest`.
Use normalized, nonnegative weights for direct null-law calls; combination
tests require strictly positive weights. The optional `check_w=False` path is
an internal fast path, not a way to validate externally supplied data.

## CombinationTest

```python
from heavily_right.combination import CombinationTest

test = CombinationTest(arg=3, method="HCauchy", level=0.05)
```

`arg` is a positive integer count or a length-`m` weight vector. Explicit
weights must be finite, strictly positive, and sum to one. `level` is a
significance level in `(0,1)`.

| Method | Score and calibration |
|---|---|
| `HCauchy` | Positive Half-Cauchy score; finite-m independence law up to `m=1000`, Landau beyond |
| `EHMP` | Reciprocal score; finite-m independence law up to `m=1000`, Landau beyond |
| `HMP` | Reciprocal score with asymptotic Landau calibration |
| `Cauchy` | Cauchy combination |
| `Levy` | Lévy combination |
| `Fisher` | Fisher score; equal weights are used |
| `Stouffer` | Weighted normal-score combination |
| `Bonferroni` | Minimum-p Bonferroni rule |
| `Simes` | Simes rule |

`get_global_score(p)` computes the combined score, `get_p_from_score(score)`
calibrates a score, `get_global_p(p)` combines both steps, and
`make_decision(p)` returns whether to reject. P-values have shape `(..., m)`;
the output has shape `(...)`. A length-`m` vector yields a scalar. Scalar input
is accepted only for `m=1`. `get_threshold_from_p(level)` returns the cutoff
without changing the object; `change_level` and `change_method` mutate it.
Read-only properties include `num_test`, `weights`, `method`, `level`, and
`threshold`.

The methods have different dependence assumptions. The finite-m laws describe
independent Uniform p-values; the API does not infer or adjust dependence from
the supplied p-values. Paper experiments assess particular dependence settings.

## Calibration and correlation

`calibration.standard_error_of_mean(sample, axis=0)` reduces the observation
axis and returns sample standard deviations with `ddof=1` divided by `sqrt(n)`.
`calibration.covariance_of_mean(sample, sample_axis=0)` treats the last axis as
the component axis. Input `(n,d)` yields `(d,d)`; input `(n,m,q)` yields
`(m,q,q)`. At least two observations are required. Both compute uncertainty of
the sample mean using the corrected denominator `n(n-1)`.

`correlation.ar1_correlation(dimension, rho)` returns entries
`rho**abs(i-j)`. `correlation.equicorrelation(dimension, rho)` returns diagonal
ones and off-diagonal `rho`. Callers must choose valid parameters: for
equicorrelation, `-1/(dimension-1) <= rho <= 1` when `dimension>1`; for AR(1),
`abs(rho)<=1`. Endpoints can be singular, so downstream factorizations may
require an explicit singular-distribution treatment.

## Scalar meta-analysis

`MetaAnalysis1D(arg, method="HCauchy", level=0.05)` inherits the combination
methods. Its principal methods take:

| Argument | Shape and meaning |
|---|---|
| `theta_hat` | `(m,)` study estimates of the same scalar parameter |
| `sigma` | `(m,)` positive standard errors, or one shared floating-point value |
| `df` | `None` for normal calibration; finite positive real scalar or `(m,)` t degrees of freedom; fractional values are not rounded |

`get_p_vector(point, ...)` returns `(m,)` two-sided p-values.
`check_cover(point, ...)` tests membership. `find_minimizer(...)` returns the
combined-score minimizer, which is the package's point estimate.
`confidence_interval_result(...)` returns a `ConfidenceIntervalResult` with
`lower`, `upper`, `estimate`, and `empty_set_diagnostics`.
`confidence_interval(...)` preserves `(lower, upper, estimate)`; append the
diagnostics with `return_diagnostics=True`.

Multi-study HCCT/EHMP interval inference requires `df>=1` when t calibration
is used, the supported convex-score regime. The p-value API still accepts any
finite positive df. A single-study t interval also accepts any positive df,
because its acceptance set is an interval without requiring a convex score.
Optimization and root inversion share translated, dimensionless coordinates;
the interval and estimate are returned in input units.

Other multi-study methods issue a warning: a single interval need not describe
a disconnected acceptance set. Their local-search compatibility behavior is
not a global optimization certificate, and automatic study removal is rejected
for those methods. Empty-set behavior is described in
[EMPTY_SETS.md](EMPTY_SETS.md).

## Vector meta-analysis and projections

`MetaAnalysisMD(arg, dim=d, method="HCauchy", level=0.05)` describes a parameter
`theta` of dimension `d>=2`. Study `j` observes an estimate of selected rows of
`projs @ theta`.

| Argument | Shape and meaning |
|---|---|
| `projs` | `(r,d)` matrix; defaults to identity |
| `sub_dim` | Prefer a list of explicit row-index arrays, one per study; equal-sized arrays may be `(m,q)` |
| `xi_hat` | `(m,q)` for equal study dimensions, otherwise a list of `(q_j,)` arrays |
| `Sigma` | `(m,q,q)` study covariances of estimates, or a list of `(q_j,q_j)` arrays; a shared `(q,q)` matrix is also accepted |
| `df` | `None` for chi-square calibration, otherwise scalar or `(m,)`; Hotelling requires `df_j > q_j-1` |
| `direction` | Finite, nonzero `(d,)` vector; bounds refer to `direction @ theta` |

An integer `sub_dim=q` uses cyclic groups of `q` projection rows; this shortcut
is intended for regular block designs. Explicit indices avoid ambiguity in
irregular networks. Indices are zero-based. Covariance matrices must be finite,
symmetric, positive definite, and in the units of the study estimates squared.
The supported multi-study convex-score regime is stricter than the validity
of a Hotelling law: scalar blocks require `df>=1`; EHMP requires `df>=q`;
HCCT requires `df>=2.5` for `q=2` and `df>=q+1` for `q>=3`. These restrictions
do not apply to the separate single-study ellipsoid path. See
[NUMERICS.md](NUMERICS.md) for the distinction between convex scores and convex
acceptance regions.
Multidimensional support rejects single-study `level<=1e-150` and multi-study
cutoff/weight combinations requiring an individual p-value at or below the
public `1e-150` floor. Extreme inputs outside this supported numerical range
are errors, not evidence of emptiness.

```python
import numpy as np
from heavily_right.meta_analysis import MetaAnalysisMD

# Two studies measure (theta_0, theta_1); another measures theta_1 - theta_0.
analysis = MetaAnalysisMD(3, dim=2)
result = analysis.support_interval(
    direction=[1.0, -1.0],
    projs=np.array([[1., 0.], [0., 1.], [-1., 1.]]),
    sub_dim=[np.array([0, 1]), np.array([0, 1]), np.array([2])],
    xi_hat=[np.array([0.2, 0.3]), np.array([0.1, 0.25]), np.array([0.1])],
    Sigma=[np.eye(2)*0.04, np.eye(2)*0.05, np.array([[0.03]])],
)
assert result.success
```

`get_p_vector(point, ...)`, `check_cover(point, ...)`, and
`find_minimizer(...)` have the analogous vector meanings.
`support_interval(...)` returns `SupportIntervalResult`, with:

- scalar `lower`, `upper`; full parameter vectors `lower_point`, `upper_point`;
- `lower_diagnostics`, `upper_diagnostics`, and Boolean `success`;
- `minimum_point`, `minimum_score`, `active_dimension`,
  `central_symmetry_used`, and optional `empty_set_diagnostics`;
- `minimum_diagnostics` for the iterative multi-study minimum, recording
  stationarity, work counts, and the conservative empty-set lower bound when
  evaluated (the analytic one-study path leaves this field `None`).

Minimum diagnostics also retain `minimum_value_lower_bound`,
`optimality_gap_bound`, `optimality_gap_tolerance`, and `cusp_anchor_correction`.
Disconnected problems include `component_indices` (parameter coordinates),
`component_weights` (original score-weight mass), and nested
`component_diagnostics`. These numerical checks prevent a tiny global gradient
from hiding an insufficiently minimized weak component.

Controls are `ftol=1e-9`, `maxiter=1000`, `newton_maxiter=80`, and
`dense_newton_max_dimension=250`. Setting `newton_maxiter=0` disables the initial
Newton attempt. `method="Powell"` names the preliminary score minimization,
not the support optimizer. Leave `format_input=True` and `find_min=True` for
ordinary calls. Supplying a feasible `x0` can help repeated work; never treat an
arbitrary starting point as evidence that the set is empty.
The candidate minimum is independently verified even with `find_min=False`.
`maxiter` limits individual numerical stages, not their combined iteration
count or elapsed time. Bounded active-set and epigraph fallbacks have their own
caps. Endpoints retain the numerical score normalization in
`diagnostics.score_scale`, with boundary errors in original score units.
`diagnostics.reconstruction_error` measures the score change after reconstructing
the returned point in input units. A failed reconstruction check leaves the
rich result unsuccessful and records recenter/rescale guidance.

`simultaneous_interval(...)` is the tuple wrapper returning
`(lower, upper, lower_point, upper_point)`; `return_diagnostics=True` appends the
rich result. It raises when endpoint certification fails. Its `solver="legacy"`
option is retained for migration comparison and does not support empty-set
intervention; selecting it emits a `RuntimeWarning` that the result is
uncertified. The compatibility `get_dif_vector` density helper emits a
`DeprecationWarning`: its legacy density behavior is not a validated derivative
of the current score and must not be used as the analytic solver derivative.
`interval_slice` and `area_slice` intersect the region with a
specified line or plane: these are slices, not support projections or marginal
confidence bounds. See their docstrings/signatures before using the legacy
slice conventions. Install the optional plotting dependencies with
`python -m pip install './Code[plot]'` when using plotting methods. The core
combination and support computations do not require Matplotlib.

## Errors, warnings, and retained information

| Type | Meaning / handling |
|---|---|
| `ValueError`, `TypeError` | Invalid method, shape, weight, or parameter; repair input |
| `NumericalCalibrationError` | Base class for failed null-law quadrature or inversion |
| `NumericalIntegrationError`, `NumericalRootError` | More specific subclasses from `null_laws` |
| `EmptySetAdjustmentWarning` | A policy selected a study for removal; keep the accompanying structured record |
| `EmptySetResolutionError` | Permitted interventions did not resolve emptiness; `.diagnostics` is serializable |
| `EmptySetNumericalError` | A required inference step could not be established numerically; this does not prove emptiness and must not authorize removal; optional `.diagnostics` preserves prior actions |
| `WorkflowFitError` | A general workflow's initial endpoint certificate failed; `.result` retains the native result |
| `WorkflowQueryError` | A later directional solve failed without changing retained studies; `.diagnostics` preserves the original intervention record and `.query_diagnostics` describes the local failed solve |
| `RuntimeError` | Other numerical failures, disabled-policy empty multivariate regions, or a failed tuple support wrapper |

An optimizer's success flag alone is insufficient. Inspect support diagnostics
and retain failed as well as successful intervention records. Detailed
serialization examples are in [EMPTY_SETS.md](EMPTY_SETS.md).
