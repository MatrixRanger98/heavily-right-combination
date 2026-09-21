# Network meta-analysis reference

`heavily_right.network` constructs and fits a joint region for a common vector of treatment effects. Every input contrast and named output contrast uses **first treatment minus second treatment**. Parameters are effects relative to one reference treatment, whose effect is fixed to zero. The module does not fit random effects, estimate between-study heterogeneity, or provide a weighted least-squares estimator.

The examples below are independent: each includes its imports and data, runs without optional dependencies, and creates no files. Run them after installing the package or with the repository root on the Python import path.

## `network_studies`

```text
network_studies(estimate, se, treat1, treat2, *, study_id=None,
                reference=None, treatments=None, covariance=None,
                df=None, multiarm="require_covariance") -> NetworkData
```

| Parameter | Meaning |
| --- | --- |
| `estimate` | Finite real vector of estimated `treat1 - treat2` effects. |
| `se` | Positive standard errors of the estimates, in the same units. Pass `None` when covariance is supplied. These are not raw-observation standard deviations. |
| `treat1`, `treat2` | Treatment labels, one per estimate. A row must compare distinct treatments. |
| `study_id` | One label per row. Repeated IDs form one study block. Without IDs, each row is treated as its own study; supply IDs explicitly for multi-arm trials. |
| `reference` | The treatment assigned effect zero. Defaults to the first observed treatment. |
| `treatments` | Optional complete, unique treatment ordering. The parameter ordering excludes the reference. Defaults to first appearance across input comparisons. |
| `covariance` | Covariance of the contrast estimates, in squared effect units. Accepts a mapping from study IDs to matrices, a sequence of matrices in first-seen study order, or a full row covariance with zero entries between study IDs. |
| `df` | `None` for known/asymptotic covariance, or finite Hotelling degrees of freedom. Accepts one scalar, a mapping by study ID, or one value per study block in first-seen order. Normal and finite-df blocks cannot be mixed. |
| `multiarm` | Default `"require_covariance"` requires covariance for repeated study IDs. `"independent"` explicitly models every contrast as a separate independent study and rejects any supplied covariance. |

Supplied covariance is authoritative, including its diagonal. It is preserved when corrected standard errors differ from `se`. Each block must be symmetric and positive definite. A repeated study ID needs its **within-trial covariance**, not just a vector of marginal SEs; different contrasts sharing an arm are generally correlated. Independence is an explicit modeling assumption, not a correction for missing covariance.

Each correlated block must have linearly independent contrast rows. All pairwise contrasts from a three-arm trial are redundant; supply two independent contrasts with their covariance, or use arm summaries. The whole treatment network must connect to the reference.

The result is `NetworkData`. Its `.studies` is an immutable-input `StudyCollection`; `.treatments`, `.reference`, `.parameter_names`, and `.treatment_index` retain the treatment mapping. `.contrast_direction(a, b)` returns the direction for `a - b`.

This example preserves the two correlated comparisons from one trial as a single block:

```python
import numpy as np
from heavily_right.network import network_studies, fit_nma

covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
data = network_studies(
    estimate=[0.2, 0.3], se=None,
    treat1=["b", "c"], treat2=["a", "a"],
    study_id=["trial", "trial"], reference="a",
    covariance={"trial": covariance},
)
assert data.parameter_names == ("b", "c")
assert len(data.studies.studies) == 1
np.testing.assert_allclose(data.studies.studies[0].covariance, covariance)
interval = fit_nma(data).contrast("b", "c")
assert interval.success and interval.lower < -0.1 < interval.upper
```

## `network_arm_studies`

```text
network_arm_studies(mean, se, treatment, study_id, *, reference=None,
                    treatments=None, df=None) -> NetworkData
```

The four positional arguments contain one entry per arm: its estimated mean, standard error of that mean, treatment label, and study ID. Each study needs at least two distinct treatment arms. The keyword arguments have the same meaning as in `network_studies`.

This constructor assumes that the estimated arm means within a trial are independent. It uses the first listed arm of each trial as that trial's baseline. If arm `0` is the baseline and `v_i = se_i**2`, it constructs estimates `mean_i - mean_0` and covariance

\[
\operatorname{Cov}(\widehat\delta_i,\widehat\delta_j)
= \mathbf{1}\{i=j\}v_i + v_0.
\]

The network reference and a trial's baseline are different choices. Changing either consistently preserves named treatment-contrast bounds. If the original arm estimates are already correlated, derive the contrast covariance from that full arm covariance and pass it to `network_studies` instead.

```python
import numpy as np
from scipy.stats import chi2
from heavily_right.network import network_arm_studies, fit_nma

data = network_arm_studies(
    mean=[0.0, 1.0, -0.5], se=[0.1, 0.2, 0.15],
    treatment=["a", "b", "c"], study_id=["trial"] * 3,
    reference="a",
)
np.testing.assert_allclose(
    data.studies.studies[0].covariance,
    [[0.05, 0.01], [0.01, 0.0325]],
)
interval = fit_nma(data).contrast("b", "c")
# A single two-coordinate normal block has an analytic ellipsoid.
half_width = np.sqrt(chi2.isf(0.05, 2) * (0.2**2 + 0.15**2))
assert interval.success
np.testing.assert_allclose(
    [interval.lower, interval.upper],
    [1.5 - half_width, 1.5 + half_width], atol=2e-9,
)
```

## `fit_nma` and `NetworkFit`

```text
fit_nma(data, *, weights=None, method="HCauchy", level=0.05,
        empty_set_policy=None, **solver_options) -> NetworkFit
```

`data` must be a `NetworkData`. Weights are positive, sum to one, and apply to **study blocks**, not contrast rows; omitted weights are equal. `method` supports `"HCauchy"` and `"EHMP"`. `level` is the significance level, so `0.05` corresponds to a nominal 95% region. Multistudy calibration assumes independent study-block p-values. Valid marginal p-values and applicable covariance/convexity conditions remain required.

For two or more parameters, supported solver controls are `ftol`, `maxiter`, `newton_maxiter`, `dense_newton_max_dimension`, and `x0`. Controls are set when fitting and reused for later directions. A two-treatment network has dimension one, uses the validated scalar workflow, and does not accept multidimensional solver controls.

Useful members of `NetworkFit` are:

| Member | Meaning |
| --- | --- |
| `.estimate`, `.parameter_names` | Fitted score minimizer and its reference-relative coordinate names. |
| `.contrast(a, b)` | Bounds for `a - b`; both treatments must be known and distinct. |
| `.support_interval(direction)` | Bounds for `direction @ theta` in parameter order; direction must be finite, nonzero, and correctly sized. Scalar fits also accept a scalar. |
| `.pairwise_intervals()` | Dictionary with one `(a, b)` key per unordered pair, signed in `.data.treatments` order. |
| `.score(point)`, `.contains(point)` | Retained-study score and membership test at a point in parameter coordinates. |
| `.inputs`, `.active_inputs` | Original and retained study collections. |
| `.original_indices`, `.weights`, `.threshold` | Original zero-based indices of retained blocks, their normalized weights, and the fitted score cutoff. |
| `.empty_set_diagnostics` | Empty-set classification and any removal history. |

Each projection returns a `WorkflowInterval` with `.lower`, `.upper`, `.lower_point`, `.upper_point`, `.success`, and `.native_result`. Scalar bounds have the units of the requested linear combination. Endpoint points always retain the reference-relative parameter coordinates; they do not become treatment labels or scalar contrast coordinates. Check `.success` for every projection before using its bounds. `.native_result` retains the numerical engine's diagnostics.

A complete scalar example also checks the sign of a reversed contrast:

```python
import numpy as np
from heavily_right.network import network_studies, fit_nma

data = network_studies([0.2], [0.1], ["b"], ["a"], reference="a")
fitted = fit_nma(data)
forward = fitted.contrast("b", "a")
reverse = fitted.contrast("a", "b")
assert forward.success and reverse.success
assert forward.lower < 0.2 < forward.upper
np.testing.assert_allclose(
    [reverse.lower, reverse.upper], [-forward.upper, -forward.lower],
)
assert fitted.contains(fitted.estimate)
assert set(fitted.pairwise_intervals()) == {("b", "a")}
```

## Errors and empty regions

Constructors raise `ValueError` for invalid numerical inputs, labels, dimensions, disconnected graphs, redundant contrast rows, missing required covariance, or non-positive-definite covariance. Fitting also rejects invalid calibration, weights, unsupported degrees of freedom, and inapplicable solver controls. Covariance matrices describe uncertainty of the **estimates**; covariance of raw individual observations needs conversion before use.

No study is removed by default. The workflow does not return an unresolved scalar empty set as a successful fitted interval. Numerical uncertainty is distinct from established emptiness. Relevant exceptions include `EmptySetResolutionError`, `EmptySetNumericalError`, and its `WorkflowFitError` subclass; the latter also retains `.result`. Inspect `.diagnostics` when available and individual projection results when `.success` is false.

An explicit `EmptySetPolicy` permits a bounded sensitivity analysis: remove entire study blocks when the engine establishes emptiness and the remaining network remains identifiable, renormalize weights, and recalibrate. The retained set is then **fixed** for every later contrast. Removal changes the analysis and supplies no post-selection coverage correction. Within-study covariance remains attached to the retained blocks.

This nonempty example shows how to pass a policy and inspect its record without removing anything or writing files:

```python
from heavily_right.empty_sets import EmptySetPolicy
from heavily_right.network import network_studies, fit_nma

data = network_studies([0.2], [0.1], ["b"], ["a"], reference="a")
fitted = fit_nma(data, empty_set_policy=EmptySetPolicy(max_removals=1))
before = fitted.original_indices
interval = fitted.contrast("a", "b")
assert interval.success and fitted.original_indices == before == (0,)
audit = fitted.empty_set_diagnostics
assert audit is not None and audit.resolved and not audit.removals
```

See [general workflows](../WORKFLOWS.md) for the shared study interface and [empty-set diagnostics](../EMPTY_SETS.md) for policy fields and interpretation.
