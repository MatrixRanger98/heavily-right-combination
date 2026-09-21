# Study, StudyCollection, and fit_studies

Import these objects from `heavily_right`. This is the general entry point for
linear study-summary designs not covered by the NMA/DAC convenience functions.

## Study(estimate, covariance, projection, df=None, label=None)

Each study estimates `projection @ theta`. For `q` observed components and
`d` common parameters, provide shapes `(q,)`, `(q,q)`, and `(q,d)`.
Covariance is the covariance of the estimate. It must be finite, symmetric,
and positive definite. Scalar summaries can use scalar estimate/covariance
and a one-dimensional projection row. Inputs are copied.

`df=None` selects known/asymptotic covariance; a finite scalar selects Hotelling
calibration and must exceed `q-1`. Stronger multi-study inference conditions
are checked when fitting. Labels, when supplied, are nonempty strings.

## StudyCollection(studies, parameter_names)

Supply a nonempty collection of `Study` objects and one unique name per
parameter. Projections must jointly identify all parameters. Missing study
labels receive defaults; resulting labels must be unique. A collection has
`.studies`, `.parameter_names`, and `.dimension`.

```python
import numpy as np
from heavily_right import Study, StudyCollection

joint = Study([0.2, 0.4], [[0.04, 0.01], [0.01, 0.09]], np.eye(2), label="joint")
inputs = StudyCollection((joint,), ("theta1", "theta2"))
assert inputs.dimension == 2
assert inputs.studies[0].label == "joint"
```

## fit_studies(inputs, *, weights=None, method="HCauchy", level=0.05, ...)

Only `HCauchy`/`EHMP` are supported here. Weights are strictly positive,
normalized, and assigned per study block. The engine supports all-known or
all-finite-df blocks; mixing them is currently rejected explicitly.
One-parameter fits require scalar study summaries with nonzero projections.

```python
import numpy as np
from heavily_right import Study, StudyCollection, fit_studies

inputs = StudyCollection(
    (Study([0.2, 0.4], np.eye(2)*0.04, np.eye(2), label="joint"),),
    ("theta1", "theta2"),
)
fit = fit_studies(inputs)
interval = fit.support_interval([1., -1.])
assert interval.success, interval.native_result
assert fit.contains(fit.estimate)
assert fit.original_indices == (0,)
assert np.isclose(interval.lower, np.array([1., -1.]) @ interval.lower_point)
```

The `WorkflowFit` retains original `.inputs`, `.active_inputs`, `.estimate`,
`.original_indices`, `.weights`, `.threshold`, and `.empty_set_diagnostics`.
Use `.score(point)`, `.contains(point)`, and `.support_interval(direction)`.
The last returns a `WorkflowInterval` with full parameter-coordinate endpoint
points and the unchanged engine result in `.native_result`.

Fitting resolves an explicitly requested `empty_set_policy` once and checks
the first coordinate support. Later queries use that same retained set and
perform their own checks. They cannot remove further studies. All query
results must be checked individually; a successful fit is not a certificate
for every later direction. Query endpoints are detached from the fit cache.

Initial failed certificates raise `WorkflowFitError` with `.result`.
Unresolved scalar emptiness raises `EmptySetResolutionError`, even when
adjustment is off. A later numerical exception is `WorkflowQueryError`:
`.diagnostics` preserves original intervention history and `.query_diagnostics`
describes the failed local retained-study solve. `fit.original_indices` maps
local indices back to original input. Invalid user inputs raise validation
errors instead. See [troubleshooting](../TROUBLESHOOTING.md).
