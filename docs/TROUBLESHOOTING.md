# Troubleshooting

Start by distinguishing invalid inputs, a genuinely empty confidence region,
and a numerical calculation that could not be certified. They require
different responses. A deterministic experiment can fail a deterministic
numerical check; repeating it unchanged does not make that check unnecessary.

The package fails explicitly when its safeguards cannot establish a usable
result. The suggestions below are diagnostic steps, not promises of convergence
or replacements for checking the statistical assumptions.
The [optimization implementation guide](OPTIMIZATION_IMPLEMENTATION.md) maps
stalls, cusps, scaling and reconstruction failures to the
implemented safeguard and its regression test. The
[finite-law implementation guide](NULL_LAW_IMPLEMENTATION.md) gives the contour,
expansion, underflow, and numerical-error details behind null-law failures.

## Find the relevant symptom

| Symptom | Meaning | Safe next step |
|---|---|---|
| `ModuleNotFoundError: heavily_right` | Package missing from the interpreter being used, or wrong import name | Check the interpreter and install the local distribution into that environment |
| Missing pandas, Matplotlib, or seaborn | A plotting/reproduction feature needs optional dependencies | Install the appropriate extra; core inference needs only NumPy and SciPy |
| Shape, weight, projection, covariance, or df `ValueError` | Inputs do not meet the API contract | Check ordering, units, identifiability, and the supported calibration regime |
| Repeated NMA study IDs require covariance | Several contrasts from one trial cannot silently be treated as independent | Supply their covariance or valid independent-arm summaries |
| Redundant contrasts or disconnected network | The supplied design does not identify the requested joint model as represented | Use independent within-trial contrasts and investigate disconnected treatment components |
| `EmptySetResolutionError` | Emptiness remains unresolved under the requested workflow/policy | Read the trace and report the empty result; do not invent an interval |
| `EmptySetNumericalError` | A numerical stage could not establish the needed result | Preserve diagnostics; do not interpret it as proof of emptiness or remove studies because of it |
| `WorkflowFitError` or `result.success == False` | Initial or later endpoint checks failed | Inspect the native result, both endpoints, and reconstruction checks |
| `WorkflowQueryError` | A later direction failed on an already fitted retained set | Preserve both original and query traces; the retained set has not changed |
| Small boundary error but failed stationarity | A candidate is near the boundary, not necessarily the directional optimum | Inspect KKT/subgradient checks and the fallback history; do not report it as a support bound |
| Tail-floor or null-law numerical error | Requested calibration is outside a supported range or inversion was unreliable | Review level/weights and report the unsupported calculation; never substitute 0 or 1 |
| Slow or nonregular solve | Large connected component, conditioning, cusps, or a degenerate region may be difficult | Inspect component sizes and model scaling; use an external time budget |
| Reproduced figure differs from a published one | Provenance, model, calibration, or solver behavior may differ | Compare recorded inputs and diagnostics; do not tune a calculation merely to match an image |

## Installation and environment

The distribution name is `heavily-right`; the import name is `heavily_right`.
The current distribution is local research software, not a promised package-index
release. From the project root, use `python -m pip install ./Code` in the intended
virtual environment. In a notebook, check which interpreter its kernel uses
before installing elsewhere. Restart the kernel after updating the package.

This small environment report performs no fitting or file writes:

```python
import sys
import numpy as np
import scipy
import heavily_right

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
print("NumPy:", np.__version__, "SciPy:", scipy.__version__)
print("heavily-right:", heavily_right.__version__)
print("Imported package:", heavily_right.__file__)
```

Python 3.11 or newer is required. Avoid naming your own script `numpy.py`,
`scipy.py`, or `heavily_right.py`, which can shadow an installed module.
Checkout-only `src` compatibility imports are not the installed public API.
Use `python -m pip install './Code[plot]'` for plotting or
`python -m pip install '.[reproduction]'` for the paper runner. Those extras
do not change the core statistical calibration.

## Validate the model before tuning a solver

For `Study`, `covariance` is the covariance of the **estimated summary**. For
NMA constructors, `se` means standard error, not raw-observation standard
deviation. Changing estimate units by a factor `s` changes standard errors by
`abs(s)` and covariance by `s**2`. For vector unit changes, transform both axes
of the covariance consistently. Reordering coordinates requires reordering
estimates, both covariance axes, and projection rows together.

Sample-mean covariance uses `W/[n(n-1)]`, where `W` is the centered cross-product
matrix. Do not divide this result by `n` again or replace it by `W/n**2`.
Use the shared calibration helpers when constructing summaries yourself.

Covariances must be finite, symmetric, and positive definite. Singular matrices
can indicate duplicated measurements, redundant contrasts, constant columns,
or too few observations. An arbitrary diagonal jitter or pseudoinverse changes
the model and must not be used simply to bypass validation. This example
deliberately rejects a singular input:

```python
import numpy as np
from heavily_right.workflows import Study

try:
    Study([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], np.eye(2))
except ValueError as error:
    print(type(error).__name__, error)
else:
    raise AssertionError("A singular covariance must not be accepted")
```

A valid Hotelling law requires `df > q-1` for a block of dimension `q`. For
multiple studies, the supported convex-score regime is stricter:

| Block dimension | HCCT / `HCauchy` | `EHMP` |
|---|---:|---:|
| `q=1` | `df>=1` | `df>=1` |
| `q=2` | `df>=2.5` | `df>=2` |
| `q>=3` | `df>=q+1` | `df>=q` |

Single-study analytic regions need a valid law, not those multi-study
convex-score conditions. Fractional df is supported. All study blocks in a
general fitted workflow must use `df=None` or all must have finite df; mixing
them is not yet implemented. Do not replace finite df by `None` merely to
make an unsupported configuration run.

For NMA, repeated study IDs group correlated contrasts into one block. Supply
within-study covariance, or use `network_arm_studies` only when its independent
arm-mean assumptions apply. Its standard errors are those of the estimated
arm means. For a three-arm trial, use two independent contrasts rather than
all three pairwise differences. `multiarm="independent"` is an explicit model
assumption, not a numerical fix for missing covariance. Check treatment labels,
reference, and the `treat1 - treat2` sign convention. A disconnected network
cannot provide cross-component comparisons without additional assumptions.

For DAC, blocks partition **columns**, exactly once, and each block uses all
rows. A final short block is retained. Each block needs more observations than
coordinates and a nonsingular covariance; `df=n-1` must also meet the applicable
conditions above. Splitting correlated columns does not establish block
independence. See [WORKFLOWS.md](WORKFLOWS.md) for all constructor conventions.

## Empty is not the same as unknown

Read `diagnostics.final_state`: `"empty"`, `"nonempty"`, or `"unknown"`.
An unknown numerical stage is not evidence for an empty set, even if a trial
optimizer returned a large score. Automatic removal requires a supported
minimum and empty-set certificate.

The general scalar workflow raises `EmptySetResolutionError` for unresolved
emptiness, including when removal is off. This differs from the legacy
`MetaAnalysis1D` tuple/interval behavior, which can return a degenerate interval
and an unresolved empty-set record. Without a policy, the multidimensional
engine raises a `RuntimeError` for an empty region. Do not classify arbitrary
`RuntimeError` messages as mathematical emptiness; inspect their context and
retain the error.

This example checks the error type and structured state without relying on
the exact wording of the message or enabling removal:

```python
from heavily_right.empty_sets import EmptySetResolutionError
from heavily_right.workflows import Study, StudyCollection, fit_studies

inputs = StudyCollection(
    (Study(-3.0, 0.25, [1.0], label="left"),
     Study(3.0, 0.25, [1.0], label="right")),
    ("theta",),
)
try:
    fit_studies(inputs)
except EmptySetResolutionError as error:
    trace = error.diagnostics
    assert trace.final_state == "empty"
    assert trace.removals == ()
    print(trace.final_state, trace.message)
else:
    raise AssertionError("These incompatible summaries should remain empty")
```

An explicit `EmptySetPolicy` is limited by `max_removals`, `min_remaining`, and
parameter identifiability. More iterations do not repair a rank-losing removal.
A grouped multi-arm trial is removed as a whole block. Ordinary DAC partition
blocks uniquely identify their coordinates, so removing one generally loses
rank. Never delete rows manually and describe that as the original fitted
analysis. Data-dependent exclusion does not automatically retain the original
nominal coverage guarantee. See [EMPTY_SETS.md](EMPTY_SETS.md).

In the fitted front ends, the retained set is fixed once. Later queries neither
reselect studies nor silently expand a removal policy. A `WorkflowQueryError`
preserves the original intervention history in `.diagnostics` and the failed
retained-model solve in `.query_diagnostics`. Query-record indices refer to
the local retained input; map a local index `j` using
`fit.original_indices[j]` on the common `WorkflowFit` (also exposed by
`NetworkFit`). Do not merge these two index systems without that mapping.

## Inspect a support result safely

The next example shows the success and error branches of a small analytic
problem. It prints diagnostics without saving files. A successful first fit
does not pre-certify every subsequent direction.

```python
import numpy as np
from heavily_right.empty_sets import EmptySetNumericalError, EmptySetResolutionError
from heavily_right.null_laws import NumericalCalibrationError
from heavily_right.workflows import (
    Study, StudyCollection, WorkflowFitError, WorkflowQueryError, fit_studies,
)

inputs = StudyCollection(
    (Study([0.2, 0.4], [[0.04, 0.01], [0.01, 0.09]], np.eye(2)),),
    ("theta1", "theta2"),
)
try:
    fit = fit_studies(inputs)
    interval = fit.support_interval([1.0, -1.0])
except WorkflowQueryError as error:
    print("Direction failed; no new studies were removed:", error.direction)
    for name in ("diagnostics", "query_diagnostics"):
        trace = getattr(error, name)
        print(name, None if trace is None else trace.to_dict())
except WorkflowFitError as error:
    print("Initial certificate failed:", error.result)
except (EmptySetNumericalError, EmptySetResolutionError) as error:
    print(type(error).__name__, error)
    if error.diagnostics is not None:
        print(error.diagnostics.to_dict())
except NumericalCalibrationError as error:
    print("Null-law calibration failed:", error)
else:
    native = interval.native_result
    for name in ("lower_diagnostics", "upper_diagnostics"):
        diagnostic = getattr(native, name)
        print(name, diagnostic.solver, diagnostic.certificate_kind,
              diagnostic.boundary_error, diagnostic.kkt_residual,
              diagnostic.reconstruction_error, diagnostic.message)
    if interval.success:
        print("Usable bounds:", interval.lower, interval.upper)
    else:
        print("Bounds withheld: inspect the failed endpoint certificate")
```

Scalar `ConfidenceIntervalResult` objects do not have multivariate endpoint
fields; inspect their empty-set record instead. The general wrappers preserve
their underlying engine result in `native_result`. Returned diagnostic NaN or
infinity is not a usable bound. JSON diagnostics turn unavailable/nonfinite
values into `null`, not zero.

## Newton stalls, KKT residuals, and conditioning

KKT denotes the constrained optimality conditions; Newton is the iteration
used to solve the smooth system. A deterministic candidate can lie almost
exactly on `score=cutoff` while its gradient points in the wrong direction.
It then fails stationarity: feasibility alone is insufficient for a support
optimum. Scalar-block cusps require a subgradient check, not a fabricated
smooth Hessian. Severe conditioning can amplify small numerical residuals.

The solver already uses coordinate/direction scaling, feasible radial starts,
positive-multiplier safeguards, merit-function line search, scaled SLSQP and
quadratic-penalty fallbacks, conditional Newton polishing, and bounded
nonsmooth refinements. Polishing is another Newton/KKT solve near a candidate,
not a statistical correction. Each candidate must pass the original-model
checks; a fallback's own success flag is insufficient.

When a solve fails:

1. Check inputs, units, labels, covariance construction, and the minimum
   diagnostics before blaming the endpoint optimizer.
2. Read both endpoint messages, `certificate_kind`, boundary error,
   `score_scale`, KKT residual, and `reconstruction_error`. Retain work counts.
3. Use a meaningful affine reparameterization if scales or offsets are extreme.
   Compare a small case with an analytic region or an independent constrained
   calculation where possible.
4. If only a stage budget was exhausted, a documented larger `maxiter` may
   help. Record the change and inspect the new certificate. Setting
   `newton_maxiter=0` is available for a controlled fallback comparison, not a
   universal remedy and not a way to bypass certification.

Do not loosen acceptance checks until a result says success, suppress a
failure, drop a difficult study, or raise the significance level just to make
the calculation finish. Tightening `ftol` alone also does not redefine all
certificate thresholds. Built-in SLSQP score normalization usually needs no
manual tuning, but it cannot correct a misspecified or unidentifiable model.

Directions must be finite, nonzero, and match parameter order. The engine
normalizes directions internally; multiplying one by a positive constant
scales its reported bounds, not its supporting point. If the requested bound
itself exceeds floating-point range, normalization cannot make that final
number representable. Prefer scientifically meaningful contrast units and
retain any overall scale separately.

### Narrow intervals around a large offset

Standardization cannot recover digits lost before fitting or make unrepresentable
absolute endpoints accurate. For example, double precision cannot preserve
changes of order `1e-13` added to `1e6`. Obtain or compute centered summaries
before that addition, fit the centered parameter, and retain the offset and
unit conversion separately. Do not round a failed endpoint and mark it valid.

This example starts with summaries already expressed as deviations in small
units; it intentionally does not reconstruct nearly identical absolute limits:

```python
from heavily_right.workflows import Study, StudyCollection, fit_studies

# Physical parameter: theta = reference + unit * phi.
# These deviations were retained before adding the large reference value.
reference = 1_000_000.0
unit = 1e-13
inputs = StudyCollection(
    (Study(0.20, 0.15**2, [1.0], label="one"),
     Study(0.25, 0.18**2, [1.0], label="two")),
    ("phi",),
)
fit = fit_studies(inputs)
interval = fit.support_interval([1.0])
assert interval.success, interval.native_result
print("Report theta as reference + unit * phi:", reference, unit)
print("Interval for phi:", interval.lower, interval.upper)
```

For a general study `estimate ≈ A @ theta`, an affine parameterization
`theta = offset + D @ z` gives centered estimate `estimate - A @ offset` and
projection `A @ D`, with the original study covariance unchanged. If the
observed summary is also rescaled, transform its covariance as well. Never
center each study by its own estimate: that would remove disagreement from
the statistical model rather than improve its numerical representation.

## Tails, degeneracy, and expensive problems

Multidimensional study p-values retain a public `1e-150` floor. Support rejects
single-study levels at/below it and multi-study cutoff/weight configurations
that would require individual tails at/below it. An extremely small weight
can therefore make a configuration unsupported even when its nominal global
level looks ordinary. The floor must not be used to certify an empty region.
Do not replace a clipped p-value with a smaller invented value or silently
discard its study. If the scientific setting genuinely needs that tail regime,
it requires a separately validated numerical method.

`NumericalIntegrationError` and `NumericalRootError` inherit from
`NumericalCalibrationError`. They indicate unreliable null-law quadrature or
inversion, not a p-value of zero/one or certain rejection/acceptance. Record
the method, weights, requested probability/score, versions, and exception.
`precision=True` exposes an estimated absolute numerical error for supported
law evaluations. Depending on the route, it includes quadrature, truncation,
and a floating-point allowance; it is not an interval-arithmetic proof.

Half-Cauchy lower tails use a shifted inverse-Laplace contour or, very near
zero, a convolution expansion. The implementation combines the oscillatory
phase and transform before direct integration for many comparable weights,
so quadrature resolves the smooth peaked product of opposing phases. Small or
skewed systems use oscillation-aware quadrature, which is also the checked
fallback if direct integration fails. Reported errors are still estimates,
not universal convergence or
floating-point certificates. A location-based cutoff does not establish that
a density or probability is zero. Retain a failed run's diagnostics; do not
round every small positive grid point to zero or weaken the integration guard.
Half-Cauchy density contours are separate from the support optimizer and
scaled-Ei routine. A numerical failure must be reported, not
converted into a zero density or certain rejection. For tiny probabilities
call `HalfCauchyMean.cdf(x, weights)` directly: `1 - sf(x, weights)` can lose a
representable small probability through subtraction. See
[NUMERICS.md](NUMERICS.md#calibration-and-finite-m-laws) for the numerical routes
and underflow limitations.

Reciprocal-score lower tails have a separate support-shifted Laplace route.
A rotated contour can lose accuracy through cancellation. The implementation
instead shifts each reciprocal score
by its exact lower support, evaluates a stable complex transform, and uses
checked tilted inversion without weakening tolerances. For example,
`HarmonicMean.pdf(2,100)` is about `3.873269497e-7`, not zero. Use
`HarmonicMean.cdf(x, weights)` directly for very small CDFs, and retain any
failed run and exception before starting a new run ID. Zero is returned below
support, or when an analytic bound establishes underflow with a positive error
allowance. This does not guarantee success for every extreme weight/argument;
new failures must still be recorded rather than clipped or suppressed. The
[reciprocal implementation section](NULL_LAW_IMPLEMENTATION.md#reciprocal-score-lower-tail-inversion)
explains the contour, support shift, and safeguards.

At machine precision near reciprocal support, an explicitly supplied weight
vector and an integer count are not interchangeable: a count defines the
equal-weight mean with support exactly one, whereas a vector retains its
supplied floating-point values without silent renormalization. The code
compensates the subtraction of their sum when testing the boundary and
evaluating the lower tail. Do not pre-round `x` or the weights to force a
boundary value; use the count form when that is the intended equal-weight
model. The analytic one-coordinate PDF also preserves representable
subnormal tails rather than first squaring a huge `x`.

Do not patch the scaled-Ei calculation by directly evaluating unscaled
`expi` at huge arguments: the production two-regime method avoids that overflow.

A region with no strict feasible slack and zero boundary gradient can be
nonregular for the iterative support solver. An error there is not permission
to return its numerical minimizer as an interval. Analytic single-study cases
have their own route; more general degeneracies need explicit analysis.

Dense Newton is limited by default to 250 active coordinates. Connected
component decomposition helps only when the model really decomposes; a large
connected problem is not made sparse by having few requested contrasts.
The constrained route and minimum preparation still have memory/runtime costs.
Sparse/Hessian-vector solvers for genuinely large components remain future
work. Increasing the dense cap is not a general scalability fix.

`maxiter` limits individual stages, not the total iteration count or elapsed
time. General NMA/DAC front ends have no five-minute hard timeout. Benchmark
representative directions, and use an external process limit for unattended
work. The reproduction runner has separate time-limit controls. Record a
timeout as an incomplete calculation, not a statistical result, and do not
overwrite an earlier completed run with partial output.

## Reproducible failure reports

Reproducibility requires the input data, ordering, model, method, weights,
degrees of freedom, seed where applicable, numerical settings, package revision,
and dependency versions—not just a deterministic script name. Changing BLAS,
SciPy, or a solver route can affect last digits and which safeguard is reached.
Compare numerical residuals and tolerances rather than demanding bitwise
identity for independently computed results.

Experiment settings and comparisons with published figures belong in the
[reproduction guides](../reproduction/README.md). Do not force visual
agreement by changing data or weakening numerical checks.

For a useful failure report, provide a small de-identified example, the full
exception chain or unsuccessful native result, original and query traces where
applicable, parameter/study ordering, covariance units, method/level/weights,
solver controls, and the environment report above. Include whether the model
was recentered and whether a fallback or timeout was involved. Avoid sending
confidential raw data when a synthetic reproducer demonstrates the same issue.
