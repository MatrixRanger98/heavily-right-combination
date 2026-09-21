# Empty confidence sets

Automatic treatment is explicit and off by default. Without a policy, scalar
intervals return a compatibility degenerate interval at the score minimizer;
`empty_set_diagnostics.encountered` identifies that case. The multidimensional
support API raises for an empty region. No study is removed implicitly.
That degenerate scalar return applies to the direct `MetaAnalysis1D` API.
The fitted front ends (`fit_studies`, `fit_nma`, and `fit_dac`) instead reject
an unresolved scalar empty set with `EmptySetResolutionError`, including when
the policy is disabled; the attached record preserves that distinction.

## Enable and control removal

Pass `empty_set_policy=EmptySetPolicy(...)` and a `study_labels` sequence to
`MetaAnalysis1D.confidence_interval_result`, `confidence_interval`, or
`MetaAnalysisMD.support_interval` / the default `simultaneous_interval` wrapper.

| Policy field | Default | Meaning |
|---|---:|---|
| `strategy` | `"remove-smallest-p"` | Rank studies by their p-value at the fitted global-score minimizer |
| `max_removals` | `1` | Maximum number of sequential removals |
| `min_remaining` | `1` | Minimum retained study count; parameter identifiability is also required |
| `warn` | `True` | Emit a visible warning for each selected removal |

Counts must be positive integers. At each stage the algorithm fits the global
score and compares its minimum with the calibrated cutoff. If the set is empty,
it ranks individual study p-values from smallest to largest. A multidimensional
candidate is eligible only if removing it leaves the retained projection rows
spanning the full parameter dimension. It removes the first eligible candidate,
renormalizes the retained weights, recalibrates for the new study count, and
refits. It stops at a nonempty region or at the policy limits. Candidate
evaluation means ranking and eligibility checking; it does not mean fitting
every possible leave-one-out model or optimizing over subsets.

Numerical minimization failure or a nonfinite score is not evidence of an empty
set. `EmptySetNumericalError` identifies such failures and prevents an automatic
removal based on an unsupported empty-set decision.
For scalar HCCT/EHMP, minimum verification includes analytic
derivative/subgradient checks and a convex lower bound before declaring
emptiness. With estimated variances this requires real `df>=1` for multiple
studies. Automatic removal is not available for the legacy multi-study
combination methods whose interval searches lack a global certificate.
Vector inference likewise independently validates the minimum and requires a
convex tangent lower bound over a necessary feasible domain to establish
emptiness. A score-value gap check also guards the minimum used to hold
inactive components fixed during support calculation. A failed preliminary minimization, an arbitrary outside starting
point, and a clipped tail cannot authorize a study removal.

## What is retained

`result.empty_set_diagnostics` records whether handling was enabled, whether
emptiness occurred, whether it was resolved, the method and significance level,
parameter dimension, policy, original labels and weights, final original indices
and weights, final score/cutoff, last minimizer, and a status message. The
`final_state` is `"nonempty"`, `"empty"`, or `"unknown"`; a numerical failure
uses `"unknown"` and records its `failure_type`.

`evaluations` retains each fitted stage, its active original study indices,
weights, individual p-values, minimum score, cutoff, minimizer, and empty/nonempty
status. Ranked candidates record original index and label, priority, p-value,
eligibility, selection, reason, and retained projection rank where applicable.
`removals` contains each selected study and the remaining indices/weights.
All indices refer to the original input and are zero-based, even after several
removals. Use labels that uniquely identify comparisons when one trial contributes
multiple comparisons.
One explicit exception is a later fitted-workflow query failure:
`WorkflowQueryError.diagnostics` retains the original intervention record,
while `.query_diagnostics` uses indices local to the retained-study solve.
Map those with `fit.original_indices`. No new deletion occurs on that path.

The record can be converted with `to_dict()`, serialized with `to_json(indent=2)`,
or written with `save_json(path)`. JSON records have `schema_version=1`.
Nonfinite or unavailable numerical values are JSON `null`, not an assertion
that the score, residual, or minimizer was zero. A failed fitted stage records
its exception text and leaves its emptiness classification unknown.
`save_json` creates a new file exclusively and refuses overwriting; choose an
existing parent directory. It does not save input covariance matrices or raw
samples, so retain your immutable input data and configuration alongside it.

```python
from pathlib import Path
from heavily_right.empty_sets import (
    EmptySetNumericalError, EmptySetPolicy, EmptySetResolutionError,
)
from heavily_right.meta_analysis import MetaAnalysis1D

analysis = MetaAnalysis1D(2)
try:
    result = analysis.confidence_interval_result(
        [-3.0, 3.0], [0.5, 0.5],
        study_labels=["A", "B"],
        empty_set_policy=EmptySetPolicy(max_removals=1, min_remaining=1),
    )
    audit = result.empty_set_diagnostics
except (EmptySetResolutionError, EmptySetNumericalError) as error:
    audit = error.diagnostics
    print(error)

if audit is not None:
    audit.save_json(Path("analysis-001-empty-set.json"))
    for removal in audit.removals:
        print(removal.description)
```

Use a fresh name for each call. If warnings are disabled with `warn=False`,
display and persist the diagnostics yourself. Unresolved records attached to
`EmptySetResolutionError` have the same serialization contract as successful
ones. `EmptySetNumericalError` can also attach an incomplete record, preserving
prior removals and an unknown final stage. Check that `.diagnostics` is not
`None`, and retain the numerical exception in the application log too.

## Interpretation and experiment-specific policies

The package preserves comparison-level provenance; removing a comparison is
not always equivalent to excluding all observations from a multi-arm trial.
The choice to enable removal in a particular analysis, and the treatment of
empty outcomes in a simulation, belong with that experiment. See the
[NMA reproduction guide](../reproduction/network_meta_analysis/README.md)
for its configuration and recorded exclusions.

Data-dependent removal changes the evidence and the target analysis. Treat the
result as a documented sensitivity procedure. The original confidence guarantee
does not automatically transfer to the selected subset, and the package does
not supply a post-selection coverage correction. Report the original empty
region, policy and limits, every excluded study/comparison, retained data, and
the resulting analysis together.
