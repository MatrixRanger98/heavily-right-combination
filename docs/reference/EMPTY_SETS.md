# EmptySetPolicy and retained diagnostics

`EmptySetPolicy(strategy="remove-smallest-p", max_removals=1,
min_remaining=1, warn=True)` enables an explicit intervention. Omitting the
policy keeps automatic removal off. Counts must be positive integers.

The algorithm first validates the score minimum and establishes emptiness.
It then ranks studies by individual p-values at that minimum. The smallest
eligible p-value is removed, ties following original index order. A vector
deletion must preserve parameter rank. Retained weights are renormalized and
the cutoff recalibrated before refitting. It does not fit and compare every
possible leave-one-out model.

```python
from heavily_right import EmptySetPolicy, MetaAnalysis1D

result = MetaAnalysis1D(2).confidence_interval_result(
    [-3., 3.], [.5, .5], study_labels=["left", "right"],
    empty_set_policy=EmptySetPolicy(max_removals=1, warn=False),
)
trace = result.empty_set_diagnostics
assert trace.encountered and trace.resolved
assert len(trace.removals) == 1
assert trace.removals[0].study_label in {"left", "right"}
assert trace.to_dict()["final_state"] == "nonempty"
assert isinstance(trace.to_json(), str)  # serialization only; no file is written
```

`EmptySetDiagnostics` includes original labels/weights, final original indices,
fitted stages, candidate decisions, selected removals, minima/cutoffs, and
`final_state` (`nonempty`, `empty`, or `unknown`). Do not interpret `unknown`
as an empty region. Unavailable/nonfinite values become JSON null.

The record can be serialized with `.to_dict()`/`.to_json()`. Explicit
`.save_json(path)` writes a new file and refuses to overwrite; retain the
input data/configuration separately. This reference's examples do not write
files. Disabling warnings requires your application to display/retain the
record itself.

`EmptySetResolutionError` means that unresolved emptiness remains;
`EmptySetNumericalError` means the required numerical conclusion was not
established. Inspect any attached `.diagnostics`, and do not delete studies
merely to work around solver failure. The higher-level fit/query exception
semantics are documented under [Study workflows](STUDIES.md).

Data-dependent removal changes the procedure and is not supplied with a
post-selection coverage correction. Report the original empty set, every
excluded study, policy limits, and resulting analysis. In NMA, a jointly
modeled multi-arm trial is one study block; independent-row
reproduction has a different deletion unit. In an ordinary DAC partition,
removing a block loses its coordinates and is therefore ineligible.

See the [complete empty-set guide](../EMPTY_SETS.md),
[mathematical methods](../METHODS.md), and [troubleshooting](../TROUBLESHOOTING.md).
