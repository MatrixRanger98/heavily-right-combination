# Python API help and runnable examples

Each page below describes inputs, units, returned information, failure behavior,
and a minimal example. Each fenced Python example is independently runnable
after installing the core package. It creates no data files or figures.

| What you need | Help page |
|---|---|
| Combine scalar or batched p-values | [CombinationTest and score transforms](COMBINATION.md) |
| Evaluate finite-sample score distributions | [HalfCauchyMean and HarmonicMean](NULL_LAWS.md) |
| Compute uncertainty of sample means | [Covariance and standard-error helpers](CALIBRATION.md) |
| Supply custom study summaries and fit a reusable region | [Study, StudyCollection, and fit_studies](STUDIES.md) |
| Work with named treatments and multi-arm trials | [Network meta-analysis](NETWORK.md) |
| Partition sample coordinates into blocks | [Divide-and-combine](DAC.md) |
| Call the scalar/vector inference engines directly | [MetaAnalysis1D and MetaAnalysisMD](INFERENCE.md) |
| Explicitly handle empty regions and retain interventions | [EmptySetPolicy and diagnostics](EMPTY_SETS.md) |

The same package entry points include executable examples in Python docstrings.
For example, `help(heavily_right.fit_nma)` and
`help(heavily_right.covariance_of_mean)` work in an installed Python session.
Examples use assertions rather than exact printed floating-point values.

For an end-to-end introduction, start with [general workflows](../WORKFLOWS.md).
For assumptions and derivations, read [mathematical methods](../METHODS.md).
For failed validation or optimization, use [troubleshooting](../TROUBLESHOOTING.md).
The [compact API reference](../API.md) retains the broader legacy interface.

## How examples are checked

`tests/test_documentation.py` runs the maintained package docstrings and
every Python block in this reference directory, the methods guide, and the
troubleshooting guide. Each Markdown block gets its own namespace and runs in
a temporary working directory. No network or plotting dependency is required.
From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m unittest discover \
  -s tests -p test_documentation.py -v
```

Keep examples small and deterministic. Catch intentional demonstration errors
and assert the expected exception or diagnostic; never weaken a numerical
check just to make an example pass. Tests also work from the source distribution.
The Markdown pages ship in that source distribution; installed-wheel users
retain the executable docstring help without needing a documentation builder.
