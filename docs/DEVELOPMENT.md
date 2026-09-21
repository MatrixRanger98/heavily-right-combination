# Development and verification

## Install and test

From the repository root:

```bash
python -m pip install -e '.[reproduction,dev]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/heavily-right-mpl \
  python -m unittest discover -s tests -v
ruff check heavily_right reproduction tests
python -m build . --outdir /tmp/heavily-right-build
```

Use a new build directory for each release check. Build/install checks
must also run outside the checkout so a missing wheel module or data resource
cannot be masked by `PYTHONPATH`. Install the wheel in an isolated environment,
import `heavily_right`, list/describe experiments, evaluate a one-study identity,
and run a short experiment with an explicit new output directory. Do not infer
support for every Python/dependency version from one successful environment;
Python 3.11+ is the declared minimum, and a multi-version test matrix remains
additional validation.

## Executable documentation

`tests/test_documentation.py` executes principal package docstrings, each
independent Python example in `docs/reference/`, and the mathematical-methods
and troubleshooting guides, plus the finite-law and optimization implementation
companions, documentation index, and recent-fixes guide. It also checks local
links. These checks are part
of the ordinary test suite; examples use temporary working directories and
must not write files. Prefer analytic assertions with meaningful tolerances to
exact printed floating-point outputs. Intentional failures must be caught and
checked rather than ignored.

Docstring examples ship in the wheel and can be read with Python `help()`.
Markdown guides/reference pages ship in the source distribution via
`MANIFEST.in`; they are not duplicated into the runtime package. No Sphinx,
MkDocs, plotting library, notebook server, or documentation-build dependency
is required to run these examples. A rendered documentation website is a
separate publishing step, not a prerequisite for package use.

## Boundaries

| Area | Responsibility |
|---|---|
| `heavily_right/scores.py`, `null_laws.py` | Reusable score transforms and finite-m / asymptotic laws |
| `heavily_right/calibration.py` | Correct covariance and standard errors of sample means |
| `heavily_right/combination.py` | Validated combination methods and decisions |
| `heavily_right/univariate.py`, `multivariate.py` | Meta-analysis orchestration and public results |
| `heavily_right/projected_region.py`, `minimum.py`, `support.py` | Projected scores/derivatives, validated convex minima, and numerical support solving |
| `heavily_right/empty_sets.py`, `multivariate_empty.py` | Opt-in study selection, structured records, failure behavior |
| `reproduction/` | Scientific scenarios, simulation summaries, plots, experiment registry |
| `tests/` | Calibration identities, shape contracts, solver comparisons, failures, packaging/reproduction checks |

Use `heavily_right` imports in personal scripts as well as packaged code.

Reusable numerical modules must not select figure names, output directories, or
paper-specific study exclusions. Experiment modules should be import-safe and
put execution in `main()`. Public functions, methods, and constructors carry
parameter and return annotations. Runtime validation and numerical regression
tests complement annotations; Ruff and annotation-presence tests are not a
complete static type proof.

## Add an experiment

1. Put the module in the appropriate `reproduction/` family. Use local random
   generators for new code, declare the seed, and separate calculations from
   figure construction where practical.
2. Add an `Experiment` entry to `manifest.py` with its ID, source, seed,
   scientific description, paper mapping, outputs, and runtime expectation.
3. Use `profile_value(paper=..., smoke=...)`. The smoke workload must still
   exercise sampling, analysis, and saving, while using a much smaller grid.
4. Reserve output paths through `ArtifactStore`, and use `save_pdf` for vector
   figures. Include the experiment name in output stems. Write input settings,
   uncertainty summaries, and failures with the output; do not depend on a
   timestamped file discovered by a loose glob.
5. Add the command and data layout to the experiment documentation; identify
   partial or reference-only mappings. Exercise a bounded smoke run.

The current experiments retain some array-based formats, global NumPy seeds, and
coupled simulate/plot entry points. A universal long-format simulation schema,
checkpoint/resume, complete Monte Carlo uncertainty metadata, and shared
cross-paper simulation specifications are not yet present. The seven long
figure workflows do provide `heavily-right-replot` (or
`python -m reproduction.replot`) for validated plot-only reuse of completed
saved data; see the [saved-data guide](../reproduction/SAVED_DATA.md).
These are explicit extensions, not capabilities supplied by packaging.

## Scientific changes

Record implemented numerical safeguards in [safeguards and validation](RECENT_FIXES.md)
and the relevant implementation guide.
Distinguish defects from deliberate changes in workload, comparisons, or plot
policy; document those experimental choices under `reproduction/`, not `docs/`.
Keep causes, safeguards, remaining limits, and regression pointers
explicit; preserve a small reproducible failure as a regression test whenever possible.

Preserve `W/[n(n-1)]` calibration. Add identity or independent-oracle tests for
changes to statistical formulas. Support changes need boundary/stationarity
checks and comparisons with analytic single-block ellipsoids and independently
constrained small systems. Test forced fallback and failed minimization paths,
not just successful Newton solves. Empty-set changes need original-index,
weight-renormalization, identifiability, no-removal-on-numerical-failure, and
JSON success/failure tests.

Tests and reproduction runs must not overwrite published reference results or
completed run artifacts. Use new immutable run IDs. A scientific numerical
discrepancy requires review before adopting a new result in a manuscript.

## Release boundary

Publishing to a public package index is a separate step from preparing this
source repository. Verify the supported platform/version matrix, distribution
contents, maintainer metadata, and dependency compatibility for each release.
Third-party data and attribution requirements remain separate from the package
license; consult the bundled provenance notices. Independent implementations
and high-precision calculations provide useful numerical oracles, but passing
a bounded comparison is not a guarantee for every input.
