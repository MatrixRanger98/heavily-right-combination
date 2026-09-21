# Python documentation

Start with [practical usage](../USAGE.md) for installation and examples, or
[general NMA/DAC workflows](WORKFLOWS.md) to analyze your own studies.

For the paper, see [arXiv:2501.01065](https://arxiv.org/abs/2501.01065)
or [read the PDF](https://arxiv.org/pdf/2501.01065).

## Methods and implementation

| Question | Guide |
|---|---|
| What numerical safeguards are implemented and how are they tested? | [Safeguards and validation](RECENT_FIXES.md) |
| What are the statistical assumptions and formulas? | [Mathematical methods](METHODS.md) |
| What accuracy and convergence limitations remain? | [Numerical contracts](NUMERICS.md) |
| How are finite-law integrals and scaled special functions evaluated? | [Null-law implementation](NULL_LAW_IMPLEMENTATION.md) |
| How do Newton, SLSQP, penalty continuation, and minimum certificates work? | [Optimization implementation](OPTIMIZATION_IMPLEMENTATION.md) |
| What happens to an empty or numerically unresolved confidence set? | [Empty-set handling](EMPTY_SETS.md) |
| How should a user interpret an error or failed certificate? | [Troubleshooting](TROUBLESHOOTING.md) |
| What inputs, shapes, units, and diagnostics does each API accept/return? | [API overview](API.md) and [individual help pages](reference/README.md) |

These guides describe the implementation and its limits, with links to the
source code and executable regression tests. Markdown guides ship in the Python source
distribution; executable help examples also ship in the runtime package.
Experimental choices, run configurations, figure selection, and manuscript
comparisons are documented under `reproduction/` and in the results README,
not in these implementation guides.

## Reproduction and development

- [Reproduction runner and commands](../reproduction/README.md)
- [Experiment settings, runtimes, and outputs](../reproduction/EXPERIMENTS.md)
- [Paper figure/table mapping and discrepancies](../reproduction/PAPER_ARTIFACTS.md)
- [Saved data, seeds, and plot-only reuse](../reproduction/SAVED_DATA.md)
- [File-by-file NPZ catalog](../reproduction/NPZ_DATA.md)
- [Development, regression checks, and packaging](DEVELOPMENT.md)

Package numerical tests, completed scientific runs, and comparisons against
published plots provide different evidence. Passing the tests does not imply
that every experiment has been rerun or that published results must match.
