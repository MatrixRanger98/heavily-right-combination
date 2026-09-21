# Numerical safeguards and validation

This is a technical summary of implemented numerical safeguards, not a claim
that every numerical input succeeds. Read the
[null-law](NULL_LAW_IMPLEMENTATION.md) and
[optimization](OPTIMIZATION_IMPLEMENTATION.md) companions for full algorithms.
Experimental settings, benchmark/plot choices, and manuscript comparisons
belong in the [reproduction guides](../reproduction/README.md), not this guide.

## Safeguard index

| Area | Implemented safeguard | Main regression file |
|---|---|---|
| Calibration | Use `W/[n(n-1)]` through one shared implementation | [test_calibration.py](../tests/test_calibration.py) |
| Explicit combination weights | Coerce iterables before array operations and validate input contracts | [test_combination.py](../tests/test_combination.py) |
| Half-Cauchy lower tail | Avoid cancellation through positive expansion / tilted inversion, with explicit underflow bounds | [test_half_cauchy_density.py](../tests/test_half_cauchy_density.py) |
| Reciprocal lower tail | Shift away the support; stabilize transform, support subtraction, and subnormal arithmetic | [test_harmonic_mean_density.py](../tests/test_harmonic_mean_density.py) |
| Precision-table reproduction | Include `2/pi` in Landau location; validate each table's own grid and explicit units | [test_numerical_table_reproduction.py](../tests/test_numerical_table_reproduction.py) |
| Disconnected NMA minimum | Allocate local gap budgets in parent units and refine without relaxing the final certificate | [test_component_minimum_budget.py](../tests/test_component_minimum_budget.py) |
| Endpoint corner cases | Scaled Newton/SLSQP, checked penalty fallback, exact scalar-cusp handling, independent certification | [test_support_solver.py](../tests/test_support_solver.py), [test_optimizer_adversarial.py](../tests/test_optimizer_adversarial.py) |
| Plot-only reuse | Protect source/destination paths; check hashes, array semantics, configuration, and required outputs | [test_replot.py](../tests/test_replot.py) |
| Plot-helper style | Set axes style before construction, scoped locally; leave saved widths and KDE curves unchanged | [test_interval_width_plotting.py](../tests/test_interval_width_plotting.py) |

## Calibration and API baseline

For observations with scatter matrix `W=sum((x_i-xbar)(x_i-xbar)')`, the
covariance of the sample mean is `W/[n(n-1)]`. The scalar standard error is
`std(x, ddof=1)/sqrt(n)`. [calibration.py](../heavily_right/calibration.py)
is the single source of truth; the univariate t and multivariate Hotelling F
identities are regression-tested. Integration and optimization routines
reuse these formulas without modifying their finite-sample calibration.

[CombinationTest](reference/COMBINATION.md) accepts valid explicit weights.
Iterables are converted before
arithmetic. Its weights must be a nonempty, finite, strictly positive vector
summing to one within the documented implementation tolerance; they are not
silently renormalized. The final p-value axis must match the number of tests,
and p-values must be finite and in `[0,1]`. Scalar and batch return shapes are
explicit. The standalone null-law APIs have their own
[weight contracts](reference/NULL_LAWS.md), including zero-weight reduction.

## Finite-law integration and scaled special functions

### Half-Cauchy: avoid lower-tail cancellation

An oscillatory upper-tail representation can suffer catastrophic cancellation
near zero even when the density is positive and representable. A warning or
an inaccurate integral must not be interpreted as a zero probability.

[_half_cauchy_lower_tail.py](../heavily_right/_half_cauchy_lower_tail.py)
uses a positive-convolution simplex expansion near zero and an exponentially
tilted inverse-Laplace contour for other lower-tail points. The positive tilt
centers the integrand near the requested value and normalization is evaluated
in logarithms. A location-based dispatch threshold only selects a numerical
method; it never declares a positive argument outside the support.
CDFs are computed directly rather than subtracting a nearly-one SF.

The checked lower-tail quadrature uses absolute tolerance `2e-13`, relative
tolerance `2e-11`, 600 subdivisions, and 100 Fourier cycles. It rejects
nonfinite results/errors, the QUADPACK maximum-float sentinel, unacceptable
error estimates, and invalid final probabilities/densities. These settings
define numerical acceptance independently of plot generation. Error estimates remain estimates, not
rigorous total-error bounds. A zero from underflow requires an analytic upper
bound and retains a positive minimum-float error; a tiny representable density
is not replaced by zero just because it is below absolute quadrature tolerance.
The [implementation guide](NULL_LAW_IMPLEMENTATION.md#underflow-error-estimates-and-failure-behavior)
specifies warning/error gates and extreme-weight limitations.

### Reciprocal: shifted transform and boundary arithmetic

For `HarmonicMean`, the statistic is `sum(w/p)`, not its reciprocal.
Write `U=1/p-1` and `z=x-sum(w)`. The transform of U is

```text
L(s) = 1 - s exp(s) E1(s) = integral_0^infinity t exp(-t)/(s+t) dt, Re(s)>0.
```

[_harmonic_mean_lower_tail.py](../heavily_right/_harmonic_mean_lower_tail.py)
uses a tilted inversion of the shifted law. For large complex arguments it
evaluates L directly by its inverse-power expansion, avoiding cancellation in
`1-s*exp(s)*E1(s)`. Near support it uses a positive simplex expansion with a
remainder estimate; see the
[coefficients and dispatch rules](NULL_LAW_IMPLEMENTATION.md#reciprocal-near-support-expansion).
The same checked quadrature adapter and tolerances apply.

Three separate floating-point fixes matter at the boundary:

- A scalar count represents an equal-weight mean with support exactly one.
  A supplied vector retains its binary floating-point weights without hidden
  renormalization; its exact sum need not be one.
- `math.fsum([x, *(-weights)])` computes the near-support residual in one
  compensated sum. Subtracting an already-rounded weight sum can erase most
  of that residual or place x on the wrong side of support.
- The one-coordinate PDF uses `(weight/x)/x`, avoiding an overflowing `x*x`
  intermediate that would destroy a representable subnormal density.

Independent 70-digit inverse-Laplace probes give
`pdf(2,100)=3.873269496959590e-7` and
`pdf(4,1000)=9.351361336938434e-6`; production agrees to about `1e-14`
relative at those points, not uniformly over all inputs. Tests include
`test_compensated_support_residual_matches_exact_input_float_oracles`,
`test_scalar_count_has_exact_mean_support_without_renormalizing_vectors`, and
`test_single_coordinate_keeps_representable_subnormal_density`.

### Real scaled Ei is a different calculation

The retained reciprocal upper-contour helper evaluates real positive
`exp(-x)*Ei(x)`: direct `exp(-x)*scipy.special.expi(x)` for `x<50`; an
optimally truncated inverse-power asymptotic expansion for `x>=50`, stopping
when terms increase or become negligible. It does not evaluate an overflowing
unscaled Ei first at large x, and it does not use Ramanujan's representation.
This helper is distinct from the complex E1/L calculation above; neither
lower-tail calculation depends on it. [test_distribution.py](../tests/test_distribution.py)
checks both real branches as well as finite-law identities and failure behavior.

## Landau parameterization and table validation

[landau_parameters()](../reproduction/numerical/distribution_precision.py)
uses `mu=(2*c/pi)*(log(m)+1-gamma)`, with `c=1` for half-Cauchy means and
`c=pi/2` for reciprocal means. The canonical Landau implementation and
`CombinationTest` use the same convention. This location/scale convention is
separate from evaluation of the scaled-Ei function.

The table assembler also validates each law's own point grid rather than
applying one grid indiscriminately to both laws. Wide-table timing fields use
seconds; long-format fields use milliseconds. Signed errors are calculated as
`Landau - finite`. Incomplete, duplicate, or inconsistent PDF/CDF pairs raise
an error rather than silently producing incomplete rows. Tests check the
parameterization against independent Fourier integration, grid contracts,
units, signed differences, and exports. Workloads, retained tables, and
manuscript comparisons are documented in the
[numerical reproduction guide](../reproduction/numerical/README.md).

## NMA minimum certificates and endpoint fallbacks

### Aggregate minimum error budgets

Individually valid component minima can have weighted gap bounds whose sum
exceeds the parent tolerance. This is a numerical certificate failure, not
evidence of an empty confidence region. The aggregate must be validated before
it can support an inference or study-selection decision.

For `G=sum(a_j*G_j)`, independently normalized local cutoffs scale like `1/a_j`.
Allowing each child its usual absolute error can spend the parent's budget
multiple times. [minimum.py](../heavily_right/minimum.py) now refines loose
children in parent score units when the aggregate alone fails. With parent
tolerance T, unrounded lower bound B, reconstructed child score S, and K
multistudy components, it reserves

```text
R = 1e-10 * max(1, abs(G), abs(B))
D = max(0, G-S)
parent allocation per multistudy component = (T-R-D)/(2*K)
local absolute gap cap = parent allocation / a_j
```

Refinement requires positive remaining budget and iteration allowance. The
cap controls initial acceptance, stopping, and final certification. A retry
replaces the old child only if independently successful with a strictly smaller
gap. Otherwise the valid old child is retained. Old and retry work counts are
accumulated even for rejected retries; `component_refinements` counts attempts,
not only accepted improvements. The final reconstructed parent is checked
against the unchanged tolerance `max(2e-6,10*ftol)*max(1,abs(G),abs(cutoff))`.
Reducing `ftol` alone cannot overcome its `2e-6` floor.

Already valid aggregates need no retry; invalid children or exhausted budgets
remain failures. Regression tests cover both support directions,
minimum reuse, weak component masses for HCauchy and EHMP, rejected retries,
and iteration exhaustion; see
[component-minimum tests](../tests/test_component_minimum_budget.py).

### Newton is a method; KKT is the certificate system

The smooth endpoint method is safeguarded Newton applied to
`grad G = eta*b, G=c`, with positive eta. Coordinate, score/slack, and direction
scaling are built in. A feasible radial start, merit-function line search,
and independent boundary/stationarity checks guard its steps. Checked SLSQP
is the first fallback; quadratic penalty continuation is the second, with
Newton polishing/restarts where applicable. Polishing is an additional solve
of the same smooth KKT equations, not a change to the statistical objective.
Optimizer-reported success alone never certifies a returned endpoint.

Scalar-study endpoints can be genuine nonsmooth cusps. The implementation uses
exact derivatives away from cusps and checked subgradient/active-set recovery
at them. A line-search stall can require a different route from an aggregate
minimum-budget refinement. Deterministic
inputs reproduce numerical difficulty; determinism does not guarantee a smooth
objective, convergence, or an accurate certificate.

See the [failure-to-regression matrix](OPTIMIZATION_IMPLEMENTATION.md#failure-modes-safeguards-and-tests)
for translation, tiny-scale, wrong-sign, weak-component, and clipped-domain
safeguards. Component decomposition is retained. Dense Newton is limited to
250 active coordinates; this does not make the other solvers sparse.
Sparse/Hessian-vector large-component solving remains future work. Neither
scaling nor these fallbacks guarantees success for every ill-conditioned input.

## Plot-only reuse and immutable-run safeguards

[replot.py](../reproduction/replot.py) and
[_replot_data.py](../reproduction/_replot_data.py) now enforce the following
in addition to never resampling:

- Resolve output paths and reject symlinked/nested run destinations that could
  write inside the source or protected trees.
- Hash metadata/schema before parsing, record hashes of consumed inputs, and
  recheck them before publishing outputs. Use a fixed experiment dispatch,
  not an arbitrary module supplied by metadata.
- Load arrays without pickle; validate keys, shapes, finiteness/ranges, axes,
  experiment identity, and configuration, not only checksums.
- For NMA, reconcile actual replication counts and available source/cache
  settings, including sample size, degrees of freedom, level, dimension,
  study count, seed, profile, and correlations. A checksum updated to match
  contradictory metadata does not make that metadata valid. Plotting uses
  the source workload, not an ambient profile default.
- Require the NMA plotted-summary NPZ and benchmark-policy JSON as well as
  the PDFs/cache before reporting completion. A failed-source recovery needs
  explicit opt-in and complete validated plot data; it is not a mechanism
  for resuming or presenting partial simulations as complete.

These checks protect reproducibility and catch accidental modification. They
are not cryptographic authentication of inputs or a guarantee against hostile
filesystem races. See the [saved-data contract](../reproduction/SAVED_DATA.md).

The interval-width plot helper sets scoped axes styles before axes construction.
Tests verify call-order
independence, unchanged global settings, and identical KDE curve coordinates.

## Empty-set API behavior is not an experiment policy

Numerical failure and certified emptiness are distinct states. A failed
minimum/support certificate cannot trigger automatic removal. Study removal
is an opt-in library policy, off by default, and retains original indices,
weights, candidate rankings, and intervention/failure records. The scalar,
multidimensional, and fitted APIs have different explicit empty-return/error
contracts; see [empty-set handling](EMPTY_SETS.md). How a simulation counts
empty outcomes is documented with that experiment in `reproduction/`.

## What regression checks establish

Run from the project root after installing the reproduction/development extras:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/heavily-right-mpl \
  python -m unittest discover -s tests -v
ruff check heavily_right reproduction tests
```

These run bounded numerical identities and independent-oracle comparisons,
saved-failure replays, forced fallback/failure tests, API/workflow contracts,
temporary-output reproduction tests, executable examples, and documentation
link checks. Ruff checks code conventions and selected static errors; it is
not a statistical or complete type proof. Neither command reruns every full
paper simulation. Full-run checks, saved-data checks, and visual comparisons
are separate evidence, recorded with the reproduction results.

Future fixes should update this summary and the relevant implementation guide,
add a small executable regression, and preserve failing inputs/diagnostics.
Do not lower acceptance standards, suppress numerical errors into zeros, or
erase failed-run provenance merely to obtain a finished plot.
