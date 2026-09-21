# Numerical methods and interpretation

For the statistical model and mathematical derivations, start with
[Methods](METHODS.md). This page records implementation details and limits.
For actionable failure handling, see [Troubleshooting](TROUBLESHOOTING.md).
The implementation companions give the [finite-law formulas and numerical
dispatch](NULL_LAW_IMPLEMENTATION.md), and the [optimization algorithms,
fallbacks, failure-mode matrix, and recorded diagnostics](OPTIMIZATION_IMPLEMENTATION.md).

## Calibration and finite-m laws

The covariance of an estimated mean is `W/[n(n-1)]`, with `W` the centered
scatter matrix. Standard errors use `std(ddof=1)/sqrt(n)`. A study with `q`
components and `df=n-1` uses the Hotelling transformation
`Q * (df-q+1)/(q*df)` against an `F(q, df-q+1)` law. With known covariance,
the quadratic statistic uses chi-square calibration. The one-dimensional
special case is the usual two-sided t interval.

`HalfCauchyMean` and `HarmonicMean` numerically invert transforms
for weighted sums under independence. "Exact finite-m" distinguishes that
distribution from a large-m approximation; it does not mean exact arithmetic.
The quadrature uses explicit absolute/relative tolerances, a finite subdivision
limit, probability-range checks, and error estimates. Quantiles use deterministic
geometric bracketing followed by Brent's method. `precision=True` returns an
estimated absolute numerical error. The contributing terms depend on the
route; this is not a rigorous bound on every floating-point or truncation error,
nor an error bound for the root returned by `ppf`.

For half-Cauchy lower tails, the original rotated characteristic-function
contour suffers severe cancellation near zero. The lower-tail route instead
uses inverse-Laplace integration on a vertical contour in the positive
half-plane. An exponential tilt centers the transformed distribution at the
requested score. With many comparable weights, direct integration combines
the transform and oscillatory phase before integration. The opposing phases
cancel near the saddle, allowing quadrature to resolve a smooth peaked product
rather than separately rapid oscillations. Small or strongly unequal-weight
systems use oscillation-aware sine/cosine quadrature; this is also the checked
fallback when direct integration fails. Neither
route may bypass the error checks; nonfinite results, excessive error, and
QUADPACK's finite failure sentinel are failures, not valid zero answers.
The error estimate includes quadrature and a floating-point accumulation
allowance. The complex Laplace transform uses scaled E1 terms
and a controlled large-argument expansion, not the reciprocal-score scaled-Ei
routine described below. The original contour remains in use above the
lower-tail dispatch threshold; that threshold selects a method, not a support
boundary.

Very close to zero, a positive-convolution expansion uses two simplex moments
and an analytic remainder bound, plus a rounding allowance. Underflow checks
use a global simplex upper bound, or a Chernoff bound for the CDF and a
tilted-convolution bound for the PDF. The latter routes include a rounding
allowance in logarithmic units; they are not outward-rounded floating-point
certificates. When the bound is below the smallest representable positive
float, zero is returned with a nonzero absolute error estimate. A numerical
location threshold alone cannot establish underflow or justify a zero answer.
Representable very small results also remain meaningful: at `x=0.4` with
1,000 equal weights the PDF is approximately `2.84394e-249`. A fixed absolute
accuracy threshold alone cannot distinguish that result from an erroneous
zero, so regression tests use relative comparisons to high-precision references.

The small CDF is evaluated directly, not by subtracting an almost-unit SF.
Call `cdf` for lower-tail probabilities: `1-sf` can round to zero. Regression
tests use independent positive two-variable convolution, high-precision
de Hoog/Cohen inversion references, simplex bounds, and both sides of contour
switches. They include upper-tail calibration reference values and injected failure
checks. These tests support the covered regimes, not universal accuracy for
all dimensions, extremely unequal weights, or arbitrary tail probabilities.

The reciprocal-score lower tail uses the same checked inversion strategy,
with a different transform. For `Y=1/p`, shift the support by writing `U=Y-1`:
`L_U(s)=1-s*exp(s)*E1(s)` in the positive half-plane. Invert at
`x-sum(weights)`, evaluated as a compensated residual near the support, after
factoring out the known support-shift exponential. At large complex arguments,
evaluate L directly through an inverse-power expansion with a Stieltjes
remainder bound, avoiding both overflow and subtraction of near-unit terms.
The PDF and small CDF are independently integrated, with explicit failure and
bounded-underflow handling; the ordinary upper-tail contour is unchanged.
The [implementation guide](NULL_LAW_IMPLEMENTATION.md#reciprocal-score-lower-tail-inversion)
gives the derivation, edge-case handling, and independent high-precision checks.
These complex scaled-E1 calculations
are distinct from the real scaled-Ei helper below.

`CombinationTest` uses the finite-m law for HCCT/EHMP up to 1,000 studies and a
Landau approximation above that; `HMP` always uses the asymptotic calibration.
The Landau code preserves the legacy parameterization and coefficient
approximations. Integration or root failure raises a typed numerical exception;
applications should report the failure, not substitute a probability of zero
or one.

## Scaled exponential integral

The internal function evaluates `exp(-x)*Ei(x)` for positive real arguments.
For `x<50`, it multiplies SciPy's `expi(x)` by `exp(-x)`. At `x>=50`, it sums
the asymptotic terms `k!/x**(k+1)` until they start increasing or become small
relative to the sum, with a finite term cap. This avoids evaluating an overflowing
unscaled Ei at large arguments. The active algorithm does not use the Ramanujan
series. The function is not a general complex-argument Ei implementation.

## Support solver

### Scalar intervals

Scalar inference translates the estimates and scales estimates and standard
errors together before either minimization or root inversion. The search is
bounded by the smallest and largest study estimates; outside that hull every
individual score increases outward. For multi-study HCCT/EHMP, the bounded
optimizer is independently checked and refined with the analytic derivative
or, at an exact study estimate, its full subgradient interval. An empty-set
decision additionally requires the convex tangent lower bound on the minimum
to exceed the cutoff. An optimizer's success flag alone cannot authorize removal.

The normal-score case is supported; with t scores, multi-study HCCT/EHMP
inference requires real `df>=1`. One-study intervals need only positive df.
Fractional df values are retained. Root searches have finite expansion and
iteration limits and an independent score-residual check. A tail-saturated
minimum, an unreliable certificate, or a failed root raises
`EmptySetNumericalError`; it is not classified as an empty interval.

Regression tests cover translations of 1,000 and 1,000,000, a unit change by
`1e-12`, fractional df 2.5, exact cusps, and injected optimizer/root failures.
This standardization does not recover precision already lost when a caller
constructs nearly equal floating-point estimates around an enormous offset.
The reconstructed endpoints are checked again in input units. If a narrow
interval cannot be represented there (for example a mean near `1e6` with
standard error `1e-13`), scalar inference raises `EmptySetNumericalError` with
recenter/rescale guidance instead of returning two rounded, misleading limits.

### Vector support

For a direction `b`, the solver maximizes `b @ z` on `G(z)<=c`. At a smooth
optimum it solves the KKT equations

```text
gradient G(z) = eta * b,    G(z) = c,    eta > 0.
```

KKT names the optimality conditions. Newton names the iteration used to solve
them. Coordinates are standardized from study information, inactive connected
components are excluded from the support optimization, and the initial point
is found by a bracketed radial boundary search from a feasible center. Newton
uses analytic derivatives, a positive-multiplier safeguard, and a backtracking
merit-function line search. Central symmetry, when detected, allows a one-sided
solve and reflection of the other endpoint.
The radial search doubles its distance up to `1e6` in standardized coordinates
and uses at most 256 Brent iterations. A failure to bracket is a numerical
failure, not evidence for an empty region.

The fallback sequence is scaled SLSQP, with conditional Newton polishing, then
scaled quadratic penalty continuation, again with polishing where available.
The direction is normalized, and the score is affinely normalized using the
initial feasible slack (or the gradient scale at a boundary start), so the
internal boundary cutoff is one. This leaves the region unchanged and prevents
a change in score units from weakening the boundary check. The normalization
is built in; ordinary callers do not choose it by hand. Diagnostics retain
`score_scale` and report score, cutoff, and boundary error in original units.
Polishing
restarts the same Newton KKT solve near a candidate endpoint to improve boundary
and stationarity accuracy. A fallback success flag cannot bypass certification.

Dense Newton is limited by default to 250 active coordinates. Large active
components take a constrained route without forming the dense Newton Hessian.
This is a practical fallback, not a claim that SLSQP scales indefinitely. Sparse
or Hessian-vector/Krylov methods for genuinely large connected components remain
future work. A problem with 2,000 coordinates split into small disconnected
components is different from one active component of size 2,000; benchmark the
actual connectivity and directions before choosing a regime.
The dimension guard does not make model preparation, minimum verification,
or the constrained fallback sparse; large-problem memory and runtime still
need separate benchmarking.

Global support certification requires a convex acceptance region. For multiple
studies the supported sufficient convex-score conditions under Hotelling
calibration are:

| Study dimension `q` | HCCT df | EHMP df |
|---|---:|---:|
| `1` | `>=1` | `>=1` |
| `2` | `>=2.5` | `>=2` |
| `>=3` | `>=q+1` | `>=q` |

These are stronger than mere distribution validity (`df>q-1`). Normal/known
covariance scores are supported without a df restriction. Estimates,
covariances, and positive normalized weights are held fixed during optimization;
projection rows must span the parameter space for bounded full-dimensional
inference. Outside a supported multi-study regime, inference is rejected rather
than labeled globally certified. A single-study Hotelling acceptance region is
an ellipsoid even where the score `-p` is not convex, so its separate analytic
support path requires only a valid law and full parameter identification.

## Certificates and nonsmooth points

The global score minimum is checked separately from support optimization.
The preliminary optimizer supplies a candidate, not a proof of emptiness.
For vector HCCT/EHMP, smooth/epigraph optimization is followed by a valid
minimum-norm subgradient check and a numerical bound on the remaining
score-value gap. Disconnected components are minimized with independently
normalized weights, then assembled and checked in the original score units.
A small global gradient alone is insufficient: a weakly weighted inactive
component can still consume enough of the score cutoff to shorten the interval.

The minimum-gap acceptance tolerance is
`max(2e-6, 10*ftol) * max(1, abs(score), abs(cutoff))`. The diagnostics retain
the estimated lower bound, gap bound, tolerance, and component checks.
Near-cusp residuals are recomputed after projection, and a conservative
intercept correction accounts for residuals that are not exactly zero in
floating-point arithmetic. These remain numerical bounds, not interval-arithmetic
proofs. If a small gap cannot be established, inference fails explicitly;
extreme weights or conditioning can therefore cause a numerical failure even
when the mathematical confidence region is nonempty.

To classify an empty set, a convex tangent
lower bound over a necessary bounded feasible domain must also exceed the
cutoff. Saturated tails invalidate this certificate. The minimum is checked
even with `find_min=False`, because fixing an inactive component at a feasible
but nonminimal point could otherwise understate the support interval.
The public multidimensional study p-values have a retained `1e-150` floor.
Support inference rejects single-study levels at or below that floor and
multi-study settings whose required individual tail probabilities reach it.
Otherwise an extreme-weight study could be capped in the public score but
treated as unbounded in a convex-domain calculation. This explicit rejection
preserves the existing p-value behavior without using it to justify a false
empty-set decision.

Each endpoint reports solver name, score/cutoff, signed boundary error, normalized
KKT residual, positive multiplier, iterations, evaluations, line-search steps,
success, a `certificate_kind` (`smooth-kkt`, `subgradient-kkt`, or
`analytic-ellipsoid`), and a message.
Check both endpoints. The certification thresholds are
part of the implementation; tightening `ftol` alone does not redefine all
acceptance criteria. The current support checks require boundary error at most
`2e-8` relative to `score_scale` and normalized KKT residual at most `2e-6`,
with a finite positive multiplier. These are numerical residual checks, not
formal proof certificates or validated bounds on error in the endpoint value.
In particular, severe conditioning can amplify a small residual; independently
check sensitive applications against analytic cases or a separate solver.
The reflected endpoint is independently checked; a failed reflection causes
a separate solve. After converting endpoints back to the caller's parameter
units, the score is evaluated again. `reconstruction_error` records the score
change caused by reconstruction. A failed returned-coordinate check makes the
rich result unsuccessful and recommends recentering/rescaling; the tuple
wrapper raises rather than silently returning such bounds.

One-dimensional study blocks can introduce cusps at exactly zero residual.
Ordinary smooth Hessian theory does not apply at those locations. A valid
subgradient certificate can still establish optimality of a convex score. The
implementation uses exact scalar-block derivatives away from the cusp, marks
its smooth Hessian unavailable at the cusp, and checks stationarity with bounded
subgradient coefficients. Small components can be polished along the active
cusp hyperplanes. Near-cusp nomination uses geometric distance to a hyperplane,
not an absolute residual that changes with the study's units.

When the smooth fallbacks do not produce a certified endpoint, the solver can
try a bounded active-set Newton search over at most 15 subsets of the four
closest scalar cusp hyperplanes. If needed, a smooth epigraph problem with
`t_j >= +residual_j` and `t_j >= -residual_j` discovers more simultaneous cusps,
followed by an original-model manifold solve and bounded-subgradient check.
Successful routes are identified as `active-set-newton` or
`epigraph-active-set-newton`. These are optimization steps, not permission to
snap an arbitrary distant point onto a cusp and declare success.

The dense-coordinate cap also applies to these nonsmooth solves; epigraph
discovery is limited to 500 total variables. Each manifold attempt has an
80-iteration cap, and epigraph SLSQP has a 1,000-iteration cap. `maxiter` is a
per-stage control, not a total optimization or wall-clock budget. Rejected
active-set/discovery work contributes to diagnostic work counters, and attempts
are named in the diagnostic message. The reproduction runner supplies the
outer experiment time limit. A degenerate region with neither strict feasible
slack nor a nonzero boundary gradient is reported as nonregular/uncertified,
not as a regular support success.

A derivative convention alone is not a convergence guarantee for Newton.
Nonsmooth handling and failure-path tests must be retained when modifying the
solver. Boundary and stationarity checks certify optimization for the supplied
score/region; they do not prove statistical coverage after adaptive study removal.

Experiment-specific choices and manuscript-result comparisons are kept in the
[reproduction guides](../reproduction/README.md).
