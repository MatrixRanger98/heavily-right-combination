# Numerical methods and limits

## Null laws

The finite-number half-Cauchy and reciprocal-score laws are computed by
characteristic-function or normalized Laplace inversion with explicit absolute
and relative tolerances. Calls can return the numerical error estimate through
`diagnostics = TRUE`; failed integration or quantile bracketing raises a typed
condition and never silently substitutes a tail probability.

The lower-tail route uses support-aware positive-convolution bounds and local
expansions, finite positive convolution for two coordinates, or a saddle-tilted
Laplace contour for larger systems. Small CDFs are evaluated directly, not as
one minus a nearly unit survival probability. Positive representable lower-tail
probabilities are not replaced by a fixed-location zero cutoff.

The native complex E1 implementation uses a power series and continued fraction;
moderate reciprocal transforms evaluate the equivalent scaled E2 fraction
directly to avoid subtraction cancellation. The
large-argument transforms use inverse-power expansions with analytic truncation
bounds. The saddle is solved in log coordinates. Many comparable weights use
phase-combined improper integration; other systems and failed direct attempts
use finite half-period integrals with checked Wynn extrapolation, at most 100
cycles. This is an R-specific backend, not a wrapper around Python's Fourier
quadrature. Failure to meet its error/convergence checks raises a typed error.

Reciprocal scores are shifted by their support before inversion. An integer
count denotes exact equal-weight support one; a supplied floating vector is not
renormalized, and its support residual is compensated before classification.
Single-coordinate densities avoid an overflowing intermediate square. Exact
support and infinite endpoints have analytic handling. Bounded underflow retains
a positive minimum-float error allowance; representable tiny outputs are not
discarded simply for being smaller than the absolute integration tolerance.

The reported absolute error combines quadrature/extrapolation, truncation where
applicable, and floating arithmetic allowances. These are numerical estimates,
not outward-rounded bounds or a uniform relative-accuracy guarantee. Extreme
scales or slowly converging oscillatory tails can still fail explicitly.

`scaled_ei()` uses the defining Ei series at moderate positive arguments and
an optimally truncated asymptotic expansion from 50 onward. There is no
Ramanujan production branch. Sine and cosine integrals use convergent series
and asymptotic expansions selected by argument size.

The Landau implementation computes quantiles by inversion of its R CDF.
Other implementations may use different location/scale conventions or
interpolation rules; compare values using the same parameterization and
appropriate numerical tolerances.

## Scalar inference

Minimization and root finding share translated, dimensionless coordinates.
HCauchy and EHMP scalar objectives have genuine cusps at study estimates; the
minimum search evaluates the bounded subgradient interval rather than smoothing
the cusp. A conservative tangent correction accompanies the minimum before an
empty set may be declared. Boundary roots are reconstructed and checked again
in the original physical units.

## Multivariate inference

A full-rank one-study region uses its analytic ellipsoid. Multi-study regions
use covariance-whitened, coordinate-scaled calculations. Each disconnected
component is minimized with its own normalized weights and natural score scale.
Analytic radial derivatives avoid finite differences at scalar residual cusps.
Absolute scalar residuals are lifted into auxiliary variables with linear
constraints for SLSQP. Each minimum is independently checked using a bounded
subgradient and a conservative tangent lower bound over an enclosing sublevel
domain. The resulting minimum-value gap must be small.

Individually valid component gaps must also fit a single aggregate gap budget.
If they do not, the solver reserves reconstruction/roundoff allowances, divides
half the remaining allowance in parent score units, and refines only loose
multistudy components. Local absolute caps are obtained by dividing by component
weight mass; they apply to initial acceptance, every restart, and final checks.
Refinement shares the component's original NLopt work budget, rather than
granting a fresh budget. Failed or non-improving retries retain the old valid
certificate. Physical-unit reconstruction and the unchanged parent tolerance
are checked again before success. An unresolved aggregate is a numerical
failure, not evidence for emptiness or study removal. Diagnostics retain the
initial gap, refinement attempts, cumulative solver iterations, and objective
callback counts. These counts are distinct: callbacks include R wrapper work.
When a failed call does not return work diagnostics, the remaining solver
budget is conservatively charged and the callback count is marked unknown.

Directional support normalizes the direction and score and fixes inactive
components at their validated minima. Epigraph SLSQP is followed by a Newton
polish with a positive multiplier and merit-decreasing line search. Failed
certificates trigger COBYLA and bounded exact active-manifold attempts. Multiple
simultaneous cusps use bounded subgradient coefficients. Nearby incompatible
hyperplanes are never treated as a common cusp.

Endpoint success requires a score-boundary residual at most `2e-8` times the
remaining feasible score budget (cutoff minus validated minimum), and a
normalized KKT residual at most `2e-6`, with a positive
multiplier. Endpoints are reconstructed in the original input units and checked
again without moving the returned point to a different cusp. Fitted minima
also undergo physical-unit reconstruction and independent component checks.
Near-empty regions whose remaining budget is smaller than the minimum accuracy
fail explicitly. A failed reconstruction cannot retain `success = TRUE`. Numerical
residuals are diagnostics, not rigorous interval-arithmetic endpoint enclosures.

The implementation uses dense matrices. Dense Newton polish is capped at
250 active coordinates. It is not a sparse or Hessian-vector solver. Very large
or ill-conditioned systems may fail explicitly. Each optimizer stage has a
finite evaluation limit; this is not a universal convergence guarantee.
Endpoint diagnostics retain optimizer attempts, errors, Newton iterations,
Newton evaluations and line-search steps.

The public p-value floor is `1e-150`. A candidate minimum with clipped tails, or
a weight/cutoff domain that can reach this floor, cannot support a convex
certificate. Such cases fail without authorizing study removal. Certified
multi-study regions support HCauchy/EHMP only. Known-covariance blocks have no
df restriction. Hotelling blocks require df >= 1 for q=1; EHMP requires df >= q
for q>=2; HCauchy requires df >= 2.5 for q=2 and df >= q+1 for q>=3.
