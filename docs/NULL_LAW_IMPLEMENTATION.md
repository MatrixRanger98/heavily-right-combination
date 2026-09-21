# Finite-law numerical implementation

This is an implementation companion to [Methods](METHODS.md), not a new
statistical approximation or an interval-arithmetic proof. It describes the
Python implementation and its numerical limits. Public usage is in
[the null-law reference](reference/NULL_LAWS.md); optimizer details are in
[Optimization implementation](OPTIMIZATION_IMPLEMENTATION.md).

## Responsibilities and public contract

| Code | Responsibility |
|---|---|
| [`scores.py`](../heavily_right/scores.py) | Positive half-Cauchy and reciprocal score transforms |
| [`_finite_null_laws.py`](../heavily_right/_finite_null_laws.py) | Scalar PDF/CDF/SF interfaces, boundaries, method dispatch, and quantiles |
| [`_half_cauchy_lower_tail.py`](../heavily_right/_half_cauchy_lower_tail.py) | Bounded near-zero expansion and tilted half-Cauchy inversion |
| [`_harmonic_mean_lower_tail.py`](../heavily_right/_harmonic_mean_lower_tail.py) | Support-shifted reciprocal-score Laplace inversion and boundary bounds |
| [`_numerics.py`](../heavily_right/_numerics.py) | Weight validation, common tolerances, checked upper-tail integration, deterministic root bracketing, typed exceptions |
| [`_special_functions.py`](../heavily_right/_special_functions.py) | Overflow-safe real scaled Ei for the reciprocal-score law |
| [`combination.py`](../heavily_right/combination.py) | Method validation, thresholds, p-values, and decisions |

For independent standard half-Cauchy variables \(H_i\), the law is that of
\(X=\sum_i w_iH_i\). Weights must be finite, nonnegative, one-dimensional,
nonempty, and sum to one (checked with `rtol=1e-10`, `atol=1e-12`). A positive
integer count means equal weights. Zero-weight coordinates are removed before
the numerical calculation. `CombinationTest` has the stricter requirement that
every weight be positive. A weight input is validated even at support endpoints.

The public argument `x` is scalar; `precision=True` returns
`(value, estimated_absolute_error)`. `ppf` takes a **CDF probability**, so a
5% upper-tail threshold uses `ppf(0.95, w)`. It returns a scalar root, not a
root-error certificate. `check_w=False` is an internal trusted-input path,
not a request to normalize invalid weights.

The finite-m target law remains an independence law. None of the numerical
changes below establish arbitrary-dependence calibration, post-selection
coverage after study removal, or exact floating-point arithmetic.

## Why lower tails use a separate contour

The rotated contour is well behaved for the tested upper-tail calibrations
but can suffer catastrophic cancellation near zero. The true density can be
positive and representable even when an oscillatory representation yields an
unreliable integral. Typed errors preserve that distinction; rounding small
positive inputs to zero or accepting a warning as a zero answer is unsafe.

An entropy-based location selects a numerical contour only; it never declares
a positive argument outside the distribution's support. Zero answers from
underflow require an analytic upper bound, not a location-based heuristic.

## Boundary and dispatch rules

The following rules describe `HalfCauchyMean`, not every other null-law class.

| Input/regime | Implemented action |
|---|---|
| NaN | Raise `ValueError` |
| `x<=0` | PDF/CDF zero, SF one, with zero error |
| `x=+inf` | PDF/SF zero, CDF one, with zero error |
| One active weight | Analytic half-Cauchy formulas; no quadrature |
| Positive x in the lower-tail range | Bounded expansion if eligible, otherwise tilted inversion |
| Remaining positive x | Retained rotated contour and common checked quadrature |

The isolated-point convention `pdf(0)=0` is retained even for one active
weight, whose density has a positive right-hand limit. This does not change
probabilities. Small one-coordinate CDFs use `atan(x/w)` directly instead of
subtracting an almost-unit survival probability.

For positive active weights, let

\[
\ell=\frac{2}{\pi}\left(-\sum_iw_i\log w_i+1-\gamma\right).
\]

The lower-tail route is selected for `x<=max(0.5, ell-1)`. This is a tested
numerical dispatch choice, not a theorem giving a unique optimal threshold.
Tests compare values and derivatives across it for m=2,10,100,1000.

## Positive-convolution expansion near zero

With m active coordinates, put

\[
U(x)=\frac{(2/\pi)^m x^{m-1}}{\Gamma(m)\prod_iw_i},\qquad
T\sim\operatorname{Dirichlet}(1,\ldots,1).
\]

A change of variables in the positive convolution gives

\[
f(x)=U(x)\,\mathbb E\prod_i
\left[1+(xT_i/w_i)^2\right]^{-1}.
\]

In particular, `U(x)` is a global upper bound, not just a local approximation.
Let \(r_i=(x/w_i)^2\) and define

\[
A_1=\frac{2\sum_i r_i}{m(m+1)},\qquad
A_2=\frac{22\sum_i r_i^2+2(\sum_i r_i)^2}
{m(m+1)(m+2)(m+3)}.
\]

The two-term approximation is \(U(1-A_1)\), with signed analytic remainder
between zero and \(UA_2\). It is used only when `x<=1e-4*min(w)`, keeping
the relative correction and remainder small. Computations of U use logarithms.
The reported error adds `16*eps*(1+m+abs(log(U)))*U` for rounding.

For the direct CDF, replace U by `x*U/m`, A1 by `A1*m/(m+2)`, and the
remainder bound A2 by `A2*m/(m+4)`. This follows by integrating the monomials;
it is not obtained from `1-sf`.

## Tilted inverse-Laplace calculation

For `Re(s)>0`, the standard half-Cauchy Laplace transform is

\[
L(s)=\frac{e^{-is}E_1(-is)-e^{is}E_1(is)}{i\pi}.
\]

Choose a positive tilt a satisfying

\[
g(a)=\sum_i w_i\frac{-L'(aw_i)}{L(aw_i)}=x.
\]

This is the mean under exponential tilting. Its derivative is minus a sum
of weighted tilted variances, so g is decreasing. Also `g(a)<=m/a`.
The implementation starts with `high=2*m/x`, halves the lower endpoint at
most 1,100 times, and uses Brent's method with `xtol=float.tiny`,
`rtol=8*eps`. The factor two avoids a rounding-sensitive tight bracket near
zero. Invalid products, overflow, or failure to bracket/solve raise a typed
numerical error. Equal weights are compressed into unique values and counts
to avoid redundant transform evaluations.

For the actual inversion, define

\[
B=ax+\sum_i\log L(aw_i),\quad v=a/\sqrt m,\quad
H(u)=\prod_i\frac{L((a+ivu)w_i)}{L(aw_i)}.
\]

Then

\[
f(x)=\frac{e^Bv}{\pi}\int_0^\infty
\operatorname{Re}\{e^{ivxu}H(u)\}\,du.
\]

For the CDF, multiply the integrand by `a/(a+i*v*u)` and the prefactor by
`1/a`. The contour change does not approximate the law; quadrature and
floating-point calculations approximate this exact representation.

The integration strategy is deliberately not unconditional:

- With at least eight active coordinates and `max(w)<=4*min(w)`, integrate
  the phase-combined real integrand directly. Opposing phases cancel near
  the saddle before the integrator sees them.
- Otherwise, use QUADPACK's sine/cosine weighted improper integration.
  This better resolves the longer algebraic tails of small or skewed systems.
- If direct quadrature fails its checks, retry the checked Fourier route.
  A failed Fourier route raises; it does not return zero or relax tolerances.
  A later nonpositive-integral or final-result failure raises without another retry.

These dimension/weight-ratio cutoffs are practical dispatch choices. They
are covered by independent numerical comparisons, not claimed globally optimal.

### Stable complex transform evaluation

Direct scaled E1 expressions are used for `abs(s)<60`. At larger arguments,

\[
L(s)\approx\frac{2}{\pi}\sum_{k=0}^{25}
\frac{(-1)^k(2k)!}{s^{2k+1}}.
\]

Rotate the defining integral by at most pi/4, toward the damped half-plane.
The finite geometric remainder is bounded by
`(2/pi)*52!/(abs(s)/sqrt(2))**53`, below `2.8e-19` at the switch. This bound
is uniform for `Re(s)>0`; it does not include all floating-point accumulation.
The large-real-s derivative is evaluated so that a representable `1/s`
tilted mean is not lost through an intermediate underflow of `1/s**2`.

This complex E1 computation is separate from real scaled Ei in reciprocal
calibration. Neither uses Ramanujan's representation.

## Underflow, error estimates, and failure behavior

The global simplex PDF/CDF bounds can establish underflow before integration.
If they are insufficient, tilting gives

\[
F(x)\le e^B,\qquad
f(x)\le e^B\min_i\frac{2}{\pi w_iL(aw_i)}.
\]

The CDF inequality is a Chernoff bound; the PDF inequality uses the fact that
the supremum of a convolution of probability densities is no larger than
the smallest component supremum. Explicit zero shortcuts require an upper
bound below the smallest positive float. They retain an absolute error of
at least `nextafter(0,1)`. Tilted bounds have a logarithmic rounding allowance.
These are analytic inequalities evaluated in ordinary floating point, not
outward-rounded interval certificates.

Final values and errors combine normalizations in logarithms before
exponentiation, preserving representable subnormal outputs where possible.
A representable tiny density is not intentionally set to zero merely because
it falls below the absolute quadrature tolerance.

The following table describes tilted lower-tail integration, not
all guards in the retained upper-tail or reciprocal routines.

| Check/setting | Current lower-tail implementation |
|---|---|
| Quadrature tolerance | Absolute `2e-13`, relative `2e-11` for direct integration |
| Work limits | 600 subdivisions; 100 Fourier tail cycles |
| QUADPACK validation | Reject nonfinite/negative error, nonfinite result, and `DBL_MAX` sentinel |
| Error-estimate gate | Reject above 25 times the target when warnings were captured, otherwise above 100 times; warnings below this gate may be accepted with their error estimate |
| Final result | Positive normalized lower-tail integral, finite/nonnegative PDF, CDF in `[0,1]`, final error no greater than 100 times target |
| Arithmetic allowance | `32*eps*(1+m+abs(log_transform)+abs(log_scale))` times the integrated magnitude |
| SF complement | Retain CDF error and add a machine-epsilon rounding allowance |

The reported error is **estimated absolute numerical error**, not a rigorous
total-error certificate or a Monte Carlo standard error. Error controls do not
establish arbitrary-tail uniform relative accuracy. Inputs near the machine
range can still fail: weights near `1e-307`/`1e-320` with comparable x exceed
the validated transform/contour numerical range and raise typed errors. Weights `1e-100`
and `1e-200` with comparable x have independent limiting-reference tests.

## Upper-tail quantiles and reciprocal calibration

Half-Cauchy quantile inversion starts at lower bound zero,
tests an upper bound of 100, and doubles it up to 200 times as needed. Brent's
root tolerances are `xtol=2e-10`, `rtol=2e-11`. A bracketing/root failure raises
`NumericalRootError`; no root uncertainty interval is currently returned.

`HarmonicMean` models `sum(w/p)` under independent uniform p-values, not its
reciprocal. Its retained upper-contour real scaled special function is `exp(-x)*Ei(x)`: for x<50,
evaluate `exp(-x)*scipy.special.expi(x)`; for x>=50, sum `k!/x**(k+1)` until
terms increase or their magnitude is at most `2e-16` times the partial sum,
with at most 79 updates after the leading term. This avoids overflowing the
unscaled Ei. This real scaled-Ei calculation is independent of the complex
scaled-E1 calculations used for lower-tail inversion.

### Reciprocal-score lower-tail inversion

Cancellation in the reciprocal rotated contour can exceed the requested
quadrature accuracy. A support-shifted contour controls that numerical issue
without changing the target law or weakening the error gate.

Write `Y=1/p`, `U=Y-1`, and `z=x-sum(w)`. The shifted transform is

\[
L(s)=1-s e^sE_1(s)
    =\int_0^\infty\frac{t e^{-t}}{s+t}\,dt,\qquad \Re s>0.
\]

For `abs(s)<50` this uses direct complex E1 with the bounded exponential.
For larger arguments it evaluates L itself from 32 terms
`sum((-1)**k*(k+1)!/s**(k+1), k=0,...,31)`, avoiding cancellation in
`1-s*exp(s)*E1(s)`. The Stieltjes integral bounds the absolute truncation
remainder by `33!/abs(s)**33`, below `7.46e-20` at the switch, uniformly in
the positive half-plane. Floating-point errors still require allowances.

The saddle solves `sum(w*(-L'(a*w)/L(a*w)))=z`, bracketed below
`high=2*m/z`. Its derivative uses the differentiated series in the large-s
regime. Invert `prod(L(w*s))` on `s=a+iv`; dividing by s gives the CDF.
The frequency scale is `a/sqrt(m)`, and the normalization is evaluated in
logarithms. Direct phase-combined integration is used for at least eight
active weights with `max(w)/min(w)<=4`; other cases use Fourier tail cycles.
The separately checked Fourier route is also the fallback on a rejected
direct quadrature. It reuses the half-Cauchy helper's checked quadrature
adapter without changing that law's implementation or tolerances.

The reciprocal dispatch range is
`x<=max(sum(w)+0.5, -sum(w*log(w))+1-gamma-1)`. This is only a contour
selector. Below support PDF/CDF are zero and SF is one, with zero error;
the existing single-coordinate PDF-at-support zero convention is preserved.
NaN is rejected, infinities are handled explicitly, and the single-coordinate
law is analytic. Tiny CDFs are evaluated directly, not as `1-sf`.

A scalar count denotes the mathematical equal-weight mean, with support
exactly at one. A supplied vector denotes its binary floating-point weights;
validation allows a small sum tolerance but does **not** renormalize them.
Near support, `math.fsum([x, *(-weights)])` computes the residual in one
compensated sum. Subtracting an already-rounded `sum(weights)` can lose most
of that residual or classify the wrong side of the boundary. The same
residual drives support checks and inversion, and quantile inversion preserves
the count/vector distinction. Single-coordinate densities use `(weight/x)/x`
so an overflowing intermediate `x*x` cannot erase a representable subnormal.

### Reciprocal near-support expansion

For m active weights and positive support residual z, define the simplex
density upper bound and correction coefficients

```text
U = z**(m-1) / (Gamma(m) * product(w))
r_j = z/w_j
A = 2*sum(r_j)/m
B = (2*sum(r_j)**2 + 4*sum(r_j**2))/(m*(m+1))
```

When `z <= 1e-6*min(w)`, return `U*(1-A)`, with estimated absolute error at
least `U*B + roundoff`. B bounds the nonnegative remainder; it is not added
as another retained term. For CDF, replace U by `U*z/m`, A by `A*m/(m+1)`,
and B by `B*m/(m+2)`. Compute the prefactor in logarithms; the roundoff
allowance is `16*eps*(1+m+abs(log(U)))*U` for the corresponding PDF/CDF U,
with a positive minimum-float error floor.

The analytic simplex underflow bound is checked before the expansion's
eligibility test. Analytic simplex, Chernoff, or tilted-density upper bounds
must establish underflow before a zero shortcut is used; a positive minimum-
float error is retained. The same finite-value, probability-range,
positive-integral, sentinel, error-estimate, and floating-allowance checks
described above apply to the shifted inversion. No unvalidated zero replaces
a numerical failure.

Independent 70-digit inverse-Laplace checks give
`pdf(2,100)=3.873269496959590e-7` and
`pdf(4,1000)=9.351361336938434e-6`; the production values agree to about
`1e-14` relative in these probes. The
[reciprocal tests](../tests/test_harmonic_mean_density.py) additionally cover
positive bivariate convolution, an analytic two-coordinate survival identity,
support boundaries (including exact-float bivariate references), extreme
weights, subnormal single-coordinate densities, underflow, contour transitions, CDF
derivatives, quantile round trips, and injected failures.

The [safeguards summary](RECENT_FIXES.md) also describes Landau parameterization
and validation of precision-table exports.

## Reproducible checks

```python
import numpy as np
from heavily_right import HalfCauchyMean

pdf, pdf_error = HalfCauchyMean.pdf(0.01, 10, precision=True)
cdf, cdf_error = HalfCauchyMean.cdf(0.01, 10, precision=True)
assert np.isclose(pdf / 3.007799724149288e-16, 1.0, rtol=2e-9, atol=0)
assert np.isclose(cdf / 3.008710733306511e-19, 1.0, rtol=2e-9, atol=0)
assert 0 <= pdf_error < pdf and 0 <= cdf_error < cdf
# Subtraction may round to zero; the direct CDF above remains representable.
assert 1.0 - HalfCauchyMean.sf(0.01, 10) == 0.0
```

The tests in [`test_half_cauchy_density.py`](../tests/test_half_cauchy_density.py)
include positive finite convolution, sum/mean Jacobian, simplex bounds,
independent high-precision references, branch transitions, unequal/extreme
weights, invalid quadrature outputs, and upper-tail reference values.
High-precision mpmath inversion is an independent validation oracle, not a
runtime dependency. These checks distinguish measured accuracy for specific
inputs from a uniform guarantee over all dimensions, weights, and tails.
