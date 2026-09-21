# HalfCauchyMean and HarmonicMean

Import either class from `heavily_right`. `HalfCauchyMean` describes a weighted
sum of independent standard half-Cauchy variables. `HarmonicMean` describes
the weighted **reciprocal score** `sum(w/p)` for independent uniform p-values,
not the reciprocal of that sum.

Both provide `pdf(x, w)`, `cdf(x, w)`, `sf(x, w)`, and `ppf(probability, w)`.
The argument `x` is scalar. `w` is an equal-weight count or a vector of finite,
nonnegative weights summing to one; zero-weight entries are discarded. This
differs from `CombinationTest`, whose weights must all be strictly positive.
Leave the public default `check_w=True` in place.

`ppf` takes a **CDF** probability in `(0,1)`: use `ppf(0.95, w)` for a 5%
upper-tail cutoff. The other three methods accept `precision=True` to return
`(value, estimated_absolute_error)` instead of a scalar. This is a numerical
error estimate, not a rigorous interval containing the exact answer.

```python
import numpy as np
from heavily_right import HalfCauchyMean

cutoff = HalfCauchyMean.ppf(0.95, [0.4, 0.6])
survival, error_estimate = HalfCauchyMean.sf(cutoff, [0.4, 0.6], precision=True)
assert np.isclose(survival, 0.05, atol=1e-7)
assert error_estimate >= 0
```

The one-coordinate identities are useful independent checks:

```python
import numpy as np
from heavily_right import HalfCauchyMean, HarmonicMean

assert np.isclose(HalfCauchyMean.sf(1.0, 1), 0.5)
assert np.isclose(HarmonicMean.sf(20.0, 1), 0.05)
assert np.isclose(HarmonicMean.ppf(0.95, 1), 20.0)
```

For a very small half-Cauchy lower-tail probability, call `cdf` directly rather
than subtracting `sf` from one. The latter can lose the entire probability to
rounding even when the CDF itself is representable:

```python
from heavily_right import HalfCauchyMean

probability, error = HalfCauchyMean.cdf(0.01, 10, precision=True)
assert 3.00e-19 < probability < 3.02e-19
assert 0 <= error < probability
```

Half-Cauchy lower tails use a shifted inverse-Laplace contour; very close to
zero they use a convolution expansion with an analytic truncation bound.
For many comparable weights, the integration combines the oscillatory phase
with the transform before quadrature, so the integrator resolves their smooth
peaked product rather than the separately rapid oscillations. Small or strongly
unequal-weight systems use oscillation-aware quadrature; this is also the
checked fallback if direct integration fails.
The upper-tail contour is retained. A location-based cutoff only selects a
numerical method. The density is positive for
every positive finite argument mathematically, although floating-point
underflow can still make extremely small results unrepresentable. Such zero
results retain a positive absolute error estimate; a small representable
density must not be replaced by zero merely because it is below an absolute
quadrature tolerance.

`HalfCauchyMean` returns PDF zero for negative arguments, at zero, and at
positive infinity. At zero this preserves the existing isolated-point
convention, including for one active weight whose right-hand density limit is
nonzero. CDF/SF have their usual support endpoints, and NaN arguments raise
`ValueError`.

Failures raise `NumericalIntegrationError` or `NumericalRootError`, both
subclasses of `NumericalCalibrationError`. Retain the inputs and exception.
Depending on the route, the estimate covers quadrature, truncation, and an
allowance for floating-point arithmetic; it does not certify every error
source or the error of `ppf`. “Finite-m” refers to the target law rather than
exact arithmetic. See [Numerics](../NUMERICS.md#calibration-and-finite-m-laws)
for the numerical routes and their limitations.

See [mathematical methods](../METHODS.md) for the transforms and inversion
principle, and [troubleshooting](../TROUBLESHOOTING.md) for failure handling.
The [implementation guide](../NULL_LAW_IMPLEMENTATION.md) records the actual
contours, expansion coefficients, tolerances, underflow bounds, and regression checks.
