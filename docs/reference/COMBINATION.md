# CombinationTest and score transforms

## CombinationTest(arg, method="HCauchy", level=0.05)

Import from `heavily_right`. `arg` is a positive integer study count or a
one-dimensional vector of strictly positive weights summing to one. `level`
is a significance level, not a confidence coefficient. P-values must be finite
and lie in `[0,1]`, with studies on the last axis: `(..., m)` returns `(...)`.
A length-`m` vector returns a scalar. Scalar input is allowed only for `m=1`.

Principal methods:

| Method | Result |
|---|---|
| `get_global_score(p)` | Combined score; not itself a p-value |
| `get_p_from_score(score)` | Calibrated p-value |
| `get_global_p(p)` | Score and calibration together |
| `make_decision(p)` | Boolean rejection decision |
| `get_threshold_from_p(level)` | Cutoff without mutating the test |

```python
import numpy as np
from heavily_right import CombinationTest

test = CombinationTest([0.4, 0.6], method="HCauchy", level=0.05)
inputs = np.array([[0.01, 0.2], [0.3, 0.4]])
combined = test.get_global_p(inputs)
assert combined.shape == (2,)
assert np.all((0 <= combined) & (combined <= 1))
assert test.make_decision(inputs).shape == (2,)
assert np.allclose(combined, test.get_p_from_score(test.get_global_score(inputs)))
```

`HCauchy` and `EHMP` use finite-m independent-uniform calibration through
1,000 studies and a Landau approximation above that count. `HMP` uses the
asymptotic calibration. The [API method table](../API.md#combinationtest)
lists other combination rules; those are not all supported by the general
confidence-region front ends. A one-study test preserves the individual
p-value and internally represents its score as `-p`.

Invalid methods, shapes, weights, or probabilities raise `ValueError` or
`TypeError`. Failed numerical calibration raises a `NumericalCalibrationError`
subclass. Do not replace such a failure by p-value zero or one. Independence
is an assumption of the finite-m reference law, not a property checked by
this interface. See [methods](../METHODS.md).

## half_cauchy_score(p) and reciprocal_score(p)

Import these array-preserving transforms from `heavily_right.scores`.
They return `cot(pi*p/2)` and `1/p`, respectively. Both map p-value zero to
positive infinity. The positive half-Cauchy transform is not the signed
Cauchy transform used by the separate Cauchy combination test.

```python
import numpy as np
from heavily_right.scores import half_cauchy_score, reciprocal_score

p = np.array([0.25, 0.5, 1.0])
assert np.allclose(half_cauchy_score(p), 1 / np.tan(np.pi*p/2))
assert np.allclose(reciprocal_score(p), [4., 2., 1.])
```

Next: [null laws](NULL_LAWS.md) or [inverting tests into regions](INFERENCE.md).
