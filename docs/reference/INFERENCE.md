# Direct scalar and vector inference

Most applications can use [NMA](NETWORK.md), [DAC](DAC.md), or
[fit_studies](STUDIES.md). These lower-level engines remain useful for repeated
custom calculations and expose their native diagnostics directly.

## MetaAnalysis1D(arg, method="HCauchy", level=0.05)

`arg` is a study count or strictly positive normalized weights.
`confidence_interval_result(theta_hat, sigma, df=None, ...)` takes one estimate
and one **standard error** per study. A scalar standard error may be shared.
`df=None` uses normal calibration; a scalar or study vector of positive real
degrees of freedom uses t calibration. Multi-study HCCT/EHMP requires `df>=1`.

The returned `ConfidenceIntervalResult` has `lower`, `upper`, `estimate`, and
`empty_set_diagnostics`. It has no `.success` property: check the diagnostics.
With removal disabled, this legacy direct API returns a degenerate interval
when the set is empty, identified by `resolved=False`/`encountered=True`.
The higher-level `fit_studies` instead rejects unresolved emptiness.

```python
import numpy as np
from scipy.stats import t
from heavily_right import MetaAnalysis1D

result = MetaAnalysis1D(1).confidence_interval_result([0.2], [0.1], df=19)
half_width = t.isf(0.025, 19) * 0.1
assert np.allclose([result.lower, result.upper], [0.2-half_width, 0.2+half_width])
assert result.empty_set_diagnostics.resolved
```

Numerical minimization/root failures raise `EmptySetNumericalError`, not a
valid empty-set decision. `confidence_interval(...)` is the older tuple form.

## MetaAnalysisMD(arg, dim, method="HCauchy", level=0.05)

`dim` is at least two. Each study estimates selected rows of `projs @ theta`.
`support_interval(direction, xi_hat, Sigma, sub_dim, df=None, projs=None, ...)`
projects the joint confidence region; it does not slice it at fixed nuisance
parameters. `Sigma` contains covariances of study **estimates**.

| Input | Meaning |
|---|---|
| `direction` | Finite nonzero parameter-length vector |
| `xi_hat` | Study vectors, potentially of unequal dimensions |
| `Sigma` | Matching positive-definite covariance matrices |
| `projs` | Matrix with `dim` columns, default identity |
| `sub_dim` | Prefer explicit zero-based projection-row indices for each study |
| `df` | `None`, shared finite value, or one finite value per study |

```python
import numpy as np
from heavily_right import MetaAnalysisMD

analysis = MetaAnalysisMD(2, dim=2)
result = analysis.support_interval(
    direction=[1., -1.],
    xi_hat=[np.array([0.2, 0.4]), np.array([0.2])],
    Sigma=[np.eye(2)*0.04, np.array([[0.03]])],
    projs=np.array([[1., 0.], [0., 1.], [-1., 1.]]),
    sub_dim=[np.array([0, 1]), np.array([2])],
)
assert result.success, (result.lower_diagnostics, result.upper_diagnostics)
assert np.isclose(result.lower, np.array([1., -1.]) @ result.lower_point)
assert result.lower < -0.2 < result.upper
```

`SupportIntervalResult` retains scalar bounds, full endpoint points, minimum
information, and separate lower/upper diagnostics. Always check `.success`.
The tuple wrapper `simultaneous_interval` raises on failed endpoint checks.
`interval_slice`/`area_slice` have different geometric meanings and are not
replacements for projections.

The sufficient convexity restrictions are stricter than valid Hotelling
degrees of freedom; see [Methods](../METHODS.md). Invalid shapes, missing rank,
or unsupported regimes are rejected. An empty multidimensional region with
intervention disabled raises; a numerical failure must not authorize removal.

Controls include `ftol`, `maxiter`, `newton_maxiter`, and
`dense_newton_max_dimension`. Newton solves KKT conditions; it is not the
only solver route. The constrained and penalty fallbacks still require
independent endpoint checks. Budgets apply per stage, not to total wall time.
See [Numerics](../NUMERICS.md) and [Troubleshooting](../TROUBLESHOOTING.md).
