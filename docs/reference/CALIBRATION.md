# Covariance and standard errors of sample means

Import both helpers from `heavily_right`. They describe uncertainty of an
estimated **mean**, not variability of individual observations. Independent
sampling units belong on the observation axis. Neither helper corrects for
clustered/serially dependent sampling or handles missing observations for you.

## standard_error_of_mean(sample, axis=0)

Computes `std(ddof=1) / sqrt(n)` along `axis`, removing that axis from the
output. At least two observations are required. A one-dimensional sample
returns a NumPy scalar; a sample of shape `(n,d)` returns shape `(d,)`.

```python
import numpy as np
from heavily_right import standard_error_of_mean

sample = np.array([[1., 2.], [2., 1.], [4., 5.]])
se = standard_error_of_mean(sample)
assert se.shape == (2,)
assert np.allclose(se, sample.std(axis=0, ddof=1) / np.sqrt(3))
```

## covariance_of_mean(sample, sample_axis=0)

Computes `W / [n*(n-1)]`, where `W` is the centered cross-product matrix.
The final axis indexes components and must be distinct from the observation
axis. Input `(n,d)` gives `(d,d)`; input `(n,m,q)` gives `(m,q,q)`.

```python
import numpy as np
from heavily_right import covariance_of_mean, standard_error_of_mean

sample = np.array([[1., 2.], [2., 1.], [4., 5.]])
covariance = covariance_of_mean(sample)
assert np.allclose(covariance, np.cov(sample, rowvar=False, ddof=1) / len(sample))
assert np.allclose(np.diag(covariance), standard_error_of_mean(sample)**2)
```

The helpers expect finite real arrays but are lightweight calculations, not
complete data validators. Inference constructors perform finiteness and
positive-definiteness checks. A mathematically singular block covariance can
still be returned here; it will be rejected for full-rank Hotelling inference.
Do not silently add a ridge to make a different statistical model pass.

For a sample mean from `n` multivariate normal observations, use this covariance
and `df=n-1`; do not divide it by `n` again. The resulting t/Hotelling identities
are derived in [Methods](../METHODS.md). [DAC](DAC.md) does these calculations
for each specified coordinate block automatically.
