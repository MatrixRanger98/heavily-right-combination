# Mathematical methods: from study summaries to projected confidence regions

This guide describes the active Python package, including the general study,
network meta-analysis (NMA), and divide-and-combine (DAC) interfaces. It explains
the statistical construction separately from the numerical checks used to
compute it. It does not introduce new statistical results or supply proofs of
the package's supported convexity regimes.

The [workflow guide](WORKFLOWS.md) provides application examples; the
[API reference](API.md) gives argument and return-value conventions; the
[numerical guide](NUMERICS.md) records implementation limits and diagnostics.
For exact implementation settings and failure-mode regressions, see the
[finite-law companion](NULL_LAW_IMPLEMENTATION.md) and
[optimization companion](OPTIMIZATION_IMPLEMENTATION.md).
Here `level=alpha` always denotes a significance level, so the intended
confidence level is `1 - alpha`.

Contents:

- [Study summaries and covariance](#study-summaries-and-the-common-parameter)
- [Gaussian and Hotelling p-values](#study-p-values-gaussian-and-hotelling-calibration)
- [Scores and combination calibration](#positive-heavy-right-scores-and-their-null-distributions)
- [Regions, projections, and simultaneous interpretation](#confidence-region-inversion-and-directional-intervals)
- [NMA design and arm covariance](#nma-design-and-shared-arm-covariance)
- [DAC coordinate partitions](#dac-is-a-coordinate-partition)
- [Convexity and cusps](#supported-convexity-and-scalar-cusps)
- [Numerical coordinates and point estimation](#numerical-coordinates-are-not-an-estimator-definition)
- [KKT, Newton, and penalty methods](#kkt-equations-newton-steps-and-penalty-continuation)
- [Minimum bounds and empty-set states](#minimum-value-bounds-and-three-different-empty-set-states)
- [Optional study removal](#optional-removal-is-a-sensitivity-analysis)
- [Numerical exactness and the scaled exponential integral](#what-exact-finite-number-does-and-does-not-mean)

## Study summaries and the common parameter

Let the common parameter be \(\theta\in\mathbb R^d\). Study block \(j\)
supplies an estimate \(\widehat\xi_j\in\mathbb R^{q_j}\), a projection
\(A_j\in\mathbb R^{q_j\times d}\), and a positive-definite covariance matrix
\(V_j\) for that estimate. Its mean model is

\[
E(\widehat\xi_j)=A_j\theta.
\]

These are `Study(estimate, covariance, projection, df, label)` fields. The
matrix `covariance` is the covariance of the estimator, or its supplied
estimate; it is not generally the covariance of individual observations.
Study dimensions may differ. The stacked projections must identify all
coordinates of \(\theta\); individual studies may identify only a subset.
The study estimates, covariance estimates, projections, and combination
weights are held fixed while candidate parameter values are evaluated.

For \(n\) independent observations \(X_i\in\mathbb R^q\), write

\[
\overline X=\frac1n\sum_{i=1}^nX_i,\qquad
W=\sum_{i=1}^n(X_i-\overline X)(X_i-\overline X)^T,\qquad
S=\frac{W}{n-1}.
\]

The estimated covariance of \(\overline X\) is

\[
\widehat{\operatorname{Var}}(\overline X)
=\frac Sn=\frac{W}{n(n-1)}.
\]

Thus a scalar standard error is the sample standard deviation with `ddof=1`
divided by \(\sqrt n\). Dividing the scatter matrix by \(n^2\) would omit
the sample-variance correction. The shared
[`covariance_of_mean`](../heavily_right/calibration.py) implements the displayed
formula, and DAC uses it for every coordinate block.

This is a common-parameter model. The NMA interface does not estimate a
between-study heterogeneity distribution, random-effects variance, or an
outcome regression. Those modeling choices must be justified before its
study summaries are supplied.

## Study p-values: Gaussian and Hotelling calibration

For a candidate \(\theta\), the quadratic discrepancy is

\[
Q_j(\theta)=(A_j\theta-\widehat\xi_j)^T
V_j^{-1}(A_j\theta-\widehat\xi_j).
\]

With an exactly Gaussian estimate and known covariance, the null law is
\(\chi^2_{q_j}\). With an asymptotically Gaussian estimate and a consistent
covariance estimate, the same calculation is an asymptotic calibration:

\[
p_j(\theta)=\Pr\{\chi^2_{q_j}\ge Q_j(\theta)\}.
\]

The public convention `df=None` selects this known/asymptotic covariance
route. Calling a covariance estimate “known” in software does not make its
sampling approximation exact.

For independent multivariate normal observations, an estimated sample-mean
covariance gives the one-sample Hotelling law. With \(\nu=n-1\),

\[
\frac{\nu-q+1}{q\nu}Q(\theta_0)
\sim F_{q,\nu-q+1},\qquad
p(\theta)=\Pr\!\left\{F_{q,\nu-q+1}
\ge\frac{\nu-q+1}{q\nu}Q(\theta)\right\}.
\]

This is the usual \(q(n-1)/(n-q)\) Hotelling-to-F conversion. Its exact
sampling interpretation requires the normal-sample assumptions, including
the covariance estimator's relationship to the mean; supplying a matrix and
a number called `df` is insufficient by itself.
[NIST's Hotelling reference](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm)
states this distributional relationship. The package evaluates its upper
tail with SciPy's `f.sf`, whose two shape arguments are numerator and
denominator degrees of freedom, respectively.
[SciPy F documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html).

The law requires \(\nu>q-1\), and the observed covariance must be positive
definite. DAC therefore needs \(n>q\) for every block, as well as a
nonsingular realized scatter matrix. The API preserves positive fractional
degrees of freedom, but their scientific interpretation must come from an
appropriate sampling or approximation argument. The general workflow currently
requires either all `df=None` blocks or all finite-df blocks.

For \(q=1\), \(Q=((\widehat\xi-a^T\theta)/s)^2\), and the formula reduces to

\[
p(\theta)=2\Pr\{t_\nu\ge|\widehat\xi-a^T\theta|/s\}.
\]

Here is an executable check of covariance calibration, the scalar
Hotelling/t identity, and the one-study DAC interval:

```python
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import f, t
from heavily_right.calibration import covariance_of_mean
from heavily_right.dac import fit_dac

x = np.array([0.4, 1.2, -0.1, 0.8, 1.1, 0.3])
n = x.size
variance = covariance_of_mean(x[:, None])[0, 0]
assert_allclose(variance, x.var(ddof=1) / n)
statistic = (x.mean() - 0.2) / np.sqrt(variance)
assert_allclose(f.sf(statistic**2, 1, n - 1),
                2 * t.sf(abs(statistic), n - 1))
fit = fit_dac(x[:, None])
interval = fit.support_interval([1.0])
radius = t.isf(0.025, n - 1) * np.sqrt(variance)
assert interval.success
assert_allclose([interval.lower, interval.upper],
                [x.mean() - radius, x.mean() + radius], atol=1e-10)
```

## Positive heavy-right scores and their null distributions

For \(m>1\), choose fixed positive block weights \(w_j\) summing to one.
They are combination weights; equal block weights are the default, and
they are not automatically inverse-variance weights. A correlated
multidimensional study receives one weight for its one block p-value.

The two general inference methods use

\[
H(p)=\cot(\pi p/2),\qquad R(p)=1/p,
\]

\[
G_H(\theta)=\sum_{j=1}^m w_jH(p_j(\theta)),\qquad
G_R(\theta)=\sum_{j=1}^m w_jR(p_j(\theta)).
\]

The API names are `HCauchy` for the half-Cauchy combination test (HCCT),
and `EHMP` for finite-number calibration of the reciprocal score. The
ordinary weighted harmonic mean of p-values is \(1/G_R\); the internal
score is its reciprocal so that larger scores always mean more evidence
against a candidate parameter. Mathematically \(H(1)=0\) and \(R(1)=1\),
so the universal score baselines are 0 and 1. At zero p-value both
transformations diverge. Floating-point evaluation at the endpoints can
differ from these mathematical identities by rounding or clipping.

The name `HCauchy` does not denote the signed Cauchy transformation
\(\tan(\pi(1/2-p))\). In particular, \(H(p)\ge0\) throughout \((0,1]\):
a large p-value cannot contribute a large negative score that cancels a
small p-value. `CombinationTest` also exposes a distinct `Cauchy` method;
it is not the score used by the general HCCT confidence-region workflow.

The null transformations can be derived directly. If \(U\) is uniform on
\((0,1)\), then for \(x>0\),

\[
\Pr\{H(U)>x\}=\frac2\pi\arctan(1/x),\qquad
f_{H(U)}(x)=\frac{2}{\pi(1+x^2)}.
\]

Thus \(H(U)\) is standard half-Cauchy. For \(x\ge1\),

\[
\Pr\{R(U)>x\}=1/x,\qquad f_{R(U)}(x)=x^{-2}.
\]

For independent study p-values under the common null, the reference score
is the weighted sum of independent variables with the corresponding law.
For example, its characteristic function factors as
\(\phi_G(t)=\prod_j\phi_{H(U)}(w_jt)\) for HCCT. The finite-number routines
evaluate inversion integrals for that distribution and obtain quantiles by
root finding. They do not replace a sum of half-Cauchy variables by a
single half-Cauchy variable.

Write \(F_w\) for this score distribution. The global p-value and cutoff are

\[
p_{\rm global}(\theta)=1-F_w(G(\theta)),\qquad
c_{\alpha,w}=F_w^{-1}(1-\alpha).
\]

Exact uniform null p-values and independence are the assumptions underlying
this exact finite-number reference law. For independent conservative
p-values the decreasing score transformations provide a stochastic
comparison, not equality with that law. If weights are selected using the
same evidence being combined, fixed-weight calibration requires a separate
justification; the code does not supply one.

[`CombinationTest`](../heavily_right/combination.py) uses these conventions:

| Number of study blocks | `HCauchy` | `EHMP` | `HMP` |
|---|---|---|---|
| One | Original study p-value | Original study p-value | Original study p-value |
| 2 through 1,000 | Finite-number independence law | Finite-number independence law | Landau approximation |
| More than 1,000 | Landau approximation | Landau approximation | Landau approximation |

The general fit interfaces expose `HCauchy` and `EHMP`; `HMP` is a
combination-layer method. For multiple tests, `HMP` and `EHMP` use the same
reciprocal score but can use different calibrations. For clarity about the
implemented asymptotic convention, define the weight entropy
\(\mathcal H_w=-\sum_jw_j\log w_j\), and let \(\gamma\) be Euler's constant.
The package calls its own `Landau` implementation with

\[
(\mathrm{loc},\mathrm{scale})=
\begin{cases}
\bigl(2(\mathcal H_w+1-\gamma)/\pi,\;1\bigr),&\text{HCCT},\\
\bigl(\mathcal H_w+1-\gamma,\;\pi/2\bigr),&\text{EHMP/HMP}.
\end{cases}
\]

This specifies the code's parameterization, including its retained location
and scale convention. It should not be substituted unchanged into a different
library's stable-law parameterization. The switch at 1,000 is an
implementation choice, not a coverage theorem or a uniform error bound over
all weight vectors. In particular, a large count with a few dominant weights
does not by itself establish a good asymptotic approximation.

For **one study**, the package deliberately stores `score = -p` and
`threshold = -alpha`. This makes acceptance equivalent to \(p\ge\alpha\)
for every method without performing a combination calibration. Consequently,
one-study score values are not on the HCCT or EHMP multi-study score scale.

## Confidence-region inversion and directional intervals

The joint confidence region is

\[
\mathcal C_\alpha=\{\theta:G(\theta)\le c_{\alpha,w}\}.
\]

Under the stated exact independent-p-value model and exact arithmetic,
\(\Pr_{\theta_0}\{\theta_0\in\mathcal C_\alpha\}=1-\alpha\) for a
continuous reference distribution. Asymptotic study calibration, asymptotic
combination calibration, numerical approximation, or model misspecification
changes what can be claimed about this probability.

For a nonzero direction \(a\), define the support function

\[
h_{\mathcal C}(a)=\sup_{\theta\in\mathcal C}a^T\theta.
\]

The directional interval is

\[
I(a)=[-h_{\mathcal C}(-a),\;h_{\mathcal C}(a)].
\]

Every nuisance coordinate is allowed to vary in these optimizations. A
coordinate slice fixes nuisance coordinates and intersects a line with the
region; it can be shorter or empty and is not the same calculation. Likewise,
separate pairwise fits generally do not project the same joint region.

The simultaneous interpretation follows from set inclusion: on the event
\(\theta_0\in\mathcal C_\alpha\), every linear contrast \(a^T\theta_0\)
belongs to its projected interval, for all directions at once. Each interval
therefore need not have exactly \(1-\alpha\) marginal coverage; the joint
region supplies the common coverage event. This argument concerns exact
supports of one valid region and does not erase failed numerical endpoints
or the effects of subsequent data-dependent study removal.

There is a useful analytic reference case. For one study with full-column-rank
\(A\), put

\[
J=A^TV^{-1}A,\qquad
\theta_c=J^{-1}A^TV^{-1}\widehat\xi,\qquad Q_0=Q(\theta_c).
\]

Completing the square gives
\(Q(\theta)=Q_0+(\theta-\theta_c)^TJ(\theta-\theta_c)\).
If \(q_\alpha\) is the chi-square cutoff, or the Hotelling-scaled F cutoff,
and \(q_\alpha>Q_0\), then

\[
I(a)=a^T\theta_c\ \mathbin{\pm}
\sqrt{(q_\alpha-Q_0)a^TJ^{-1}a}.
\]

The package uses a separate analytic ellipsoid path for this case. When
\(A=I\), \(Q_0=0\), \(J^{-1}=V\), and this becomes the familiar ellipsoid
projection formula. Distribution validity, rather than multi-study
score convexity, is sufficient for that analytic path.

## NMA design and shared-arm covariance

Choose a reference treatment and write its effect as zero. The parameter
vector contains the other treatment effects relative to that reference.
Each contrast row estimates `treat1 - treat2`: its design row has +1 in the
first treatment's coordinate, -1 in the second's, and no coordinate for the
reference. For treatments reference, A, and B, the contrasts A-reference,
B-reference, and B-A have rows

\[
(1,0),\qquad(0,1),\qquad(-1,1),
\]

respectively. Network connectivity to the reference is the corresponding
identifiability requirement. A named contrast is projected from the fitted
joint region by passing its design row as the support direction.

Several contrasts from one trial can be dependent even when different trials
are independent. The default constructor groups repeated study IDs and
requires their joint covariance; the group contributes one Hotelling or
Gaussian block p-value. A complete set of pairwise contrasts within a
multi-arm trial is redundant. Supply an independent contrast basis and its
covariance, rather than labeling redundant rows as separate studies.

The arm-summary constructor assumes independent estimated arm means. If arm
0 is the first arm within a trial and \(s_i\) is the standard error of arm
\(i\)'s estimated mean, then the vector of differences from arm 0 has

\[
\widehat\xi_i=\widehat\mu_i-\widehat\mu_0,\qquad
V=\operatorname{diag}(s_1^2,\ldots,s_q^2)+s_0^2\mathbf1\mathbf1^T.
\]

The off-diagonal term is the variance of the shared baseline estimate.
The local baseline need not equal the network reference. Paired or otherwise
dependent arms need an appropriately derived covariance instead. A common
Hotelling df is not inferred from arm SEs: arbitrary separately estimated
arm variances do not automatically have the covariance sampling law required
for exact Hotelling calibration.

This example checks both the shared-arm covariance and an analytic support
interval. The variance of B-A excludes the shared baseline variance, although
that variance remains in the joint two-dimensional covariance:

```python
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import chi2
from heavily_right.network import network_arm_studies, fit_nma

data = network_arm_studies(
    mean=[0.0, 0.2, 0.5], se=[0.1, 0.2, 0.15],
    treatment=["control", "A", "B"], study_id=["trial"] * 3,
    reference="control", treatments=["control", "A", "B"],
)
study = data.studies.studies[0]
expected_covariance = np.diag([0.2**2, 0.15**2]) + 0.1**2
assert_allclose(study.covariance, expected_covariance)
direction = data.contrast_direction("B", "A")
assert_allclose(direction, [-1, 1])
assert_allclose(direction @ study.covariance @ direction, 0.2**2 + 0.15**2)
interval = fit_nma(data).contrast("B", "A")
radius = np.sqrt(chi2.isf(0.05, 2) * (0.2**2 + 0.15**2))
assert interval.success
assert_allclose([interval.lower, interval.upper], [0.3 - radius, 0.3 + radius])
```

Setting `multiarm="independent"` changes the model to independent scalar
contrast blocks. It is an explicit additional assumption, not an estimate or
correction of the missing within-trial covariance. The constructor's
covariance support concerns dependence within a block; the combination law
still needs its assumptions about dependence between blocks.

## DAC is a coordinate partition

For an observation-by-coordinate sample \(X\in\mathbb R^{n\times d}\), DAC
chooses disjoint column sets \(B_1,\ldots,B_m\) whose union is all coordinates.
The study projection \(A_j\) selects coordinates in \(B_j\), and the study
summary is that block's sample mean, covariance \(W_j/[n(n-1)]\), and
\(\nu_j=n-1\). All blocks use all observations. This interface does not split
the observations into computational shards.

Explicit index/name partitions preserve their supplied order. With
`block_size=k`, consecutive blocks are formed and the last may be shorter;
columns are not repeated to fill it. A one-column sample reduces to scalar
inference. Each block covariance must be positive definite; finite-df
convexity can impose stronger sample-size requirements than covariance
invertibility alone.

A partition does not make correlated coordinates independent. Independence
of the full block data across \(B_j\), for example independent coordinate
blocks in independent multivariate normal observations, is sufficient to
make their block p-values independent. Zero cross-covariance alone is not
generally independence outside an appropriate distributional model. The
finite-number combination law does not provide a general correction for
correlated DAC blocks. An independence-based cutoff applied there must not be
reported as an exact-coverage guarantee merely because the covariance within
each block was computed correctly.

## Supported convexity and scalar cusps

The multi-study support calculation relies on convexity of the supplied
score. The active implementation accepts these sufficient finite-df regimes:

| Block dimension \(q\) | HCCT | EHMP |
|---|---:|---:|
| 1 | \(\nu\ge1\) | \(\nu\ge1\) |
| 2 | \(\nu\ge2.5\) | \(\nu\ge2\) |
| 3 or more | \(\nu\ge q+1\) | \(\nu\ge q\) |

These are the package's supported sufficient conditions, not a claim that
every excluded case is nonconvex, and not a proof of necessity or optimal
constants. They are stronger than \(\nu>q-1\), which only makes the F law
well defined. Known-covariance Gaussian scores are also supported. The
one-study analytic ellipsoid exception was described above. The checks are
implemented in [`projected_region.py`](../heavily_right/projected_region.py).

The relevant geometry can be seen from a whitened residual
\(r=\|V^{-1/2}(A\theta-\widehat\xi)\|\) and a radial score \(h(r)\).
For \(r>0\), with residual unit vector \(v\), its Hessian in whitened
residual coordinates is

\[
h''(r)vv^T+\frac{h'(r)}r(I-vv^T).
\]

Nonnegative \(h'\) and \(h''\) therefore imply convexity, which is preserved
by affine composition and positive weighted sums. To see the derivatives
used numerically, let \(p(r)\) be the radial survival probability and
\(f(r)=-p'(r)\). Away from zero,

\[
h_R'(r)=\frac{f(r)}{p(r)^2},\qquad
h_R''(r)=\frac{f'(r)}{p(r)^2}+\frac{2f(r)^2}{p(r)^3},
\]

\[
h_H'(r)=\frac\pi2 f(r)\csc^2\!\left(\frac{\pi p(r)}2\right),
\]

\[
h_H''(r)=\frac\pi2\csc^2\!\left(\frac{\pi p(r)}2\right)
\left[f'(r)+\pi f(r)^2\cot\!\left(\frac{\pi p(r)}2\right)\right].
\]

The normal radial density has logarithmic derivative
\((q-1)/r-r\); under the displayed Hotelling parameterization it is
\((q-1)/r-(\nu+1)r/(\nu+r^2)\). These identities explain the analytic
derivative implementation without establishing every global convexity
inequality in the table.

For a scalar block the signed whitened residual is \(u=a^Tz-b\), and its
contribution is \(h(|u|)\). At \(u=0\), a nonzero right derivative
\(h'(0+)\) gives a cusp. Its subgradients are

\[
\{\beta h'(0+)a:-1\le\beta\le1\}.
\]

Selecting a zero derivative arbitrarily can miss a stationary minimum or
support point, and there need not be a classical Hessian at the cusp. The
code uses exact scalar radial derivatives away from zero, bounded
subgradient coefficients at active cusps, and, when needed, epigraph
variables \(t\ge u\), \(t\ge-u\) or optimization on cusp hyperplanes.
Nearby incompatible hyperplanes are not treated as one exact intersection.

A particularly transparent identity occurs for scalar \(t_1\) p-values:
\(p(r)=2\arctan(1/r)/\pi\) for \(r>0\), so \(H(p(r))=r\). A weighted
HCCT score then becomes a weighted absolute-deviation objective. This also
shows why a score minimizer need not equal a least-squares center:

```python
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import t
from heavily_right.scores import half_cauchy_score
from heavily_right.workflows import Study, StudyCollection, fit_studies

radii = np.array([0.0, 0.1, 1.0, 5.0])
assert_allclose(half_cauchy_score(2 * t.sf(radii, 1)), radii, atol=1e-14)
inputs = StudyCollection(
    (Study(0.0, 1.0, [1.0], df=1), Study(0.7, 1.0, [1.0], df=1)),
    ("theta",),
)
fit = fit_studies(inputs, weights=[0.9, 0.1])
assert_allclose(fit.estimate, [0.0], atol=1e-10)
assert_allclose(fit.score([0.0]), 0.07, atol=1e-12)
# The weighted quadratic center would instead be 0.9 * 0 + 0.1 * 0.7 = 0.07.
```

## Numerical coordinates are not an estimator definition

The vector solver constructs a weighted information matrix and center,
component by component,

\[
J=\sum_jw_jA_j^TV_j^{-1}A_j,\qquad
\theta_c=J^{-1}\sum_jw_jA_j^TV_j^{-1}\widehat\xi_j.
\]

It uses a diagonal scale \(D\) with entries
\(D_{kk}=\sqrt{(J^{-1})_{kk}}\), and represents physical coordinates by
\(\theta=\theta_c+Dz\). This improves scale balance; it is not full matrix
whitening and cannot remove every form of ill-conditioning.

The reported multi-study point estimate is a validated minimizer of the
combined score, \(\widehat\theta\in\arg\min_\theta G(\theta)\). The GLS-like
center above provides numerical coordinates and an initial candidate. It
does not replace that minimization or turn the method into conventional
weighted least squares. Multiple score minimizers can exist, as the
absolute-deviation identity illustrates.

When the projection structure separates into disconnected coordinate
components, the score separates into component contributions. Inactive
components must be held at validated minima to maximize the score budget
available to an active support direction. The code normalizes weights within
each component for minimum validation, then recombines contributions in the
original score units. Structural separation used by the optimizer is a
different question from stochastic independence used for calibration.

## KKT equations, Newton steps, and penalty continuation

In standardized coordinates a directional upper endpoint solves
\(\max_z b^Tz\) subject to \(G(z)\le c\), where \(b=Da\) for a physical
direction \(a\); the constant \(a^T\theta_c\) is restored afterward. For a
smooth regular boundary point, minimizing \(-b^Tz\) gives the Lagrangian

\[
\mathcal L(z,\lambda)=-b^Tz+\lambda(G(z)-c).
\]

Stationarity and an active boundary yield

\[
-b+\lambda\nabla G(z)=0,\qquad G(z)=c,\qquad \lambda>0.
\]

The code writes these equations as
\(\nabla G(z)=\eta b\), with \(\eta=1/\lambda>0\). The sign is essential:
the opposite boundary can satisfy an unsigned alignment check. A zero
multiplier does not certify a regular support maximum. The general convex
KKT framework, and the role of constraint qualifications for necessity, are
described in
[Boyd's Stanford lecture on duality and KKT conditions](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html).

For this particular problem, sufficiency has a short direct argument. If a
valid subgradient \(g\in\partial G(z_*)\) satisfies \(g=\eta b\),
\(\eta>0\), and \(G(z_*)=c\), then any feasible \(y\) satisfies

\[
c\ge G(y)\ge G(z_*)+g^T(y-z_*)
=c+\eta b^T(y-z_*).
\]

Hence \(b^Ty\le b^Tz_*\). This is a mathematical implication with exact
values and a valid subgradient. Small floating-point residuals are a
numerical approximation to it, not automatically a rigorous bound on error
in the support value.

At a smooth candidate, Newton's method linearizes both equations together:

\[
\begin{pmatrix}\nabla^2G(z)&-b\\\nabla G(z)^T&0\end{pmatrix}
\begin{pmatrix}\Delta z\\\Delta\eta\end{pmatrix}
=-
\begin{pmatrix}\nabla G(z)-\eta b\\G(z)-c\end{pmatrix}.
\]

Backtracking requires a positive updated \(\eta\) and a smaller residual
merit. A bracketed radial search from a feasible point supplies an initial
boundary candidate. Newton solves equations; “KKT” names the conditions
being solved. At cusps, the bounded-subgradient/manifold route replaces
an invalid smooth Hessian calculation.

The actual score normalization matters when reading solver formulas. With
a feasible starting point \(z_0\), the solver sets

\[
s=c-G(z_0)>0,\qquad
\widetilde G(z)=1+\frac{G(z)-c}{s},\qquad
\widetilde c=1,\qquad u=b/\|b\|.
\]

At a boundary start with zero slack it instead tries
\(s=\|\nabla G(z_0)\|\); if no positive usable scale exists it reports a
nonregular start. The affine score transform preserves the region. If
\(\widetilde\eta\) is the normalized equation's coefficient, the coefficient
in \(\nabla G=\eta b\) is
\(\eta=s\widetilde\eta/\|b\|\). Diagnostics retain the original score,
cutoff, boundary error, this unnormalized \(\eta\), and `score_scale=s`.

When Newton does not produce a checked endpoint, constrained SLSQP and
optional Newton polishing are attempted. The next fallback minimizes the
one-sided quadratic-penalty objective

\[
P_\mu(z)=-u^Tz+\frac\mu2
\left[\max\!\left\{\frac{G(z)-c}{s},0\right\}\right]^2,
\qquad \mu=1,10,\ldots,10^8.
\]

Its gradient is

\[
\nabla P_\mu(z)=-u+
\mu\max\!\left\{\frac{G(z)-c}{s},0\right\}\frac{\nabla G(z)}s.
\]

This is the objective in
[`support.py`](../heavily_right/support.py), expressed before the affine
normalization. It penalizes only violations of the inequality. Each stage
uses L-BFGS-B and warm-starts the next stage. Finite \(\mu\) can leave an
exterior candidate; the implementation intersects the segment from the
feasible start to that candidate with \(G=c\), then checks the original
boundary and stationarity conditions. Radial projection alone is not a
support certificate. The continuation coefficient \(\mu\) is not the KKT
multiplier \(\lambda\), and an optimizer's success flag cannot replace the
endpoint checks.

The current checks require a finite positive \(\eta\), scaled boundary
error at most \(2\times10^{-8}\), and normalized KKT residual at most
\(2\times10^{-6}\), together with the route's success conditions. The
alignment residual minimizes the distance from \(b\) to the span of a
gradient/subgradient; the independent sign check excludes the wrong
orientation. Endpoints are also checked after conversion back to input
units. Large offsets can round away a small displacement that was
representable internally.

Dense Newton has a default active-dimension limit of 250. Its alternative
constrained routes do not make the whole workflow sparse. Limits on
iterations and fallback stages are numerical budgets, not convergence
theorems or an overall wall-clock guarantee. A failed diagnostic means the
endpoint was not established by the implemented checks.

## Minimum-value bounds and three different empty-set states

Finding a feasible point establishes nonemptiness. Finding a candidate
minimum above the cutoff does not establish emptiness: the optimizer might
have stopped too early. The minimum validator therefore checks both
stationarity and a bound on the possible score-value gap.

For a convex score and \(g\in\partial G(z)\),

\[
G(y)\ge G(z)+g^T(y-z).
\]

If a necessary domain is enclosed in \(\|y\|\le R\), this gives the lower
bound

\[
L=G(z)-g^Tz-R\|g\|.
\]

The radius is derived from individual score budgets. Let \(\beta=0\) for
HCCT or \(\beta=1\) for EHMP. If \(G(y)\le c\), then each block obeys

\[
T_j(p_j(y))\le B_j=\frac{c-(1-w_j)\beta}{w_j}.
\]

This implies a lower p-value limit \(1/B_j\) for EHMP, or
\(2\arctan(1/B_j)/\pi\) for HCCT, and hence an upper bound on its
Mahalanobis statistic. In standardized coordinates, summing those weighted
quadratic bounds gives

\[
y^TJy+2\ell^Ty+K\le M,\qquad K\ge0.
\]

If a conservative positive eigenvalue floor is \(\lambda_0\le
\lambda_{\min}(J)\), dropping \(K\) yields the enclosing radius

\[
R=\frac{\|\ell\|+\sqrt{\|\ell\|^2+\lambda_0M}}{\lambda_0}.
\]

The implementation adds numerical margins and can decline to construct
this radius when the eigenvalue or tail calculation is unreliable. For
minimum accuracy it uses a sublevel containing the candidate, takes the
larger of the tangent bound and the universal score baseline, and reports
\(\max\{G(z)-L,0\}\) as a gap bound. Its acceptance tolerance is

\[
\max\{2\times10^{-6},10\,\mathrm{ftol}\}
\max\{1,|G(z)|,|c|\}.
\]

Near a scalar cusp, a chosen cusp subgradient is anchored at residual zero,
which may differ slightly from the represented candidate. The validator
lowers the tangent intercept by an anchor correction. For a weighted
radial contribution at residual \(u\), the correction includes

\[
w\bigl(|h(|u|)-h(0)|+h'(0+)|u|\bigr).
\]

This prevents pretending that an approximate snap produces an exact tangent
at the represented point. Disconnected components are validated separately
so a small global gradient from a weakly weighted direction cannot conceal
a material unused score budget. See
[`minimum.py`](../heavily_right/minimum.py) for the bound assembly and
diagnostic fields.

The reporting distinction is:

| State | What the calculation establishes |
|---|---|
| Nonempty | A candidate in the acceptance region; support additionally needs the validated minima and endpoint checks |
| Certified empty | A usable lower bound over the necessary feasible domain exceeds the cutoff, with the numerical margin |
| Unknown/numerical failure | A minimum, tail, domain, root, or endpoint requirement could not be established |

An endpoint failure can occur after nonemptiness has already been established;
it does not undo the existence of a feasible point. In that situation the
uncertainty concerns the directional bound, not whether the region is empty.

These are numerical certificates, not interval-arithmetic proofs. A very
small gradient is insufficient on its own, and an unknown state must not be
converted into a study-removal decision. The scalar solver applies the
same principle using its one-dimensional convex bounds and derivative or
subgradient checks.

The public multidimensional p-value calculation has a retained floor of
\(10^{-150}\). Flooring would cap a study score, whereas the underlying
convex radial score is unbounded. Consequently the minimum/support code
rejects candidate tails or cutoff domains that require reaching that floor;
single-study levels at or below it are rejected as well. This avoids using
an uncapped convex bound to certify the semantics of a capped public score.
Extreme weights can trigger this guard even with an ordinary overall level.

## Optional removal is a sensitivity analysis

Study removal is disabled by default. With an explicit `EmptySetPolicy`,
the algorithm can remove the smallest-p study at a validated incompatible
minimum, subject to the policy's limits and, in multiple dimensions,
preservation of parameter identifiability. It removes a whole correlated
study block, renormalizes retained weights, recalibrates the cutoff, and
records original indices, labels, minima, candidates, and removals.

The general workflows resolve that intervention during fitting. Subsequent
NMA contrasts or DAC directions use the same retained study set; they do
not select a new subset for each query. This makes their geometric
interpretation coherent as projections of one fitted region.

It does not recover the original fixed-study coverage claim. The retained
set was selected using the same data, so the independent fixed-weight null
calibration is not automatically the conditional law after selection.
Report the full removal record and the result as a sensitivity analysis;
numerical endpoint success is not evidence of nominal post-selection
coverage. Further details are in [the empty-set guide](EMPTY_SETS.md).

## What “exact finite-number” does and does not mean

There are separate sources of approximation: the study's sampling law,
the reference distribution of the combined score, floating-point numerical
evaluation of that reference law, and optimization of the resulting region.
An exact finite-number distribution specifies the reference probability
model; it does not mean exact arithmetic or a uniform numerical guarantee.

The finite-law inversion uses SciPy quadrature with absolute tolerance
\(2\times10^{-13}\), relative tolerance \(2\times10^{-11}\), and subdivision
limit 600. Quantile inversion uses geometric bracketing and Brent's method
with absolute tolerance \(2\times10^{-10}\) and relative tolerance
\(2\times10^{-11}\). Additional finiteness, error-estimate, and probability
range checks can raise typed numerical errors. `precision=True` returns
an estimated absolute numerical error (quadrature plus a rounding allowance
in the half-Cauchy lower-tail route); it does not bound every modeling,
truncation, rounding, or quantile error. See the active
[`_numerics.py`](../heavily_right/_numerics.py) and
[SciPy quadrature documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).

The half-Cauchy lower tail uses an exponentially tilted inverse-Laplace
contour, with a bounded positive-convolution expansion very close to zero.
The tilt solves for a transformed mean equal to the requested score; it is
a numerical change of contour, not a change to the null distribution. The
small CDF is evaluated directly, avoiding cancellation in \(1-\mathrm{SF}\).
An entropy-based location selects a contour; it is not a support boundary.
Explicit zero shortcuts in this route require an analytic upper bound
below floating-point range, with a nonzero error estimate. See
[Numerics](NUMERICS.md#calibration-and-finite-m-laws) for route selection,
underflow handling, and independent validation. The Landau approximation
itself retains its piecewise coefficient approximations. Broad
accuracy claims for extreme quantiles or highly unequal weights require
additional validation beyond ordinary upper-tail examples and solver tests.

One special function used in reciprocal-score inversion is

\[
E(x)=e^{-x}\operatorname{Ei}(x),\qquad x>0.
\]

Evaluating \(\operatorname{Ei}(x)\) first can overflow even when \(E(x)\)
is moderate. The active helper uses SciPy's real `expi` multiplied by
\(e^{-x}\) for \(x<50\). For \(x\ge50\) it sums the asymptotic expansion

\[
E(x)\sim\sum_{k\ge0}\frac{k!}{x^{k+1}}
=\frac1x\left(1+\frac1x+\frac{2!}{x^2}+\cdots\right).
\]

The recurrence is \(t_0=1/x\), \(t_k=t_{k-1}k/x\). The code stops before
terms begin increasing, when a term is at most \(2\times10^{-16}\) of the
accumulated sum, or after its finite cap of 80 terms. This uses an
asymptotic expansion, not a convergent series for all positive arguments
and not the Ramanujan series. The asymptotic identity is documented in
[NIST DLMF, equation 6.12.2](https://dlmf.nist.gov/6.12.E2);
the implementation is
[`_special_functions.py`](../heavily_right/_special_functions.py).

Finally, a small numerical residual can correspond to a larger parameter
error in an ill-conditioned problem. A positive KKT multiplier and passing
residual checks support the reported computation under its assumptions;
they are not a new theorem about arbitrary dependence, tail clipping,
adaptive removal, or all floating-point inputs. For sensitive applications,
the one-study ellipsoid, scalar t, shared-arm covariance, and \(t_1\)-HCCT
identities above provide independent analytic checks of different layers.
