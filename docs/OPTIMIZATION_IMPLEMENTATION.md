# Convex optimization implementation and robustness

This is an implementation guide to the current Python confidence-region
solvers, not a claim of convergence for arbitrary objectives. Read
[Methods](METHODS.md) for the statistical construction and
[Troubleshooting](TROUBLESHOOTING.md) for user-facing remedies. Calibration of
the cutoff is separate from optimization; a better optimizer does not change
the supplied covariance, degrees of freedom, weights, or significance level.

## Source map and supported problem

| Responsibility | Implementation |
| --- | --- |
| Generic support-point KKT/Newton and fallbacks | [support.py](../heavily_right/support.py), `solve_support_point` |
| Projected score, derivatives, components, scalar cusps | [projected_region.py](../heavily_right/projected_region.py), `ProjectedScoreRegion`, `ProjectedScoreComponent` |
| Checked minima and convex lower bounds | [minimum.py](../heavily_right/minimum.py), `minimize_score_region` |
| Analytic ellipsoids, both endpoints, physical-coordinate checks | [multivariate.py](../heavily_right/multivariate.py), `_support_interval_once` |
| Scalar minimum and boundary roots | [univariate.py](../heavily_right/univariate.py) |
| Opt-in empty-set procedure and retained diagnostics | [multivariate_empty.py](../heavily_right/multivariate_empty.py), [empty_sets.py](../heavily_right/empty_sets.py) |

For a direction `b`, support inference solves

```text
maximize b' theta subject to G(theta) <= c.
lower endpoint = -support(-b), upper endpoint = support(b).
```

A study block has residual `r_j=P_j theta-hat_theta_j`, covariance `Sigma_j`,
and quadratic statistic `Q_j=r_j' Sigma_j^-1 r_j`. Known-covariance blocks use
`p_j=chi2.sf(Q_j,q_j)`. A block with residual degrees of freedom `nu_j` uses
`F.sf(a_j Q_j,q_j,nu_j+1-q_j)`, where
`a_j=(nu_j+1-q_j)/(q_j nu_j)`. Multiple-study scores are
`sum w_j cot(pi*p_j/2)` for HCauchy and `sum w_j/p_j` for EHMP.
The one-study internal score is instead `-p`; it has a separate analytic route.

Construction checks finite compatible arrays, symmetric positive-definite
covariances, positive normalized active weights, valid projection indices,
positive Hotelling denominator degrees of freedom, and collective full rank.
For multiple studies, global support certification is restricted to HCauchy/EHMP
and the implemented sufficient convexity regimes:

| Block dimension q | EHMP residual df | HCauchy residual df |
| --- | --- | --- |
| 1 | at least 1 | at least 1 |
| 2 | at least 2 | at least 2.5 |
| greater than 2 | at least q | at least q+1 |

Normal blocks need no residual-df check. A valid F distribution alone is not a
convexity certificate. A full-rank one-study ellipsoid is exempt from these
multi-study restrictions because its sublevel set is handled analytically.

## Coordinates, components, and a genuinely minimized base

Projection rows define an overlap graph: coordinates occurring in the same
block are joined using **exact nonzero entries**, not a numerical coefficient
cutoff. Each connected component supplies a weighted information matrix and
right-hand side. The generalized least-squares center `theta_0` solves that
system; coordinate scales are square roots of the diagonal of its inverse.
Every computed finite positive scale is retained, including very small scales.
Optimization uses `theta=theta_0+D z`, with diagonal `D`; this is diagonal
scaling, **not full whitening** of a rotated ill-conditioned matrix.

Only components touched by `b` enter an endpoint solve. Other components remain
at a checked minimum and contribute a constant score. This is why a merely
feasible base is insufficient: unminimized inactive coordinates can consume
score budget and falsely narrow an interval. Even `find_min=False` does not
disable independent minimum validation. The preliminary Powell/default search
is a starting-point provider, not a certificate.

`minimize_score_region` handles disconnected components in their own normalized
weight units, then aggregates contributions in the original units. A child's
one-study `-p` score is never substituted for its parent's HCauchy/EHMP term.
Compatible centers and one-study minima have analytic paths. Otherwise scalar
absolute residuals are lifted to epigraph variables, and scaled SLSQP minimizes
the smooth lifted objective with linear inequalities. At most three optimizer
passes share the supplied `maxiter` budget; objective rescaling after a large
score decrease does not reset that budget. These passes use SLSQP
`ftol=min(requested_ftol,1e-14)`. Checked starts can return with zero
optimization iterations.

A minimum needs both a small normalized valid subgradient and a conservative
objective-gap bound. For `tau=max(2e-6,10*ftol)`, the stationarity threshold is
`tau`, with gradient normalized by `max(1,abs(c),abs(G))`; the gap tolerance is
`tau*max(1,abs(G),abs(c))` (omit c when not supplied). A near-cusp subgradient
anchored at zero includes an intercept correction for residual floating-point
mismatch. Small stationarity alone is not enough in weakly weighted directions.

The lower-bound construction in [minimum.py](../heavily_right/minimum.py) is:

```text
Necessary study bounds imply z' A z + 2 h' z + q0 <= S, with q0 >= 0.
R = (||h|| + sqrt(||h||^2 + lambda*S))/lambda, lambda <= lambda_min(A).
For a valid subgradient g at x:
G(z) >= G(x) - anchor_correction - g' x - R*||g||, for ||z|| <= R.
```

The implementation reduces the eigenvalue bound and subtracts numerical safety
margins. It uses a sublevel containing the candidate for its minimum-gap bound,
and the requested cutoff's necessary domain for an emptiness certificate.
Emptiness requires the latter bound to exceed `c+1e-8*max(1,abs(c))`, not just
an optimizer's above-cutoff answer. Unavailable bounds or unvalidated minima
produce a numerical failure, not permission to delete studies.

The public multidimensional p-value floor is `1e-150`. Minimum validation checks
both candidate tails and whether the **whole requested cutoff domain** can
reach that floor. Otherwise the unclipped convex kernel could certify a
different problem from the public clipped score. Such domains are refused,
including analytically centered/symmetric cases with extreme weights.

### Aggregate minimum error budgets

Local success is not enough for a disconnected score `G=sum(a_j*G_j)`:
the weighted local minimum-gap bounds must also sum to a valid parent bound.
Local cutoffs grow as `1/a_j`, so otherwise valid local tolerances can together
exceed the parent tolerance. If this happens with individually valid children,
the code reserves its lower-bound rounding margin and reconstruction discrepancy,
then allocates half the remaining parent gap budget equally among multistudy
components in **parent score units**. Only loose children are refined, with
their allocation divided by `a_j` as a stricter local absolute gap cap.

The cap controls initial-point acceptance, optimization stopping, and final
certification. Each retry shares its component's original `maxiter`; an invalid
retry cannot promote a child to success. The reconstructed parent score and
gap are checked again with the **unchanged parent tolerance formula**, evaluated
at the reconstructed parent score. Analytic
one-study components and already-valid aggregates incur no extra solve.
`MinimumDiagnostics.component_refinements`, child work counts/messages, and
the aggregate gap expose this recovery. Refinements count attempted retries,
including rejected ones. A retry must be independently successful and strictly
improve its gap; otherwise the old valid child is retained. Work counts include
both attempts even when the retry is rejected. The
[safeguards summary](RECENT_FIXES.md#aggregate-minimum-error-budgets) gives the exact
reserve/allocation formulas and distinguishes aggregate checks from cusp stalls.
An exhausted or still-invalid budget
remains a numerical failure, never permission to remove a study.

See the [component-minimum regressions](../tests/test_component_minimum_budget.py)
for successful refinements, invalid retries, and unchanged final tolerances.

## Analytic one-study support and symmetry

For one identified block, let `M=P' Sigma^-1 P`, let `Q_min` be its quadratic
residual at `theta_0`, and let `q_alpha` be the chi-square cutoff or the F cutoff
divided by `a_j`. With `B=q_alpha-Q_min>0`, the physical endpoints are

```text
theta_plus/minus = theta_0 +/- sqrt(B/(b' M^-1 b)) * M^-1 b.
```

The implementation evaluates this in standardized coordinates and independently
checks both endpoints. Negative B establishes emptiness; zero B raises a
degenerate-boundary error rather than pretending an ordinary positive-multiplier
certificate applies. One-study levels at/below `1e-150` are refused. Its
boundary scale is `min(level,abs(c-G_min))`, protecting small tail probabilities.
This route also avoids an important penalty pathology: `G=-p` is bounded, so a
finite exterior penalty on its violation need not control an unbounded linear
objective, even though the confidence ellipsoid itself is bounded.

For multiple studies, the region is nominated centrally symmetric only if the
maximum Mahalanobis residual at the GLS center is at most `1e-10`. This test
does not grow with an arbitrary common translation. An upper endpoint is
reflected only after it passes; the reflected lower endpoint receives its own
score, gradient, positive-multiplier and stationarity checks. Failed reflection
triggers a separate lower support solve. Both endpoints subsequently undergo
the physical-coordinate check below.

## Generic support scaling and the KKT system

`solve_support_point` safely normalizes the direction by its largest absolute
entry and then its Euclidean norm. Zero/nonfinite directions and a norm beyond
floating-point range are rejected. If `z0` is the finite feasible start, define

```text
s = abs(c-G(z0)); use ||grad G(z0)|| only when this slack is zero.
G_tilde(z) = 1 + (G(z)-c)/s; cutoff_tilde = 1; direction = unit b.
```

Zero slack and zero gradient give `success=False`, solver `nonregular-start`;
there is no general singleton solver. Positive multiplication of the original
score or a change in direction magnitude must not weaken the acceptance rule.
The physical support value may be negative; its **sign is not a certificate**.

A radial start follows the unit direction from the feasible point. The distance
starts at at most 1, doubles until the boundary is bracketed, and is capped at
`1e6` standardized units. Brent's method uses `xtol=finfo(float).tiny`,
`rtol=1e-11`, and at most 256 iterations. Failure to bracket is explicit.

In the following equations `G,c,b` denote the scaled problem. The multiplier
convention is `grad G=eta*b`, so `eta>0` is the reciprocal of the usual positive
constraint multiplier. Newton solves

```text
F(z,eta) = [grad G(z)-eta*b; G(z)-c] = 0,
[ Hessian G(z)  -b ] [delta_z  ] = -F(z,eta).
[ grad G(z)'     0 ] [delta_eta]
```

The initial eta is `max(grad G'b/(b'b),machine_epsilon)`. A singular direct
linear solve falls back to least squares with `rcond=1e-12`. Nonfinite curvature
or numerical derivative/linear-algebra failure disables that Newton attempt;
it does not disable the independent first-derivative fallbacks. Programming
errors in callbacks are not generally swallowed.

The merit is the Euclidean combination of `||grad G-eta*b||/||b||` and
`(G-c)/max(abs(c),tiny)`. Each step tries at most 24 halvings, accepting only
finite points/curvature, positive eta, and **strictly smaller** merit. No Armijo
constant or trust-region claim is implied. Newton's stopping threshold is
`max(ftol,2e-9)`; endpoint acceptance is a separate check.

For q>1, analytic derivatives use `v=P_scaled' Sigma^-1 r` and the radial
transform derivatives with respect to Q:
`grad G_j=2*w_j*T'_j*v` and
`H_j=w_j*(2*T'_j*P_scaled' Sigma^-1 P_scaled+4*T''_j*v*v')`.
Finite chi-square/F density derivative limits, including q=4 at zero, are
implemented. Scalar q=1 terms instead use exact normal/t radial derivatives in
absolute standardized residual units; the singular quadratic-density route
would create an artificial flat neighborhood and cancellation.

## Checked fallback sequence

The regular sequence in [support.py](../heavily_right/support.py) is:

```text
radial start -> KKT Newton -> scaled SLSQP -> Newton restart
             -> quadratic-penalty continuation -> Newton restart
             -> bounded nonsmooth active-set / epigraph discovery if needed
```

Successful candidates stop the regular sequence; optional near-cusp refinement
also runs around its candidate checks. A disabled dense Newton stage is skipped.

SLSQP minimizes `-unit_b'z`, subject to
`(c-G(z))/max(abs(c),1)>=0`, with analytic gradients and an exact-point cache.
It starts from the last finite Newton point or the radial start. Normal solver
termination alone does not suffice. If its candidate fails the common
certificate, a Newton restart attempts to polish it before penalty continuation.

The second fallback uses L-BFGS-B on

```text
-unit_b'z + (mu/2)*max((G(z)-c)/max(abs(c),1),0)^2,
mu = 1,10,...,1e8 (nine stages).
```

Continuation warm-starts from finite penalty candidates. An exterior candidate
is projected back to the boundary along its segment from the feasible base
using Brent's method (`xtol=rtol=1e-11`). Boundary projection alone is not
optimality: the resulting point must still pass stationarity and positive eta,
and the accepting penalty stage must report success. The best unsigned merit
candidate is tracked for possible polishing, but when a candidate passes, that
**same candidate and its diagnostics** replace the stored result. This prevents
a wrong-side point with negative eta from inheriting another point's success.
Numerical exceptions allow later stages or an unsuccessful result, never an
unchecked endpoint. The compatibility `solver="legacy"` penalty implementation is
separate, explicitly warned as uncertified, and is not this automatic fallback.

## Genuine scalar cusps and bounded active-set recovery

At a zero scalar residual, the classical Hessian does not exist: the score
routine selects the zero subgradient and marks the Hessian unavailable. A valid
support certificate may instead select any subgradient in the cusp interval.

`certify_support_candidate` nominates scalar hyperplanes whose geometric
distance `abs(standardized_residual)/||standardized_row||` is at most `1e-8`.
A joint least-squares projection must move by at most `1e-8` and satisfy the
equalities to `64*eps*max(1,||point||)`. Nearby incompatible planes are rejected.
When allowed, a nullspace Newton polish optimizes along the exact manifold;
actual residuals are checked again afterward. A bounded least-squares solve
then chooses cusp coefficients in `[-1,1]` and positive eta. Original boundary,
support-value change, and the final KKT residual must still pass. Nomination or
snapping is not itself a certificate.

If this local refinement is insufficient, `solve_nonsmooth_support_candidate`
performs a **new solve**, allowed to move farther: it tries the at most 15
nonempty subsets of the four closest scalar hyperplanes. If necessary, an
epigraph problem introduces `t_j>=|a_j'z+r_j|` and discovers more simultaneous
cusps. Lifted SLSQP only nominates an active set (`t_j<=sqrt(1e-11)`); an exact
manifold solve and original-score subgradient checks must certify the result.
Labels include `active-set-newton` and `epigraph-active-set-newton`, with
`certificate_kind="subgradient-kkt"`. Dense active-set discovery is disabled
when dense Newton is disabled, and has its own 250-coordinate cap; the lifted
problem additionally has a 500-variable cap.

## Acceptance, reconstruction, and diagnostics

Every regular endpoint needs finite support value, score/error, eta and KKT
residual; an acceptable generating stage or independently validated refinement;
`eta>0`; scaled boundary error at most `2e-8`; and KKT residual at most `2e-6`.
The diagnostic residual is the relative distance of b from the gradient span:

```text
lambda_hat = (b'g)/max(g'g,tiny),
kkt_residual = ||b-lambda_hat*g||/max(||b||,tiny).
```

For a cusp, g is the selected valid subgradient. A tiny unsigned residual at the
opposite endpoint is insufficient because eta would have the wrong sign. In
the original score units the boundary check is `abs(G-c)/score_scale<=2e-8`.
The wrapper reevaluates that original score after solving the normalized problem.

`MetaAnalysisMD` also evaluates the public score at the **actual returned
physical point**, not just the more accurate standardized representation.
Adding a huge center can round away a displacement; then success is revoked
and the message requests recentering/rescaling. `reconstruction_error` records
the score discrepancy between the two representations. This is a boundary
recheck, not a second full KKT solve in physical coordinates.

`SupportDiagnostics` records `success`, `solver`, `certificate_kind`, original
`score`, `cutoff`, `boundary_error`, `kkt_residual`, `eta`, `score_scale`,
`reconstruction_error`, iteration/evaluation/line-search counters and messages.
The interval retains both endpoint diagnostics, minimum diagnostics, active
dimension and whether reflection was used. `support_interval` can return an
unsuccessful endpoint result for inspection; `simultaneous_interval` raises
instead of returning the legacy tuple when either endpoint fails. Numerical
certificates use floating-point tolerances, not interval-arithmetic proofs.

This small, independently runnable example inspects both certificates without
writing any files; covariance inputs describe the study estimates themselves.

```python
import numpy as np
from heavily_right import MetaAnalysisMD

analysis = MetaAnalysisMD(2, dim=2)
result = analysis.support_interval(
    [1.0, -1.0], xi_hat=[[0.2, 0.4], [0.2, 0.4]],
    Sigma=[0.04 * np.eye(2)], sub_dim=2,
)
assert result.success
for diagnostic in (result.lower_diagnostics, result.upper_diagnostics):
    assert diagnostic.success and diagnostic.eta > 0
    print(diagnostic.solver, diagnostic.certificate_kind,
          diagnostic.boundary_error, diagnostic.kkt_residual)
```

## Work limits and the large-component boundary

| Stage/control | Current default or fixed limit |
| --- | --- |
| Main support controls | `ftol=1e-9`, `maxiter=1000`, `newton_maxiter=80` |
| Each ordinary Newton attempt | `min(maxiter,newton_maxiter)`; 24 backtracking trials per iteration |
| SLSQP and each of nine penalty stages | each receives `maxiter`; not a shared global budget |
| Dense support cutoff | configurable `dense_newton_max_dimension=250` |
| Near-cusp/active-manifold Newton | at most 80 iterations per attempt, `ftol=1e-11`, at most 250 coordinates |
| Bounded cusp enumeration | at most 15 subsets; epigraph discovery at most 500 variables and 1000 SLSQP iterations |
| Cusp bounded least squares | `tol=1e-13`, at most 1000 iterations |
| Minimum refinement | up to three scaled SLSQP passes per solve; any aggregate-gap retry shares that component's original `maxiter` |

Larger active components set the support Newton budget to zero and genuinely
avoid its dense Hessian callback, including the optional cusp polish. They use
first-derivative constrained/penalty methods. **This is not a fully matrix-free
or sparse solver:** SLSQP may use dense internal storage, and construction,
information matrices, and minimum/domain checks can still allocate dense
component matrices. Sparse Hessians, Hessian-vector/Krylov KKT solvers and full
whitening remain future work, not implemented promises. A decomposable
2000-coordinate problem with only 50 active coordinates is not evidence of
performance on one coupled 2000-coordinate component.

Counters expose accumulated endpoint/refinement work but are not an end-to-end
wall-clock budget or a complete accounting of preliminary minimum work. The
reproduction runner's separate `--max-seconds` bounds whole experiments; public
solver iteration limits do not provide that wall-clock guarantee.

The algorithm uses fixed initialization rules, bounded stage sequences and
deterministic tie breaking rather than random restarts. Deterministic does not
mean successful: the same difficult input can reproducibly fail a certificate.
Nor is bitwise agreement across BLAS/SciPy versions promised; small arithmetic
changes can alter the fallback path while the same acceptance rules apply.

## Scalar intervals and opt-in empty-set intervention

Scalar inference translates by the first estimate and jointly scales estimates
and standard errors by `max(estimate_range,max_SE)`. The minimum search stays
on the estimate hull: bounded search uses `xatol=1e-14`, at most 500 iterations,
followed by at most 100 monotone-subgradient refinement steps and exact checks
of nearby study-estimate cusps. The normalized subgradient tolerance is `1e-7`;
saturated tails (`p<=2e-150`) cannot establish emptiness. Boundary searches allow
200 doubling expansions and 200 Brent iterations, with an internal relative
score check `1e-8` and reconstructed-coordinate check `1e-7`. A candidate
nonempty interval that collapses through rounding or cannot be represented
fails explicitly. These are scalar routines, not a
call to the multidimensional KKT solver.

An `EmptySetPolicy` is absent by default. Its sole strategy removes the smallest
p-value study at a certified empty minimum; ties use original zero-based index.
Defaults are `max_removals=1`, `min_remaining=1`, `warn=True`. Multidimensional
removal skips blocks whose deletion loses projection rank. Candidates are
p-value/eligibility ranked, **not** exhaustively leave-one-out optimized.
An accepted whole-block deletion renormalizes weights, recalibrates the cutoff
and refits. Unknown numerical states never authorize another deletion. Earlier
removals, labels, ranks, p-values, cutoffs and failures remain in the trace.
This adaptive procedure is a disclosed sensitivity intervention; original
nominal coverage is not automatically preserved after data-driven selection.

API contracts differ: a default scalar structured result can retain an empty,
collapsed interval with `empty_set_diagnostics.resolved=False`; this is not a
certified singleton. The low-level multidimensional no-policy path raises for
confirmed emptiness. General [workflow fits](../heavily_right/workflows.py)
reject unresolved fits and fix the retained study set once; later contrast or
support queries cannot trigger fresh removals. Scalar traces currently do not
persist the internal tangent bound as `MinimumDiagnostics`, whereas
multidimensional traces retain the minimum certificate. Validation errors may
raise before any intervention trace exists. Inference does not write files;
trace serialization and exclusive, non-overwriting saving are explicit calls.

## Failure modes, safeguards, and tests

The following failure modes are covered by executable regressions. Test names
identify the corresponding checks in the linked files.

| Failure mode | Current safeguard | Regression |
| --- | --- | --- |
| Senn metformin direction: Newton stall; SLSQP success/near-boundary penalty still nonstationary | scaled constrained solve, Newton restarts, full KKT/subgradient checks | [support tests](../tests/test_support_solver.py): `test_network_meta_analysis_metformin_direction_is_certified`, `test_penalty_is_used_only_after_slsqp_failure` |
| Scalar q=1 clipping invented smooth curvature; multiple NMA cusp directions failed | exact radial derivatives, geometric cusp nomination, bounded active-set and epigraph recovery | [support tests](../tests/test_support_solver.py): `test_q1_derivatives_remain_exact_near_the_residual_cusp`, `test_nearby_incompatible_cusps_are_not_snapped_to_one_point`, `test_epigraph_fallback_handles_five_simultaneous_scalar_cusps` |
| Disabling Newton still evaluated a Hessian | explicit score/gradient-only path | [support tests](../tests/test_support_solver.py): `test_disabled_newton_never_evaluates_a_hessian`; [adversarial tests](../tests/test_optimizer_adversarial.py): `test_redundant_active_cusps_respect_disabled_dense_hessian` |
| Wrong-side point inherited another penalty point's success | accept point and diagnostics together; positive eta | [adversarial tests](../tests/test_optimizer_adversarial.py): `test_wrong_sign_optimizer_success_is_never_certified` |
| Tiny score units falsely certified an exterior point; zero-radius set gained width | slack/direction normalization; explicit nonregular failure | [adversarial tests](../tests/test_optimizer_adversarial.py): `test_tiny_score_units_do_not_certify_a_point_outside_the_set`, `test_singleton_does_not_return_a_large_uncertified_interval` |
| Large translation created false symmetry or erased returned displacement | Mahalanobis symmetry test; independent reflection and physical-point check | [adversarial tests](../tests/test_optimizer_adversarial.py): `test_translated_inconsistent_regions_do_not_become_symmetric`, `test_unrepresentable_offset_endpoint_is_not_reported_certified` |
| Rotated one-block ellipsoid exhausted iterative routes | analytic normal/Hotelling ellipsoid support | [adversarial tests](../tests/test_optimizer_adversarial.py): `test_rotated_ill_conditioned_single_block_matches_ellipsoid` |
| Weak inactive component wasted score and narrowed support despite tiny gradient | normalized component minima and objective-gap bound | [minimum tests](../tests/test_minimum.py): `test_disconnected_weak_component_cannot_consume_support_budget`, `test_connected_weak_direction_recovers_or_fails_explicitly` |
| Individually valid component gaps exceed the parent budget | allocate/refine in parent units; reserve rounding/reconstruction error; retain unchanged final tolerance | [component-budget tests](../tests/test_component_minimum_budget.py): `test_saved_nma_minimum_meets_unchanged_parent_gap_after_refinement`, `test_failed_refinement_cannot_promote_child_or_parent_to_success` |
| Unclipped internal tangent falsely implied emptiness for publicly clipped scores | candidate-tail and entire-cutoff-domain floor guards | [minimum tests](../tests/test_minimum.py): `test_extreme_weight_clipped_domain_cannot_certify_false_emptiness`, `test_extreme_weight_domain_guard_also_applies_to_symmetric_center` |
| Scalar translation broke bracketing; tiny units manufactured emptiness | shared centered/scaled minimum and roots | [scalar tests](../tests/test_univariate_optimization.py), [adversarial tests](../tests/test_optimizer_adversarial.py): `test_scalar_translation_preserves_interval_and_estimate`, `test_scalar_tiny_units_cannot_create_an_empty_set` |
| Optimizer success or later failure triggered an unjustified deletion/lost trace | validated lower bound; unknown distinct from empty; persistent intervention history | [empty-set tests](../tests/test_empty_sets.py): `test_failed_multivariate_minimization_cannot_trigger_deletion`, `test_multivariate_failure_after_removal_retains_complete_trace` |

Scalar cusp endpoints require subgradient certificates rather than a fictitious
smooth Hessian. A small score-boundary error alone is not an independent oracle
for a support width. Forced-failure tests exercise SLSQP and penalty fallbacks,
including routes that ordinary well-conditioned examples may not require.
Analytic one-block ellipsoids, finite-difference derivatives, and independently
constrained small systems provide complementary checks; these bounded tests
do not establish a universal convergence or accuracy guarantee.

## Remaining limits and appropriate next steps

Convexity and identifiability checks, approximate KKT conditions and minimum
bounds do not guarantee convergence for all ill-conditioned data. A finite KKT
residual is not a conditioning-independent support-value error bound or a
promise of a fixed number of accurate digits. Weak connected
directions may still fail their objective-gap check. Dense nonsmooth recovery
examines a bounded set of candidates, not every possible active set. Degenerate
regions, unresolved rank, extreme directions/translations, unrepresentable tails
and clipped-score domains may be refused. Increasing iteration counts cannot
repair missing rank, a nonconvex unsupported regime or destroyed input precision.
Use failure diagnostics to distinguish these cases; do not loosen certificates,
silently delete studies or substitute uncertified penalty endpoints. Further
sparse/whitened algorithms would require their own derivative, invariance,
independent-solver and adversarial validation before being called implemented.
