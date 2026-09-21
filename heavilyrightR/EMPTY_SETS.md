# Empty confidence sets

Study removal is disabled by default. Passing `empty_set_policy()` opts into a
sensitivity analysis; it does not create an automatic post-selection coverage
guarantee.

For scalar HCauchy/EHMP inference, every stage records active original study
indices, normalized weights, individual p-values at the fitted minimum, the
minimum score and conservative lower bound, candidate ranking, selected study,
and final status. Candidates are ranked by individual p-value, with original
index as the tie breaker. Weights are renormalized and the finite-number cutoff
is recalibrated after every removal.

Use `write_empty_trace()` to save a JSON record. It refuses to overwrite an
existing file.

Multivariate removal is implemented by `fit_meta_md()` and inherited by
`fit_nma()` and `fit_dac()`. Each iteration requires a validated minimum and a
conservative convex lower bound above the cutoff. Candidate blocks are ranked
by individual p-value, with original index as the tie breaker. A candidate is
skipped if deleting its projection rows would lose parameter identifiability.
The first eligible candidate is removed, weights are renormalized, the cutoff
is recalibrated, and the remaining model is fitted again.

The removal unit is the supplied `study()` block. Joint multivariate trials
therefore remain intact. In the explicit comparison-level NMA model, removal
acts on comparisons, which need not coincide with whole clinical trials.

The fit's `empty_set_diagnostics` records original labels and indices,
candidates and rank checks, selected removals, weights, minima, bounds, cutoffs,
and every intermediate state. Numerical failure after a removal retains prior
history. `write_empty_trace()` writes the same record to JSON.

`heavilyright_empty_region_error` means a certified empty set with adjustment
disabled. `heavilyright_empty_resolution_error` means the policy limit or rank
constraint prevents resolution. `heavilyright_empty_numerical_error` means
the mathematical decision remains unknown; no further deletion is allowed.
The latter includes clipped tails, failed calibration, and unvalidated minima.
Errors carry the complete record in their `diagnostics` field.
