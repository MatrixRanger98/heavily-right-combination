#' Heavily right-tailed combination and confidence-region inference
#'
#' Native R methods for p-value combination, scalar meta-analysis, multivariate
#' confidence regions, treatment networks, and coordinate divide-and-combine
#' (DAC) inference. Python is not required at runtime.
#'
#' @section Starting points:
#' Use [new_combination()] and [combine_p()] for p-values; [fit_meta_1d()] for
#' scalar summaries; [study()], [fit_meta_md()], and [support_interval()] for
#' general multivariate summaries. Treatment networks use [network_studies()]
#' or [network_arm_studies()], followed by [fit_nma()] and [nma_contrasts()].
#' Coordinate partitions of an observation matrix use [fit_dac()].
#'
#' @section Statistical and numerical boundaries:
#' Study covariance matrices describe estimates, not individual observations.
#' Finite-number HCauchy/EHMP calibration assumes independent null p-values.
#' Convexity and numerical success do not establish independence or nominal
#' coverage under a different model. Study removal via [empty_set_policy()]
#' is opt-in sensitivity analysis without a post-selection coverage guarantee.
#' Always check support-interval `success` and retain diagnostics.
#'
#' @section Guides:
#' The installed vignettes are `getting-started`, `workflows-and-diagnostics`,
#' `mathematical-methods`, and `troubleshooting`. List them with
#' `vignette(package = "heavilyright")`. The package is licensed under
#' GPL version 3 or later; installed COPYING and NOTICE files provide license
#' and third-party attribution information.
#'
#' @seealso [combine_p()], [fit_meta_1d()], [fit_nma()], [fit_dac()]
#' @examples
#' rule <- new_combination("hcauchy", m = 3, calibration = "finite")
#' combine_p(c(0.01, 0.2, 0.4), rule)
"_PACKAGE"
