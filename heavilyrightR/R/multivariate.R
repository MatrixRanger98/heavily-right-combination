.hr_pinv <- function(x, tolerance = 1e-12) {
  x <- as.matrix(x)
  decomposition <- svd(x)
  keep <- decomposition$d > max(decomposition$d, 0) * tolerance
  if (!any(keep)) return(matrix(0, ncol(x), nrow(x)))
  decomposition$v[, keep, drop = FALSE] %*%
    (t(decomposition$u[, keep, drop = FALSE]) / decomposition$d[keep])
}

#' Construct a multivariate study summary
#' @param estimate Study estimate vector.
#' @param vcov Positive-definite covariance matrix of the estimate, not raw observations.
#' @param projection Matrix mapping the common parameter to the study estimand.
#' @param df `NULL` for chi-square calibration or Hotelling degrees of freedom.
#' @param label Stable study identifier. Study removal acts on this entire block.
#' @return A `heavilyright_study` object.
#' @details If omitted, `projection` is the identity map at fitting time.
#'   For a sample mean based on `n` independent multivariate normal observations,
#'   use [covariance_of_mean()] and `df = n - 1`. A supplied Hotelling degree
#'   of freedom must exceed the study dimension minus one; multiple-study
#'   convex inference imposes additional restrictions in [fit_meta_md()].
#' @seealso [fit_meta_md()], [covariance_of_mean()]
#' @examples
#' study(c(0.2, -0.1), diag(c(0.04, 0.09)), df = 29, label = "trial-1")
#' # A scalar contrast of two common parameters:
#' study(0.3, matrix(0.04), projection = matrix(c(1, -1), nrow = 1))
#' @export
study <- function(estimate, vcov, projection = NULL, df = NULL, label = NULL) {
  .hr_assert_numeric(estimate, "estimate"); .hr_assert_numeric(vcov, "vcov")
  estimate <- as.numeric(estimate); vcov <- as.matrix(vcov); q <- length(estimate)
  if (!identical(dim(vcov), c(q, q))) stop("vcov dimensions must match estimate", call. = FALSE)
  if (max(abs(vcov - t(vcov))) > 1e-12 * max(abs(vcov))) stop("vcov must be symmetric", call. = FALSE)
  factor <- tryCatch(chol(vcov), error = function(e) NULL)
  if (is.null(factor)) stop("vcov must be positive definite", call. = FALSE)
  if (!is.null(projection)) {
    .hr_assert_numeric(projection, "projection"); projection <- as.matrix(projection)
    if (nrow(projection) != q || ncol(projection) < 1L || !any(projection != 0)) stop("projection must have one row per estimate and nonempty coordinate support", call. = FALSE)
  }
  if (!is.null(df) && (length(df) != 1L || !is.numeric(df) || !is.finite(df) || df <= q - 1)) stop("Hotelling df must exceed study dimension minus one", call. = FALSE)
  if (!is.null(label) && (length(label) != 1L || is.na(label) || !nzchar(as.character(label)))) stop("label must be a nonempty identifier", call. = FALSE)
  structure(list(estimate = estimate, vcov = vcov, precision = chol2inv(factor),
    projection = projection, df = df, label = if (is.null(label)) NA_character_ else as.character(label), q = q),
    class = "heavilyright_study")
}

.hr_prepare_studies <- function(studies, parameter_dim = NULL) {
  if (!is.list(studies) || !length(studies) || !all(vapply(studies, inherits, logical(1), "heavilyright_study"))) stop("studies must be a nonempty list of study() objects", call. = FALSE)
  known <- unique(vapply(studies, function(s) if (is.null(s$projection)) NA_integer_ else ncol(s$projection), integer(1)))
  known <- known[!is.na(known)]
  if (is.null(parameter_dim)) {
    if (!length(known)) known <- unique(vapply(studies, `[[`, integer(1), "q"))
    if (length(known) != 1L) stop("parameter_dim is required for ambiguous projections", call. = FALSE)
    parameter_dim <- known
  }
  parameter_dim <- .hr_positive_integer(parameter_dim, "parameter_dim")
  for (i in seq_along(studies)) {
    if (is.null(studies[[i]]$projection)) {
      if (studies[[i]]$q != parameter_dim) stop("identity projection dimension does not match parameter_dim", call. = FALSE)
      studies[[i]]$projection <- diag(parameter_dim)
    }
    if (ncol(studies[[i]]$projection) != parameter_dim) stop("projection matrices disagree on parameter dimension", call. = FALSE)
    if (is.na(studies[[i]]$label)) studies[[i]]$label <- as.character(i)
  }
  list(studies = studies, dimension = parameter_dim)
}

.hr_projection_rank <- function(studies, dimension) {
  if (!length(studies)) return(0L)
  A <- do.call(rbind, lapply(studies, `[[`, "projection"))
  scales <- sqrt(colSums(A^2)); scales[scales == 0] <- 1
  qr(sweep(A, 2, scales, "/"), tol = 1e-11)$rank
}

.hr_md_standardization <- function(studies, dimension) {
  B <- do.call(rbind, lapply(studies, function(s) forwardsolve(t(chol(s$vcov)), s$projection)))
  y <- unlist(lapply(studies, function(s) as.numeric(forwardsolve(t(chol(s$vcov)), s$estimate))))
  preliminary <- sqrt(colSums(B^2))
  if (any(!is.finite(preliminary)) || any(preliminary <= 0)) stop("positive-weight projections do not span the parameter space", call. = FALSE)
  A <- sweep(B, 2, preliminary, "/")
  decomposition <- svd(A)
  if (length(decomposition$d) < dimension || min(decomposition$d) <= 1e-12 * max(decomposition$d)) stop("positive-weight projections do not span the parameter space reliably", call. = FALSE)
  center <- as.numeric(decomposition$v %*% (crossprod(decomposition$u, y) / decomposition$d)) / preliminary
  inverse_scaled <- tcrossprod(sweep(decomposition$v, 2, decomposition$d, "/"))
  inverse <- inverse_scaled / outer(preliminary, preliminary)
  scale <- sqrt(diag(inverse))
  if (any(!is.finite(scale)) || any(scale <= 0) || any(!is.finite(center))) .hr_stop("coordinate standardization failed", "heavilyright_numerical_error")
  list(center = center, scale = scale, information = crossprod(B), inverse = inverse, rank = dimension)
}

.hr_md_pvalues <- function(theta, studies) {
  vapply(studies, function(s) {
    residual <- as.numeric(forwardsolve(t(chol(s$vcov)), s$projection %*% theta - s$estimate))
    Q <- sum(residual^2)
    p <- if (is.null(s$df)) stats::pchisq(Q, s$q, lower.tail = FALSE) else {
      stats::pf(Q * (s$df + 1 - s$q) / (s$q * s$df), s$q, s$df + 1 - s$q, lower.tail = FALSE)
    }
    max(p, .hr_probability_floor)
  }, numeric(1))
}

.hr_md_score <- function(theta, fit) as.numeric(combination_score(.hr_md_pvalues(theta, fit$studies), fit$specification))

.hr_components <- function(studies, dimension) {
  parent <- seq_len(dimension)
  root <- function(i) { while (parent[i] != i) i <- parent[i]; i }
  for (s in studies) {
    indices <- which(colSums(abs(s$projection)) > 0)
    if (length(indices) > 1L) for (j in indices[-1L]) parent[root(j)] <- root(indices[1L])
  }
  labels <- vapply(seq_len(dimension), root, integer(1))
  unname(split(seq_len(dimension), match(labels, unique(labels))))
}

.hr_validate_convex_regime <- function(studies, method) {
  if (length(studies) <= 1L) return(invisible(TRUE))
  if (!method %in% c("hcauchy", "ehmp")) stop("certified multi-study regions support only hcauchy and ehmp", call. = FALSE)
  for (s in studies) {
    if (is.null(s$df)) next
    required <- if (s$q == 1L) 1 else if (method == "ehmp") s$q else if (s$q == 2L) 2.5 else s$q + 1
    if (s$df < required) stop(sprintf("%s Hotelling block dimension %d requires df >= %g", method, s$q, required), call. = FALSE)
  }
  invisible(TRUE)
}

.hr_md_fit_once <- function(studies, dimension, spec) {
  fit <- structure(list(studies = studies, dimension = dimension, specification = spec,
    standardization = .hr_md_standardization(studies, dimension),
    components = .hr_components(studies, dimension)), class = "heavilyright_md_fit")
  fit$kernel <- .hr_make_kernel(fit)
  if (spec$alpha <= .hr_probability_floor) .hr_stop("significance level is at or below the public probability floor", "heavilyright_numerical_error")
  fit$minimum <- .hr_md_minimum(fit)
  fit
}

#' Fit a multidimensional heavily-right confidence region
#'
#' Study removal is disabled by default. An opt-in policy removes the entire
#' input block with the smallest individual p-value at a validated minimum,
#' skipping deletions that lose parameter identifiability. Every removal needs
#' a convex lower bound above the cutoff. The procedure is a sensitivity
#' analysis, not a post-selection coverage correction.
#'
#' @param studies List of [study()] objects.
#' @param parameter_dim Common parameter dimension, including dimension one.
#' @param method,weights,alpha,calibration Arguments to [new_combination()].
#' @param empty_policy Optional [empty_set_policy()].
#' @return A `heavilyright_md_fit`. Certified empty sets and unresolved numerical
#'   cases raise distinct typed errors with complete `diagnostics` records.
#'   Successful fits contain retained `studies`, `specification`, `minimum`,
#'   and `empty_set_diagnostics`. `coef()` extracts the minimum-score point;
#'   use [support_interval()] for directional bounds, not `confint()`.
#' @details Multi-study fits support `"hcauchy"` and `"ehmp"` only. For
#'   Hotelling study dimension `q`, both require `df >= 1` when `q = 1`;
#'   EHMP requires `df >= q` when `q >= 2`; HCauchy requires `df >= 2.5`
#'   when `q = 2` and `df >= q + 1` when `q >= 3`. Known-covariance blocks
#'   have no df restriction. The combined projection matrices must span the
#'   common parameter space. Independence calibration does not justify treating
#'   dependent study blocks as independent.
#' @seealso [study()], [support_interval()], [empty_set_policy()]
#' @examples
#' studies <- list(
#'   study(c(0.2, -0.1), diag(c(0.04, 0.09))),
#'   study(c(0.1, 0), diag(c(0.05, 0.08)))
#' )
#' fit <- fit_meta_md(studies)
#' coef(fit)
#' contains(fit, coef(fit))
#' support_interval(fit, c(1, -1))
#' @export
fit_meta_md <- function(studies, parameter_dim = NULL, method = "hcauchy", weights = NULL,
                        alpha = 0.05, calibration = c("auto", "finite", "landau"), empty_policy = NULL) {
  prepared <- .hr_prepare_studies(studies, parameter_dim)
  original <- new_combination(method, length(prepared$studies), weights, alpha, match.arg(calibration))
  if (original$m != length(prepared$studies)) stop("weights must match the number of studies", call. = FALSE)
  .hr_validate_convex_regime(prepared$studies, original$method)
  if (.hr_projection_rank(prepared$studies, prepared$dimension) < prepared$dimension) stop("positive-weight projections do not span the parameter space", call. = FALSE)
  if (!is.null(empty_policy) && !inherits(empty_policy, "heavilyright_empty_policy")) stop("empty_policy has the wrong class", call. = FALSE)
  active <- seq_along(prepared$studies); labels <- vapply(prepared$studies, `[[`, character(1), "label")
  trace <- list(); removals <- list(); spec <- original
  record <- function(state, message) {
    out <- .hr_empty_diagnostics(!is.null(empty_policy), any(vapply(trace, function(t) isTRUE(t$empty), logical(1))),
      state == "nonempty", trace, active, labels, original$weights, spec, message, state)
    out$removals <- removals; out$policy <- empty_policy
    out$parameter_dimension <- prepared$dimension
    out$final_threshold <- spec$cutoff
    out
  }
  repeat {
    current <- prepared$studies[active]
    failure <- NULL
    fit <- tryCatch(.hr_md_fit_once(current, prepared$dimension, spec), error = function(e) { failure <<- e; NULL })
    if (is.null(fit)) {
      trace[[length(trace) + 1L]] <- list(step = length(removals), original_indices = active,
        weights = spec$weights, cutoff = spec$cutoff, empty = NA, message = conditionMessage(failure))
      .hr_stop(conditionMessage(failure), "heavilyright_empty_numerical_error",
               diagnostics = record("unknown", conditionMessage(failure)), cause = failure)
    }
    minimum <- fit$minimum
    empty <- minimum$score > spec$cutoff
    if (empty && !isTRUE(minimum$empty_certified)) {
      message <- "minimum exceeds cutoff but a valid convex lower bound did not establish emptiness"
      trace[[length(trace) + 1L]] <- list(step = length(removals), original_indices = active,
        weights = spec$weights, cutoff = spec$cutoff, empty = NA, minimum_diagnostics = minimum)
      .hr_stop(message, "heavilyright_empty_numerical_error", diagnostics = record("unknown", message))
    }
    p <- .hr_md_pvalues(minimum$point, current)
    allowed <- !is.null(empty_policy) && empty && length(removals) < empty_policy$max_removals && length(active) > empty_policy$min_remaining
    selected <- NA_integer_; candidates <- list()
    if (empty) for (local in order(p, active)) {
      rank <- .hr_projection_rank(current[-local], prepared$dimension)
      eligible <- allowed && rank == prepared$dimension
      chosen <- eligible && is.na(selected)
      if (chosen) selected <- local
      candidates[[length(candidates) + 1L]] <- list(original_index = active[local], study_label = labels[active[local]],
        individual_pvalue = p[local], priority = length(candidates) + 1L, eligible = eligible, selected = chosen,
        retained_projection_rank = rank,
        reason = if (!allowed) "policy disabled or limit reached" else if (rank < prepared$dimension) "removal loses parameter identifiability" else if (chosen) "smallest admissible p-value; ties by original index" else "eligible but lower priority")
    }
    trace[[length(trace) + 1L]] <- list(step = length(removals), original_indices = active,
      weights = spec$weights, individual_pvalues = p, minimum_score = minimum$score,
      minimum_value_lower_bound = minimum$minimum_value_lower_bound,
      feasible_set_score_lower_bound = minimum$feasible_set_score_lower_bound,
      cutoff = spec$cutoff, global_minimizer = minimum$point, empty = empty,
      minimum_diagnostics = minimum, candidates = candidates)
    if (!empty) {
      fit$empty_set_diagnostics <- record("nonempty", if (length(removals)) sprintf("resolved after removing %d study blocks", length(removals)) else "confidence region was nonempty")
      fit$call <- match.call()
      return(fit)
    }
    if (is.na(selected)) {
      message <- if (is.null(empty_policy)) "confidence region is certified empty; study removal is disabled" else if (!allowed) "empty region remains after permitted removals" else "empty region cannot be resolved without losing parameter identifiability"
      .hr_stop(message, if (is.null(empty_policy)) "heavilyright_empty_region_error" else "heavilyright_empty_resolution_error", diagnostics = record("empty", message))
    }
    removed <- active[selected]; active <- active[-selected]
    next_weights <- original$weights[active]; next_weights <- next_weights / sum(next_weights)
    removals[[length(removals) + 1L]] <- list(step = length(removals) + 1L, original_index = removed,
      study_label = labels[removed], individual_pvalue = p[selected], minimum_score = minimum$score,
      cutoff = spec$cutoff, global_minimizer = minimum$point, remaining_original_indices = active, remaining_weights = next_weights)
    if (empty_policy$warn) warning(sprintf("removing study block %s in empty-set sensitivity analysis", labels[removed]), call. = FALSE)
    calibration_error <- NULL
    next_spec <- tryCatch(new_combination(original$method, weights = next_weights, alpha = alpha,
      calibration = original$calibration), error = function(e) { calibration_error <<- e; NULL })
    if (is.null(next_spec)) {
      spec$weights <- next_weights; spec$m <- length(next_weights); spec$cutoff <- NA_real_
      trace[[length(trace) + 1L]] <- list(step = length(removals), original_indices = active, weights = next_weights, empty = NA, message = conditionMessage(calibration_error))
      .hr_stop("cutoff recalibration failed after removal", "heavilyright_empty_numerical_error", diagnostics = record("unknown", conditionMessage(calibration_error)))
    }
    spec <- next_spec
  }
}

#' Region score
#' @param object Fitted region.
#' @param ... Additional arguments.
#' @return A numeric score; region membership is `score <= cutoff`.
#' @seealso [contains()], [fit_meta_md()]
#' @examples
#' fit <- fit_meta_md(list(study(c(0.2, -0.1), diag(2))))
#' score(fit, c(0, 0))
#' fit$specification$cutoff
#' @export
score <- function(object, ...) UseMethod("score")
#' @param theta Parameter vector.
#' @rdname score
#' @export
score.heavilyright_md_fit <- function(object, theta, ...) {
  .hr_assert_numeric(theta, "theta")
  if (length(theta) != object$dimension) stop("theta has the wrong dimension", call. = FALSE)
  .hr_md_score(theta, object)
}

#' Test confidence-region membership
#' @param object Fitted region.
#' @param theta Parameter vector.
#' @param ... Additional arguments.
#' @return A single logical value using the fitted cutoff without an added
#'   numerical tolerance. A reconstructed endpoint may differ from the cutoff
#'   by a small floating-point residual; consult its support diagnostics.
#' @seealso [score()], [support_interval()]
#' @examples
#' fit <- fit_meta_md(list(study(c(0.2, -0.1), diag(2))))
#' contains(fit, c(0, 0))
#' contains(fit, c(10, 10))
#' @export
contains <- function(object, theta, ...) UseMethod("contains")
#' @export
contains.heavilyright_md_fit <- function(object, theta, ...) score(object, theta) <= object$specification$cutoff

#' Directional support interval for a fitted confidence region
#'
#' This projects the entire confidence region; it is not a line slice.
#' Check `success` before using numerical endpoints. Every success requires
#' independent boundary, positive-multiplier KKT/subgradient, and physical-unit
#' reconstruction checks. These residual checks are not interval arithmetic.
#' @param object A fit from [fit_meta_md()], [fit_nma()], or [fit_dac()].
#' @param direction Finite nonzero parameter-space direction.
#' @param maxeval Maximum evaluations per constrained optimizer stage.
#' @return A `heavilyright_support_interval` with `lower`, `upper`,
#'   `lower_point`, `upper_point`, `success`, `lower_diagnostics`,
#'   `upper_diagnostics`, `minimum_diagnostics`, and `empty_set_diagnostics`.
#'   A failed certificate can return `success = FALSE`; numerical failures
#'   can instead raise a `heavilyright_numerical_error`. Neither outcome
#'   establishes an empty confidence region.
#' @seealso [fit_meta_md()], [nma_contrasts()]
#' @examples
#' fit <- fit_meta_md(list(study(c(0.2, -0.1), diag(c(0.04, 0.09)))))
#' interval <- support_interval(fit, c(1, -1))
#' stopifnot(interval$success)
#' c(lower = interval$lower, upper = interval$upper)
#' interval$upper_diagnostics
#' @export
support_interval <- function(object, direction, maxeval = 2000L) {
  if (!inherits(object, "heavilyright_md_fit")) stop("object must be a multidimensional fit", call. = FALSE)
  .hr_assert_numeric(direction, "direction")
  if (length(direction) != object$dimension || max(abs(direction)) == 0) stop("direction must be finite, nonzero, and match parameter dimension", call. = FALSE)
  maxeval <- .hr_positive_integer(maxeval, "maxeval")
  if (length(object$studies) == 1L) return(.hr_analytic_support(object, direction))
  endpoint <- function(d) tryCatch(.hr_support_endpoint(object, d, maxeval), error = function(e) {
    diagnostics <- object$empty_set_diagnostics
    diagnostics$inference_failure <- list(stage = "support", direction = d, message = conditionMessage(e))
    .hr_stop(conditionMessage(e), "heavilyright_numerical_error", diagnostics = diagnostics, cause = e)
  })
  upper <- endpoint(direction)
  lower <- endpoint(-direction)
  lower_point <- object$standardization$center + object$standardization$scale * lower$point_z
  upper_point <- object$standardization$center + object$standardization$scale * upper$point_z
  active <- unique(unlist(object$components[vapply(object$components, function(i) any(direction[i] != 0), logical(1))]))
  structure(list(lower = sum(direction * lower_point), upper = sum(direction * upper_point),
    lower_point = lower_point, upper_point = upper_point, lower_diagnostics = lower$diagnostics,
    upper_diagnostics = upper$diagnostics, success = lower$diagnostics$success && upper$diagnostics$success,
    direction = direction, minimum_point = object$minimum$point, minimum_score = object$minimum$score,
    minimum_diagnostics = object$minimum, active_dimension = length(active),
    central_symmetry_used = FALSE, specification = object$specification,
    empty_set_diagnostics = object$empty_set_diagnostics), class = "heavilyright_support_interval")
}

#' @export
print.heavilyright_md_fit <- function(x, ...) {
  cat(sprintf("Heavily-right %d-dimensional region from %d study blocks\nMinimum score: %.8g  cutoff: %.8g\n",
    x$dimension, length(x$studies), x$minimum$score, x$specification$cutoff))
  invisible(x)
}
#' @export
coef.heavilyright_md_fit <- function(object, ...) {
  if (is.null(object$parameter_names)) object$minimum$point else stats::setNames(object$minimum$point, object$parameter_names)
}
#' @export
print.heavilyright_support_interval <- function(x, ...) {
  cat(sprintf("Directional %.1f%% support interval: [%.8g, %.8g]\nCertified: %s\n",
    100 * (1 - x$specification$alpha), x$lower, x$upper, if (x$success) "yes" else "no"))
  invisible(x)
}
