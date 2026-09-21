.hr_validate_univariate <- function(estimate, se, df, m) {
  .hr_assert_numeric(estimate, "estimate")
  estimate <- as.numeric(estimate)
  if (length(estimate) != m) stop("estimate must contain one value per study", call. = FALSE)
  if (length(se) == 1L) se <- rep(as.numeric(se), m)
  .hr_assert_numeric(se, "se")
  se <- as.numeric(se)
  if (length(se) != m || any(se <= 0)) stop("se must contain positive standard errors", call. = FALSE)
  if (is.null(df)) return(list(estimate = estimate, se = se, df = NULL))
  if (length(df) == 1L) df <- rep(as.numeric(df), m)
  .hr_assert_numeric(df, "df")
  df <- as.numeric(df)
  if (length(df) != m || any(df <= 0)) stop("df must contain positive degrees of freedom", call. = FALSE)
  list(estimate = estimate, se = se, df = df)
}

.hr_study_p_1d <- function(theta, estimate, se, df = NULL) {
  radius <- abs(theta - estimate) / se
  probability <- if (is.null(df)) {
    2 * stats::pnorm(radius, lower.tail = FALSE)
  } else {
    2 * stats::pt(radius, df, lower.tail = FALSE)
  }
  .hr_clip_probability(probability)
}

.hr_scalar_problem <- function(estimate, se, df, spec) {
  if (spec$m > 1L && spec$method %in% c("hcauchy", "ehmp") && !is.null(df) && any(df < 1)) {
    stop("multi-study hcauchy/ehmp interval inference requires df >= 1", call. = FALSE)
  }
  origin <- estimate[1L]
  centered <- estimate - origin
  scale <- max(diff(range(centered)), max(se))
  standardized <- centered / scale
  errors <- se / scale
  if (!is.finite(scale) || scale <= 0 || any(!is.finite(standardized)) || any(!is.finite(errors)) || any(errors <= 0)) {
    .hr_stop("scalar inputs cannot be reliably standardized", "heavilyright_numerical_error")
  }
  list(origin = origin, scale = scale, estimate = standardized, se = errors, df = df, spec = spec)
}

.hr_scalar_score <- function(z, problem) {
  p <- .hr_study_p_1d(z, problem$estimate, problem$se, problem$df)
  as.numeric(combination_score(p, problem$spec))
}

.hr_scalar_subgradient <- function(z, problem) {
  residual <- (z - problem$estimate) / problem$se
  radius <- abs(residual)
  if (is.null(problem$df)) {
    log_p <- log(2) + stats::pnorm(radius, lower.tail = FALSE, log.p = TRUE)
    log_density <- stats::dnorm(radius, log = TRUE)
  } else {
    log_p <- log(2) + stats::pt(radius, problem$df, lower.tail = FALSE, log.p = TRUE)
    log_density <- stats::dt(radius, problem$df, log = TRUE)
  }
  if (problem$spec$method == "hcauchy") {
    log_sine <- ifelse(
      log_p < log(1e-7),
      log(pi / 2) + log_p,
      log(sin(pi / 2 * exp(log_p)))
    )
    log_slopes <- log(pi) + log_density - 2 * log_sine
  } else {
    log_slopes <- log(2) + log_density - 2 * log_p
  }
  log_slopes <- log_slopes + log(problem$spec$weights) - log(problem$se)
  log_scale <- max(log_slopes)
  if (!is.finite(log_scale)) .hr_stop("scalar derivative is numerically saturated", "heavilyright_numerical_error")
  slopes <- exp(log_slopes - log_scale)
  signed <- sum(sign(residual) * slopes)
  cusp_radius <- sum(slopes[residual == 0])
  closest <- max(signed - cusp_radius, 0) + min(signed + cusp_radius, 0)
  c(slope = closest, log_scale = log_scale)
}

.hr_scalar_minimum <- function(problem) {
  lower <- min(problem$estimate)
  upper <- max(problem$estimate)
  score <- function(z) .hr_scalar_score(z, problem)
  if (lower == upper) {
    value <- score(lower)
    return(list(point = lower, score = value, lower_bound = value, subgradient = 0))
  }
  optimized <- stats::optimize(score, c(lower, upper), tol = 1e-12)
  if (!is.finite(optimized$minimum) || !is.finite(optimized$objective)) {
    .hr_stop("global-score minimization failed; emptiness was not established", "heavilyright_numerical_error")
  }
  point <- optimized$minimum
  convex <- problem$spec$method %in% c("hcauchy", "ehmp")
  if (convex) {
    left <- lower
    right <- upper
    for (iteration in seq_len(120L)) {
      slope <- .hr_scalar_subgradient(point, problem)[1L]
      if (slope == 0) break
      if (slope > 0) right <- point else left <- point
      midpoint <- left + (right - left) / 2
      if (midpoint == left || midpoint == right) break
      point <- midpoint
    }
    nearest <- order(abs(problem$estimate - point))[seq_len(min(2L, length(problem$estimate)))]
    candidates <- unique(c(point, left, right, problem$estimate[nearest]))
    slopes <- vapply(candidates, function(value) abs(.hr_scalar_subgradient(value, problem)[1L]), numeric(1L))
    point <- candidates[which.min(slopes)]
  }
  value <- score(point)
  p <- .hr_study_p_1d(point, problem$estimate, problem$se, problem$df)
  if (!is.finite(value) || any(p <= 2e-150)) {
    .hr_stop("minimum score is nonfinite or tail-saturated; emptiness was not established", "heavilyright_numerical_error")
  }
  if (!convex) return(list(point = point, score = value, lower_bound = -Inf, subgradient = NA_real_))
  subgradient <- .hr_scalar_subgradient(point, problem)
  if (abs(subgradient[1L]) > 1e-7) {
    .hr_stop("scalar minimum failed its subgradient certificate", "heavilyright_numerical_error")
  }
  distance <- max(point - lower, upper - point)
  correction <- if (subgradient[1L] == 0 || distance == 0) 0 else {
    exponent <- subgradient[2L] + log(abs(subgradient[1L])) + log(distance)
    if (exponent < 709) exp(exponent) else Inf
  }
  list(point = point, score = value, lower_bound = value - correction,
       subgradient = subgradient[1L], cusp_anchor_correction = correction)
}

.hr_scalar_interval_once <- function(estimate, se, df, spec) {
  problem <- .hr_scalar_problem(estimate, se, df, spec)
  if (spec$m == 1L) {
    quantile <- if (is.null(df)) stats::qnorm(1 - spec$alpha / 2) else stats::qt(1 - spec$alpha / 2, df)
    return(list(
      lower = estimate - se * quantile,
      upper = estimate + se * quantile,
      estimate = estimate,
      minimum = list(point = 0, score = -1, lower_bound = -1, subgradient = 0),
      empty = FALSE
    ))
  }
  minimum <- .hr_scalar_minimum(problem)
  physical_estimate <- problem$origin + problem$scale * minimum$point
  if (!is.finite(physical_estimate)) .hr_stop("scalar estimate is not representable", "heavilyright_numerical_error")
  if (minimum$score > spec$cutoff) {
    if (spec$method %in% c("hcauchy", "ehmp") && minimum$lower_bound <= spec$cutoff) {
      .hr_stop("minimum lower bound cannot establish an empty set", "heavilyright_numerical_error")
    }
    return(list(lower = physical_estimate, upper = physical_estimate,
                estimate = physical_estimate, minimum = minimum, empty = TRUE))
  }
  endpoint <- function(direction) {
    inside <- minimum$point
    step <- max(problem$se)
    outside <- inside
    for (iteration in seq_len(200L)) {
      outside <- minimum$point + direction * step
      candidate_score <- .hr_scalar_score(outside, problem)
      if (!is.finite(candidate_score)) .hr_stop("nonfinite boundary-bracketing score", "heavilyright_numerical_error")
      if (candidate_score >= spec$cutoff) break
      inside <- outside
      step <- step * 2
      if (iteration == 200L) .hr_stop("boundary bracketing exceeded 200 expansions", "heavilyright_numerical_error")
    }
    root <- stats::uniroot(
      function(z) .hr_scalar_score(z, problem) - spec$cutoff,
      sort(c(inside, outside)), tol = 1e-12, maxiter = 300L
    )$root
    residual <- abs(.hr_scalar_score(root, problem) - spec$cutoff)
    if (!is.finite(residual) || residual > 1e-8 * max(abs(spec$cutoff), .Machine$double.xmin)) {
      .hr_stop("boundary root failed its score-residual check", "heavilyright_numerical_error")
    }
    physical <- problem$origin + problem$scale * root
    reconstructed <- (physical - problem$origin) / problem$scale
    reconstruction_error <- abs(.hr_scalar_score(reconstructed, problem) - spec$cutoff)
    if (!is.finite(physical) || !is.finite(reconstruction_error) ||
        reconstruction_error > 1e-7 * max(abs(spec$cutoff), .Machine$double.xmin)) {
      .hr_stop("boundary loses accuracy in input units; recenter or rescale inputs", "heavilyright_numerical_error")
    }
    physical
  }
  lower <- endpoint(-1)
  upper <- endpoint(1)
  if (!(lower <= physical_estimate && physical_estimate <= upper)) {
    .hr_stop("interval is not representable in input units", "heavilyright_numerical_error")
  }
  list(lower = lower, upper = upper, estimate = physical_estimate,
       minimum = minimum, empty = FALSE)
}

#' Fit a one-dimensional heavily-right meta-analysis
#'
#' @param estimate Study estimates.
#' @param se Positive study standard errors.
#' @param df `NULL` for normal calibration or positive Student-t degrees of
#'   freedom.
#' @param method,weights,alpha,calibration Combination-rule arguments.
#' @param empty_policy Optional policy from [empty_set_policy()]. Omission keeps
#'   study removal off.
#' @param study_labels Optional stable study identifiers.
#' @return A `heavilyright_ci` object with `lower`, `upper`, `estimate`,
#'   `empty`, `specification`, `minimum`, and `empty_set_diagnostics`.
#'   `coef()` extracts the estimate and `confint()` extracts the endpoints.
#'   Always inspect `empty`: when true, equal stored endpoints are a sentinel
#'   for an empty set, not a singleton confidence interval. Refit to change
#'   the confidence level. Numerical failures raise an error.
#' @details Multiple-study HCauchy/EHMP inference with Student-t calibration
#'   requires all degrees of freedom to be at least one. Other scalar
#'   combination rules do not carry the same convexity guarantee.
#' @seealso [empty_set_policy()], [fit_meta_md()], [new_combination()]
#' @examples
#' fit <- fit_meta_1d(c(0.10, 0.14, 0.08), c(0.04, 0.05, 0.03))
#' fit$empty
#' coef(fit)
#' confint(fit)
#' fit_meta_1d(0.2, se = 0.1, df = 19)  # One-study Student-t interval
#' @export
fit_meta_1d <- function(estimate, se, df = NULL, method = "hcauchy",
                        weights = NULL, alpha = 0.05,
                        calibration = c("auto", "finite", "landau"),
                        empty_policy = NULL, study_labels = NULL) {
  m <- length(estimate)
  spec <- new_combination(method, m, weights, alpha, calibration)
  input <- .hr_validate_univariate(estimate, se, df, m)
  labels <- if (is.null(study_labels)) as.character(seq_len(m)) else as.character(study_labels)
  if (length(labels) != m || anyNA(labels)) stop("study_labels must identify every study", call. = FALSE)
  if (!is.null(empty_policy) && !inherits(empty_policy, "heavilyright_empty_policy")) stop("empty_policy has the wrong class", call. = FALSE)
  if (!is.null(empty_policy) && m > 1L && !spec$method %in% c("hcauchy", "ehmp")) {
    stop("automatic empty-set treatment requires convex hcauchy/ehmp inference", call. = FALSE)
  }
  active <- seq_len(m)
  trace <- list()
  current_spec <- spec
  current <- input
  removals <- 0L
  repeat {
    state <- .hr_scalar_interval_once(current$estimate, current$se, current$df, current_spec)
    p <- .hr_study_p_1d(state$estimate, current$estimate, current$se, current$df)
    order_local <- order(p, active)
    allowed <- !is.null(empty_policy) && state$empty &&
      removals < empty_policy$max_removals && length(active) > empty_policy$min_remaining
    selected <- if (allowed) order_local[1L] else NA_integer_
    candidates <- lapply(seq_along(order_local), function(priority) {
      local <- order_local[priority]
      list(original_index = active[local], study_label = labels[active[local]],
           individual_pvalue = p[local], priority = priority,
           eligible = allowed, selected = !is.na(selected) && local == selected,
           reason = if (!allowed) "policy disabled or limit reached" else if (local == selected) "smallest p-value; ties by original index" else "eligible but lower priority")
    })
    trace[[length(trace) + 1L]] <- list(
      step = removals,
      original_indices = active,
      weights = current_spec$weights,
      individual_pvalues = p,
      minimum_score = state$minimum$score,
      minimum_value_lower_bound = state$minimum$lower_bound,
      cusp_anchor_correction = state$minimum$cusp_anchor_correction %||% 0,
      cutoff = current_spec$cutoff,
      global_minimizer = state$estimate,
      empty = state$empty,
      candidates = candidates
    )
    if (!state$empty) break
    if (is.null(empty_policy)) break
    if (!allowed) {
      diagnostics <- .hr_empty_diagnostics(TRUE, TRUE, FALSE, trace, active, labels,
        spec$weights, current_spec, "empty set remains after permitted removals", "empty")
      .hr_stop(diagnostics$message, "heavilyright_empty_resolution_error", diagnostics = diagnostics)
    }
    if (empty_policy$warn) warning(sprintf("removing study %s in empty-set sensitivity analysis", labels[active[selected]]), call. = FALSE)
    keep <- seq_along(active) != selected
    active <- active[keep]
    current$estimate <- current$estimate[keep]
    current$se <- current$se[keep]
    if (!is.null(current$df)) current$df <- current$df[keep]
    new_weights <- current_spec$weights[keep]
    new_weights <- new_weights / sum(new_weights)
    current_spec <- new_combination(spec$method, weights = new_weights, alpha = spec$alpha,
                                    calibration = spec$calibration)
    removals <- removals + 1L
  }
  diagnostics <- .hr_empty_diagnostics(
    !is.null(empty_policy), state$empty || removals > 0L, !state$empty,
    trace, active, labels, spec$weights, current_spec,
    if (!state$empty && removals == 0L) "confidence set was nonempty" else if (!state$empty) sprintf("resolved after removing %d study/studies", removals) else "confidence set is empty; adjustment is disabled",
    if (state$empty) "empty" else "nonempty"
  )
  out <- list(
    lower = state$lower, upper = state$upper, estimate = state$estimate,
    empty = state$empty, specification = current_spec,
    minimum = state$minimum, empty_set_diagnostics = diagnostics,
    call = match.call()
  )
  class(out) <- "heavilyright_ci"
  out
}

`%||%` <- function(x, y) if (is.null(x)) y else x

#' @export
print.heavilyright_ci <- function(x, ...) {
  if (x$empty) {
    cat(sprintf("Empty %.1f%% confidence set; score minimum %.8g exceeds cutoff %.8g\n",
      100 * (1 - x$specification$alpha), x$minimum$score, x$specification$cutoff))
  } else {
    cat(sprintf("%.1f%% confidence interval: [%.8g, %.8g]\nEstimate: %.8g\n",
      100 * (1 - x$specification$alpha), x$lower, x$upper, x$estimate))
  }
  invisible(x)
}

#' @export
coef.heavilyright_ci <- function(object, ...) object$estimate

#' @export
confint.heavilyright_ci <- function(object, parm, level = 1 - object$specification$alpha, ...) {
  if (!isTRUE(all.equal(level, 1 - object$specification$alpha))) stop("refit to request a different confidence level", call. = FALSE)
  matrix(c(object$lower, object$upper), nrow = 1L,
         dimnames = list("parameter", c("lower", "upper")))
}
