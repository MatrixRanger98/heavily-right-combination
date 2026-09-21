#' Define a p-value combination rule
#'
#' @param method Combination method: `"hcauchy"`, `"ehmp"`, `"hmp"`,
#'   `"cauchy"`, `"levy"`, `"fisher"`, `"stouffer"`, `"bonferroni"`, or
#'   `"simes"`.
#' @param m Number of studies when `weights` is omitted.
#' @param weights Strictly positive weights summing to one.
#' @param alpha Significance level.
#' @param calibration `"finite"` for the finite-number independence law,
#'   `"landau"` for its asymptotic approximation, or `"auto"` (finite through
#'   1,000 studies). The returned specification records the resolved choice.
#' @return A `heavilyright_combination` specification.
#' @details Finite HCauchy/EHMP calibration assumes independent uniform null
#'   p-values. HMP always uses the Landau approximation. Other rules use their
#'   method-specific calibration; choosing a rule does not validate dependence
#'   assumptions. Fisher uses equal weights, warning if unequal weights are
#'   supplied. With one study every rule returns the original p-value.
#' @seealso [combine_p()], [combination_score()], [combination_cutoff()]
#' @examples
#' rule <- new_combination("hcauchy", weights = c(0.2, 0.3, 0.5),
#'                         calibration = "finite")
#' rule
#' combine_p(c(0.01, 0.2, 0.4), rule)
#' @export
new_combination <- function(method = "hcauchy", m = NULL, weights = NULL,
                            alpha = 0.05,
                            calibration = c("auto", "finite", "landau")) {
  method <- .hr_method(method)
  alpha <- .hr_level(alpha)
  weights <- .hr_weights(weights, m)
  calibration <- .hr_calibration(match.arg(calibration), method, length(weights))
  if (method == "fisher" && max(weights) - min(weights) > 1e-14) {
    warning("Fisher calibration uses equal study contributions; supplied weights are ignored", call. = FALSE)
    weights <- rep.int(1 / length(weights), length(weights))
  }
  out <- list(
    method = method,
    weights = weights,
    m = length(weights),
    alpha = alpha,
    calibration = calibration
  )
  out$cutoff <- combination_cutoff(alpha, out)
  class(out) <- "heavilyright_combination"
  out
}

.hr_combination_spec <- function(spec = NULL, method = "hcauchy", m = NULL,
                                 weights = NULL, alpha = 0.05,
                                 calibration = "auto") {
  if (!is.null(spec)) {
    if (!inherits(spec, "heavilyright_combination")) stop("spec must be a heavilyright combination specification", call. = FALSE)
    return(spec)
  }
  new_combination(method, m, weights, alpha, calibration)
}

.hr_score_row <- function(p, spec) {
  m <- spec$m
  if (m == 1L) return(-p[1L])
  switch(spec$method,
    hcauchy = sum(spec$weights * half_cauchy_score(p)),
    ehmp = sum(spec$weights / p),
    hmp = sum(spec$weights / p),
    cauchy = sum(spec$weights / tan(pi * p)),
    levy = sum(spec$weights / stats::qnorm((1 + p) / 2)^2) /
      sum(sqrt(spec$weights))^2,
    fisher = -sum(log(p)),
    stouffer = -sum(spec$weights * stats::qnorm(p)) / sqrt(sum(spec$weights^2)),
    bonferroni = -min(p),
    simes = -min(sort(p) / seq_len(m)),
    stop("unsupported combination method", call. = FALSE)
  )
}

#' Compute a method-specific combined score
#' @param p Numeric p-values. For batched input, studies occupy the final
#'   dimension.
#' @param spec A specification from [new_combination()].
#' @param ... Arguments forwarded to [new_combination()] when `spec` is absent.
#' @return A scalar or array with the study dimension removed.
#' @details Larger scores give stronger evidence against the global null.
#'   Scores from different methods are not directly comparable. For one study
#'   the internal score is `-p`, regardless of the selected method.
#' @seealso [combination_pvalue()], [combine_p()]
#' @examples
#' rule <- new_combination("hcauchy", m = 3)
#' p <- rbind(c(0.01, 0.2, 0.4), c(0.2, 0.5, 0.8))
#' combination_score(p, rule)
#' @export
combination_score <- function(p, spec = NULL, ...) {
  spec <- .hr_combination_spec(spec, ...)
  p <- .hr_pvalues(p, spec$m)
  .hr_row_apply(p, spec$m, function(row) .hr_score_row(row, spec))
}

.hr_landau_location <- function(spec) {
  entropy <- -sum(spec$weights * log(spec$weights)) + 1 - .hr_euler_gamma
  if (spec$method == "hcauchy") entropy * 2 / pi else entropy
}

#' Convert a combination score to a global p-value
#' @param score Numeric global score.
#' @inheritParams combination_score
#' @return Numeric global p-values.
#' @seealso [combination_score()], [combine_p()]
#' @examples
#' rule <- new_combination("hcauchy", m = 3)
#' value <- combination_score(c(0.01, 0.2, 0.4), rule)
#' combination_pvalue(value, rule)
#' @export
combination_pvalue <- function(score, spec = NULL, ...) {
  spec <- .hr_combination_spec(spec, ...)
  .hr_assert_numeric(score, "score", finite = FALSE)
  if (any(is.nan(score))) stop("score must not contain NaN", call. = FALSE)
  if (spec$m == 1L) return(-score)
  method <- spec$method
  if (method == "hcauchy") {
    if (spec$calibration == "finite") {
      return(vapply(score, function(value) if (is.infinite(value) && value > 0) 0 else phcmean(value, spec$weights, lower.tail = FALSE), numeric(1L)))
    }
    return(plandau(score, .hr_landau_location(spec), 1, lower.tail = FALSE))
  }
  if (method == "ehmp") {
    if (spec$calibration == "finite") {
      return(vapply(score, function(value) if (is.infinite(value) && value > 0) 0 else pehmean(value, spec$weights, lower.tail = FALSE), numeric(1L)))
    }
    return(plandau(score, .hr_landau_location(spec), pi / 2, lower.tail = FALSE))
  }
  switch(method,
    hmp = plandau(score, .hr_landau_location(spec), pi / 2, lower.tail = FALSE),
    cauchy = stats::pcauchy(score, lower.tail = FALSE),
    levy = 2 * stats::pnorm(1 / sqrt(score)) - 1,
    fisher = stats::pgamma(score, shape = spec$m, lower.tail = FALSE),
    stouffer = stats::pnorm(score, lower.tail = FALSE),
    bonferroni = pmin(1, -score * spec$m),
    simes = pmin(1, -score * spec$m),
    stop("unsupported combination method", call. = FALSE)
  )
}

#' Global-score cutoff
#' @param alpha Significance level.
#' @param spec A combination specification.
#' @return Numeric score cutoff.
#' @details This computes a cutoff without changing `spec$alpha` or
#'   `spec$cutoff`. Create a new specification to change rejection decisions
#'   made by [combine_p()].
#' @seealso [new_combination()], [combine_p()]
#' @examples
#' rule <- new_combination("hcauchy", m = 3, alpha = 0.05)
#' combination_cutoff(0.01, rule)
#' rule$cutoff  # Still the configured 0.05 cutoff
#' @export
combination_cutoff <- function(alpha, spec) {
  alpha <- .hr_level(alpha)
  if (!inherits(spec, "heavilyright_combination") && !is.list(spec)) stop("spec must describe a combination rule", call. = FALSE)
  if (spec$m == 1L) return(-alpha)
  method <- spec$method
  if (method == "hcauchy") {
    if (spec$calibration == "finite") return(qhcmean(1 - alpha, spec$weights))
    return(qlandau(1 - alpha, .hr_landau_location(spec), 1))
  }
  if (method == "ehmp") {
    if (spec$calibration == "finite") return(qehmean(1 - alpha, spec$weights))
    return(qlandau(1 - alpha, .hr_landau_location(spec), pi / 2))
  }
  switch(method,
    hmp = qlandau(1 - alpha, .hr_landau_location(spec), pi / 2),
    cauchy = stats::qcauchy(1 - alpha),
    levy = 1 / stats::qnorm((1 + alpha) / 2)^2,
    fisher = stats::qgamma(1 - alpha, shape = spec$m),
    stouffer = stats::qnorm(1 - alpha),
    bonferroni = -alpha / spec$m,
    simes = -alpha / spec$m,
    stop("unsupported combination method", call. = FALSE)
  )
}

#' Combine p-values
#'
#' @param decision If `TRUE`, return whether the score reaches the configured
#'   rejection cutoff instead of a global p-value.
#' @inheritParams combination_score
#' @return Global p-values or logical rejection decisions.
#' @seealso [new_combination()], [combination_score()]
#' @examples
#' rule <- new_combination("hcauchy", m = 3, calibration = "finite")
#' p <- c(0.01, 0.2, 0.4)
#' combine_p(p, rule)
#' combine_p(p, rule, decision = TRUE)
#' combine_p(rbind(p, c(0.2, 0.5, 0.8)), rule)
#' @export
combine_p <- function(p, spec = NULL, ..., decision = FALSE) {
  spec <- .hr_combination_spec(spec, ...)
  score <- combination_score(p, spec)
  if (decision) return(score >= spec$cutoff)
  combination_pvalue(score, spec)
}

#' @export
print.heavilyright_combination <- function(x, ...) {
  cat(sprintf(
    "Heavily-right combination: %s (%s calibration)\nStudies: %d  alpha: %g  cutoff: %.10g\n",
    x$method, x$calibration, x$m, x$alpha, x$cutoff
  ))
  invisible(x)
}
