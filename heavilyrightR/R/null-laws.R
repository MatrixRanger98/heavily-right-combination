.hr_null_count <- function(weights) {
  length(weights) == 1L && is.numeric(weights) && is.finite(weights) &&
    weights >= 1 && weights <= .Machine$integer.max && weights == trunc(weights)
}

.hr_null_weights <- function(weights) {
  if (.hr_null_count(weights)) {
    return(rep.int(1 / as.integer(weights), as.integer(weights)))
  }
  .hr_weights(weights = weights, allow_zero = TRUE)
}

.hr_integrate_checked <- function(fun, lower, value_bounds, diagnostics) {
  result <- tryCatch(
    stats::integrate(
      Vectorize(fun, "z"), lower = lower, upper = Inf,
      subdivisions = 600L, rel.tol = 2e-11, abs.tol = 2e-13,
      stop.on.error = FALSE
    ),
    error = function(error) .hr_stop(
      sprintf("characteristic-function inversion failed: %s", conditionMessage(error)),
      "heavilyright_integration_error"
    )
  )
  tolerance <- max(2e-13, 2e-11 * abs(result$value))
  if (!is.finite(result$value) || !is.finite(result$abs.error) ||
      result$abs.error < 0 || result$abs.error > 100 * tolerance ||
      !identical(result$message, "OK")) {
    .hr_stop(
      sprintf(
        "characteristic-function inversion failed its error check: value=%.17g, error=%.3g, message=%s",
        result$value, result$abs.error, result$message
      ),
      "heavilyright_integration_error",
      value = result$value,
      estimated_error = result$abs.error
    )
  }
  range_tolerance <- max(5e-9, 25 * tolerance)
  if (!is.null(value_bounds[[1L]]) && result$value < value_bounds[[1L]] - range_tolerance) {
    .hr_stop("inversion result is below its mathematical range", "heavilyright_integration_error")
  }
  if (!is.null(value_bounds[[2L]]) && result$value > value_bounds[[2L]] + range_tolerance) {
    .hr_stop("inversion result is above its mathematical range", "heavilyright_integration_error")
  }
  value <- result$value
  if (!is.null(value_bounds[[1L]])) value <- max(value_bounds[[1L]], value)
  if (!is.null(value_bounds[[2L]])) value <- min(value_bounds[[2L]], value)
  if (diagnostics) {
    return(structure(
      list(value = value, estimated_error = result$abs.error, message = result$message),
      class = "heavilyright_probability"
    ))
  }
  value
}

.hr_probability_result <- function(value, error, diagnostics, message = "analytic branch") {
  if (diagnostics) {
    structure(list(value = value, estimated_error = error, message = message),
      class = "heavilyright_probability")
  } else value
}

.hr_hc_integrand <- function(z, x, weights, density = FALSE) {
  weights <- weights[weights != 0]
  location <- (-sum(weights * log(weights)) + 1 - .hr_euler_gamma) * 2 / pi
  scale_factor <- if (x > location + 4) (location + 4) * 10 / x else 16
  argument <- scale_factor * z * weights
  sici <- .hr_sici(argument)
  f_star <- -2 / pi * (
    cos(argument) * (sici$si - pi / 2) - sin(argument) * sici$ci
  )
  components <- -f_star + 2 * cos(argument) + 2i * sin(argument)
  log_product <- sum(log(components))
  if (density) {
    2 * Im(exp(log_product - x * scale_factor * z)) * scale_factor / (2 * pi)
  } else {
    2 * Im(exp(log_product - x * scale_factor * z)) / (2 * pi * z)
  }
}

.hr_eh_integrand <- function(z, x, weights, density = FALSE) {
  weights <- weights[weights != 0]
  location <- (-sum(weights * log(weights)) + 1 - .hr_euler_gamma) * 2 / pi
  scale_factor <- if (x > location + 4) (location + 4) * 10 / x else 16
  argument <- scale_factor * z * weights
  transformed <- argument * scaled_ei(argument) - 1
  components <- -transformed + 1i * pi * exp(log(argument) - argument)
  log_product <- sum(log(components) + argument)
  if (density) {
    2 * Im(exp(log_product - x * scale_factor * z)) * scale_factor / (2 * pi)
  } else {
    2 * Im(exp(log_product - x * scale_factor * z)) / (2 * pi * z)
  }
}

.hr_null_eval <- function(x, weights, law, quantity, diagnostics = FALSE) {
  if (length(x) != 1L || !is.numeric(x) || is.na(x)) stop("x must be a non-missing scalar", call. = FALSE)
  count <- .hr_null_count(weights)
  weights <- .hr_null_weights(weights)
  active <- weights[weights != 0]
  result <- function(value, error = 0, message = "analytic branch") {
    .hr_probability_result(as.numeric(value), as.numeric(error), diagnostics, message)
  }
  if (x == Inf) return(result(if (quantity == "cdf") 1 else 0))
  distance <- if (law == "hcauchy") x else if (count) x - 1 else {
    if (x == -Inf) -Inf else .hr_compensated_sum(c(x, -active))
  }
  if (distance <= 0) return(result(if (quantity == "sf") 1 else 0))
  if (law == "hcauchy") {
    if (length(active) == 1L) {
      value <- switch(quantity,
        sf = 2 / pi * atan(active / x),
        cdf = 2 / pi * atan(x / active),
        pdf = if (x > active) (2 / pi) * ((active / x) / x) / (1 + (active / x)^2)
          else (2 / pi / active) / (1 + (x / active)^2))
      return(result(value))
    }
    location <- (-sum(active * log(active)) + 1 - .hr_euler_gamma) * 2 / pi
    lower <- x <= max(0.5, location - 1)
  } else {
    if (length(active) == 1L) {
      value <- switch(quantity, sf = active / x, cdf = distance / x,
        pdf = (active / x) / x)
      return(result(value))
    }
    lower <- distance <= 0.5 || x <= -sum(active * log(active)) - .hr_euler_gamma
  }
  if (lower) {
    evaluated <- .hr_lower_tail(distance, active, law, cdf = quantity != "pdf")
    value <- evaluated[1L]; error <- evaluated[2L]
    if (quantity == "sf") { value <- 1 - value; error <- error + .Machine$double.eps }
    return(result(value, error, "lower-tail convolution or normalized Laplace inversion"))
  }
  integrand <- if (law == "hcauchy") .hr_hc_integrand else .hr_eh_integrand
  evaluated <- .hr_integrate_checked(
    function(z) integrand(z, x, weights, density = quantity == "pdf"),
    if (law == "hcauchy") 0 else 1e-150,
    if (quantity == "pdf") list(0, NULL) else list(0, 1), TRUE
  )
  # The quadrature estimate excludes floating evaluation of the transform
  # product. Account for its study-count-dependent arithmetic separately.
  evaluated$estimated_error <- evaluated$estimated_error +
    64 * .Machine$double.eps * (1 + length(active)) * abs(evaluated$value)
  if (evaluated$estimated_error > 100 * max(2e-13, 2e-11 * abs(evaluated$value))) {
    .hr_lower_error("upper-tail result failed its total numerical error check")
  }
  if (quantity == "cdf") {
    evaluated$value <- 1 - evaluated$value
    evaluated$estimated_error <- evaluated$estimated_error + .Machine$double.eps
  }
  if (diagnostics) evaluated else evaluated$value
}

#' Finite-number weighted half-Cauchy null law
#'
#' These functions numerically invert the finite-number independence law. The
#' calculation is numerical rather than exact arithmetic. Set `diagnostics`
#' to receive the numerical error estimate. Failed quadrature raises a typed
#' condition. Lower tails use positive convolution or normalized Laplace
#' inversion, with direct CDF evaluation. Analytic support boundaries and
#' bounded underflow can return zero without quadrature; bounded underflow
#' retains a positive error allowance. See the mathematical-methods vignette.
#'
#' @param x,q Numeric scalar score.
#' @param p CDF probability.
#' @param weights A positive integer for mathematical equal weights, or a
#'   nonnegative weight vector whose sum is one within numerical tolerance.
#'   Vectors are not renormalized. Zero weights are removed. For reciprocal
#'   scores, a count has support exactly one; an explicit vector has support
#'   at the exact real sum of its supplied binary floating-point entries.
#' @param log Return log density.
#' @param lower.tail,log.p Standard R distribution arguments.
#' @param diagnostics Return a diagnostic list instead of a bare value.
#' @return A number, or a diagnostic list with `value`, `estimated_error`, and
#'   `message` when requested. The error is on the unlogged probability/density
#'   scale even when `log.p` or `log` is true; it is not a rigorous error bound.
#'   Quantiles accept one probability strictly between zero and one after
#'   applying `lower.tail` and `log.p`.
#' @seealso [pehmean()], [new_combination()]
#' @examples
#' phcmean(5, weights = c(0.2, 0.3, 0.5),
#'         lower.tail = FALSE, diagnostics = TRUE)
#' dhcmean(2, weights = 1)
#' threshold <- qhcmean(0.95, weights = 2)
#' phcmean(threshold, weights = 2)
#' phcmean(0.01, weights = 10, diagnostics = TRUE)  # A tiny direct CDF
#' @export
phcmean <- function(q, weights, lower.tail = TRUE, log.p = FALSE, diagnostics = FALSE) {
  result <- .hr_null_eval(q, weights, "hcauchy", if (lower.tail) "cdf" else "sf", diagnostics)
  if (diagnostics) {
    if (log.p) result$value <- log(result$value)
    return(result)
  }
  if (log.p) log(result) else result
}

#' @rdname phcmean
#' @export
dhcmean <- function(x, weights, log = FALSE, diagnostics = FALSE) {
  result <- .hr_null_eval(x, weights, "hcauchy", "pdf", diagnostics)
  if (diagnostics) {
    if (log) result$value <- log(result$value)
    return(result)
  }
  if (log) log(result) else result
}

.hr_quantile <- function(p, cdf, lower) {
  if (length(p) != 1L || !is.numeric(p) || !is.finite(p) || p <= 0 || p >= 1) {
    stop("p must lie strictly between zero and one", call. = FALSE)
  }
  low <- lower
  high <- max(100, lower + 1)
  for (i in seq_len(200L)) {
    if (cdf(high) >= p) break
    low <- high
    high <- high * 2
    if (i == 200L) .hr_stop("failed to bracket null-law quantile", "heavilyright_root_error")
  }
  tryCatch(
    stats::uniroot(function(value) cdf(value) - p, c(low, high), tol = 2e-10, maxiter = 300L)$root,
    error = function(error) .hr_stop(
      sprintf("failed to invert null-law CDF: %s", conditionMessage(error)),
      "heavilyright_root_error"
    )
  )
}

#' @rdname phcmean
#' @export
qhcmean <- function(p, weights, lower.tail = TRUE, log.p = FALSE) {
  if (log.p) p <- exp(p)
  if (!lower.tail) p <- 1 - p
  .hr_null_weights(weights)
  .hr_quantile(p, function(value) phcmean(value, weights), 0)
}

#' Finite-number weighted reciprocal-score null law
#'
#' These functions evaluate the independence law of a weighted sum of `1/U`
#' for independent uniform p-values. A scalar count has support boundary one;
#' a supplied vector has support at the sum of its entries, without
#' renormalization. Support distance is evaluated with compensated summation.
#' Numerical and diagnostic conventions are the same as [phcmean()].
#'
#' @inheritParams phcmean
#' @return A numeric value or, for density/probability calls with diagnostics
#'   enabled, a list with `value`, `estimated_error`, and `message`. Error
#'   estimates remain on the unlogged scale. Quantiles are scalar.
#' @seealso [phcmean()], [new_combination()]
#' @examples
#' pehmean(5, weights = 2, lower.tail = FALSE, diagnostics = TRUE)
#' dehmean(2, weights = 1)
#' threshold <- qehmean(0.95, weights = 2)
#' pehmean(threshold, weights = 2)
#' pehmean(2, weights = 100, diagnostics = TRUE)  # Stable lower-tail CDF
#' @export
pehmean <- function(q, weights, lower.tail = TRUE, log.p = FALSE, diagnostics = FALSE) {
  result <- .hr_null_eval(q, weights, "ehmp", if (lower.tail) "cdf" else "sf", diagnostics)
  if (diagnostics) {
    if (log.p) result$value <- log(result$value)
    return(result)
  }
  if (log.p) log(result) else result
}

#' @rdname pehmean
#' @export
dehmean <- function(x, weights, log = FALSE, diagnostics = FALSE) {
  result <- .hr_null_eval(x, weights, "ehmp", "pdf", diagnostics)
  if (diagnostics) {
    if (log) result$value <- log(result$value)
    return(result)
  }
  if (log) log(result) else result
}

#' @rdname pehmean
#' @export
qehmean <- function(p, weights, lower.tail = TRUE, log.p = FALSE) {
  if (log.p) p <- exp(p)
  if (!lower.tail) p <- 1 - p
  checked <- .hr_null_weights(weights)
  lower <- if (.hr_null_count(weights)) 1 else sum(checked) * (1 - 2 * .Machine$double.eps)
  .hr_quantile(p, function(value) pehmean(value, weights), lower)
}

#' @export
print.heavilyright_probability <- function(x, ...) {
  cat(sprintf("value: %.17g\nestimated numerical error: %.3g\n", x$value, x$estimated_error))
  invisible(x)
}
