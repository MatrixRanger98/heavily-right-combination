#' Scaled exponential integral
#'
#' Computes `exp(-x) * Ei(x)` for positive `x`. Moderate arguments use the
#' convergent defining series; large arguments use an optimally truncated
#' asymptotic expansion. The former production Ramanujan route is not used.
#'
#' @param x Positive numeric values.
#' @return Numeric values with the same length as `x`.
#' @examples
#' scaled_ei(c(1, 50, 1000))
#' @export
scaled_ei <- function(x) {
  .hr_assert_numeric(x, "x")
  if (any(x <= 0)) stop("x must be positive", call. = FALSE)
  vapply(as.numeric(x), function(value) {
    if (value < 50) {
      term <- value
      total <- .hr_euler_gamma + log(value) + term
      for (k in 2:10000) {
        term <- term * value / k
        add <- term / k
        total_new <- total + add
        if (abs(add) <= 2e-16 * max(1, abs(total_new))) {
          total <- total_new
          break
        }
        total <- total_new
      }
      return(exp(-value) * total)
    }
    term <- 1 / value
    total <- term
    previous <- abs(term)
    for (order in 1:79) {
      term <- term * order / value
      magnitude <- abs(term)
      if (magnitude > previous) break
      total <- total + term
      if (magnitude <= 2e-16 * abs(total)) break
      previous <- magnitude
    }
    total
  }, numeric(1L), USE.NAMES = FALSE)
}

.hr_sici_scalar <- function(x) {
  if (!is.finite(x) || x < 0) stop("sine/cosine integral requires a finite nonnegative argument", call. = FALSE)
  if (x == 0) return(c(si = 0, ci = -Inf))
  if (x <= 12) {
    si_term <- x
    si <- si_term
    ci_term <- -x * x / 2
    ci_sum <- ci_term / 2
    for (k in 1:500) {
      si_term <- si_term * (-x * x) / ((2 * k) * (2 * k + 1))
      si_add <- si_term / (2 * k + 1)
      si <- si + si_add
      ci_term <- ci_term * (-x * x) / ((2 * k + 1) * (2 * k + 2))
      ci_add <- ci_term / (2 * k + 2)
      ci_sum <- ci_sum + ci_add
      if (max(abs(si_add), abs(ci_add)) <= 2e-16 * max(1, abs(si), abs(ci_sum))) break
    }
    return(c(si = si, ci = .hr_euler_gamma + log(x) + ci_sum))
  }
  f_term <- 1 / x
  f <- f_term
  g_term <- 1 / (x * x)
  g <- g_term
  prev_f <- abs(f_term)
  prev_g <- abs(g_term)
  for (k in 1:100) {
    next_f <- f_term * (-(2 * k - 1) * (2 * k)) / (x * x)
    next_g <- g_term * (-(2 * k) * (2 * k + 1)) / (x * x)
    if (abs(next_f) <= prev_f) {
      f <- f + next_f
      f_term <- next_f
      prev_f <- abs(next_f)
    }
    if (abs(next_g) <= prev_g) {
      g <- g + next_g
      g_term <- next_g
      prev_g <- abs(next_g)
    }
    if (abs(next_f) > prev_f && abs(next_g) > prev_g) break
  }
  c(si = pi / 2 - cos(x) * f - sin(x) * g, ci = sin(x) * f - cos(x) * g)
}

.hr_sici <- function(x) {
  values <- vapply(x, .hr_sici_scalar, numeric(2L))
  list(si = values[1L, ], ci = values[2L, ])
}
