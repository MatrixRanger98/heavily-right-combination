# Native R lower-tail calculations. Formulas and remainder bounds are recorded
# in the mathematical-methods vignette. No Python or external special-function
# runtime is used. Numerical error estimates are not interval enclosures.
.hr_smallest <- .Machine$double.xmin * .Machine$double.eps

# Error-free addition of partials avoids rounding the support before subtracting
# x. Sorting also makes the result invariant to input order.
.hr_compensated_sum <- function(x) {
  partials <- numeric()
  for (value in x[order(abs(x))]) {
    next_partials <- numeric()
    for (part in partials) {
      if (abs(value) < abs(part)) { temporary <- value; value <- part; part <- temporary }
      high <- value + part
      low <- part - (high - value)
      if (low != 0) next_partials <- c(next_partials, low)
      value <- high
    }
    partials <- c(next_partials, value)
  }
  sum(partials)
}

.hr_lower_error <- function(message, ...) .hr_stop(message, "heavilyright_integration_error", ...)

# Contracted incomplete-gamma fraction for exp(z)*E_order(z), order 1 or 2.
# DLMF 8.19.1 and 8.9.2. Evaluating E2 directly avoids 1-z*exp(z)*E1(z)
# cancellation in the reciprocal transform at moderate arguments.
.hr_scaled_en_fraction <- function(value, order = 1L) {
  b <- value + order
  c <- 1e300 + 0i
  d <- 1 / b
  answer <- d
  stable <- 0L
  for (k in seq_len(2000L)) {
    numerator <- -as.double(k) * (k + order - 1)
    b <- b + 2
    d <- b + numerator * d
    c <- b + numerator / c
    if (Mod(d) < 1e-300) d <- 1e-300 + 0i
    if (Mod(c) < 1e-300) c <- 1e-300 + 0i
    d <- 1 / d
    delta <- c * d
    answer <- answer * delta
    if (!is.finite(answer)) .hr_lower_error("complex exponential-integral fraction overflowed")
    stable <- if (Mod(delta - 1) <= 4 * .Machine$double.eps) stable + 1L else 0L
    if (stable >= 3L) return(answer)
  }
  .hr_lower_error(sprintf("complex exponential-integral fraction did not converge at %s", format(value, digits = 17)))
}

# exp(z)*E1(z), principal branch: DLMF 6.6.2 and the contracted continued
# fraction in 6.9.1. Near the negative real cut, the series preserves its side
# through the principal complex logarithm; no asymptotic branch is substituted.
.hr_scaled_e1 <- function(z) {
  z <- as.complex(z)
  if (any(!is.finite(z) | z == 0)) .hr_lower_error("invalid complex E1 argument")
  vapply(z, function(value) {
    series <- Mod(value) <= 1.5 || (Re(value) < 0 &&
      (Mod(value) <= 4 || abs(Im(value)) <= -0.2 * Re(value)))
    if (series) {
      term <- -value
      total <- term
      correction <- 0i
      for (k in 2:1000) {
        term <- term * (-value / k)
        add <- term / k - correction
        next_total <- total + add
        correction <- (next_total - total) - add
        total <- next_total
        if (Mod(term / k) <= 2e-16 * max(Mod(total), .hr_smallest)) {
          answer <- exp(value) * (-.hr_euler_gamma - log(value) - total)
          if (!is.finite(answer)) .hr_lower_error("complex E1 series overflowed")
          return(answer)
        }
      }
      .hr_lower_error("complex E1 series did not converge")
    }
    .hr_scaled_en_fraction(value)
  }, complex(1), USE.NAMES = FALSE)
}

.hr_laplace <- function(s, law, derivative = FALSE) {
  s <- as.complex(s)
  if (any(!is.finite(s) | Re(s) <= 0)) .hr_lower_error("Laplace argument is outside the positive half-plane")
  threshold <- if (law == "hcauchy") 60 else 50
  large <- Mod(s) >= threshold
  value <- slope <- complex(length(s))
  if (any(!large)) {
    z <- s[!large]
    if (law == "hcauchy") {
      minus <- .hr_scaled_e1(-1i * z)
      plus <- .hr_scaled_e1(1i * z)
      value[!large] <- (minus - plus) / (1i * pi)
      slope[!large] <- z * (minus + plus) / pi
    } else {
      factors <- vapply(z, function(argument) {
        if (Mod(argument) <= 1.5) 1 - argument * .hr_scaled_e1(argument)
        else .hr_scaled_en_fraction(argument, order = 2L)
      }, complex(1))
      value[!large] <- factors
      if (derivative) slope[!large] <- z * (.hr_scaled_e1(z) - factors)
    }
  }
  if (any(large)) {
    z <- s[large]
    term <- 1 / z
    total <- differentiated <- term
    count <- if (law == "hcauchy") 25L else 31L
    for (k in seq_len(count)) {
      if (law == "hcauchy") {
        term <- term * (-(2 * k) * (2 * k - 1)) * (1 / z)^2
        power <- 2 * k + 1
      } else {
        term <- term * (-(k + 1) / z)
        power <- k + 1
      }
      total <- total + term
      differentiated <- differentiated + power * term
    }
    factor <- if (law == "hcauchy") 2 / pi else 1
    value[large] <- factor * total
    slope[large] <- factor * differentiated
  }
  if (any(!is.finite(value) | value == 0)) .hr_lower_error("nonfinite or zero Laplace transform")
  if (!derivative) return(value)
  mean <- Re(slope / value) / Re(s)
  if (any(Im(s) != 0 | !is.finite(mean) | mean <= 0)) .hr_lower_error("invalid real tilted mean")
  mean
}

.hr_lower_quad <- function(...) stats::integrate(...)

.hr_lower_integrate <- function(fun, lower = 0, upper = Inf, absolute = 2e-13) {
  result <- tryCatch(.hr_lower_quad(function(x) vapply(x, fun, numeric(1)),
    lower, upper, subdivisions = 600L, rel.tol = 2e-11, abs.tol = absolute,
    stop.on.error = FALSE), error = function(e) .hr_lower_error(conditionMessage(e)))
  target <- max(absolute, 2e-11 * abs(result$value))
  if (!is.finite(result$value) || abs(result$value) == .Machine$double.xmax ||
      !is.finite(result$abs.error) || result$abs.error < 0 ||
      result$abs.error > 100 * target || !identical(result$message, "OK")) {
    .hr_lower_error("lower-tail quadrature failed its convergence/error check",
      value = result$value, estimated_error = result$abs.error, quadrature_message = result$message)
  }
  c(value = result$value, error = result$abs.error)
}

# Even columns of Wynn's epsilon table accelerate the sequence of cycle sums.
# Singular differences invalidate that extrapolant rather than yielding a zero.
.hr_epsilon <- function(sequence) {
  previous <- rep(0, length(sequence) + 1L)
  current <- sequence
  answer <- utils::tail(sequence, 1)
  for (order in seq_len(min(12L, length(sequence) - 1L))) {
    delta <- diff(current)
    next_column <- previous[seq.int(2L, length(current))] + 1 / delta
    previous <- current
    current <- next_column
    if (order %% 2L == 0L && is.finite(utils::tail(current, 1))) answer <- utils::tail(current, 1)
  }
  answer
}

.hr_lower_cycles <- function(fun, frequency, max_cycles = 100L) {
  if (!is.finite(frequency) || frequency <= 0) .hr_lower_error("invalid inversion frequency")
  width <- pi / frequency
  sums <- numeric()
  estimates <- numeric()
  error <- magnitude <- 0
  for (cycle in seq_len(max_cycles)) {
    block <- .hr_lower_integrate(fun, (cycle - 1) * width, cycle * width, absolute = 2e-14)
    magnitude <- magnitude + abs(block[1L])
    error <- error + block[2L]
    sums <- c(sums, if (length(sums)) utils::tail(sums, 1) + block[1L] else block[1L])
    if (cycle >= 8L) {
      extrapolated <- .hr_epsilon(utils::tail(sums, 20L))
      estimates <- c(estimates, extrapolated)
      if (length(estimates) >= 5L) {
        recent <- utils::tail(estimates, 5L)
        residual <- max(abs(diff(recent)), abs(extrapolated - recent[1L]))
        estimate_error <- 10 * residual + error + 64 * .Machine$double.eps * magnitude
        if (is.finite(extrapolated) && estimate_error <= max(2e-13, 2e-11 * abs(extrapolated))) {
          return(c(value = extrapolated, error = estimate_error, accumulation = magnitude))
        }
      }
    }
  }
  .hr_lower_error("oscillatory lower-tail cycles did not converge", cycles = max_cycles)
}

.hr_boundary_expansion <- function(x, weights, law, cdf) {
  m <- length(weights)
  log_upper <- (m - 1) * log(x) - lgamma(m) - sum(log(weights))
  if (law == "hcauchy") log_upper <- log_upper + m * log(2 / pi)
  if (cdf) log_upper <- log_upper + log(x) - log(m)
  allowance <- 32 * .Machine$double.eps * (1 + m + abs(log_upper))
  if (log_upper + allowance < log(.hr_smallest)) return(c(value = 0, error = .hr_smallest))
  limit <- if (law == "hcauchy") 1e-4 else 1e-6
  if (x > limit * min(weights)) return(NULL)
  ratios <- x / weights
  if (law == "hcauchy") {
    ratios <- ratios^2
    first <- 2 * sum(ratios) / (m * (m + 1))
    second <- (22 * sum(ratios^2) + 2 * sum(ratios)^2) / (m * (m + 1) * (m + 2) * (m + 3))
    if (cdf) { first <- first * m / (m + 2); second <- second * m / (m + 4) }
  } else {
    first <- 2 * sum(ratios) / m
    second <- (2 * sum(ratios)^2 + 4 * sum(ratios^2)) / (m * (m + 1))
    if (cdf) { first <- first * m / (m + 1); second <- second * m / (m + 2) }
  }
  upper <- exp(log_upper)
  c(value = upper * (1 - first), error = max(upper * (second + allowance), .hr_smallest))
}

# A finite positive convolution is simpler and better conditioned than an
# oscillatory transform for two coordinates, especially at extreme weight ratios.
.hr_two_score <- function(x, weights, law, cdf) {
  w <- sort(weights); a <- w[1L]; b <- w[2L]
  if (law == "hcauchy") {
    bound <- atan(x / a)
    integrand <- function(t) {
      remaining <- max(0, x - a * tan(bound * t))
      if (cdf) (2 / pi) * atan(remaining / b) else (1 / b) / (1 + (remaining / b)^2)
    }
    scale <- (2 / pi) * bound
    if (!cdf) scale <- scale * 2 / pi
  } else {
    bound <- x / (a + x)
    integrand <- function(t) {
      v <- bound * t
      remaining <- max(0, x - a * v / (1 - v))
      if (cdf) remaining / (b + remaining) else (b / (b + remaining)) / (b + remaining)
    }
    scale <- bound
  }
  # Integrate on [0,1] and factor out the typical height; absolute tolerances
  # then cannot silently erase a tiny but representable CDF.
  height <- max(integrand(0), integrand(0.5))
  if (!is.finite(height) || height <= 0 || !is.finite(scale) || scale <= 0) {
    .hr_lower_error("two-coordinate convolution cannot be represented")
  }
  result <- .hr_lower_integrate(function(t) integrand(t) / height, 0, 1)
  factor <- scale * height
  value <- factor * result[1L]
  error <- max(factor * (result[2L] + 64 * .Machine$double.eps * abs(result[1L])), .hr_smallest)
  if (!is.finite(value) || value <= 0 || (cdf && value > 1) || !is.finite(error) ||
      error > 100 * max(2e-13, 2e-11 * value)) .hr_lower_error("two-coordinate convolution failed its final error check")
  c(value = unname(value), error = unname(error))
}

.hr_lower_tail <- function(x, weights, law, cdf = FALSE) {
  if (!is.finite(x) || x <= 0) .hr_lower_error("lower-tail inversion needs a positive finite distance from support")
  boundary <- .hr_boundary_expansion(x, weights, law, cdf)
  if (!is.null(boundary)) return(boundary)
  if (length(weights) == 2L) return(.hr_two_score(x, weights, law, cdf))
  unique <- sort(unique(weights))
  counts <- tabulate(match(weights, unique))
  m <- length(weights)
  log_high <- log(2 * m) - log(x)
  if (log_high > log(.Machine$double.xmax)) .hr_lower_error("tilt bracket overflowed")
  mean_gap <- function(log_a) {
    arguments <- exp(log_a) * unique
    mean <- sum(counts * unique * .hr_laplace(arguments, law, derivative = TRUE))
    if (!is.finite(mean) || mean <= 0) .hr_lower_error("tilted mean cannot be represented")
    log(mean) - log(x)
  }
  log_low <- log_high
  for (iteration in seq_len(1100L)) {
    if (mean_gap(log_low) >= 0) break
    log_low <- log_low - log(2)
    if (iteration == 1100L) .hr_lower_error("could not bracket the lower-tail tilt")
  }
  root <- tryCatch(stats::uniroot(mean_gap, c(log_low, log_high), tol = 2e-14, maxiter = 200L)$root,
    error = function(e) .hr_lower_error(paste("tilt root failed:", conditionMessage(e))))
  tilt <- exp(root)
  factors <- Re(.hr_laplace(tilt * unique, law))
  if (any(!is.finite(factors) | factors <= 0)) .hr_lower_error("nonpositive real Laplace transform")
  logs <- log(factors)
  log_transform <- sum(counts * logs)
  log_prefactor <- tilt * x + log_transform
  log_bound <- log_prefactor
  if (!cdf) log_bound <- log_bound + (if (law == "hcauchy") log(2 / pi) else 0) - max(log(unique) + logs)
  allowance <- 64 * .Machine$double.eps * (1 + m + abs(log_transform) + abs(log_bound))
  if (log_bound + allowance < log(.hr_smallest)) return(c(value = 0, error = .hr_smallest))
  frequency_scale <- tilt / sqrt(m)
  frequency <- frequency_scale * x
  log_scale <- log_prefactor + log(frequency_scale / pi) - if (cdf) log(tilt) else 0
  integrand <- function(u) {
    s <- complex(real = tilt, imaginary = u * frequency_scale)
    exponent <- sum(counts * (log(.hr_laplace(s * unique, law)) - logs)) + 1i * frequency * u
    value <- exp(exponent)
    if (cdf) value <- value * tilt / s
    Re(value)
  }
  result <- NULL
  if (m >= 8L && max(unique) / min(unique) <= 4) {
    result <- tryCatch(.hr_lower_integrate(integrand), heavilyright_integration_error = function(e) NULL)
  }
  if (is.null(result)) result <- .hr_lower_cycles(integrand, frequency)
  normalized <- unname(result[1L])
  if (!is.finite(normalized) || normalized <= 0) .hr_lower_error("lower-tail integral is not positive")
  accumulation <- if (length(result) == 3L) result[3L] else abs(normalized)
  rounding <- 64 * .Machine$double.eps * (1 + m + abs(log_transform) + abs(log_scale))
  value <- exp(log_scale + log(normalized))
  error <- max(exp(log_scale + log(result[2L] + rounding * accumulation)), .hr_smallest)
  if (!is.finite(value) || !is.finite(error) || value < 0 || (cdf && value > 1) ||
      error > 100 * max(2e-13, 2e-11 * value)) .hr_lower_error("lower-tail result failed its final error check")
  c(value = value, error = unname(error))
}
