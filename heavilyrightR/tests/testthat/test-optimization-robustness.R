test_that("analytic radial gradients and Hessians match independent differences", {
  for (method in c("hcauchy", "ehmp")) for (q in c(1L, 2L, 4L)) for (df in list(NULL, 9.5)) {
    kernel <- list(dimension = q, method = method, blocks = list(
      list(q = q, df = df, B = diag(q), b = rep(.12, q), weight = .4),
      list(q = q, df = df, B = diag(q) * .8, b = rep(-.07, q), weight = .6)))
    z <- seq(.2, .5, length.out = q); h <- 2e-6
    actual <- heavilyright:::.hr_kernel_eval(z, kernel, hessian = TRUE)
    fd_gradient <- numeric(q); fd_H <- matrix(0, q, q)
    for (j in seq_len(q)) {
      plus <- minus <- z; plus[j] <- plus[j] + h; minus[j] <- minus[j] - h
      p <- heavilyright:::.hr_kernel_eval(plus, kernel)
      m <- heavilyright:::.hr_kernel_eval(minus, kernel)
      fd_gradient[j] <- (p$value - m$value) / (2 * h)
      fd_H[, j] <- (p$gradient - m$gradient) / (2 * h)
    }
    expect_equal(actual$gradient, fd_gradient, tolerance = 2e-7)
    expect_equal(actual$hessian, fd_H, tolerance = 2e-7)
  }
})

test_that("direction magnitude and coordinate units preserve support", {
  summaries <- list(study(c(.1, .2), diag(c(.04, .09))), study(c(.14, .1), diag(c(.05, .07))))
  fit <- fit_meta_md(summaries)
  baseline <- support_interval(fit, c(1, .3))
  expect_true(baseline$success)
  for (factor in c(1e-100, 1e100)) {
    result <- support_interval(fit, c(1, .3) * factor)
    expect_true(result$success)
    expect_equal(c(result$lower, result$upper) / factor, c(baseline$lower, baseline$upper), tolerance = 2e-7)
  }
  scale <- c(1e-10, 1e6)
  converted <- lapply(summaries, function(s) study(s$estimate * scale, s$vcov * outer(scale, scale)))
  result <- support_interval(fit_meta_md(converted), c(1, .3) / scale)
  expect_true(result$success)
  expect_equal(c(result$lower, result$upper), c(baseline$lower, baseline$upper), tolerance = 2e-7)
})

test_that("weak disconnected components cannot consume the support budget", {
  mass <- 1e-12
  studies <- list(study(0, matrix(1), matrix(c(1, 0), 1), 1),
    study(.7, matrix(1), matrix(c(1, 0), 1), 1),
    study(0, matrix(1), matrix(c(0, 1), 1), 1))
  fit <- fit_meta_md(studies, weights = c(.9 - mass, .1, mass))
  result <- support_interval(fit, c(1, 0))
  expect_true(result$success)
  threshold <- fit$specification$cutoff
  expect_equal(c(result$lower, result$upper), c(-(threshold - .07), threshold + .07) / (1 - mass), tolerance = 1e-7)
  expect_equal(fit$minimum$point, c(0, 0), tolerance = 1e-9)
})

test_that("tail floors and nonconvex methods never authorize deletion", {
  studies <- list(study(c(0, 0), diag(2)), study(c(3, 0), diag(2) * 1e-10))
  caught <- tryCatch(fit_meta_md(studies, weights = c(1, 1e-151),
    empty_policy = empty_set_policy(warn = FALSE)), heavilyright_empty_numerical_error = identity)
  expect_s3_class(caught, "heavilyright_empty_numerical_error")
  expect_length(caught$diagnostics$removals, 0)
  expect_equal(caught$diagnostics$final_state, "unknown")
  expect_error(fit_meta_md(studies, method = "fisher"), "only hcauchy and ehmp")
})

test_that("unrepresentable endpoints fail reconstruction explicitly", {
  fit <- fit_meta_md(list(study(c(1e16, 1e16), diag(2) * 1e-4)))
  result <- support_interval(fit, c(1, 0))
  expect_false(result$success)
  expect_match(result$upper_diagnostics$message, "reconstruction")
  rotation <- matrix(c(1, 1, -1, 1), 2) / sqrt(2)
  covariance <- rotation %*% diag(c(1, 1e-6)) %*% t(rotation)
  fit <- fit_meta_md(list(study(c(0, 0), covariance)))
  direction <- c(1, -.9)
  result <- support_interval(fit, direction)
  radius <- sqrt(qchisq(.95, 2) * as.numeric(crossprod(direction, covariance %*% direction)))
  expect_true(result$success)
  expect_equal(c(result$lower, result$upper), c(-radius, radius), tolerance = 2e-7)
})

test_that("incompatible nearby cusps do not acquire a fictitious subgradient", {
  kernel <- list(dimension = 1L, method = "hcauchy", blocks = list(
    list(q = 1L, df = NULL, B = matrix(1), b = 0, weight = .5),
    list(q = 1L, df = NULL, B = matrix(1), b = -5e-10, weight = .5)))
  result <- heavilyright:::.hr_subgradient(0, kernel)
  expect_length(result$active, 0)
  expect_equal(result$point, 0)
})

test_that("numerical failure after removal preserves the entire audit", {
  actual <- heavilyright:::.hr_md_fit_once
  testthat::local_mocked_bindings(.hr_md_fit_once = function(studies, dimension, spec) {
    if (length(studies) == 1L) stop("injected failure after first removal")
    actual(studies, dimension, spec)
  }, .package = "heavilyright")
  summaries <- list(study(c(-4, 0), diag(2)), study(c(4, 0), diag(2)))
  caught <- tryCatch(fit_meta_md(summaries, empty_policy = empty_set_policy(warn = FALSE)), heavilyright_empty_numerical_error = identity)
  expect_s3_class(caught, "heavilyright_empty_numerical_error")
  expect_length(caught$diagnostics$removals, 1)
  expect_length(caught$diagnostics$trace, 2)
  expect_true(caught$diagnostics$trace[[1]]$empty)
  expect_true(is.na(caught$diagnostics$trace[[2]]$empty))
  expect_equal(caught$diagnostics$final_state, "unknown")
})

test_that("near-empty support uses feasible slack rather than absolute cutoff", {
  kernel <- list(dimension = 1L, method = "hcauchy", blocks = list(
    list(q = 1L, df = NULL, B = matrix(1), b = 1, weight = .5),
    list(q = 1L, df = NULL, B = matrix(1), b = -1, weight = .5)))
  minimum <- heavilyright:::.hr_kernel_eval(0, kernel)$value
  cutoff <- minimum + 1e-10
  false_candidate <- heavilyright:::.hr_endpoint_certificate(1e-4, 1, kernel, cutoff, cutoff - minimum)
  expect_false(false_candidate$diagnostics$success)
  expect_gt(false_candidate$diagnostics$scaled_boundary_error, 1)
})

test_that("a linearized cusp projection must also pass actual floating-point residuals", {
  kernel <- list(dimension = 1L, method = "hcauchy", blocks = list(
    list(q = 1L, df = NULL, B = matrix(1e12), b = -1e12 - 2^-13, weight = 1)))
  actual <- heavilyright:::.hr_subgradient(1, kernel)
  expect_length(actual$active, 0)
  expect_equal(actual$point, 1)
})

test_that("an unrepresentable physical minimum fails without removal", {
  summaries <- list(study(2^54, matrix((4 / 3)^2)), study(2^54 + 4, matrix((4 / 3)^2)))
  error <- tryCatch(fit_meta_md(summaries, empty_policy = empty_set_policy(warn = FALSE)), heavilyright_empty_numerical_error = identity)
  expect_s3_class(error, "heavilyright_empty_numerical_error")
  expect_match(conditionMessage(error), "reconstruction|reconstructed")
  expect_length(error$diagnostics$removals, 0)
})

test_that("universal score baseline can establish emptiness", {
  summaries <- list(study(c(0, 0), diag(2)), study(c(0, 0), diag(2)))
  error <- tryCatch(fit_meta_md(summaries, alpha = .99, calibration = "landau"), heavilyright_empty_region_error = identity)
  expect_s3_class(error, "heavilyright_empty_region_error")
  expect_gt(error$diagnostics$trace[[1]]$feasible_set_score_lower_bound, error$diagnostics$trace[[1]]$cutoff)
})

test_that("overflowing requested support values are not certified", {
  result <- support_interval(fit_meta_md(list(study(3, matrix(1)))), 1e308)
  expect_false(result$success)
  expect_match(result$upper_diagnostics$message, "overflow")
})

test_that("optimizer success alone cannot certify a minimum or deletion", {
  testthat::local_mocked_bindings(.hr_slsqp = function(x0, ...) {
    list(par = x0, convergence = 1L, message = "injected false success")
  }, .package = "heavilyright")
  summaries <- list(study(c(0, 0), diag(2)), study(c(1, 0), diag(2)))
  error <- tryCatch(fit_meta_md(summaries, weights = c(.1, .9),
    empty_policy = empty_set_policy(warn = FALSE)), heavilyright_empty_numerical_error = identity)
  expect_s3_class(error, "heavilyright_empty_numerical_error")
  expect_match(conditionMessage(error), "stationarity|gap")
  expect_length(error$diagnostics$removals, 0)
})
