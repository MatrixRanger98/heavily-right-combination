test_that("single-study support uses an analytic ellipsoid", {
  covariance <- matrix(c(0.04, 0.01, 0.01, 0.09), 2)
  fit <- fit_meta_md(list(study(c(0.2, -0.1), covariance, df = 29)), parameter_dim = 2)
  result <- support_interval(fit, c(1, 0))
  expect_true(result$success)
  expect_equal(c(result$lower, result$upper), c(-0.3260939111953708, 0.7260939111953707), tolerance = 3e-9)
  expect_equal(result$upper_diagnostics$solver, "analytic-single-study-ellipsoid")
  expect_lte(result$upper_diagnostics$boundary_error, 1e-10)
})

test_that("mixed-dimensional projections produce certified support", {
  studies <- list(
    study(c(0.2, 0.3), diag(2) * 0.04, diag(2), label = "a"),
    study(c(0.1, 0.25), diag(2) * 0.05, diag(2), label = "b"),
    study(0.1, matrix(0.03, 1), matrix(c(-1, 1), 1), label = "c")
  )
  fit <- fit_meta_md(studies, parameter_dim = 2)
  expect_true(contains(fit, coef(fit)))
  result <- support_interval(fit, c(1, -1), maxeval = 5000)
  expect_true(result$success, info = paste(result$lower_diagnostics$message, result$upper_diagnostics$message))
  expect_lte(result$lower_diagnostics$scaled_boundary_error, 2e-6)
  expect_lte(result$upper_diagnostics$kkt_residual, 2e-6)
  expect_equal(result$lower, -0.51721219, tolerance = 3e-6)
  expect_equal(result$upper, 0.31601946, tolerance = 3e-6)
})

test_that("invalid covariance and Hotelling inputs fail early", {
  expect_error(study(c(0, 0), matrix(c(1, 2, 2, 1), 2)), "positive definite")
  expect_error(study(c(0, 0), diag(2), df = 1), "exceed")
})

test_that("multivariate emptiness fails closed without study deletion", {
  studies <- list(
    study(c(-10, 0), diag(2) * 1e-4, label = "left"),
    study(c(10, 0), diag(2) * 1e-4, label = "right")
  )
  caught <- tryCatch(
    fit_meta_md(studies, 2, empty_policy = empty_set_policy(warn = FALSE)),
    heavilyright_empty_numerical_error = identity
  )
  expect_s3_class(caught, "heavilyright_empty_numerical_error")
  expect_length(caught$diagnostics$trace, 1)
  expect_equal(caught$diagnostics$final_original_indices, 1:2)
  expect_match(caught$diagnostics$message, "clipped or underflowed tails")
})
