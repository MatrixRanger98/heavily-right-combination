test_that("scalar interval matches the validated baseline", {
  fit <- fit_meta_1d(c(0.1, 0.14, 0.08), c(0.04, 0.05, 0.03))
  expect_s3_class(fit, "heavilyright_ci")
  expect_equal(c(fit$lower, fit$upper, fit$estimate),
    c(0.027683700431938574, 0.15206471399215454, 0.1), tolerance = 3e-9)
  expect_false(fit$empty)
})

test_that("translation and tiny units preserve scalar inference", {
  estimate <- c(0.1, 0.2, 0.3)
  se <- c(0.1, 0.12, 0.08)
  baseline <- fit_meta_1d(estimate, se)
  for (case in list(c(1000, 1), c(0, 1e-12))) {
    shifted <- fit_meta_1d(case[1] + case[2] * estimate, case[2] * se)
    expect_equal((c(shifted$lower, shifted$upper, shifted$estimate) - case[1]) / case[2],
      c(baseline$lower, baseline$upper, baseline$estimate), tolerance = 3e-8)
  }
})

test_that("single-study intervals are analytic for normal and fractional t", {
  for (df in list(NULL, 0.5, 2.5)) {
    fit <- fit_meta_1d(0.2, 0.1, df = df)
    quantile <- if (is.null(df)) qnorm(0.975) else qt(0.975, df)
    expect_equal(c(fit$lower, fit$upper), 0.2 + c(-1, 1) * 0.1 * quantile, tolerance = 1e-12)
  }
})

test_that("empty-set treatment is off by default and retains complete traces", {
  disabled <- fit_meta_1d(c(-3, 3), 0.5)
  expect_true(disabled$empty)
  expect_false(disabled$empty_set_diagnostics$enabled)
  expect_length(disabled$empty_set_diagnostics$trace, 1)

  adjusted <- suppressWarnings(fit_meta_1d(c(-3, 3), 0.5,
    empty_policy = empty_set_policy(max_removals = 1, warn = FALSE),
    study_labels = c("left", "right")))
  expect_false(adjusted$empty)
  expect_true(adjusted$empty_set_diagnostics$encountered)
  expect_length(adjusted$empty_set_diagnostics$trace, 2)
  expect_equal(length(adjusted$empty_set_diagnostics$final_original_indices), 1)
  expect_true(adjusted$empty_set_diagnostics$trace[[1]]$candidates[[1]]$selected)
})

test_that("unsupported t convexity fails before optimization", {
  expect_error(fit_meta_1d(c(0, 0.1), 1, df = 0.5), "df >= 1")
})

