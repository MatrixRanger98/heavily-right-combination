test_that("DAC partitions coordinates and preserves calibrated covariance", {
  set.seed(21)
  x <- matrix(rnorm(120), 30, 4, dimnames = list(NULL, letters[1:4]))
  inputs <- dac_studies(x, blocks = list(first = c("a", "c"), second = c("b", "d")))
  expect_equal(inputs[[1]]$estimate, unname(colMeans(x[, c(1, 3)])))
  expect_equal(unname(inputs[[1]]$vcov), unname(stats::cov(x[, c(1, 3)]) / 30))
  expect_equal(inputs[[1]]$projection, diag(4)[c(1, 3), ])
  expect_equal(inputs[[1]]$df, 29)
  expect_equal(lengths(attr(dac_studies(x, block_size = 3), "blocks")), c(3L, 1L))
  expect_error(dac_studies(x, blocks = list(c(1, 2), c(2, 3, 4))), "partition")
  expect_error(dac_studies(x, block_size = 1.5), "integer")
  fit <- fit_dac(x, block_size = 2)
  manual <- fit_meta_md(dac_studies(x, block_size = 2), 4)
  expect_equal(unname(coef(fit)), coef(manual))
  interval <- support_interval(fit, c(1, 0, 0, 0))
  expect_true(interval$success)
  expect_equal(interval$active_dimension, 2L)
  expect_equal(fit$sample_size, 30L)
  expect_true(contains(fit, colMeans(x)))
})

test_that("two-treatment NMA reproduces scalar inference", {
  estimates <- c(0.1, 0.14, 0.08); se <- c(0.04, 0.05, 0.03)
  network <- network_studies(estimates, se, rep("B", 3), rep("A", 3), reference = "A")
  fit <- fit_nma(network)
  contrasts <- nma_contrasts(fit, matrix(c("B", "A"), 1))
  scalar <- fit_meta_1d(estimates, se)
  expect_equal(c(contrasts$lower, contrasts$upper), c(scalar$lower, scalar$upper), tolerance = 2e-7)
  expect_equal(names(coef(fit)), "B")
})

test_that("NMA contrasts are reference and label invariant", {
  make_network <- function(reference, labels = c("A", "B", "C")) {
    network_studies(c(0.2, 0.25, 0.1), c(0.2, 0.18, 0.22),
      labels[c(2, 3, 3)], labels[c(1, 1, 2)], reference = reference)
  }
  first <- fit_nma(make_network("A"))
  second <- fit_nma(make_network("C"))
  renamed <- fit_nma(make_network("one", c("one", "two", "three")))
  a <- nma_contrasts(first, matrix(c("B", "C"), 1))
  b <- nma_contrasts(second, matrix(c("B", "C"), 1))
  c <- nma_contrasts(renamed, matrix(c("two", "three"), 1))
  expect_equal(a[c("lower", "upper")], b[c("lower", "upper")], tolerance = 3e-7)
  expect_equal(a[c("lower", "upper")], c[c("lower", "upper")], tolerance = 3e-7)
  expect_equal(nrow(nma_contrasts(first)), 3L)
  expect_error(network_studies(c(0, 0), c(1, 1), c("A", "C"), c("B", "D")), "disconnected")
})

test_that("multi-arm studies retain joint covariance and block identity", {
  network <- network_arm_studies(c(1, 1.2, 0.8, 1, 1.15), rep(0.1, 5),
    c("A", "B", "C", "A", "B"), c("trial1", "trial1", "trial1", "trial2", "trial2"), reference = "A")
  expect_length(network$studies, 2)
  expect_equal(network$studies[[1]]$vcov, matrix(c(.02, .01, .01, .02), 2), tolerance = 1e-15)
  expect_equal(network$studies[[1]]$label, "trial1")
  fit <- fit_nma(network)
  expect_true(all(nma_contrasts(fit)$success))
  wls <- nma_wls(network)
  expect_equal(dim(wls$vcov), c(2L, 2L))
  expect_equal(wls$reference, "A")
  expect_error(network_studies(c(0, 0), c(1, 1), c("B", "C"), c("A", "A"), c("trial", "trial")), "joint vcov")
  independent <- network_studies(c(0, 0), c(1, 1), c("B", "C"), c("A", "A"), c("trial", "trial"), multiarm = "independent")
  expect_length(independent$studies, 2)
})

test_that("multivariate removals require a lower bound and retain the history", {
  summaries <- list(study(c(-4, 0), diag(2), label = "left"), study(c(4, 0), diag(2), label = "right"))
  error <- tryCatch(fit_meta_md(summaries), heavilyright_empty_region_error = identity)
  expect_s3_class(error, "heavilyright_empty_region_error")
  expect_equal(error$diagnostics$final_state, "empty")
  expect_length(error$diagnostics$removals, 0)
  fit <- fit_meta_md(summaries, empty_policy = empty_set_policy(warn = FALSE))
  record <- fit$empty_set_diagnostics
  expect_length(record$removals, 1)
  expect_length(record$trace, 2)
  expect_true(record$trace[[1]]$minimum_diagnostics$empty_certified)
  expect_gt(record$trace[[1]]$feasible_set_score_lower_bound, record$trace[[1]]$cutoff)
  expect_equal(record$final_weights, 1)
  expect_equal(record$final_state, "nonempty")
  expect_true(support_interval(fit, c(1, 0))$success)
  limited <- tryCatch(fit_meta_md(summaries, empty_policy = empty_set_policy(max_removals = 0, warn = FALSE)), heavilyright_empty_resolution_error = identity)
  expect_s3_class(limited, "heavilyright_empty_resolution_error")
})

test_that("removal skips a smaller-p block when its deletion loses rank", {
  summaries <- list(study(c(7, 0), diag(2), label = "rank-critical"),
    study(0, matrix(1), matrix(c(1, 0), 1), label = "x1"),
    study(0, matrix(1), matrix(c(1, 0), 1), label = "x2"))
  fit <- fit_meta_md(summaries, weights = c(.05, .45, .5),
    empty_policy = empty_set_policy(max_removals = 2, warn = FALSE))
  candidates <- fit$empty_set_diagnostics$trace[[1]]$candidates
  expect_equal(candidates[[1]]$original_index, 1L)
  expect_false(candidates[[1]]$eligible)
  expect_equal(candidates[[1]]$retained_projection_rank, 1L)
  expect_true(candidates[[2]]$selected)
  expect_equal(fit$empty_set_diagnostics$final_original_indices, 1L)
  expect_true(support_interval(fit, c(0, 1))$success)
})
