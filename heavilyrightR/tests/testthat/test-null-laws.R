test_that("score transforms preserve support and shape", {
  expect_equal(half_cauchy_score(c(0, 0.5, 1)), c(Inf, 1, 0), tolerance = 1e-15)
  expect_equal(reciprocal_score(c(0, 0.25, 1)), c(Inf, 4, 1))
  expect_error(half_cauchy_score(-0.1), "[0, 1]", fixed = TRUE)
})

test_that("scaled Ei uses stable moderate and large branches", {
  expect_equal(scaled_ei(50), 0.02041704555594399, tolerance = 2e-15)
  expect_equal(scaled_ei(1000), 0.0010010020060241204, tolerance = 2e-18)
})

test_that("finite laws satisfy analytic identities", {
  for (x in c(0.5, 2, 20)) {
    expect_equal(phcmean(x, 1, lower.tail = FALSE), 2 / pi * atan(1 / x), tolerance = 1e-14)
    expect_equal(pehmean(x, 1, lower.tail = FALSE), if (x <= 1) 1 else 1 / x, tolerance = 1e-14)
  }
  for (x in c(2, 5, 20)) {
    exact <- 1 / x + log(4 * (x - 0.5)^2) / (4 * x^2)
    expect_equal(pehmean(x, c(0.5, 0.5), lower.tail = FALSE), exact, tolerance = 2e-11)
  }
  expect_equal(qhcmean(0.95, 2), 13.68751142121654, tolerance = 3e-8)
})

test_that("Landau functions match snapshots and invert their own CDF", {
  points <- c(-10, -5.5, -1, 0, 1, 5, 50, 300)
  expect_equal(dlandau(points), c(0, 0, 0.221762209, 0.2622401260878838,
    0.163531241, 0.0265588931, 0.000274514184, 7.21869609e-6), tolerance = 8e-9)
  probabilities <- c(0.001, 0.01, 0.1, 0.5, 0.9, 0.99)
  expect_equal(plandau(qlandau(probabilities)), probabilities, tolerance = 3e-9)
})

test_that("quadrature diagnostics are explicit", {
  result <- phcmean(5, c(0.2, 0.3, 0.5), lower.tail = FALSE, diagnostics = TRUE)
  expect_s3_class(result, "heavilyright_probability")
  expect_equal(result$value, 0.15356066843681823, tolerance = 3e-8)
  expect_true(is.finite(result$estimated_error))
})
