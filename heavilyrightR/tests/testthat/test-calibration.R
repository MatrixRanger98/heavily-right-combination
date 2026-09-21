test_that("mean uncertainty uses n(n-1) and ddof one", {
  sample <- matrix(c(1, 2, 4, 7, 2, 5, 8, 11), ncol = 2)
  n <- nrow(sample)
  centered <- sweep(sample, 2, colMeans(sample), "-")
  expected <- crossprod(centered) / (n * (n - 1))
  expect_equal(covariance_of_mean(sample), expected, tolerance = 0)
  expect_equal(standard_error_of_mean(sample)^2, diag(expected), tolerance = 1e-15)
  expect_error(covariance_of_mean(matrix(1:2, 1)), "two observations")
})

test_that("block covariance has explicit q by q by m layout", {
  sample <- array(seq_len(5 * 3 * 2), c(5, 3, 2))
  actual <- covariance_of_mean(sample)
  expect_equal(dim(actual), c(2, 2, 3))
  for (j in 1:3) expect_equal(actual[, , j], covariance_of_mean(sample[, j, ]))
})

test_that("correlation constructors validate their domains", {
  expect_equal(ar1_correlation(3, 0.5), outer(1:3, 1:3, function(i, j) 0.5^abs(i - j)))
  expect_equal(diag(equicorrelation(4, 0.2)), rep(1, 4))
  expect_error(equicorrelation(4, -0.5), "valid")
})

