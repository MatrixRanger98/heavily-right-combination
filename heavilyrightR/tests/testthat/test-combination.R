test_that("all nine combination rules match validated fixed inputs", {
  p <- c(0.02, 0.2, 0.7)
  expected <- list(
    hcauchy = c(11.802574980147883, 0.06096154291551871),
    ehmp = c(18.809523809523807, 0.06104468109104662),
    hmp = c(18.809523809523807, 0.06600071912027838),
    cauchy = c(5.514794745443705, 0.05709884040002967),
    levy = c(178.63633674957526, 0.0596417140704903),
    fisher = c(5.8781358618009785, 0.06763225410674976),
    stouffer = c(1.3688799549850457, 0.08551840126339533),
    bonferroni = c(-0.02, 0.06), simes = c(-0.02, 0.06)
  )
  for (method in names(expected)) {
    spec <- suppressWarnings(new_combination(method, m = 3))
    actual_score <- combination_score(p, spec)
    expect_equal(actual_score, expected[[method]][1], tolerance = 3e-8, info = method)
    expect_equal(combination_pvalue(actual_score, spec), expected[[method]][2], tolerance = 3e-8, info = method)
  }
})

test_that("calibration choice is named and immutable in results", {
  expect_equal(new_combination("hcauchy", m = 2)$calibration, "finite")
  expect_equal(new_combination("hcauchy", m = 1001)$calibration, "landau")
  expect_equal(new_combination("hcauchy", m = 2, calibration = "landau")$calibration, "landau")
})

test_that("batched p-values use the final dimension", {
  spec <- new_combination("stouffer", m = 3)
  p <- rbind(c(0.1, 0.2, 0.3), c(0.3, 0.4, 0.5))
  expect_equal(combination_score(p, spec), apply(p, 1, combination_score, spec = spec))
  expect_length(combine_p(p, spec), 2)
})

test_that("weights and public p-value support fail explicitly", {
  expect_error(new_combination("hcauchy", weights = c(0.2, 0.3, 0.4)), "sum")
  expect_error(new_combination("hcauchy", weights = c(0.2, 0.8, 0)), "positive")
  expect_error(combination_score(c(0.2, 1.1), new_combination("stouffer", m = 2)), "[0, 1]", fixed = TRUE)
})
