# Fixed NMA component-budget regression inputs; no reproduction imports or
# generated artifacts are needed to exercise the minimum/support certificates.
saved_component_budget_studies <- function() {
  means <- c(
    -.9235547877177038, -1.2123716842226262, -1.0413815245098765, -.9081055261327049,
    -1.2430190732874253, -.45515077333688586, -1.231551357005721, .4856965856580656,
    .4992083447201604, -.815073318927908, -1.0385136997996947, -1.0545400404054852,
    .15124543115137423, -.5284622211531385, .421072549309604, .03451511797749109,
    .07768668221117091, -1.0424529170847947, -.5786284461615806, .029457035007630285,
    -.09579944443419365, .1534489704627513, .0630815287912311, -.8767096824440432,
    -.6891993921854633, -.5452829920777958, -.9424942925151076, -.09113978694325173)
  variances <- c(
    .009541104589970598, .011045536097184207, .010409111449021359, .008178959933424617,
    .011681371935128821, .01106917799440211, .006860444299186636, .00909572543577906,
    .011171912157005949, .011663689347595005, .010194828177729019, .009880066093024184,
    .00958081684822472, .009091926501806357, .008192868312363313, .009528547184720803,
    .009213917871079181, .011592352564315055, .008999905273741563, .007956370755338506,
    .010755855039327763, .010075966598386917, .010101704225241263, .011073652710447435,
    .013589406855185077, .008527668044300675, .010472112555212798, .007617548267542314)
  design <- rbind(
    c(0,0,1,0,0,0,0,0,0), c(0,0,1,0,0,0,0,0,0), c(-1,0,1,0,0,0,0,0,0),
    c(0,0,0,0,0,1,0,0,0), c(0,0,0,0,0,1,0,0,0), c(0,0,0,0,1,0,0,0,0),
    c(0,0,0,0,0,1,0,0,0), c(0,0,-1,0,1,0,0,0,0), c(0,0,0,0,1,-1,0,0,0),
    c(0,0,0,0,0,1,0,0,0), c(0,0,0,0,0,1,0,0,0), c(0,0,0,0,0,1,0,0,0),
    c(0,0,-1,0,0,1,0,0,0), c(0,0,0,0,0,1,0,-1,0), c(1,0,0,0,0,0,0,-1,0),
    c(1,0,0,0,0,0,0,0,0), c(0,0,0,0,0,0,1,0,0), c(0,0,0,0,0,0,0,0,1),
    c(0,0,1,0,0,0,0,-1,0), c(0,0,0,1,0,0,0,0,0), c(0,0,0,1,0,0,0,0,0),
    c(0,0,-1,0,0,1,0,0,0), c(0,0,0,1,0,0,0,0,0), c(0,0,1,0,0,0,0,0,0),
    c(0,1,0,0,0,0,0,0,0), c(0,1,0,0,0,0,0,0,0), c(0,0,1,0,0,0,0,0,0),
    c(1,0,0,0,0,0,0,0,0))
  lapply(seq_along(means), function(i)
    study(means[i], matrix(variances[i]), design[i, , drop = FALSE], df = 99))
}

test_that("saved NMA input validates the minimum and both support directions", {
  fit <- fit_meta_md(saved_component_budget_studies())
  expect_equal(fit$specification$cutoff, 16.24181581228568, tolerance = 1e-11)
  expect_true(fit$minimum$success)
  expect_lte(fit$minimum$optimality_gap_bound, fit$minimum$optimality_gap_tolerance)
  expect_equal(fit$minimum$optimality_gap_tolerance, 2e-6 * fit$specification$cutoff)
  expect_equal(fit$minimum$score, 3.213019157430995, tolerance = 1e-8)
  expected <- list(c(-.2753011120137878, .18785467543640175),
                   c(-.8423592725890696, -.31820975856203104))
  for (j in 1:2) {
    result <- support_interval(fit, diag(9)[, j])
    expect_true(result$success)
    expect_equal(c(result$lower, result$upper), expected[[j]], tolerance = 2e-7)
    for (endpoint in list(result$lower_diagnostics, result$upper_diagnostics)) {
      expect_true(endpoint$success)
      expect_lte(endpoint$scaled_boundary_error, 2e-8)
      expect_lte(endpoint$kkt_residual, 2e-6)
      expect_gt(endpoint$eta, 0)
    }
  }
  expect_length(fit$empty_set_diagnostics$removals, 0)
  # Natural-score normalization satisfies the aggregate certificate directly.
  expect_equal(fit$minimum$component_refinements, 0)
})

component_budget_fit <- function(mass = .1, method = "hcauchy", center = c(8, 8), df = 99) {
  studies <- lapply(seq_len(4), function(i) study(c(-.2, .2, -.1, .1)[i],
    matrix(1), matrix(if (i <= 2) c(1, 0) else c(0, 1), 1), df = df))
  spec <- new_combination(method, weights = c(rep((1-mass)/2, 2), rep(mass/2, 2)))
  spec$cutoff <- 14
  # A valid, deliberately poor origin exercises large initial natural scores.
  # Covariances, projections, true minima, and all acceptance limits are real.
  fit <- list(studies = studies, dimension = 2L, specification = spec,
    standardization = list(center = center, scale = c(1, 1)), components = list(1L, 2L))
  fit$kernel <- .hr_make_kernel(fit)
  fit
}

loose_component_budget_run <- function(method = "hcauchy", mass = .1,
                                      exhausted = FALSE, reject = FALSE, nonimproving = FALSE,
                                      throws = FALSE, selective = FALSE) {
  origin <- if (selective) 20 else 8
  fit <- component_budget_fit(mass, method, center = rep(origin, 2), df = if (selective) 1 else 99)
  actual <- .hr_minimize_kernel
  observed <- list()
  solve <- function(kernel, cutoff, maxeval = 5000L, start = NULL, maximum_gap = Inf) {
    if (is.finite(maximum_gap)) {
      observed[[length(observed) + 1L]] <<- list(cap = maximum_gap, maxeval = maxeval)
      if (throws) stop("injected refinement failure")
      result <- actual(kernel, cutoff, maxeval, start, maximum_gap)
      if (reject) result$success <- FALSE
      if (nonimproving) result$optimality_gap_bound <- 1
      return(result)
    }
    result <- actual(kernel, cutoff, maxeval, start = -origin)
    # Weaken a valid lower bound, while remaining inside the actual local
    # tolerance. This never invents an overly optimistic convex certificate.
    gap <- .9 * result$optimality_gap_tolerance
    stopifnot(gap > result$optimality_gap_bound)
    result$minimum_value_lower_bound <- result$score - gap
    result$optimality_gap_bound <- gap
    result$iterations <- if (exhausted) maxeval else 3L
    result
  }
  testthat::local_mocked_bindings(.hr_minimize_kernel = solve, .package = "heavilyright")
  result <- tryCatch(.hr_md_minimum(fit, maxeval = 20L), heavilyright_numerical_error = identity)
  list(result = result, budgets = observed)
}

test_that("loose local bounds are refined in parent score units", {
  for (method in c("hcauchy", "ehmp")) for (mass in c(.1, .5, 1e-8)) {
    run <- loose_component_budget_run(method, mass)
    result <- run$result
    expect_true(isTRUE(result$success))
    expect_equal(result$optimality_gap_tolerance, 2e-6 * 14)
    expect_gt(result$initial_optimality_gap_bound, result$optimality_gap_tolerance)
    expect_lte(result$optimality_gap_bound, result$optimality_gap_tolerance)
    expect_equal(result$component_refinements, 2L)
    expect_length(run$budgets, 2L)
    expect_equal(vapply(run$budgets, `[[`, numeric(1), "maxeval"), c(17, 17))
    allocation <- vapply(run$budgets, `[[`, numeric(1), "cap") * c(1-mass, mass)
    expect_equal(allocation[1], allocation[2])
    expect_lt(sum(allocation), result$optimality_gap_tolerance)
    expect_true(all(vapply(result$component_refinement_attempts, `[[`, logical(1), "accepted")))
    expect_true(all(vapply(result$component_diagnostics, `[[`, numeric(1), "iterations") <= 20))
    expect_false(result$empty_certified)
  }
})

test_that("natural-scale weak components and valid aggregates need no refinement", {
  for (method in c("hcauchy", "ehmp")) for (mass in c(.1, 1e-8)) {
    testthat::local_mocked_bindings(.hr_slsqp = function(...) stop("unnecessary optimizer"),
                                  .package = "heavilyright")
    result <- .hr_md_minimum(component_budget_fit(mass, method, center = c(0, 0)))
    expect_true(result$success)
    expect_equal(result$component_refinements, 0L)
    expect_equal(result$iterations, 0L)
    expect_equal(result$objective_evaluations, 0L)
  }
})

test_that("only children exceeding their parent-unit allocation are retried", {
  run <- loose_component_budget_run(selective = TRUE)
  expect_true(run$result$success)
  expect_length(run$budgets, 1)
  expect_equal(run$result$component_refinement_attempts[[1]]$component, 1L)
  expect_equal(run$result$component_diagnostics[[2]]$maximum_gap, Inf)
  expect_equal(run$result$component_diagnostics[[2]]$iterations, 3L)
})

test_that("exhausted and invalid refinements cannot authorize success or emptiness", {
  for (mode in c("exhausted", "reject", "nonimproving", "throws")) {
    run <- loose_component_budget_run(exhausted = mode == "exhausted",
      reject = mode == "reject", nonimproving = mode == "nonimproving", throws = mode == "throws")
    expect_s3_class(run$result, "heavilyright_numerical_error")
    expect_match(conditionMessage(run$result), "component minima validated but aggregate gap")
    result <- run$result$minimum_diagnostics
    expect_false(result$success)
    expect_false(result$empty_certified)
    expect_gt(result$optimality_gap_bound, result$optimality_gap_tolerance)
    expect_length(run$budgets, if (mode == "exhausted") 0L else 2L)
    expect_true(all(vapply(result$component_refinement_attempts, function(x) !x$accepted, logical(1))))
    expect_true(all(vapply(result$component_diagnostics, `[[`, logical(1), "success")))
    if (mode == "throws") expect_true(is.na(result$objective_evaluations))
  }
})

test_that("an unresolved aggregate stays unknown without study removal", {
  failure <- loose_component_budget_run(exhausted = TRUE)$result
  testthat::local_mocked_bindings(.hr_md_fit_once = function(...) stop(failure),
                                  .package = "heavilyright")
  result <- tryCatch(fit_meta_md(component_budget_fit()$studies,
    empty_policy = empty_set_policy(warn = FALSE)), heavilyright_empty_numerical_error = identity)
  expect_s3_class(result, "heavilyright_empty_numerical_error")
  expect_equal(result$diagnostics$final_state, "unknown")
  expect_length(result$diagnostics$removals, 0)
  expect_false(result$cause$minimum_diagnostics$empty_certified)
})

test_that("a tighter cap triggers a real solve for a locally acceptable start", {
  kernel <- list(dimension = 1L, method = "hcauchy", blocks = list(
    list(B = matrix(1), b = -.2, q = 1L, df = NULL, weight = .5),
    list(B = matrix(1), b = .2, q = 1L, df = NULL, weight = .5)))
  initial <- .hr_minimize_kernel(kernel, 2, start = 5e-7)
  expect_true(initial$success)
  expect_equal(initial$iterations, 0L)
  expect_gt(initial$optimality_gap_bound, 1e-8)
  tightened <- .hr_minimize_kernel(kernel, 2, start = initial$point_z, maximum_gap = 1e-8)
  expect_true(tightened$success)
  expect_gt(tightened$iterations, 0L)
  expect_gt(tightened$objective_evaluations, 0L)
  expect_lte(tightened$optimality_gap_bound, 1e-8)
  expect_equal(tightened$optimality_gap_tolerance, 1e-8)
})

test_that("an absolute cap applies to initial acceptance and every restart", {
  kernel <- list(dimension = 1L, method = "hcauchy", blocks = list(
    list(B = matrix(1), b = -.2, q = 1L, df = NULL, weight = .5),
    list(B = matrix(1), b = .2, q = 1L, df = NULL, weight = .5)))
  budgets <- numeric()
  testthat::local_mocked_bindings(.hr_slsqp = function(x0, control, ...) {
    budgets <<- c(budgets, control$maxeval)
    list(par = x0, iter = min(3L, control$maxeval), convergence = 1L,
         message = "deliberately stationary solver")
  }, .package = "heavilyright")
  expect_true(.hr_minimize_kernel(kernel, 2, start = 0)$success)
  expect_length(budgets, 0)
  result <- tryCatch(.hr_minimize_kernel(kernel, 2, maxeval = 8L, start = 0,
    maximum_gap = 1e-12), heavilyright_numerical_error = identity)
  expect_s3_class(result, "heavilyright_numerical_error")
  expect_equal(budgets, c(8, 5, 2))
  expect_equal(result$minimum_diagnostics$iterations, 8)
  expect_equal(result$minimum_diagnostics$optimality_gap_tolerance, 1e-12)
  expect_false(result$minimum_diagnostics$success)
})
