expect_tail_reference <- function(result, expected, relative = 2e-9) {
  expect_gt(result$value, 0)
  expect_true(is.finite(result$estimated_error))
  expect_gt(result$estimated_error, 0)
  expect_lt(abs(result$value / expected - 1), relative)
  expect_lte(abs(result$value - expected), result$estimated_error +
    64 * .Machine$double.eps * abs(expected) + .hr_smallest)
}

test_that("native complex transforms agree with independent 70-digit values", {
  s <- c(.1+1i, 10+20i, 49+1i, 50+0i, .0001+50i, 100+200i,
         1e6+1e6i, 1e-8+2i, 2+1e-8i)
  eh <- c(.380588473672988316-.316148064537851568i,
    .0219097637643217419-.0367809403434896533i,
    .0196148236640680239-.000385165288622254362i,
    .0192445034942564817,
    .000796244798878913946-.0199523745569949659i,
    .00202347484007256499-.00396791306822823314i,
    .000000499999999998500006-.000000499999000001500000i,
    .201958023385759500-.289090604975361015i,
    .277342766223554828-.000000000839858506646678i)
  hc <- c(.369351403093116509-.376132432525877381i,
    .0128464093961859750-.0254831345127195797i,
    .0129760890842611630-.000264381109696914364i,
    .0127222578493759541,
    .0000000255264037783368224-.0127426308564080293i,
    .00127335160985926553-.00254649944270760977i,
    .000000318309886184108981-.000000318309886183472362i,
    .135335284220680726-.328435744604761579i,
    .254024650928716589-.000000000920203979164296i)
  # mpmath E1 formulas at 70 digits, independent of the R series/fraction.
  expect_lt(max(Mod(.hr_laplace(s, "ehmp") / eh - 1)), 3e-12)
  expect_lt(max(Mod(.hr_laplace(s, "hcauchy") / hc - 1)), 3e-12)
  for (law in c("hcauchy", "ehmp")) {
    for (x in c(.1, 2, 49, 60, 1000)) {
      h <- x * 1e-4
      mean <- -(log(Re(.hr_laplace(x + h, law))) -
                  log(Re(.hr_laplace(x - h, law)))) / (2 * h)
      expect_equal(.hr_laplace(x, law, derivative = TRUE), mean, tolerance = 2e-7)
    }
  }
})

test_that("reciprocal lower tails agree with independent deHoog inversions", {
  # Independent 70-digit deHoog inversion, verified at two inversion degrees.
  for (case in list(c(100, 2, 3.8732694969595899849e-7, 1.4570141691622174263e-8),
                    c(1000, 4, 9.3513613369384336208e-6, 6.7033392953993451592e-7))) {
    expect_tail_reference(dehmean(case[2], case[1], diagnostics = TRUE), case[3])
    expect_tail_reference(pehmean(case[2], case[1], diagnostics = TRUE), case[4])
    expect_equal(pehmean(case[2], case[1]) + pehmean(case[2], case[1], lower.tail = FALSE), 1)
  }
})

test_that("half-Cauchy lower tails retain tiny representable probabilities", {
  x <- c(.01, .1, .2, .5)
  pdf <- c(3.0077997241492881812e-16, 2.5302244695915063995e-7,
           8.1541631828556601332e-5, .034037379736545367572)
  cdf <- c(3.0087107333065109274e-19, 2.6037233244268005771e-9,
           1.8051385443885068624e-6, .0026247015692195846242)
  for (i in seq_along(x)) {
    expect_tail_reference(dhcmean(x[i], 10, diagnostics = TRUE), pdf[i])
    expect_tail_reference(phcmean(x[i], 10, diagnostics = TRUE), cdf[i])
  }
  expect_equal(1 - phcmean(.01, 10, lower.tail = FALSE), 0)
  expect_equal(phcmean(.01, 10, log.p = TRUE), log(cdf[1]), tolerance = 2e-12)
  expect_tail_reference(dhcmean(.4, 1000, diagnostics = TRUE), 2.8439421334728674250e-249, 2e-8)
  expect_tail_reference(dhcmean(.5, 1000, diagnostics = TRUE), 5.1774117530277950472e-187, 2e-8)
  former_cutoff <- 2.1667664604475318268
  expect_tail_reference(dhcmean(former_cutoff, 1000, diagnostics = TRUE), 2.5857730958474028568e-6, 2e-8)
  expect_tail_reference(phcmean(former_cutoff, 1000, diagnostics = TRUE), 1.040069978848716663e-7, 2e-8)
})

test_that("support residuals preserve the supplied binary weights", {
  x <- 1 + .Machine$double.eps
  w <- c(1e-12, 1-1e-12)
  expect_tail_reference(dehmean(x, w, diagnostics = TRUE), .00019988292363303158068)
  expect_tail_reference(pehmean(x, w, diagnostics = TRUE), 1.9981916750177780043e-20)
  expect_tail_reference(dehmean(x, c(.5,.5+.Machine$double.eps/2), diagnostics=TRUE),
    4.4408920985006232035e-16)
  expect_tail_reference(pehmean(x, c(.5,.5+.Machine$double.eps/2), diagnostics=TRUE),
    2.4651903288156606147e-32)
  below <- c(1e-12, 1-1e-12-.Machine$double.eps/2)
  expect_tail_reference(dehmean(1, below, diagnostics=TRUE), 8.8892679730126766233e-5)
  expect_tail_reference(pehmean(1, below, diagnostics=TRUE), 3.9514225828469864622e-21)
  for (fun in list(dehmean, pehmean)) {
    expect_equal(fun(1, 3), 0)
    expect_gt(fun(1, rep(1/3, 3)), 0)
    expect_equal(fun(1, c(.5, .5 + .Machine$double.eps/2)), 0)
    expect_gt(fun(x, c(.5, .5 + .Machine$double.eps/2)), 0)
  }
  expect_equal(.hr_compensated_sum(c(1, -rep(1/3, 3))), 2^-54)
  for (x in c(1e155, 1e160)) {
    expect_gt(dehmean(x, 1), 0)
    expect_equal(dehmean(x, 1), (1 / x) / x, tolerance = 0)
    expect_gt(dhcmean(x, 1), 0)
  }
  expect_gt(phcmean(1e-200, 1), 0)
})

test_that("cycle inversion agrees with independent small and skewed system oracles", {
  # 70-digit mpmath deHoog inversion, degrees 60 and 80 agree to 27 digits.
  cases <- list(
    list("hcauchy", .5, 3, .382906636767160814826384487, .0872456818488393765501700472),
    list("ehmp", 1.5, 3, .350624439780921742682515122, .103660721844370954751227792),
    list("hcauchy", .1, c(.2,.3,.5), .0403424250539869715387865701, .00137905024385249684300202451),
    list("ehmp", 1.3, c(.01,.19,.8), .481828326653501065792758285, .101055482552996557442339432),
    list("hcauchy", .5, 7, .104115846664193072357880872, .0111581500015008942079003453)
  )
  for (case in cases) {
    d <- if (case[[1]] == "hcauchy") dhcmean else dehmean
    p <- if (case[[1]] == "hcauchy") phcmean else pehmean
    expect_tail_reference(d(case[[2]], case[[3]], diagnostics = TRUE), case[[4]])
    expect_tail_reference(p(case[[2]], case[[3]], diagnostics = TRUE), case[[5]])
  }
})

test_that("upper-contour diagnostics include transform evaluation roundoff", {
  # Independent 70-digit deHoog inversions at degrees 60 and 80 agree.
  expected <- .378440282000501265642994593212172331891189576
  expect_tail_reference(dhcmean(1, c(.01,.19,.8), diagnostics=TRUE), expected)
})

test_that("analytic boundaries are handled before any integration", {
  for (m in c(1, 2, 100)) {
    for (law in c("hcauchy", "ehmp")) {
      p <- if (law == "hcauchy") phcmean else pehmean
      d <- if (law == "hcauchy") dhcmean else dehmean
      boundary <- if (law == "hcauchy") 0 else 1
      expect_equal(p(-Inf, m), 0)
      expect_equal(p(Inf, m), 1)
      expect_equal(p(Inf, m, lower.tail = FALSE), 0)
      expect_equal(d(Inf, m), 0)
      expect_equal(d(boundary, m, diagnostics = TRUE)$estimated_error, 0)
      expect_error(p(NaN, m), "non-missing")
      expect_error(p(Inf, c(.2,.2)), "sum to one")
    }
  }
})

test_that("two reciprocal scores obey an independent survival identity", {
  for (x in c(1.01, 1.1, 1.5, 2, 5, 20)) {
    expected <- 1/x + log(2*x-1)/(2*x*x)
    expect_equal(pehmean(x, 2, lower.tail = FALSE), expected, tolerance = 2e-12)
  }
  # Positive convolution in the original score coordinate, independently
  # transformed from the production support-distance integrand.
  for (w in list(c(.5,.5), c(.2,.8), c(1e-6, 1-1e-6))) {
    for (x in c(1.01, 1.1, 1.5)) {
      a <- min(w); b <- max(w)
      upper <- (x-a-b)/(x-b)
      pdf <- integrate(function(v) b/(x-a/(1-v))^2, 0, upper,
        abs.tol = 2e-13, rel.tol = 2e-11)$value
      expect_equal(dehmean(x, w), pdf, tolerance = 3e-10)
    }
  }
})

test_that("extreme weights and underflow retain honest error semantics", {
  for (a in c(1e-100, 1e-200)) {
    expect_tail_reference(dhcmean(a, c(a,1), diagnostics = TRUE), 1/pi)
    expect_tail_reference(phcmean(a, c(a,1), diagnostics = TRUE),
      a * (1/pi - 2*log(2)/pi^2))
    expect_tail_reference(dehmean(1.1, c(a,1), diagnostics = TRUE), 1/1.1^2)
    expect_tail_reference(pehmean(1.1, c(a,1), diagnostics = TRUE), 1-1/1.1)
  }
  for (x in c(.0001, .1, .3)) {
    for (fun in list(dhcmean, phcmean)) {
      result <- fun(x, 1000, diagnostics = TRUE)
      expect_equal(result$value, 0)
      expect_gte(result$estimated_error, .hr_smallest)
    }
  }
  for (fun in list(dehmean, pehmean)) {
    result <- fun(1.1, 1000, diagnostics = TRUE)
    expect_equal(result$value, 0)
    expect_gte(result$estimated_error, .hr_smallest)
    expect_gt(fun(1.5, 1000), 0)
  }
})

test_that("zero weights, order, and diagnostic flags preserve values", {
  for (fun in list(dhcmean, phcmean, dehmean, pehmean)) {
    x <- if (identical(fun, dhcmean) || identical(fun, phcmean)) .1 else 1.1
    value <- fun(x, c(.2,.3,.5))
    expect_equal(fun(x, c(0,.5,.3,.2)), value, tolerance = 1e-12)
    expect_equal(fun(x, c(.2,.3,.5), diagnostics = TRUE)$value, value, tolerance = 0)
  }
})

test_that("contours meet smoothly and CDF derivatives agree with densities", {
  for (m in c(2, 10, 100, 1000)) {
    for (law in c("hcauchy", "ehmp")) {
      limit <- if (law == "hcauchy") max(.5, 2/pi*(log(m)+1-.hr_euler_gamma)-1)
        else max(1.5, log(m)-.hr_euler_gamma)
      p <- if (law == "hcauchy") phcmean else pehmean
      d <- if (law == "hcauchy") dhcmean else dehmean
      h <- limit * 1e-5
      for (fun in list(p,d)) {
        left <- 2*fun(limit-h,m)-fun(limit-2*h,m)
        right <- 2*fun(limit+h,m)-fun(limit+2*h,m)
        expect_lt(abs(left-right), 3e-8)
      }
      derivative <- (p(limit+h,m)-p(limit-h,m))/(2*h)
      expect_lt(abs(derivative-d(limit,m)), 3e-8)
    }
  }
})

test_that("lower and upper quantiles preserve calibration", {
  for (m in c(2, 10, 100)) {
    for (probability in c(1e-4, .01, .95)) {
      expect_equal(pehmean(qehmean(probability,m),m), probability, tolerance = 2e-8)
      expect_equal(phcmean(qhcmean(probability,m),m), probability, tolerance = 2e-8)
    }
  }
})

test_that("invalid quadrature, transform, and cycle outputs fail explicitly", {
  for (bad in list(c(NA,NA), c(Inf,0), c(1,-1), c(1,1), c(.Machine$double.xmax,0))) {
    local({
      testthat::local_mocked_bindings(.hr_lower_quad = function(...) {
        # Exercise the adapter's actual guards, never mutate stats' namespace.
        list(value=bad[1], abs.error=bad[2], message="OK")
      })
      expect_error(dehmean(2,100), class="heavilyright_integration_error")
      expect_error(phcmean(.1,10), class="heavilyright_integration_error")
    })
  }
  expect_error(.hr_laplace(0, "ehmp"), class="heavilyright_integration_error")
  expect_error(.hr_lower_cycles(function(u) exp(-u), 1, max_cycles=1),
    class="heavilyright_integration_error")
  local({
    testthat::local_mocked_bindings(.hr_lower_quad = function(...) {
      list(value=-1, abs.error=0, message="OK")
    })
    expect_error(dehmean(1.1,2), class="heavilyright_integration_error")
    expect_error(dhcmean(.1,10), class="heavilyright_integration_error")
  })
  local({
    testthat::local_mocked_bindings(.hr_scaled_e1 = function(z) rep(NA_complex_, length(z)))
    expect_error(dehmean(2,100), class="heavilyright_integration_error")
  })
})
