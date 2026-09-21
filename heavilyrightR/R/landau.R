# Landau density/CDF coefficient families follow CERN ROOT's numerical
# approximations: https://root.cern.ch/doc/master/namespaceROOT_1_1Math.html
# See the installed NOTICE and COPYING files, and ../THIRD_PARTY.md in the
# source repository, for attribution and licensing information.
.hr_landau_base_pdf_one <- function(x) {
  if (is.infinite(x)) return(0)
  if (x < -5.5) {
    u <- exp(x + 1)
    if (u < 1e-10) return(0)
    a <- c(0.04166666667, -0.01996527778, 0.02709538966)
    return(0.3989422803 * exp(-1 / u) / sqrt(u) * (1 + (a[1] + (a[2] + a[3] * u) * u) * u))
  }
  if (x < -1) {
    p <- c(0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
    q <- c(1, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
    u <- exp(-x - 1)
    return(exp(-u) * sqrt(u) * (p[1] + (p[2] + (p[3] + (p[4] + p[5] * x) * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + (q[4] + q[5] * x) * x) * x) * x))
  }
  if (x < 1) {
    p <- c(0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
    q <- c(1, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
    return((p[1] + (p[2] + (p[3] + (p[4] + p[5] * x) * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + (q[4] + q[5] * x) * x) * x) * x))
  }
  if (x < 5) {
    p <- c(0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
    q <- c(1, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
    return((p[1] + (p[2] + (p[3] + (p[4] + p[5] * x) * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + (q[4] + q[5] * x) * x) * x) * x))
  }
  u <- 1 / x
  if (x < 12) {
    p <- c(0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
    q <- c(1, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
  } else if (x < 50) {
    p <- c(1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
    q <- c(1, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
  } else if (x < 300) {
    p <- c(1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
    q <- c(1, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
  } else {
    u <- 1 / (x - x * log(x) / (x + 1))
    return(u * u * (1 + (-1.845568670 - 4.284640743 * u) * u))
  }
  u * u * (p[1] + (p[2] + (p[3] + (p[4] + p[5] * u) * u) * u) * u) /
    (q[1] + (q[2] + (q[3] + (q[4] + q[5] * u) * u) * u) * u)
}

.hr_landau_base_cdf_one <- function(x) {
  if (is.infinite(x)) return(if (x < 0) 0 else 1)
  if (x < -5.5) {
    u <- exp(x + 1)
    if (u < 1e-10) return(0)
    a <- c(0, -0.4583333333, 0.6675347222, -1.641741416)
    return(0.3989422803 * exp(-1 / u) * sqrt(u) *
      (1 + (a[1] + (a[2] + (a[3] + a[4] * u) * u) * u) * u))
  }
  if (x < -1) {
    p <- c(0.2514091491, -0.06250580444, 0.0145838123, -0.002108817737, 0.000741124729)
    q <- c(1, -0.005571175625, 0.06225310236, -0.003137378427, 0.001931496439)
    u <- exp(-x - 1)
    return(exp(-u) / sqrt(u) * (p[1] + (p[2] + (p[3] + (p[4] + p[5] * x) * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + (q[4] + q[5] * x) * x) * x) * x))
  }
  if (x < 1) {
    p <- c(0.2868328584, 0.3564363231, 0.1523518695, 0.02251304883)
    q <- c(1, 0.6191136137, 0.1720721448, 0.02278594771)
    return((p[1] + (p[2] + (p[3] + p[4] * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + q[4] * x) * x) * x))
  }
  if (x < 4) {
    p <- c(0.2868329066, 0.3003828436, 0.09950951941, 0.008733827185)
    q <- c(1, 0.4237190502, 0.1095631512, 0.008693851567)
    return((p[1] + (p[2] + (p[3] + p[4] * x) * x) * x) /
      (q[1] + (q[2] + (q[3] + q[4] * x) * x) * x))
  }
  u <- 1 / x
  if (x < 12) {
    p <- c(1.00035163, 4.503592498, 10.8588388, 7.536052269)
    q <- c(1, 5.539969678, 19.33581111, 27.21321508)
  } else if (x < 50) {
    p <- c(1.000006517, 49.09414111, 85.05544753, 153.2153455)
    q <- c(1, 50.09928881, 139.9819104, 420.0002909)
  } else if (x < 300) {
    p <- c(1.000000983, 132.9868456, 916.2149244, -960.5054274)
    q <- c(1, 133.9887843, 1055.990413, 553.2224619)
  } else {
    u <- 1 / (x - x * log(x) / (x + 1))
    return(1 - (1 + (-0.4227843351 - 2.043403138 * u) * u) * u)
  }
  (p[1] + (p[2] + (p[3] + p[4] * u) * u) * u) /
    (q[1] + (q[2] + (q[3] + q[4] * u) * u) * u)
}

#' Landau approximation used by asymptotic calibrations
#'
#' This retains the parameterization used by the validated Python package.
#' It is an asymptotic approximation, not the finite-number null law.
#'
#' @param x,q Numeric evaluation points.
#' @param p Numeric probabilities.
#' @param location Location parameter.
#' @param scale Positive scale parameter.
#' @param log Return log density.
#' @param lower.tail,log.p Standard R distribution arguments.
#' @return Numeric values.
#' @seealso [phcmean()], [pehmean()], [new_combination()]
#' @examples
#' dlandau(c(0, 1, 5))
#' plandau(5, lower.tail = FALSE)
#' threshold <- qlandau(0.95)
#' plandau(threshold)
#' @export
dlandau <- function(x, location = 0, scale = 1, log = FALSE) {
  if (length(scale) != 1L || !is.finite(scale) || scale <= 0) stop("scale must be finite and positive", call. = FALSE)
  z <- (x - location) * pi / (2 * scale) + log(pi / (2 * scale))
  value <- vapply(z, .hr_landau_base_pdf_one, numeric(1L)) * pi / (2 * scale)
  if (log) log(value) else value
}

#' @rdname dlandau
#' @export
plandau <- function(q, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  if (length(scale) != 1L || !is.finite(scale) || scale <= 0) stop("scale must be finite and positive", call. = FALSE)
  z <- (q - location) * pi / (2 * scale) + log(pi / (2 * scale))
  value <- vapply(z, .hr_landau_base_cdf_one, numeric(1L))
  if (!lower.tail) value <- 1 - value
  if (log.p) log(value) else value
}

#' @rdname dlandau
#' @export
qlandau <- function(p, location = 0, scale = 1, lower.tail = TRUE, log.p = FALSE) {
  if (length(scale) != 1L || !is.finite(scale) || scale <= 0) stop("scale must be finite and positive", call. = FALSE)
  if (log.p) p <- exp(p)
  if (!lower.tail) p <- 1 - p
  if (any(!is.finite(p) | p < 0 | p > 1)) stop("p must lie in [0, 1]", call. = FALSE)
  vapply(p, function(probability) {
    if (probability == 0) return(-Inf)
    if (probability == 1) return(Inf)
    low <- -8
    while (.hr_landau_base_cdf_one(low) > probability) low <- low * 2
    high <- 8
    while (.hr_landau_base_cdf_one(high) < probability) high <- high * 2
    base <- stats::uniroot(
      function(value) .hr_landau_base_cdf_one(value) - probability,
      c(low, high), tol = 2e-10, maxiter = 300L
    )$root
    base * 2 * scale / pi + location + 2 * scale / pi * log(2 * scale / pi)
  }, numeric(1L))
}
