.hr_euler_gamma <- 0.5772156649015328606
.hr_probability_floor <- 1e-150

.hr_stop <- function(message, class = "heavilyright_error", ...) {
  condition <- structure(
    c(list(message = message, call = NULL), list(...)),
    class = c(class, "error", "condition")
  )
  stop(condition)
}

.hr_assert_numeric <- function(x, name, finite = TRUE, nonempty = TRUE) {
  if (!is.numeric(x) || (nonempty && length(x) == 0L)) {
    stop(sprintf("%s must be a%s numeric object", name, if (nonempty) " nonempty" else ""), call. = FALSE)
  }
  if (finite && any(!is.finite(x))) {
    stop(sprintf("%s must be finite", name), call. = FALSE)
  }
  invisible(x)
}

.hr_weights <- function(weights = NULL, m = NULL, allow_zero = FALSE) {
  if (is.null(weights)) {
    if (length(m) != 1L || !is.numeric(m) || !is.finite(m) || m < 1 || m != as.integer(m)) {
      stop("m must be a positive integer", call. = FALSE)
    }
    return(rep.int(1 / as.integer(m), as.integer(m)))
  }
  .hr_assert_numeric(weights, "weights")
  weights <- as.numeric(weights)
  if ((allow_zero && any(weights < 0)) || (!allow_zero && any(weights <= 0))) {
    stop(sprintf("weights must be %s", if (allow_zero) "nonnegative" else "strictly positive"), call. = FALSE)
  }
  if (!isTRUE(all.equal(sum(weights), 1, tolerance = 1e-10))) {
    stop("weights must sum to one", call. = FALSE)
  }
  weights
}

.hr_method <- function(method) {
  aliases <- c(
    hcauchy = "hcauchy", HCauchy = "hcauchy",
    ehmp = "ehmp", EHMP = "ehmp", hmp = "hmp", HMP = "hmp",
    cauchy = "cauchy", Cauchy = "cauchy", levy = "levy", Levy = "levy",
    fisher = "fisher", Fisher = "fisher", stouffer = "stouffer", Stouffer = "stouffer",
    bonferroni = "bonferroni", Bonferroni = "bonferroni", simes = "simes", Simes = "simes"
  )
  method <- as.character(method)[1L]
  if (is.na(aliases[method])) {
    stop("unknown combination method", call. = FALSE)
  }
  unname(aliases[method])
}

.hr_level <- function(alpha) {
  if (length(alpha) != 1L || !is.numeric(alpha) || !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("alpha must lie strictly between zero and one", call. = FALSE)
  }
  as.numeric(alpha)
}

.hr_calibration <- function(calibration, method, m) {
  calibration <- match.arg(calibration, c("auto", "finite", "landau"))
  if (method %in% c("hcauchy", "ehmp")) {
    if (calibration == "auto") return(if (m <= 1000L) "finite" else "landau")
    return(calibration)
  }
  if (method == "hmp") return("landau")
  "closed-form"
}

.hr_pvalues <- function(p, m = NULL) {
  .hr_assert_numeric(p, "p")
  if (any(p < 0 | p > 1)) stop("p-values must lie in [0, 1]", call. = FALSE)
  if (!is.null(m)) {
    if (is.null(dim(p))) {
      if (length(p) != m) stop("p must contain one value per study", call. = FALSE)
    } else if (dim(p)[length(dim(p))] != m) {
      stop("the final p-value dimension must match the number of studies", call. = FALSE)
    }
  }
  p
}

.hr_row_apply <- function(x, m, fun) {
  if (is.null(dim(x))) return(fun(as.numeric(x)))
  d <- dim(x)
  matrix_x <- matrix(x, ncol = m)
  out <- apply(matrix_x, 1L, fun)
  if (length(d) == 2L) return(out)
  array(out, dim = d[-length(d)])
}

.hr_clip_probability <- function(p) pmax(as.numeric(p), .hr_probability_floor)

.hr_positive_integer <- function(x, name, minimum = 1L) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) || x < minimum ||
      x != trunc(x) || x > .Machine$integer.max) {
    stop(sprintf("%s must be an integer >= %d", name, minimum), call. = FALSE)
  }
  as.integer(x)
}
