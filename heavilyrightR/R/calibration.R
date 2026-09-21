#' Standard error of a sample mean
#'
#' Uses the sample standard deviation with Bessel's correction (`n - 1`) and
#' divides by `sqrt(n)`.
#'
#' @param sample A numeric vector, matrix, or array.
#' @param margin Observation margin (ignored for a vector).
#' @return Componentwise standard errors.
#' @seealso [covariance_of_mean()]
#' @examples
#' x <- matrix(c(1, 2, 4, 7, 2, 5, 8, 11), ncol = 2)
#' standard_error_of_mean(x)
#' sqrt(diag(covariance_of_mean(x)))
#' standard_error_of_mean(x[, 1])
#' @export
standard_error_of_mean <- function(sample, margin = 1L) {
  .hr_assert_numeric(sample, "sample")
  d <- dim(sample)
  if (is.null(d)) {
    n <- length(sample)
    if (n < 2L) stop("at least two observations are required", call. = FALSE)
    return(stats::sd(sample) / sqrt(n))
  }
  margin <- as.integer(margin)
  if (length(margin) != 1L || margin < 1L || margin > length(d)) stop("invalid observation margin", call. = FALSE)
  n <- d[margin]
  if (n < 2L) stop("at least two observations are required", call. = FALSE)
  apply(sample, setdiff(seq_along(d), margin), stats::sd) / sqrt(n)
}

#' Covariance of one or more sample means
#'
#' Computes centered scatter divided by `n * (n - 1)`. This is covariance of
#' the sample mean, not covariance of individual observations. A matrix is
#' interpreted as observations by components. An array of shape `n x m x q`
#' returns `m` covariance matrices of shape `q x q`.
#'
#' @param sample Numeric matrix or three-dimensional array.
#' @param sample_margin Observation margin; currently must be the first margin.
#' @return A covariance matrix or a `q x q x m` array.
#' @seealso [standard_error_of_mean()], [study()]
#' @examples
#' x <- matrix(c(1, 2, 4, 7, 2, 5, 8, 11), ncol = 2)
#' covariance_of_mean(x)
#' stopifnot(isTRUE(all.equal(covariance_of_mean(x), stats::cov(x) / nrow(x))))
#' @export
covariance_of_mean <- function(sample, sample_margin = 1L) {
  .hr_assert_numeric(sample, "sample")
  d <- dim(sample)
  if (is.null(d) || length(d) < 2L || length(d) > 3L) {
    stop("sample must be a matrix or three-dimensional array", call. = FALSE)
  }
  if (sample_margin != 1L) stop("sample_margin must currently be 1", call. = FALSE)
  n <- d[1L]
  if (n < 2L) stop("at least two observations are required", call. = FALSE)
  if (length(d) == 2L) {
    centered <- sweep(sample, 2L, colMeans(sample), "-")
    return(crossprod(centered) / (n * (n - 1)))
  }
  m <- d[2L]
  q <- d[3L]
  out <- array(NA_real_, dim = c(q, q, m))
  for (j in seq_len(m)) {
    block <- sample[, j, , drop = FALSE]
    dim(block) <- c(n, q)
    centered <- sweep(block, 2L, colMeans(block), "-")
    out[, , j] <- crossprod(centered) / (n * (n - 1))
  }
  out
}
