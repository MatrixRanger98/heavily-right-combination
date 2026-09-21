#' Heavy-right score transforms
#'
#' `half_cauchy_score()` computes `cot(pi * p / 2)`. `reciprocal_score()`
#' computes `1 / p`. Zero p-values map to positive infinity.
#'
#' @param p Numeric p-values in `[0, 1]`.
#' @return A numeric object with the same shape as `p`.
#' @seealso [combine_p()]
#' @examples
#' p <- c(0, 0.01, 0.5, 1)
#' half_cauchy_score(p)
#' reciprocal_score(p)
#' @export
half_cauchy_score <- function(p) {
  p <- .hr_pvalues(p)
  out <- 1 / tan(pi * p / 2)
  out[p == 1] <- 0
  out
}

#' @rdname half_cauchy_score
#' @export
reciprocal_score <- function(p) {
  p <- .hr_pvalues(p)
  1 / p
}
