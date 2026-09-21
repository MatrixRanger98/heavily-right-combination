#' Correlation-matrix constructors
#' @param dimension Positive integer dimension.
#' @param rho Correlation parameter. AR(1) accepts `[-1, 1]`;
#'   equicorrelation accepts `[-1/(dimension-1), 1]` for dimension greater
#'   than one. Boundary values can produce singular matrices, which [study()]
#'   does not accept as covariance matrices.
#' @return A numeric correlation matrix.
#' @examples
#' ar1_correlation(3, 0.5)
#' equicorrelation(3, 0.2)
#' @export
ar1_correlation <- function(dimension, rho) {
  dimension <- as.integer(dimension)
  if (length(dimension) != 1L || is.na(dimension) || dimension < 1L) stop("dimension must be positive", call. = FALSE)
  if (length(rho) != 1L || !is.finite(rho) || abs(rho) > 1) stop("rho must lie in [-1, 1]", call. = FALSE)
  index <- seq_len(dimension)
  rho ^ abs(outer(index, index, "-"))
}

#' @rdname ar1_correlation
#' @export
equicorrelation <- function(dimension, rho) {
  dimension <- as.integer(dimension)
  if (length(dimension) != 1L || is.na(dimension) || dimension < 1L) stop("dimension must be positive", call. = FALSE)
  lower <- if (dimension == 1L) -1 else -1 / (dimension - 1)
  if (length(rho) != 1L || !is.finite(rho) || rho < lower || rho > 1) stop("rho is outside the valid equicorrelation range", call. = FALSE)
  matrix(rho, dimension, dimension) + diag(1 - rho, dimension)
}
