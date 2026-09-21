#' Construct coordinate-block studies for divide-and-combine inference
#'
#' Rows are observations and columns are parameters. Blocks partition columns,
#' not observations. Each block uses the sample mean, covariance of that mean
#' `W/[n(n-1)]`, and Hotelling degrees of freedom `n-1`. The finite combination
#' calibration assumes independent block p-values; splitting correlated columns
#' does not itself establish that assumption or guarantee nominal coverage.
#'
#' @param sample Finite numeric observation-by-coordinate matrix.
#' @param blocks List of coordinate indices or column names forming a partition.
#' @param block_size Positive integer for consecutive blocks; the final block
#'   may be smaller. Omit both block arguments for singleton coordinate blocks.
#' @return A list of [study()] objects with block and sample metadata.
#' @seealso [fit_dac()], [covariance_of_mean()]
#' @examples
#' set.seed(21)
#' x <- matrix(rnorm(120), 30, 4, dimnames = list(NULL, letters[1:4]))
#' blocks <- dac_studies(x, blocks = list(first = c("a", "c"),
#'                                      second = c("b", "d")))
#' blocks[[1]]$vcov
#' blocks[[1]]$projection
#' @export
dac_studies <- function(sample, blocks = NULL, block_size = NULL) {
  .hr_assert_numeric(sample, "sample")
  if (!is.matrix(sample) || nrow(sample) < 2L || ncol(sample) < 1L) {
    stop("sample must be a matrix with at least two observations", call. = FALSE)
  }
  d <- ncol(sample)
  coordinate_names <- colnames(sample)
  if (is.null(coordinate_names)) coordinate_names <- paste0("theta", seq_len(d))
  if (anyNA(coordinate_names) || anyDuplicated(coordinate_names)) stop("column names must be unique and complete", call. = FALSE)
  if (!is.null(blocks) && !is.null(block_size)) stop("supply blocks or block_size, not both", call. = FALSE)
  if (is.null(blocks)) {
    if (is.null(block_size)) block_size <- 1L
    block_size <- .hr_positive_integer(block_size, "block_size")
    blocks <- unname(split(seq_len(d), ceiling(seq_len(d) / block_size)))
  }
  if (!is.list(blocks) || !length(blocks)) stop("blocks must be a nonempty list", call. = FALSE)
  blocks <- lapply(blocks, function(indices) {
    if (is.character(indices)) indices <- match(indices, coordinate_names)
    if (!is.numeric(indices) || !length(indices) || anyNA(indices) ||
        any(!is.finite(indices)) || any(indices != trunc(indices)) ||
        any(indices < 1 | indices > d)) stop("invalid block coordinate indices", call. = FALSE)
    as.integer(indices)
  })
  indices <- unlist(blocks, use.names = FALSE)
  if (anyDuplicated(indices) || !identical(sort(indices), seq_len(d))) {
    stop("blocks must partition all coordinates exactly once", call. = FALSE)
  }
  labels <- names(blocks)
  if (is.null(labels)) labels <- paste0("block", seq_along(blocks))
  out <- lapply(seq_along(blocks), function(j) {
    index <- blocks[[j]]
    x <- sample[, index, drop = FALSE]
    study(colMeans(x), covariance_of_mean(x), diag(d)[index, , drop = FALSE],
          df = nrow(sample) - 1, label = labels[j])
  })
  structure(out, class = c("heavilyright_dac_studies", "list"),
            blocks = blocks, parameter_names = coordinate_names, sample_size = nrow(sample))
}

#' Fit a general divide-and-combine confidence region
#' @inheritParams dac_studies
#' @param method,weights,alpha,calibration,empty_policy Arguments to [fit_meta_md()].
#' @return A multidimensional fit usable with [contains()], [score()], and
#'   [support_interval()], retaining the coordinate partition and sample size.
#' @seealso [dac_studies()], [fit_meta_md()]
#' @examples
#' set.seed(21)
#' x <- matrix(rnorm(120), 30, 4)
#' fit <- fit_dac(x, block_size = 2)
#' coef(fit)
#' contains(fit, colMeans(x))
#' support_interval(fit, c(1, 0, -1, 0))
#' @export
fit_dac <- function(sample, blocks = NULL, block_size = NULL, method = "hcauchy",
                    weights = NULL, alpha = 0.05, calibration = c("auto", "finite", "landau"),
                    empty_policy = NULL) {
  inputs <- dac_studies(sample, blocks, block_size)
  fit <- fit_meta_md(inputs, ncol(sample), method, weights, alpha,
                     match.arg(calibration), empty_policy)
  fit$blocks <- attr(inputs, "blocks")
  fit$sample_size <- attr(inputs, "sample_size")
  fit$parameter_names <- attr(inputs, "parameter_names")
  fit$call <- match.call()
  class(fit) <- c("heavilyright_dac_fit", class(fit))
  fit
}

.hr_network_labels <- function(treat1, treat2, reference, treatments) {
  first <- as.character(treat1); second <- as.character(treat2)
  if (!length(first) || length(first) != length(second) || anyNA(first) || anyNA(second) ||
      any(!nzchar(first)) || any(!nzchar(second)) || any(first == second)) {
    stop("treatment pairs must be complete, nonempty, and distinct", call. = FALSE)
  }
  observed <- unique(c(first, second))
  if (is.null(reference)) reference <- observed[1L]
  reference <- as.character(reference)
  if (length(reference) != 1L || is.na(reference) || !reference %in% observed) stop("reference must name an observed treatment", call. = FALSE)
  if (is.null(treatments)) treatments <- c(reference, sort(setdiff(observed, reference)))
  treatments <- as.character(treatments)
  if (anyNA(treatments) || anyDuplicated(treatments) || !setequal(treatments, observed)) {
    stop("treatments must list every observed treatment exactly once", call. = FALSE)
  }
  parameters <- setdiff(treatments, reference)
  design <- matrix(0, length(first), length(parameters), dimnames = list(NULL, parameters))
  for (i in seq_along(first)) {
    if (first[i] != reference) design[i, match(first[i], parameters)] <- 1
    if (second[i] != reference) design[i, match(second[i], parameters)] <- -1
  }
  if (qr(design)$rank < ncol(design)) stop("treatment network is disconnected from the reference", call. = FALSE)
  list(first = first, second = second, reference = reference, treatments = treatments,
       parameters = parameters, design = design)
}

#' Construct a general contrast-based treatment network
#'
#' A positive estimate means `treat1 - treat2`. Repeated study identifiers
#' require a covariance matrix for their jointly modelled contrasts. Use an
#' independent, nonredundant contrast basis for each multivariate trial. The
#' explicit `multiarm = "independent"` option instead treats every row as a
#' separate scalar test in an independent-comparison model;
#' it does not correct dependence within a multi-arm trial.
#'
#' @param estimate Finite contrast estimates.
#' @param se Positive contrast standard errors when `vcov` is omitted.
#' @param treat1,treat2 Treatment labels for each contrast.
#' @param study_id Trial identifiers; omitted identifiers make rows independent.
#' @param reference Reference treatment; defaults to the first observed treatment.
#' @param treatments Optional complete treatment order, including reference.
#' @param vcov List of positive-definite covariance matrices, one per unique
#'   study identifier in first-appearance order; each follows that trial's rows.
#' @param df `NULL` for known covariance or a scalar/vector of Hotelling degrees
#'   of freedom, one per resulting study block.
#' @param multiarm `"require_covariance"` or explicit `"independent"` row model.
#' @return A `heavilyright_network` carrying study summaries and treatment maps.
#' @seealso [network_arm_studies()], [fit_nma()], [nma_wls()]
#' @examples
#' network <- network_studies(
#'   c(0.2, 0.25, 0.1), c(0.2, 0.18, 0.22),
#'   treat1 = c("B", "C", "C"), treat2 = c("A", "A", "B"),
#'   reference = "A"
#' )
#' nma_wls(network)
#' # Two nonredundant contrasts from one three-arm trial:
#' joint <- network_studies(c(0.2, 0.25),
#'   treat1 = c("B", "C"), treat2 = c("A", "A"),
#'   study_id = c("trial-1", "trial-1"), reference = "A",
#'   vcov = list(matrix(c(0.04, 0.01, 0.01, 0.05), 2)))
#' joint$studies[[1]]$vcov
#' @export
network_studies <- function(estimate, se = NULL, treat1, treat2, study_id = NULL,
                            reference = NULL, treatments = NULL, vcov = NULL,
                            df = NULL, multiarm = c("require_covariance", "independent")) {
  .hr_assert_numeric(estimate, "estimate")
  network <- .hr_network_labels(treat1, treat2, reference, treatments)
  m <- length(estimate)
  if (m != nrow(network$design)) stop("estimates must match treatment pairs", call. = FALSE)
  multiarm <- match.arg(multiarm)
  if (is.null(study_id)) study_id <- as.character(seq_len(m))
  study_id <- as.character(study_id)
  if (length(study_id) != m || anyNA(study_id) || any(!nzchar(study_id))) stop("study_id must identify every contrast", call. = FALSE)
  ids <- unique(study_id)
  groups <- lapply(ids, function(id) which(study_id == id))
  if (!is.null(vcov) && multiarm == "independent") stop("vcov and independent row modelling cannot be combined", call. = FALSE)
  if (is.null(vcov)) {
    .hr_assert_numeric(se, "se")
    if (length(se) != m || any(se <= 0)) stop("se must contain one positive value per contrast", call. = FALSE)
    if (any(lengths(groups) > 1L) && multiarm != "independent") {
      stop("repeated study_id requires joint vcov; use network_arm_studies() or explicitly request multiarm='independent'", call. = FALSE)
    }
    groups <- as.list(seq_len(m))
    ids <- paste(study_id, network$first, network$second, sep = ":")
    covariance <- lapply(se, function(s) matrix(s^2, 1L))
  } else {
    if (!is.list(vcov) || length(vcov) != length(groups)) stop("vcov must contain one matrix per study block", call. = FALSE)
    covariance <- vcov
  }
  if (!is.null(df)) {
    .hr_assert_numeric(df, "df")
    if (length(df) == 1L) df <- rep(df, length(groups))
    if (length(df) != length(groups)) stop("df must match study blocks", call. = FALSE)
  }
  summaries <- lapply(seq_along(groups), function(j) {
    rows <- groups[[j]]
    study(estimate[rows], covariance[[j]], network$design[rows, , drop = FALSE],
          if (is.null(df)) NULL else df[j], ids[j])
  })
  structure(c(network, list(studies = summaries, groups = groups, study_id = study_id,
                            estimate = as.numeric(estimate), multiarm = multiarm)),
            class = "heavilyright_network")
}

#' Construct correlated contrasts from independent treatment arms
#'
#' Within each trial the first arm is the local baseline. Covariances include
#' the shared baseline variance. Inputs are means and standard errors of arm
#' means, not standard deviations. Known/asymptotic covariance is the default;
#' supplying Hotelling `df` requires a scientifically justified common covariance
#' estimator and is not inferred from unequal arm sample sizes.
#'
#' @param mean,se Arm means and positive standard errors of the means.
#' @param treatment,study_id Treatment and trial identifiers per arm.
#' @inheritParams network_studies
#' @return A `heavilyright_network` for [fit_nma()].
#' @seealso [network_studies()], [fit_nma()]
#' @examples
#' network <- network_arm_studies(
#'   mean = c(1, 1.2, 0.8, 1, 1.15), se = rep(0.1, 5),
#'   treatment = c("A", "B", "C", "A", "B"),
#'   study_id = c("trial1", "trial1", "trial1", "trial2", "trial2"),
#'   reference = "A"
#' )
#' network$studies[[1]]$vcov  # Shared-baseline off-diagonal covariance
#' nma_wls(network)
#' @export
network_arm_studies <- function(mean, se, treatment, study_id, reference = NULL,
                                treatments = NULL, df = NULL) {
  .hr_assert_numeric(mean, "mean"); .hr_assert_numeric(se, "se")
  n <- length(mean)
  if (length(se) != n || any(se <= 0) || length(treatment) != n || length(study_id) != n ||
      anyNA(treatment) || anyNA(study_id)) stop("arm vectors must have matching complete entries and positive se", call. = FALSE)
  ids <- unique(as.character(study_id))
  estimates <- first <- second <- contrast_ids <- vector("list", length(ids))
  covariances <- vector("list", length(ids))
  for (j in seq_along(ids)) {
    index <- which(study_id == ids[j])
    names <- as.character(treatment[index])
    if (length(index) < 2L || anyDuplicated(names)) stop("each trial needs at least two distinct treatment arms", call. = FALSE)
    base <- index[1L]; others <- index[-1L]
    estimates[[j]] <- mean[others] - mean[base]
    first[[j]] <- as.character(treatment[others]); second[[j]] <- rep(as.character(treatment[base]), length(others))
    contrast_ids[[j]] <- rep(ids[j], length(others))
    covariances[[j]] <- diag(se[others]^2, length(others)) + se[base]^2
  }
  network_studies(unlist(estimates), treat1 = unlist(first), treat2 = unlist(second),
                  study_id = unlist(contrast_ids), reference = reference, treatments = treatments,
                  vcov = covariances, df = df)
}

#' Fit a treatment-network confidence region
#' @param network Result of [network_studies()] or [network_arm_studies()].
#' @param method,weights,alpha,calibration,empty_policy Arguments to [fit_meta_md()].
#' @return A fitted network for [nma_contrasts()] or [support_interval()].
#' @details Treatment coefficients are relative to the reference. This is a
#'   common-effect confidence region, not a random-effects heterogeneity model.
#'   Trials must satisfy the study-calibration assumptions in [fit_meta_md()].
#' @seealso [network_studies()], [network_arm_studies()], [nma_contrasts()]
#' @examples
#' network <- network_studies(c(0.2, 0.25, 0.1), c(0.2, 0.18, 0.22),
#'   treat1 = c("B", "C", "C"), treat2 = c("A", "A", "B"), reference = "A")
#' fit <- fit_nma(network)
#' coef(fit)
#' nma_contrasts(fit)
#' @export
fit_nma <- function(network, method = "hcauchy", weights = NULL, alpha = 0.05,
                    calibration = c("auto", "finite", "landau"), empty_policy = NULL) {
  if (!inherits(network, "heavilyright_network")) stop("network must come from a network constructor", call. = FALSE)
  fit <- fit_meta_md(network$studies, length(network$parameters), method, weights,
                     alpha, match.arg(calibration), empty_policy)
  fit$network <- network
  fit$parameter_names <- network$parameters
  fit$call <- match.call()
  class(fit) <- c("heavilyright_nma_fit", class(fit))
  fit
}

#' Certified treatment contrasts from a fitted network
#' @param object A result of [fit_nma()].
#' @param comparisons Optional two-column matrix/data frame of treatment labels;
#'   defaults to all unordered pairs including the reference.
#' @param maxeval Maximum optimizer evaluations per stage.
#' @return A data frame of contrast estimates, lower/upper simultaneous support
#'   bounds and widths. Its `intervals` attribute retains full certificates.
#'   Any uncertified contrast raises an error carrying completed results.
#' @seealso [fit_nma()], [support_interval()]
#' @examples
#' network <- network_studies(c(0.2, 0.25, 0.1), c(0.2, 0.18, 0.22),
#'   treat1 = c("B", "C", "C"), treat2 = c("A", "A", "B"), reference = "A")
#' fit <- fit_nma(network)
#' result <- nma_contrasts(fit, comparisons = rbind(c("B", "A"), c("C", "B")))
#' result
#' attr(result, "intervals")[[1]]$upper_diagnostics
#' @export
nma_contrasts <- function(object, comparisons = NULL, maxeval = 2000L) {
  if (!inherits(object, "heavilyright_nma_fit")) stop("object must be a fitted network", call. = FALSE)
  network <- object$network
  if (is.null(comparisons)) comparisons <- t(utils::combn(network$treatments, 2L))
  comparisons <- as.matrix(comparisons)
  if (ncol(comparisons) != 2L || !nrow(comparisons) || anyNA(comparisons) ||
      any(!comparisons %in% network$treatments) || any(comparisons[, 1] == comparisons[, 2])) stop("comparisons must contain valid distinct treatment pairs", call. = FALSE)
  intervals <- vector("list", nrow(comparisons))
  rows <- vector("list", nrow(comparisons))
  for (i in seq_len(nrow(comparisons))) {
    direction <- numeric(object$dimension)
    for (k in 1:2) if (comparisons[i, k] != network$reference) {
      direction[match(comparisons[i, k], network$parameters)] <- c(1, -1)[k]
    }
    interval <- tryCatch(support_interval(object, direction, maxeval), error = function(e) {
      .hr_stop("NMA contrast failed during support calculation", "heavilyright_numerical_error",
        comparison = comparisons[i, ], intervals = intervals, diagnostics = object$empty_set_diagnostics, cause = e)
    })
    intervals[[i]] <- interval
    if (!interval$success) .hr_stop("NMA contrast did not pass endpoint certification", "heavilyright_numerical_error", comparison = comparisons[i, ], intervals = intervals)
    rows[[i]] <- data.frame(treat1 = comparisons[i, 1], treat2 = comparisons[i, 2],
      estimate = sum(direction * object$minimum$point), lower = interval$lower,
      upper = interval$upper, width = interval$upper - interval$lower, success = TRUE)
  }
  out <- do.call(rbind, rows)
  attr(out, "intervals") <- intervals
  attr(out, "empty_set_diagnostics") <- object$empty_set_diagnostics
  out
}

#' Common-effect generalized least squares for a treatment network
#' @param network Result of a network constructor.
#' @return Named treatment effects relative to the reference and their covariance.
#' @details The return value is a list with `coefficients`, `vcov`, and
#'   `reference`. This comparator uses the supplied covariance matrices and
#'   does not estimate a between-trial heterogeneity variance.
#' @seealso [network_studies()], [fit_nma()]
#' @examples
#' network <- network_studies(c(0.2, 0.25, 0.1), c(0.2, 0.18, 0.22),
#'   treat1 = c("B", "C", "C"), treat2 = c("A", "A", "B"), reference = "A")
#' nma_wls(network)
#' @export
nma_wls <- function(network) {
  if (!inherits(network, "heavilyright_network")) stop("network must come from a network constructor", call. = FALSE)
  standard <- .hr_md_standardization(network$studies, length(network$parameters))
  dimnames(standard$inverse) <- list(network$parameters, network$parameters)
  list(coefficients = stats::setNames(standard$center, network$parameters),
       vcov = standard$inverse, reference = network$reference)
}
