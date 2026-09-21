#' Configure opt-in empty-set sensitivity treatment
#'
#' Study removal is disabled unless this object is passed explicitly to a fit.
#' Removal is a sensitivity procedure and does not provide post-selection
#' coverage guarantees.
#'
#' @param max_removals Maximum number of removals.
#' @param min_remaining Minimum number of retained studies.
#' @param warn Emit a warning for each selected removal.
#' @return An `heavilyright_empty_policy` object.
#' @seealso [fit_meta_1d()], [fit_meta_md()], [write_empty_trace()]
#' @examples
#' original <- fit_meta_1d(c(-3, 3), se = 0.5)
#' original$empty
#' sensitivity <- fit_meta_1d(c(-3, 3), se = 0.5,
#'   empty_policy = empty_set_policy(max_removals = 1, warn = FALSE))
#' sensitivity$empty_set_diagnostics$final_original_indices
#' sensitivity$empty_set_diagnostics$trace
#' @export
empty_set_policy <- function(max_removals = 1L, min_remaining = 1L, warn = TRUE) {
  max_removals <- .hr_positive_integer(max_removals, "max_removals", 0L)
  min_remaining <- .hr_positive_integer(min_remaining, "min_remaining")
  if (!is.logical(warn) || length(warn) != 1L || is.na(warn)) stop("warn must be TRUE or FALSE", call. = FALSE)
  structure(
    list(strategy = "smallest_p_at_certified_minimum", max_removals = max_removals,
         min_remaining = min_remaining, warn = isTRUE(warn)),
    class = "heavilyright_empty_policy"
  )
}

.hr_empty_diagnostics <- function(enabled, encountered, resolved, trace,
                                  active, labels, original_weights,
                                  final_spec, message, final_state) {
  structure(list(
    enabled = enabled,
    encountered = encountered,
    resolved = resolved,
    strategy = if (enabled) "smallest_p_at_certified_minimum" else NULL,
    original_study_count = length(original_weights),
    final_original_indices = as.integer(active),
    original_weights = as.numeric(original_weights),
    final_weights = as.numeric(final_spec$weights),
    study_labels = as.character(labels),
    trace = trace,
    message = message,
    final_state = final_state
  ), class = "heavilyright_empty_diagnostics")
}

#' Save an empty-set trace without overwriting existing evidence
#' @param diagnostics Empty-set diagnostics returned by a fit.
#' @param path Destination JSON path.
#' @return The normalized output path, invisibly. Requires the optional
#'   `jsonlite` package and refuses to overwrite an existing file.
#' @seealso [empty_set_policy()]
#' @examples
#' if (requireNamespace("jsonlite", quietly = TRUE)) {
#'   fit <- fit_meta_1d(c(-3, 3), se = 0.5)
#'   path <- tempfile(fileext = ".json")
#'   write_empty_trace(fit$empty_set_diagnostics, path)
#'   record <- jsonlite::read_json(path, simplifyVector = TRUE)
#'   print(record$final_state)
#'   unlink(path)
#' }
#' @export
write_empty_trace <- function(diagnostics, path) {
  if (!inherits(diagnostics, "heavilyright_empty_diagnostics")) stop("diagnostics has the wrong class", call. = FALSE)
  if (file.exists(path)) stop("refusing to overwrite an existing diagnostics file", call. = FALSE)
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required to write JSON", call. = FALSE)
  jsonlite::write_json(unclass(diagnostics), path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  invisible(normalizePath(path, mustWork = TRUE))
}
