.hr_analytic_support <- function(fit, direction) {
  s <- fit$studies[[1L]]; inverse <- fit$standardization$inverse
  center <- fit$standardization$center
  residual <- as.numeric(forwardsolve(t(chol(s$vcov)), s$projection %*% center - s$estimate))
  qmin <- sum(residual^2)
  qcut <- if (is.null(s$df)) stats::qchisq(fit$specification$alpha, s$q, lower.tail = FALSE) else {
    s$q * s$df / (s$df + 1 - s$q) * stats::qf(fit$specification$alpha, s$q, s$df + 1 - s$q, lower.tail = FALSE)
  }
  if (!is.finite(qcut) || qcut < qmin) .hr_stop("single-study ellipsoid is empty or not representable", "heavilyright_numerical_error")
  unit <- direction / max(abs(direction)); unit <- unit / .hr_norm(unit)
  variance <- as.numeric(crossprod(unit, inverse %*% unit))
  shift <- as.numeric(inverse %*% unit) * sqrt((qcut - qmin) / variance)
  lower <- center - shift; upper <- center + shift
  diagnostic <- function(point) {
    r <- as.numeric(forwardsolve(t(chol(s$vcov)), s$projection %*% point - s$estimate))
    error <- abs(sum(r^2) - qcut) / max(qcut, .Machine$double.xmin)
    boundary <- abs(score(fit, point) - fit$specification$cutoff)
    list(success = is.finite(error) && error <= 2e-7 && is.finite(boundary) && boundary <= 2e-7 * fit$specification$alpha,
      solver = "analytic-single-study-ellipsoid", certificate_kind = "analytic",
      boundary_error = boundary, scaled_boundary_error = boundary / fit$specification$alpha,
      reconstruction_error = error, kkt_residual = 0, multiplier = NA_real_, iterations = 0L,
      message = if (error <= 2e-7) "analytic ellipsoid and reconstructed quadratic checked" else "physical-unit reconstruction failed")
  }
  ld <- diagnostic(lower); ud <- diagnostic(upper)
  lower_value <- sum(direction * lower); upper_value <- sum(direction * upper)
  if (!is.finite(lower_value) || !is.finite(upper_value)) {
    ld$success <- ud$success <- FALSE
    ld$message <- ud$message <- "requested support values overflow; rescale the direction"
  }
  structure(list(lower = lower_value, upper = upper_value,
    lower_point = lower, upper_point = upper, lower_diagnostics = ld, upper_diagnostics = ud,
    success = ld$success && ud$success, direction = direction, minimum_point = fit$minimum$point,
    minimum_score = fit$minimum$score, minimum_diagnostics = fit$minimum,
    central_symmetry_used = TRUE, active_dimension = fit$dimension,
    specification = fit$specification, empty_set_diagnostics = fit$empty_set_diagnostics),
    class = "heavilyright_support_interval")
}

.hr_ray_start <- function(kernel, center, direction, cutoff) {
  unit <- direction / .hr_norm(direction)
  objective <- function(t) .hr_kernel_eval(center + t * unit, kernel)$value - cutoff
  outside <- 1
  for (iteration in seq_len(100L)) {
    if (objective(outside) >= 0) break
    outside <- 2 * outside
    if (iteration == 100L) .hr_stop("support ray failed to bracket a boundary", "heavilyright_numerical_error")
  }
  root <- stats::uniroot(objective, c(0, outside), tol = 1e-12, maxiter = 200L)$root
  center + root * unit
}

.hr_null_basis <- function(A, d) {
  if (!nrow(A)) return(diag(d))
  s <- svd(A, nu = 0, nv = d)
  rank <- sum(s$d > max(s$d) * 1e-12)
  if (rank == d) return(matrix(0, d, 0))
  s$v[, seq.int(rank + 1L, d), drop = FALSE]
}

.hr_polish_support <- function(z, direction, kernel, cutoff, active = integer(), maxiter = 60L, score_scale) {
  work <- list(iterations = 0L, evaluations = 0L, line_search_steps = 0L, active_cusps = active)
  finished <- function(point, message) { attr(point, "work") <- c(work, list(message = message)); point }
  if (length(z) > 250L) return(finished(z, "dense Newton disabled above 250 coordinates"))
  d <- length(z); origin <- z
  if (length(active)) {
    A <- do.call(rbind, lapply(kernel$blocks[active], `[[`, "B"))
    b <- vapply(kernel$blocks[active], function(s) s$b[1L], numeric(1))
    origin <- origin - as.numeric(.hr_pinv(A) %*% (A %*% origin + b))
    if (max(abs(A %*% origin + b)) > 1e-10) return(finished(z, "active hyperplanes are incompatible"))
    basis <- .hr_null_basis(A, d)
  } else basis <- diag(d)
  if (!ncol(basis)) return(finished(origin, "active hyperplanes determine a single point"))
  target <- as.numeric(crossprod(basis, direction))
  if (.hr_norm(target) <= 1e-14) return(finished(origin, "no tangent support direction on this manifold"))
  target <- target / .hr_norm(target)
  evaluate <- function(v) {
    work$evaluations <<- work$evaluations + 1L
    point <- origin + as.numeric(basis %*% v)
    e <- .hr_kernel_eval(point, kernel, skip = active, hessian = TRUE)
    for (j in active) {
      block <- kernel$blocks[[j]]
      e$value <- e$value + block$weight * .hr_radial(0, 1L, block$df, kernel$method)$value
    }
    list(value = (e$value - cutoff) / score_scale,
         gradient = as.numeric(crossprod(basis, e$gradient)) / score_scale,
         hessian = crossprod(basis, e$hessian %*% basis) / score_scale,
         point = point, tail_valid = e$tail_valid)
  }
  v <- numeric(ncol(basis)); current <- evaluate(v)
  eta <- max(sum(current$gradient * target), 1e-8)
  for (iteration in seq_len(maxiter)) {
    work$iterations <- iteration
    residual <- c(current$gradient - eta * target, current$value)
    merit <- .hr_norm(residual)
    if (!is.finite(merit) || !current$tail_valid || merit < 2e-11) break
    J <- rbind(cbind(current$hessian, -target), c(current$gradient, 0))
    delta <- tryCatch(as.numeric(solve(J, -residual)), error = function(e) NULL)
    if (is.null(delta) || any(!is.finite(delta))) break
    accepted <- FALSE; step <- 1
    for (line in seq_len(40L)) {
      work$line_search_steps <- work$line_search_steps + 1L
      proposed_eta <- eta + step * delta[length(delta)]
      if (proposed_eta > 0) {
        proposed_v <- v + step * delta[-length(delta)]
        proposed <- evaluate(proposed_v)
        new_merit <- .hr_norm(c(proposed$gradient - proposed_eta * target, proposed$value))
        if (proposed$tail_valid && is.finite(new_merit) && new_merit <= (1 - 1e-4 * step) * merit) {
          v <- proposed_v; eta <- proposed_eta; current <- proposed; accepted <- TRUE; break
        }
      }
      step <- step / 2
    }
    if (!accepted) break
  }
  finished(current$point, "bounded Newton refinement completed; independent certificate still required")
}

.hr_endpoint_certificate <- function(z, direction, kernel, cutoff, score_scale, snap = TRUE) {
  sub <- .hr_subgradient(z, kernel, snap = snap)
  z <- sub$point
  scale <- score_scale
  g <- sub$smooth_gradient / scale
  columns <- sub$columns / scale
  if (!ncol(columns)) {
    eta <- sum(g * direction) / sum(direction^2)
    residual <- if (eta > 0) .hr_norm(g - eta * direction) / max(.hr_norm(g), abs(eta) * .hr_norm(direction), .Machine$double.xmin) else Inf
  } else {
    # g + C a = eta d with |a| <= 1 and eta > 0. The zero
    # multiplier solution cannot certify a support maximum.
    A <- cbind(columns, -direction)
    parameters <- .hr_bounded_ls(A, -g, c(rep(-1, ncol(columns)), 0), c(rep(1, ncol(columns)), Inf),
      c(rep(0, ncol(columns)), max(sum(g * direction) / sum(direction^2), 1e-8)))
    eta <- parameters[length(parameters)]
    gradient <- g + as.numeric(columns %*% parameters[-length(parameters)])
    residual <- if (eta > 0) .hr_norm(gradient - eta * direction) /
      max(.hr_norm(gradient), eta * .hr_norm(direction), .Machine$double.xmin) else Inf
  }
  boundary <- abs(.hr_kernel_eval(z, kernel)$value - cutoff)
  success <- is.finite(scale) && scale > 0 && sub$tail_valid && is.finite(residual) && residual <= 2e-6 &&
    boundary <= 2e-8 * scale && sub$correction <= 2e-8 * scale
  list(point = z, diagnostics = list(success = success,
    certificate_kind = if (ncol(columns)) "subgradient-kkt" else "smooth-kkt",
    boundary_error = boundary, scaled_boundary_error = boundary / scale,
    kkt_residual = residual, eta = eta, multiplier = if (eta > 0) 1 / (eta * scale) else NA_real_,
    cusp_anchor_correction = sub$correction, active_cusps = sub$active,
    message = if (success) "independent boundary and positive-multiplier KKT checks passed" else "endpoint failed boundary or KKT certification"))
}

.hr_support_endpoint <- function(fit, direction, maxeval) {
  active_coordinates <- sort(unique(unlist(fit$components[vapply(fit$components, function(i) any(direction[i] != 0), logical(1))])))
  selected <- which(vapply(fit$kernel$blocks, function(b) any(b$B[, active_coordinates, drop = FALSE] != 0), logical(1)))
  base <- fit$minimum$point_z
  inactive <- setdiff(seq_along(fit$kernel$blocks), selected)
  constant <- .hr_kernel_eval(base, fit$kernel, skip = selected)$value
  kernel <- list(dimension = length(active_coordinates), method = fit$kernel$method,
    blocks = lapply(fit$kernel$blocks[selected], function(b) { b$B <- b$B[, active_coordinates, drop = FALSE]; b }))
  cutoff <- fit$specification$cutoff - constant
  start <- base[active_coordinates]
  # Normalize before taking a norm, preserving endpoints for tiny/large directions.
  raw <- direction / max(abs(direction)) * fit$standardization$scale
  target <- raw[active_coordinates] / .hr_norm(raw[active_coordinates])
  scale <- cutoff - .hr_kernel_eval(start, kernel)$value
  if (!is.finite(scale) || scale <= 0 || fit$minimum$optimality_gap_bound > 0.01 * scale) {
    .hr_stop("degenerate or unresolved support region: remaining score budget is below minimum accuracy", "heavilyright_numerical_error")
  }
  ray <- .hr_ray_start(kernel, start, target, cutoff)
  lift <- .hr_lift(kernel, ray); d <- kernel$dimension
  objective_gradient <- c(-target, rep(0, length(lift$scalar)))
  attempts <- list(); candidates <- list(ray)
  slsqp <- tryCatch(.hr_slsqp(lift$initial,
    fn = function(x) sum(objective_gradient * x), gr = function(x) objective_gradient,
    lower = lift$lower,
    hin = function(x) c((lift$evaluate(x)$value - cutoff) / scale,
                        as.numeric(lift$matrix %*% x + lift$constant)),
    hinjac = function(x) rbind(lift$evaluate(x)$gradient / scale, lift$matrix),
    deprecatedBehavior = FALSE,
    control = list(xtol_rel = 1e-11, ftol_abs = 1e-12, maxeval = maxeval)), error = identity)
  if (!inherits(slsqp, "error")) {
    attempts[[1L]] <- list(solver = "epigraph-slsqp", convergence = slsqp$convergence, iterations = slsqp$iter, message = slsqp$message)
    if (all(is.finite(slsqp$par))) candidates[[length(candidates) + 1L]] <- slsqp$par[seq_len(d)]
  } else attempts[[1L]] <- list(solver = "epigraph-slsqp", message = conditionMessage(slsqp))
  best <- NULL
  try_candidate <- function(z, solver) {
    tryCatch({
      sub <- .hr_subgradient(as.numeric(z), kernel)
      polished <- .hr_polish_support(sub$point, target, kernel, cutoff, sub$active, score_scale = scale)
      attempts[[length(attempts) + 1L]] <<- c(list(solver = solver), attr(polished, "work"))
      for (point in list(as.numeric(polished), sub$point)) {
        checked <- .hr_endpoint_certificate(point, target, kernel, cutoff, scale)
        if (isTRUE(checked$diagnostics$success) && (is.null(best) || sum(target * checked$point) > sum(target * best$point))) {
          checked$diagnostics$solver <- solver
          best <<- checked
        }
      }
    }, error = function(e) { attempts[[length(attempts) + 1L]] <<- list(solver = solver, error = conditionMessage(e)) })
  }
  for (z in rev(candidates)) try_candidate(z, "epigraph-slsqp-newton")
  if (is.null(best)) {
    fallback <- tryCatch(nloptr::cobyla(ray, fn = function(z) -sum(target * z),
      hin = function(z) (.hr_kernel_eval(z, kernel)$value - cutoff) / scale,
      deprecatedBehavior = FALSE, control = list(xtol_rel = 1e-11, maxeval = maxeval)), error = identity)
    if (!inherits(fallback, "error")) {
      attempts[[length(attempts) + 1L]] <- list(solver = "cobyla", convergence = fallback$convergence, iterations = fallback$iter, message = fallback$message)
      if (all(is.finite(fallback$par))) { candidates[[length(candidates) + 1L]] <- fallback$par; try_candidate(fallback$par, "cobyla-newton") }
    } else attempts[[length(attempts) + 1L]] <- list(solver = "cobyla", message = conditionMessage(fallback))
  }
  # Exact active-manifold solves are optimization attempts, not permissive snaps.
  if (is.null(best) && d <= 250L) {
    candidate <- candidates[[length(candidates)]]
    scalar <- which(vapply(kernel$blocks, function(b) b$q == 1L, logical(1)))
    distances <- vapply(kernel$blocks[scalar], function(b) abs(b$b + sum(b$B * candidate)) / .hr_norm(b$B), numeric(1))
    nominated <- scalar[utils::head(order(distances), 4L)]
    for (count in seq_along(nominated)) {
      sets <- utils::combn(nominated, count, simplify = FALSE)
      for (indices in sets) {
        point <- tryCatch(.hr_polish_support(candidate, target, kernel, cutoff, indices, score_scale = scale), error = identity)
        if (inherits(point, "error")) attempts[[length(attempts) + 1L]] <- list(solver = "active-manifold-newton", error = conditionMessage(point)) else {
          attempts[[length(attempts) + 1L]] <- c(list(solver = "active-manifold-discovery"), attr(point, "work"))
          try_candidate(as.numeric(point), "active-manifold-newton")
        }
        if (!is.null(best)) break
      }
      if (!is.null(best)) break
    }
  }
  if (is.null(best)) {
    best <- .hr_endpoint_certificate(candidates[[length(candidates)]], target, kernel, cutoff, scale)
    best$diagnostics$success <- FALSE
    best$diagnostics$solver <- "exhausted-constrained-fallbacks"
  }
  full <- base; full[active_coordinates] <- best$point
  physical <- fit$standardization$center + fit$standardization$scale * full
  reconstructed <- (physical - fit$standardization$center) / fit$standardization$scale
  original_score <- .hr_md_score(physical, fit)
  internal_score <- .hr_kernel_eval(full, fit$kernel)$value
  reconstruction <- abs(original_score - internal_score) / scale
  rebuilt <- .hr_endpoint_certificate(reconstructed[active_coordinates], target, kernel, cutoff, scale, snap = FALSE)
  best$diagnostics$success <- best$diagnostics$success && is.finite(reconstruction) && reconstruction <= 2e-8 && rebuilt$diagnostics$success &&
    abs(original_score - fit$specification$cutoff) <= 2e-8 * scale && is.finite(sum(direction * physical))
  best$diagnostics$reconstruction_error <- reconstruction
  best$diagnostics$score_scale <- scale
  best$diagnostics$attempts <- attempts
  best$diagnostics$iterations <- sum(vapply(attempts, function(a) a$iterations %||% 0, numeric(1)))
  best$diagnostics$newton_evaluations <- sum(vapply(attempts, function(a) a$evaluations %||% 0, numeric(1)))
  best$diagnostics$line_search_steps <- sum(vapply(attempts, function(a) a$line_search_steps %||% 0, numeric(1)))
  best$diagnostics$inactive_components_at_validated_minima <- length(inactive)
  if (!best$diagnostics$success) best$diagnostics$message <- "endpoint failed independent KKT, boundary, or physical-unit reconstruction checks"
  list(point_z = full, diagnostics = best$diagnostics)
}
