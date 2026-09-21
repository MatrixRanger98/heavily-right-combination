# Internal convex radial kernel. All derivatives use dimensionless coordinates.
.hr_norm <- function(x) {
  largest <- max(abs(x), 0)
  if (largest == 0) return(0)
  largest * sqrt(sum((x / largest)^2))
}

.hr_radial <- function(r, q, df, method) {
  r <- max(0, r)
  if (q == 1L) {
    if (is.null(df)) {
      lp <- log(2) + stats::pnorm(r, lower.tail = FALSE, log.p = TRUE)
      ld <- log(2) + stats::dnorm(r, log = TRUE)
      log_slope <- -r
    } else {
      lp <- log(2) + stats::pt(r, df, lower.tail = FALSE, log.p = TRUE)
      ld <- log(2) + stats::dt(r, df, log = TRUE)
      log_slope <- -(df + 1) * r / (df + r^2)
    }
  } else if (r == 0) {
    second <- if (q == 2L) {
      density_q0 <- if (is.null(df)) 0.5 else (df - 1) / (2 * df)
      if (method == "hcauchy") pi * density_q0 else 2 * density_q0
    } else 0
    return(list(value = if (method == "hcauchy") 0 else 1, first = 0,
                second = second, probability = 1, tail_valid = TRUE))
  } else if (is.null(df)) {
    lp <- stats::pchisq(r^2, q, lower.tail = FALSE, log.p = TRUE)
    ld <- stats::dchisq(r^2, q, log = TRUE) + log(2 * r)
    log_slope <- (q - 1) / r - r
  } else {
    argument_scale <- (df + 1 - q) / (q * df)
    lp <- stats::pf(argument_scale * r^2, q, df + 1 - q, lower.tail = FALSE, log.p = TRUE)
    ld <- stats::df(argument_scale * r^2, q, df + 1 - q, log = TRUE) + log(2 * argument_scale * r)
    log_slope <- (q - 1) / r - (df + 1) * r / (df + r^2)
  }
  # Trials outside the representable convex domain receive a finite barrier;
  # no certificate is ever accepted there.
  valid <- is.finite(lp) && lp > log(.hr_probability_floor)
  lp_safe <- max(lp, log(.hr_probability_floor))
  probability <- exp(lp_safe)
  if (method == "ehmp") {
    value <- exp(-lp_safe)
    first <- exp(min(690, ld - 2 * lp_safe))
    second <- first * (log_slope + 2 * exp(min(345, ld - lp_safe)))
  } else {
    if (probability < 1e-7) {
      value <- 2 / pi * exp(-lp_safe)
      first <- 2 / pi * exp(min(690, ld - 2 * lp_safe))
      second <- first * (log_slope + 2 * exp(min(345, ld - lp_safe)))
    } else {
      sine <- sin(pi / 2 * probability)
      value <- if (r == 0) 0 else 1 / tan(pi / 2 * probability)
      first <- pi / 2 * exp(ld) / sine^2
      second <- first * (log_slope + pi * exp(ld) * value)
    }
  }
  list(value = value, first = first, second = second,
       probability = exp(lp), tail_valid = valid)
}

.hr_make_kernel <- function(fit) {
  blocks <- lapply(seq_along(fit$studies), function(i) {
    item <- fit$studies[[i]]
    root <- t(chol(item$vcov))
    list(B = forwardsolve(root, sweep(item$projection, 2, fit$standardization$scale, "*")),
         b = as.numeric(forwardsolve(root, item$projection %*% fit$standardization$center - item$estimate)),
         q = item$q, df = item$df, weight = fit$specification$weights[i])
  })
  list(blocks = blocks, dimension = fit$dimension, method = fit$specification$method)
}

.hr_kernel_eval <- function(z, kernel, skip = integer(), hessian = FALSE) {
  d <- length(z)
  value <- 0; gradient <- numeric(d); H <- if (hessian) matrix(0, d, d) else NULL
  tail_valid <- TRUE
  for (i in setdiff(seq_along(kernel$blocks), skip)) {
    block <- kernel$blocks[[i]]
    residual <- as.numeric(block$b + block$B %*% z)
    r <- .hr_norm(residual)
    radial <- .hr_radial(r, block$q, block$df, kernel$method)
    value <- value + block$weight * radial$value
    tail_valid <- tail_valid && radial$tail_valid
    if (r > 0) {
      g <- as.numeric(crossprod(block$B, residual / r))
      gradient <- gradient + block$weight * radial$first * g
      if (hessian) H <- H + block$weight * (
        radial$second * tcrossprod(g) + radial$first / r *
          (crossprod(block$B) - tcrossprod(g)))
    } else if (hessian && block$q != 1L) {
      H <- H + block$weight * radial$second * crossprod(block$B)
    }
  }
  list(value = value, gradient = gradient, hessian = H, tail_valid = tail_valid)
}

.hr_lift <- function(kernel, start) {
  scalar <- which(vapply(kernel$blocks, function(b) b$q == 1L, logical(1L)))
  d <- kernel$dimension; k <- length(scalar)
  rows <- if (k) do.call(rbind, lapply(kernel$blocks[scalar], `[[`, "B")) else matrix(0, 0, d)
  offsets <- if (k) vapply(kernel$blocks[scalar], function(b) b$b[1], numeric(1)) else numeric()
  matrix <- rbind(cbind(rows, -diag(k)), cbind(-rows, -diag(k)))
  evaluate <- function(x) {
    result <- .hr_kernel_eval(x[seq_len(d)], kernel, skip = scalar)
    gradient <- c(result$gradient, numeric(k))
    for (j in seq_len(k)) {
      block <- kernel$blocks[[scalar[j]]]
      radial <- .hr_radial(x[d + j], 1L, block$df, kernel$method)
      result$value <- result$value + block$weight * radial$value
      gradient[d + j] <- block$weight * radial$first
    }
    list(value = result$value, gradient = gradient)
  }
  list(initial = c(start, abs(as.numeric(rows %*% start) + offsets)),
       lower = c(rep(-Inf, d), rep(0, k)), scalar = scalar, rows = rows,
       offsets = offsets, matrix = matrix, constant = c(offsets, -offsets),
       evaluate = evaluate)
}

.hr_bounded_ls <- function(matrix, target, lower, upper, start = NULL) {
  n <- ncol(matrix)
  if (!n) return(numeric())
  if (is.null(start)) start <- pmin(upper, pmax(lower, rep(0, n)))
  norm <- max(.hr_norm(target), max(abs(matrix)), .Machine$double.xmin)
  A <- matrix / norm; b <- target / norm
  result <- stats::optim(start,
    function(x) sum((A %*% x - b)^2),
    function(x) 2 * as.numeric(crossprod(A, A %*% x - b)),
    method = "L-BFGS-B", lower = lower, upper = upper,
    control = list(maxit = 2000L, factr = 1, pgtol = 1e-13))
  result$par
}

.hr_subgradient <- function(z, kernel, snap = TRUE, tolerance = 1e-8) {
  active <- which(vapply(kernel$blocks, function(b) {
    b$q == 1L && abs(b$b + sum(b$B * z)) <= tolerance * .hr_norm(b$B)
  }, logical(1)))
  if (length(active) && snap) {
    A <- do.call(rbind, lapply(kernel$blocks[active], `[[`, "B"))
    r <- vapply(kernel$blocks[active], function(b) as.numeric(b$b + b$B %*% z), numeric(1))
    delta <- as.numeric(.hr_pinv(A, 1e-12) %*% -r)
    proposed <- z + delta
    actual <- vapply(kernel$blocks[active], function(b) as.numeric(b$b + b$B %*% proposed), numeric(1))
    rounding_tolerance <- 128 * .Machine$double.eps * max(1, .hr_norm(proposed))
    if (.hr_norm(delta) <= 2 * tolerance && max(abs(A %*% delta + r)) <= rounding_tolerance &&
        max(abs(actual)) <= rounding_tolerance) z <- proposed else active <- integer()
  } else if (length(active)) {
    # Reconstructed endpoints must be checked at their actual returned point.
    residuals <- vapply(kernel$blocks[active], function(b) as.numeric(b$b + b$B %*% z), numeric(1))
    active <- active[abs(residuals) <= 128 * .Machine$double.eps * max(1, .hr_norm(z))]
  }
  result <- .hr_kernel_eval(z, kernel, skip = active, hessian = FALSE)
  columns <- matrix(0, kernel$dimension, length(active))
  correction <- 0
  for (j in seq_along(active)) {
    block <- kernel$blocks[[active[j]]]
    radial <- .hr_radial(0, 1L, block$df, kernel$method)
    residual <- abs(as.numeric(block$b + block$B %*% z))
    actual <- .hr_radial(residual, 1L, block$df, kernel$method)
    result$value <- result$value + block$weight * actual$value
    columns[, j] <- block$weight * radial$first * as.numeric(block$B)
    correction <- correction + block$weight * (abs(actual$value - radial$value) + radial$first * residual)
  }
  coefficients <- .hr_bounded_ls(columns, -result$gradient, rep(-1, length(active)), rep(1, length(active)))
  list(point = z, value = result$value, smooth_gradient = result$gradient,
       gradient = result$gradient + as.numeric(columns %*% coefficients),
       columns = columns, active = active, correction = correction,
       hessian = result$hessian, tail_valid = result$tail_valid)
}

.hr_domain_radius <- function(kernel, level) {
  baseline <- if (kernel$method == "ehmp") 1 else 0
  if (!is.finite(level) || level <= baseline) return(NULL)
  H <- matrix(0, kernel$dimension, kernel$dimension)
  linear <- numeric(kernel$dimension); radius_sum <- 0
  for (block in kernel$blocks) {
    allowed <- (level - (1 - block$weight) * baseline) / block$weight
    probability <- if (kernel$method == "ehmp") 1 / allowed else 2 / pi * atan(1 / allowed)
    if (!is.finite(probability) || probability <= .hr_probability_floor || probability >= 1) return(NULL)
    Q <- if (is.null(block$df)) stats::qchisq(probability, block$q, lower.tail = FALSE) else {
      stats::qf(probability, block$q, block$df + 1 - block$q, lower.tail = FALSE) *
        block$q * block$df / (block$df + 1 - block$q)
    }
    radius_sum <- radius_sum + block$weight * Q
    H <- H + block$weight * crossprod(block$B)
    linear <- linear + block$weight * as.numeric(crossprod(block$B, block$b))
  }
  eigenvalues <- eigen(H, symmetric = TRUE, only.values = TRUE)$values
  floor <- min(eigenvalues) - 64 * .Machine$double.eps * kernel$dimension * max(abs(eigenvalues))
  if (!is.finite(floor) || floor <= 0 || !is.finite(radius_sum)) return(NULL)
  L <- .hr_norm(linear)
  (L + sqrt(L^2 + floor * radius_sum)) / floor * (1 + 1e-10)
}

.hr_tail_domain_valid <- function(kernel, level) {
  baseline <- if (kernel$method == "ehmp") 1 else 0
  all(vapply(kernel$blocks, function(block) {
    budget <- level - (1 - block$weight) * baseline
    if (budget <= 0) return(TRUE)
    inverse <- block$weight / budget
    p <- if (kernel$method == "ehmp") inverse else 2 / pi * atan(inverse)
    is.finite(p) && p > .hr_probability_floor
  }, logical(1)))
}

.hr_minimum_check <- function(z, kernel, cutoff, snap = TRUE, maximum_gap = Inf) {
  sub <- .hr_subgradient(z, kernel, snap = snap)
  radius <- .hr_domain_radius(kernel, max(sub$value, cutoff))
  baseline <- if (kernel$method == "ehmp") 1 else 0
  bound <- baseline
  if (!is.null(radius)) bound <- max(bound, sub$value - sub$correction - sum(sub$gradient * sub$point) - radius * .hr_norm(sub$gradient))
  bound <- bound - 1e-10 * max(1, abs(bound), abs(sub$value))
  feasible_radius <- .hr_domain_radius(kernel, cutoff)
  lower <- baseline
  if (!is.null(feasible_radius)) lower <- max(lower, sub$value - sub$correction -
    sum(sub$gradient * sub$point) - feasible_radius * .hr_norm(sub$gradient))
  if (is.finite(lower)) lower <- lower - 1e-10 * max(1, abs(lower), abs(sub$value))
  gap <- max(0, sub$value - bound)
  tolerance <- min(maximum_gap, 2e-6 * max(1, abs(sub$value), abs(cutoff)))
  stationarity <- .hr_norm(sub$gradient) / max(1, abs(sub$value), abs(cutoff))
  list(point_z = sub$point, score = sub$value, normalized_gradient = stationarity,
       minimum_value_lower_bound = bound, feasible_set_score_lower_bound = lower,
       optimality_gap_bound = gap, optimality_gap_tolerance = tolerance,
       cusp_anchor_correction = sub$correction,
       success = sub$tail_valid && is.finite(gap) && gap <= tolerance && stationarity <= 2e-6,
       empty_certified = is.finite(lower) && lower > cutoff + 1e-8 * max(1, abs(cutoff)))
}

# Keep the solver boundary local so failure-injection tests never alter a
# dependency's namespace. The wrapper does not change arguments or results.
.hr_slsqp <- function(...) nloptr::slsqp(...)

.hr_minimize_kernel <- function(kernel, cutoff, maxeval = 5000L, start = NULL, maximum_gap = Inf) {
  if (is.null(start)) start <- numeric(kernel$dimension)
  if (length(maximum_gap) != 1L || is.na(maximum_gap) || maximum_gap <= 0)
    stop("maximum_gap must be positive", call. = FALSE)
  if (!.hr_tail_domain_valid(kernel, cutoff)) .hr_stop("cutoff permits a probability at or below the public probability floor", "heavilyright_numerical_error")
  initial <- .hr_kernel_eval(start, kernel)
  if (!initial$tail_valid) .hr_stop("initial minimum has clipped or underflowed tails; emptiness is unknown", "heavilyright_numerical_error")
  check <- .hr_minimum_check(start, kernel, cutoff, maximum_gap = maximum_gap)
  attempts <- list(); iterations <- 0L; evaluations <- 0L
  if (!check$success) for (restart in seq_len(3L)) {
    remaining <- maxeval - iterations
    if (remaining <= 0) break
    lift <- .hr_lift(kernel, start)
    scale <- max(1, cutoff, check$score)
    before <- evaluations
    result <- tryCatch(.hr_slsqp(lift$initial,
      fn = function(x) { evaluations <<- evaluations + 1L; lift$evaluate(x)$value / scale },
      gr = function(x) lift$evaluate(x)$gradient / scale,
      lower = lift$lower,
      hin = if (nrow(lift$matrix)) function(x) as.numeric(lift$matrix %*% x + lift$constant) else NULL,
      hinjac = if (nrow(lift$matrix)) function(x) lift$matrix else NULL,
      deprecatedBehavior = FALSE,
      control = list(xtol_rel = 1e-12, ftol_abs = 1e-14, maxeval = remaining)),
      error = function(e) list(par = lift$initial, convergence = -999L, message = conditionMessage(e)))
    # NLopt's reported iteration count is the unit of its maxeval limit.
    # If unavailable after a failure, conservatively spend the remaining budget.
    used <- if (length(result$iter) == 1L && is.finite(result$iter) && result$iter >= 0)
      max(1, result$iter) else remaining
    iterations <- iterations + used
    attempts[[restart]] <- list(convergence = result$convergence, message = result$message,
      iterations = used, objective_evaluations = evaluations - before, maxeval = remaining)
    point <- result$par[seq_len(kernel$dimension)]
    if (all(is.finite(point)) && .hr_kernel_eval(point, kernel)$value <= check$score * (1 + 1e-12)) start <- point
    check <- .hr_minimum_check(start, kernel, cutoff, maximum_gap = maximum_gap)
    if (check$success) break
  }
  check$optimizer <- list(method = "analytic-gradient epigraph SLSQP", attempts = attempts)
  check$certification_level <- cutoff
  check$maximum_gap <- maximum_gap
  check$iterations <- iterations
  check$objective_evaluations <- evaluations
  if (!check$success) .hr_stop("minimum failed independent stationarity or minimum-value gap certification", "heavilyright_numerical_error", minimum_diagnostics = check)
  check
}

.hr_component_summary <- function(fit, children, masses, local_kernels) {
  point <- numeric(fit$dimension)
  for (i in seq_along(children)) point[fit$components[[i]]] <- children[[i]]$point_z
  value <- .hr_kernel_eval(point, fit$kernel)$value
  physical <- fit$standardization$center + fit$standardization$scale * point
  reconstructed <- (physical - fit$standardization$center) / fit$standardization$scale
  reconstructed_score <- .hr_md_score(physical, fit)
  if (!is.finite(reconstructed_score) || abs(reconstructed_score - value) > 2e-8 * max(1, abs(value)) ||
      any(.hr_md_pvalues(physical, fit$studies) <= .hr_probability_floor)) {
    .hr_stop("minimum fails physical-unit reconstruction; recenter or rescale the input", "heavilyright_numerical_error")
  }
  bounds <- vapply(children, `[[`, numeric(1), "minimum_value_lower_bound")
  for (i in seq_along(children)) {
    check <- .hr_minimum_check(reconstructed[fit$components[[i]]], local_kernels[[i]],
      children[[i]]$certification_level, snap = FALSE, maximum_gap = children[[i]]$maximum_gap)
    if (!isTRUE(children[[i]]$success) || !isTRUE(check$success))
      .hr_stop("reconstructed component minimum fails stationarity or gap certification", "heavilyright_numerical_error")
    # Keep both the supplied certificate and the reconstructed-point check
    # conservative. Reconstruction must not silently tighten a loose child.
    bounds[i] <- min(bounds[i], check$minimum_value_lower_bound)
  }
  bound <- sum(masses * bounds)
  rounding <- 1e-10 * max(1, abs(value), abs(reconstructed_score), abs(bound))
  bound <- bound - rounding
  lower <- sum(masses * vapply(children, `[[`, numeric(1), "feasible_set_score_lower_bound"))
  lower <- lower - 1e-10 * max(1, abs(lower), abs(value), abs(reconstructed_score))
  gap <- max(0, max(value, reconstructed_score) - bound)
  tolerance <- 2e-6 * max(1, value, abs(fit$specification$cutoff))
  list(point_z = point, point = physical, score = value,
    reconstruction_error = abs(reconstructed_score - value),
    reconstruction_reserve = max(0, max(value, reconstructed_score) -
      sum(masses * vapply(children, `[[`, numeric(1), "score"))),
    rounding_reserve = rounding, minimum_value_lower_bound = bound,
    feasible_set_score_lower_bound = lower, optimality_gap_bound = gap,
    optimality_gap_tolerance = tolerance, success = is.finite(gap) && gap <= tolerance)
}

.hr_md_minimum <- function(fit, maxeval = 5000L) {
  if (length(fit$studies) == 1L) {
    value <- .hr_md_score(fit$standardization$center, fit)
    return(list(point_z = numeric(fit$dimension), point = fit$standardization$center,
      score = value, success = TRUE, normalized_gradient = 0,
      minimum_value_lower_bound = value, feasible_set_score_lower_bound = value,
      optimality_gap_bound = 0, optimality_gap_tolerance = 0,
      empty_certified = value > fit$specification$cutoff,
      cusp_anchor_correction = 0, optimizer = list(method = "analytic-single-study")))
  }
  kernel <- fit$kernel; cutoff <- fit$specification$cutoff
  if (!.hr_tail_domain_valid(kernel, cutoff)) .hr_stop("cutoff permits a probability at or below the public probability floor", "heavilyright_numerical_error")
  children <- list(); masses <- numeric(); local_kernels <- list(); local_cutoffs <- numeric()
  for (i in seq_along(fit$components)) {
    indices <- fit$components[[i]]
    selected <- which(vapply(kernel$blocks, function(b) any(b$B[, indices, drop = FALSE] != 0), logical(1)))
    blocks <- kernel$blocks[selected]
    mass <- sum(vapply(blocks, `[[`, numeric(1), "weight")); masses[i] <- mass
    child <- list(dimension = length(indices), method = kernel$method,
      blocks = lapply(blocks, function(b) { b$B <- b$B[, indices, drop = FALSE]; b$weight <- b$weight / mass; b }))
    local_kernels[[i]] <- child
    baseline <- if (kernel$method == "ehmp") 1 else 0
    local_cutoff <- (cutoff - (1 - mass) * baseline) / mass
    local_cutoffs[i] <- local_cutoff
    # Solve components at their own natural score scale. A huge inherited
    # cutoff must not relax minimum accuracy for a weakly weighted component.
    natural <- max(baseline + 1, .hr_kernel_eval(numeric(length(indices)), child)$value)
    result <- .hr_minimize_kernel(child, min(local_cutoff, natural), maxeval = maxeval)
    feasible_check <- .hr_minimum_check(result$point_z, child, local_cutoff)
    result$feasible_set_score_lower_bound <- feasible_check$feasible_set_score_lower_bound
    result$empty_certified <- feasible_check$empty_certified
    children[[i]] <- result
  }
  summary <- .hr_component_summary(fit, children, masses, local_kernels)
  initial_gap <- summary$optimality_gap_bound
  refinements <- list()
  if (!summary$success) {
    eligible <- which(vapply(local_kernels, function(k) length(k$blocks) > 1L, logical(1)))
    fixed <- setdiff(seq_along(children), eligible)
    fixed_gap <- sum(masses[fixed] * vapply(children[fixed], `[[`, numeric(1), "optimality_gap_bound"))
    available <- summary$optimality_gap_tolerance - summary$rounding_reserve -
      summary$reconstruction_reserve - fixed_gap
    if (length(eligible) && is.finite(available) && available > 0) {
      # Allocate in PARENT score units; only then divide by component mass.
      # Half the available budget leaves headroom, without relaxing acceptance.
      allocation <- available / (2 * length(eligible))
      for (i in eligible) {
        old <- children[[i]]; remaining <- maxeval - old$iterations
        if (masses[i] * old$optimality_gap_bound <= allocation || remaining <= 0) next
        cap <- allocation / masses[i]
        failure <- NULL
        retry <- tryCatch(.hr_minimize_kernel(local_kernels[[i]], old$certification_level,
          maxeval = remaining, start = old$point_z, maximum_gap = cap),
          error = function(e) { failure <<- e; e$minimum_diagnostics })
        accepted <- is.null(failure) && isTRUE(retry$success) &&
          is.finite(retry$optimality_gap_bound) && retry$optimality_gap_bound <= cap &&
          retry$optimality_gap_bound < old$optimality_gap_bound
        if (accepted) {
          indices <- fit$components[[i]]
          physical <- fit$standardization$center[indices] + fit$standardization$scale[indices] * retry$point_z
          reconstructed <- (physical - fit$standardization$center[indices]) / fit$standardization$scale[indices]
          validation <- .hr_minimum_check(reconstructed, local_kernels[[i]], old$certification_level,
            snap = FALSE, maximum_gap = cap)
          accepted <- isTRUE(validation$success)
        }
        if (accepted) {
          feasible <- .hr_minimum_check(retry$point_z, local_kernels[[i]], local_cutoffs[i])
          retry$feasible_set_score_lower_bound <- feasible$feasible_set_score_lower_bound
          retry$empty_certified <- feasible$empty_certified
          children[[i]] <- retry
        }
        used <- if (is.null(retry$iterations)) remaining else retry$iterations
        evaluations <- if (is.null(retry$objective_evaluations)) NA_real_ else retry$objective_evaluations
        children[[i]]$iterations <- old$iterations + used
        children[[i]]$objective_evaluations <- old$objective_evaluations + evaluations
        children[[i]]$optimizer$attempts <- c(old$optimizer$attempts, retry$optimizer$attempts)
        refinements[[length(refinements) + 1L]] <- list(component = i,
          parent_allocation = allocation, maximum_gap = cap, maxeval = remaining,
          iterations = used, objective_evaluations = evaluations, accepted = accepted,
          initial_gap = old$optimality_gap_bound,
          final_gap = children[[i]]$optimality_gap_bound,
          message = if (accepted) "parent gap-budget refinement accepted" else
            if (is.null(failure)) "parent gap-budget refinement not accepted" else conditionMessage(failure))
      }
      summary <- .hr_component_summary(fit, children, masses, local_kernels)
    }
  }
  summary <- c(summary, list(normalized_gradient = max(vapply(children, `[[`, numeric(1), "normalized_gradient")),
    empty_certified = isTRUE(summary$success) && is.finite(summary$feasible_set_score_lower_bound) &&
      summary$feasible_set_score_lower_bound > cutoff + 1e-8 * max(1, abs(cutoff)),
    component_indices = fit$components, component_weights = masses, component_diagnostics = children,
    component_refinements = length(refinements), component_refinement_attempts = refinements,
    initial_optimality_gap_bound = initial_gap,
    iterations = sum(vapply(children, `[[`, numeric(1), "iterations")),
    objective_evaluations = sum(vapply(children, `[[`, numeric(1), "objective_evaluations")),
    cusp_anchor_correction = sum(masses * vapply(children, `[[`, numeric(1), "cusp_anchor_correction")),
    optimizer = list(method = "independently normalized convex component minima")))
  if (!summary$success) .hr_stop("component minima validated but aggregate gap exceeds tolerance",
    "heavilyright_numerical_error", minimum_diagnostics = summary)
  summary
}
