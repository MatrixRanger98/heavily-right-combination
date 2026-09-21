# heavilyright

`heavilyright` is a native R implementation of the heavily-right p-value
combination and confidence-region methods in this project. It is deliberately
functional and does not call Python or translate the Python classes directly.

The package is distributed under the GNU General Public License, version 3 or
later. See [COPYING](inst/COPYING) and [third-party notices](inst/NOTICE).
It is installed from source and has not been published to CRAN.

## Install and find help

R 4.2 or later and `nloptr` 2.1.0 or later are required. From the repository root, install the
runtime dependency and then the local source directory:

```r
install.packages("nloptr")
install.packages("heavilyrightR", repos = NULL, type = "source")
library(heavilyright)
help(package = "heavilyright")
help("fit_nma", package = "heavilyright")
example("fit_nma", package = "heavilyright")
```

The dependency installation requires access to an R package repository. On
systems without an appropriate `nloptr` binary, its source installation also
requires its platform build prerequisites. Python is never a runtime dependency.
The optional `jsonlite` package enables JSON trace export.

To include rendered guides when installing from the source tree, install
the documentation tools and use `devtools::install()` with vignette building
enabled (Pandoc must also be available):

```r
install.packages(c("devtools", "knitr", "rmarkdown", "jsonlite"))
devtools::install("heavilyrightR", build_vignettes = TRUE,
                  upgrade = "never")
vignette(package = "heavilyright")
```

Alternatively, build a source archive with `R CMD build heavilyrightR`
and install that archive; built vignettes are then available without rebuilding
them during installation. Direct installation of the source directory does
not itself build vignettes.

## Documentation map

| Guide | What it covers |
|:--|:--|
| `vignette("getting-started", package = "heavilyright")` | First p-value, covariance, scalar, and multivariate examples |
| `vignette("workflows-and-diagnostics", package = "heavilyright")` | End-to-end NMA, DAC, and opt-in empty-set sensitivity analysis |
| `vignette("mathematical-methods", package = "heavilyright")` | Statistical model, exact covariance convention, calibration, convexity restrictions, support geometry, and numerical limits |
| `vignette("troubleshooting", package = "heavilyright")` | Input errors, failure classes, diagnostic interpretation, trace export, and reproducible issue reports |

Every exported function has reference help and belongs to a topic with a
runnable example. Package overview: `help("heavilyright", package = "heavilyright")`.
Guides are installed with built source archives; the Markdown notes below are
additional developer-facing source documentation.

## Core examples

```r
library(heavilyright)

rule <- new_combination("hcauchy", m = 3, calibration = "finite")
combine_p(c(0.01, 0.20, 0.40), rule)

ci <- fit_meta_1d(
  estimate = c(0.10, 0.14, 0.08),
  se = c(0.04, 0.05, 0.03)
)
confint(ci)
```

Multivariate inputs use explicit study objects:

```r
fit <- fit_meta_md(
  list(study(
    estimate = c(0.2, -0.1),
    vcov = matrix(c(0.04, 0.01, 0.01, 0.09), 2),
    df = 29
  )),
  parameter_dim = 2
)
support_interval(fit, c(1, 0))
```

## General NMA and divide-and-combine workflows

```r
# Any treatment names and reference; estimates are treat1 minus treat2.
network <- network_studies(
  estimate = c(0.2, 0.25, 0.1), se = c(0.2, 0.18, 0.22),
  treat1 = c("B", "C", "C"), treat2 = c("A", "A", "B"), reference = "A"
)
nma <- fit_nma(network)
nma_contrasts(nma)  # Every treatment pair, including the reference
nma_wls(network)   # Common-effect GLS comparator

# DAC divides parameter coordinates, retaining all observation rows.
set.seed(1)
sample <- matrix(rnorm(120), nrow = 30, ncol = 4)
dac <- fit_dac(sample, blocks = list(c(1, 3), c(2, 4)))
support_interval(dac, c(1, 0, -1, 0))
```

`network_arm_studies()` constructs joint contrasts and shared-baseline
covariances from independent treatment arms. `network_studies()` also accepts
explicit covariance blocks. Repeated trial identifiers require joint covariance
unless the caller explicitly selects the comparison-level
`multiarm = "independent"` model. Empty-set removal acts on whole input blocks.

DAC coordinate groups form an explicit partition. Unequal final blocks are
retained without padding or duplicating coordinates. Custom summary estimates,
overlapping projections, and covariance structures can be supplied directly
through `study()` and `fit_meta_md()`.

Finite-number calibration is the independence law. Neither dividing correlated
coordinates nor modelling multiple correlated comparisons as independent rows
automatically establishes nominal coverage. The package does not implement a
random-effects heterogeneity model or a post-selection coverage correction.

## Statistical contracts

- `standard_error_of_mean()` uses sample SD with divisor `n - 1`, followed by
  division by `sqrt(n)`.
- `covariance_of_mean()` returns centered scatter `W/[n(n-1)]`.
- `calibration = "finite"` names finite-number characteristic-function
  inversion. `calibration = "landau"` names the asymptotic approximation.
- Study covariance matrices are covariances of estimates, not covariances of
  individual observations.
- Empty-set study removal is off by default. Scalar and multivariate treatment
  retain complete traces. Multivariate removal requires a validated convex
  minimum, a lower bound exceeding the cutoff, and retained parameter rank.
- Every reported multivariate endpoint carries a boundary and KKT or
  subgradient-KKT diagnostic. Optimizer termination by itself is not accepted.
- Certified multi-study support is limited to HCauchy and EHMP in their proven
  convexity regimes. One-study ellipsoids use a separate analytic route.

See [NUMERICS.md](NUMERICS.md) and [EMPTY_SETS.md](EMPTY_SETS.md) for numerical
and intervention boundaries. For runnable R workflows, use the included
reference examples, such as `example("fit_nma", package = "heavilyright")`,
and the `getting-started` and `workflows-and-diagnostics` vignettes listed above.

## Validate the package

From the repository root, run the package tests without installing the source:

```sh
Rscript -e 'testthat::test_local("heavilyrightR", reporter = "summary", stop_on_failure = TRUE)'
```

Install `testthat` and `pkgload` first if needed. To include executable examples
and vignettes in a package check, use `R CMD build heavilyrightR` followed by
`R CMD check --no-manual heavilyright_0.1.0.9000.tar.gz`. Building vignettes
requires `knitr`, `rmarkdown`, and Pandoc. Run these build/check commands in a
separate working directory, using an absolute source path, to keep generated
archives and check reports out of the source tree.
