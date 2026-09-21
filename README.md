# heavily-right-combination

Python and native R implementations of heavily-right p-value combination,
confidence regions, network meta-analysis (NMA), and divide-and-combine (DAC),
with reproducible experiments and saved paper results.

## Paper

Tianle Liu, Xiao-Li Meng, and Natesh S. Pillai (2025).
[A Heavily Right Strategy for Statistical Inference with Dependent Studies in Arbitrary Dimensions](https://arxiv.org/abs/2501.01065).
arXiv:2501.01065. [Read the PDF](https://arxiv.org/pdf/2501.01065).

## Install

Commands below start at this repository's root. Python requires 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[reproduction,dev]'
```

```python
from heavily_right.combination import CombinationTest

test = CombinationTest([0.2, 0.3, 0.5], method="HCauchy", level=0.05)
print(test.get_global_p([0.01, 0.20, 0.40]))
```

R requires 4.2 or newer; it does not use Python as a computational backend.

```r
install.packages("nloptr")
install.packages("heavilyrightR", repos = NULL, type = "source")
library(heavilyright)
example("fit_nma", package = "heavilyright")
```

## Repository guide

| Directory | Contents |
|---|---|
| [heavily_right/](heavily_right/) | Reusable Python statistical package |
| [docs/](docs/README.md) | Python API, methods, runnable examples, numerical safeguards and troubleshooting |
| [tests/](tests/) | Python statistical, numerical, workflow and reproduction tests |
| [heavilyrightR/](heavilyrightR/README.md) | Native R package, reference help, tests and vignettes |
| [reproduction/](reproduction/README.md) | Python experiment configurations, simulation, summaries and plotting |
| [results/](results/README.md) | Saved results and complete figure/table-to-code mapping |
| [geogebra/](geogebra/README.md) | Eight illustration constructions and PDF-export instructions |

The NMA input CSV and its provenance live in
[reproduction/network_meta_analysis/data/](reproduction/network_meta_analysis/data/README.md).
Saved numerical arrays are documented individually in the
[NPZ catalog](reproduction/NPZ_DATA.md). See [USAGE.md](USAGE.md) for worked
package examples and [the experiment catalog](reproduction/EXPERIMENTS.md)
for each generator, configuration and output filename.

## Reproduce and test

```bash
python -m reproduction.run --list
python -m reproduction.run illustrations/connectivity-scores \
  --profile smoke --max-seconds 300 --run-id first-check --results-root my-results
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
ruff check heavily_right reproduction tests
python tools/check_release.py
```

`--results-root` is the directory under which experiment/run subdirectories
are created. A run ID cannot overwrite an existing run. Saved-data replotting
avoids repeating expensive simulations; see [reproduction/SAVED_DATA.md](reproduction/SAVED_DATA.md).
Full profiles can be expensive. Consult the runtime guidance before running them.

The Python package and experiment layer are separate. The Python wheel includes
the runtime package, experiment modules and NMA CSV; the repository additionally
contains both languages' tests/docs, saved results and external-tool sources.
R currently implements selected validation workflows rather than every Python
paper experiment. GeoGebra export requires an external installation.

## Scientific qualifications

- Covariance of an estimated mean is centered scatter divided by `n*(n-1)`.
- Finite-number calibration uses the independence null law; dependent inputs
  require the stated statistical assumptions.
- Empty-set study removal is opt-in and diagnostic. Coverage experiments count
  certified empty sets as noncoverage without removing studies or resampling.
- Convex support calculations apply only in the documented normal/t and
  multivariate degree-of-freedom regimes. Numerical uncertainty is not emptiness.
- Passing package tests does not establish that every saved result matches a
  manuscript figure or that every full experiment has been rerun.

## License and attribution

Project-owned material is distributed under **GPL-3.0-or-later**; see
[LICENSE](LICENSE). Third-party components retain their own attribution and
terms; consult [THIRD_PARTY.md](THIRD_PARTY.md), including the ROOT coefficient
provenance qualification and Senn2013 dataset citation.

Maintainer: Tianle Liu, [tylerliuthu@gmail.com](mailto:tylerliuthu@gmail.com).
