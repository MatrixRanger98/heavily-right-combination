# Third-party provenance

Project-owned source, documentation and results are offered under
GPL-3.0-or-later. Third-party material retains the attribution and terms below.
No external runtime dependency is vendored except the identified adapted
Landau algorithms and the Senn2013 data export.

## Landau approximation algorithms

Files: `heavily_right/_landau.py` and `heavilyrightR/R/landau.R`.

The coefficient tables and piecewise algorithms are adapted from CERN ROOT's
MathCore Landau routines, which trace the density algorithm to CERNLIB G110
DENLAN. Upstream MathCore attribution is **Copyright (c) 2005, LCG ROOT MathLib
Team**; the density source identifies Andras Zsenei and Lorenzo Moneta.

- [Density source](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html)
- [CDF source](https://root.cern.ch/doc/master/ProbFuncMathCore_8cxx_source.html)
- [Quantile source](https://root.cern.ch/doc/master/QuantFuncMathCore_8cxx_source.html)
- [ROOT license](https://root.cern.ch/about/license/): LGPL-2.1-or-later;
  the text is included in [licenses/LGPL-2.1.txt](licenses/LGPL-2.1.txt).

The translations adapt these algorithms to NumPy or native R, package-specific
parameterization and scalar/vector handling; they are not unmodified ROOT
distributions. The native R quantile uses numerical inversion of its CDF.
The source revision originally used for the coefficient extraction was not
recorded, so an exact upstream commit is not asserted. The published upstream
algorithm attribution and license are supplied, but the precise extraction
revision remains a provenance limitation requiring maintainer review.

ROOT's LGPL section 3 permits distributing a modified copy under the ordinary
GPL version 2 or a later version. This combined distribution uses GPL-3.0-or-later
while retaining upstream copyright attribution and the LGPL text. No ROOT
runtime or optional MathMore/GSL component is bundled.

## Senn2013 network meta-analysis data

File: `reproduction/network_meta_analysis/data/senn2013.csv`.

Source: the `Senn2013` dataset distributed with **netmeta 3.6-1**, 28 rows and
seven fields, exported without changing the observations. netmeta declares
`GPL (>= 2) | file LICENSE`; its additional LICENSE notice concerns MIT-licensed
hasseDiagram code, which is not included here. The GPLv2 text is provided in
[licenses/GPL-2.txt](licenses/GPL-2.txt); this distribution uses the permitted
GPLv3-or-later option. Upstream source:
[netmeta](https://CRAN.R-project.org/package=netmeta).

Dataset citation: Senn S, Gavini F, Magrez D, Scheen A (2013). Issues in
performing a network meta-analysis. *Statistical Methods in Medical Research*,
22, 169-189. [DOI](https://doi.org/10.1177/0962280211432220).

See the [data guide](reproduction/network_meta_analysis/data/README.md) for
field semantics, multi-arm handling, checksum and validation fixture.
The study data are comparison-level summary statistics, not individual records.

## Dependencies and external tools

Python depends on NumPy and SciPy; reproduction additionally uses Matplotlib,
pandas and seaborn. R uses nloptr plus standard R packages, with optional
jsonlite, knitr, rmarkdown and testthat. They are installed separately and
remain under their respective upstream licenses.

GeoGebra and Wolfram software are not bundled. The repository includes project
construction/script files and scientific PDF exports; running the external
tools requires a separate installation under their own terms.

Maintainer: Tianle Liu, tylerliuthu@gmail.com.
