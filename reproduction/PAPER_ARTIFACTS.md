# Paper figures and tables

The authoritative public mapping is the [results index](../results/README.md).
It combines paper tables and additional numerical tables, and includes every
distributed PDF, including outputs unused by the paper. Its
[machine-readable map](../results/figure-map.json) records the same associations.

Figures are ordered by appearance in the main paper, followed by the supplement;
unused figures follow in creation-time order. Multiple included versions of a
panel remain grouped. A figure association identifies an experiment and panel;
it does not by itself assert numerical or visual identity with the manuscript.

## Selection and interpretation

Main Figures **10 and 14 are fixed manuscript selections**. Their PDFs are
included as reference outputs, and their checksums are recorded in the
[selection policy](../results/figure-selection-policy.json). All included
reproduction counterparts for those figures are comparison-only.

Other generated figures are linked to their current preferred results. The
eight GeoGebra figures include editable sources and vector PDF exports; these
exports are not claimed to be new Python executions. See the
[GeoGebra guide](../geogebra/README.md).

For numerical tables, use full-precision CSV values rather than extracting
rounded numbers from a PDF. Precision/error and runtime columns depend on the
numerical procedure and environment. Analytical tables have no numerical
generator; their output column is NA. Numerical CAtr columns are omitted from
this release's reproduction scope.

The [experiment catalog](EXPERIMENTS.md) gives commands and output filenames.
The [saved-data guide](SAVED_DATA.md) explains repeat counts, seeds, uncertainty,
and the distinction between raw summaries and NMA's manual-width plot benchmark.
