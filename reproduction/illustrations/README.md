# Mathematical illustrations

Run the Python IDs using the [common command](../README.md). Each generated
figure is a vector PDF under the run's `figures/` directory.

| Experiment | Module | Outputs |
|---|---|---|
| `illustrations/hcct-cct-geometry` | [hcct_cct_geometry.py](hcct_cct_geometry.py) | Four `hcct-cct-geometry-{hcct-region,cct-region,cct-horn-magnified,cct-infinite-area-tubes}.pdf` panels and a configuration JSON |
| `illustrations/connectivity-scores` | [connectivity_scores.py](connectivity_scores.py) | `connectivity-scores.pdf` |
| `illustrations/connectivity-regions` | [connectivity_regions.py](connectivity_regions.py) | `connectivity-regions.pdf` |
| `illustrations/two-pvalue-regions` | [pvalue_regions.py](pvalue_regions.py) | Six `pvalue-regions-{fisher,stouffer,bonferroni,cauchy,hcauchy,harmonic}.pdf` panels |

The blockwise geometry uses the rounded HCCT cutoff **13.68751** explicitly;
both profiles produce the same deterministic panels. Its saved configuration
records the mathematical settings. The connectivity experiments contrast CCT
and HCCT score curves/regions. A separate
[Wolfram implementation](../../wolfram/README.md) provides a cross-language
check of the one-dimensional score illustration.

Four external illustration IDs use the eight editable GeoGebra constructions:

| Experiment | Source stems |
|---|---|
| `illustrations/convexity` | `convexity-{half-cauchy-student-t,cauchy-student-t,fisher-normal,fisher-student-t}` |
| `illustrations/distribution-densities` | `density-{cauchy-score-transform,half-cauchy-score-transform}` |
| `illustrations/simultaneous-projection` | `simultaneous-interval-projection` |
| `illustrations/contour-integration` | `contour-integration` |

Each source stem is shared by its `.ggb` and corresponding `.pdf`. The Python
runner lists these IDs but does not execute them. Follow the
[GeoGebra command-line export guide](../../geogebra/README.md); Classic 5 export
options are not supported by every Classic 6 build.
