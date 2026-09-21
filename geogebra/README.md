# GeoGebra illustration sources

This directory contains only the eight constructions mapped to current paper
illustrations. Every source has a descriptive name, and its future reproduction
PDF uses the same filename stem.

All eight retained PDFs are included in the
[paper-figure map](../results/README.md#geogebra-figures-and-sources), with paper
numbers, LaTeX labels, PDF links, and links to these `.ggb` sources. They remain
historical exports until the external export workflow is rerun. The 2026-09-21
results cleanup retained every mapped GeoGebra source and PDF.

## File format

A `.ggb` file is a ZIP archive. Its `geogebra.xml` member records the
construction, view bounds, visibility, and styles as readable text. For audit
purposes, inspect it without unpacking the source:

```bash
unzip -p Code/geogebra/convexity-half-cauchy-student-t.ggb geogebra.xml
```

The archive itself is binary, and hand-editing the XML is not recommended for
normal figure work: object labels and dependencies make uncontrolled changes
fragile. Use GeoGebra Classic for substantive edits, then save a new `.ggb`.

## Source, output, and paper mapping

| Reproduction ID | GeoGebra source and run PDF | Current paper PDF |
|---|---|---|
| `illustrations/convexity` | `convexity-half-cauchy-student-t.{ggb,pdf}` | `convexity_hc_t.pdf` |
| `illustrations/convexity` | `convexity-cauchy-student-t.{ggb,pdf}` | `convexity_c_t.pdf` |
| `illustrations/convexity` | `convexity-fisher-normal.{ggb,pdf}` | `convexity_f_n.pdf` |
| `illustrations/convexity` | `convexity-fisher-student-t.{ggb,pdf}` | `convexity_f_t.pdf` |
| `illustrations/distribution-densities` | `density-cauchy-score-transform.{ggb,pdf}` | `cauchy_density.pdf` |
| `illustrations/distribution-densities` | `density-half-cauchy-score-transform.{ggb,pdf}` | `hcauchy_density.pdf` |
| `illustrations/simultaneous-projection` | `simultaneous-interval-projection.{ggb,pdf}` | `simultaneous-1.pdf` |
| `illustrations/contour-integration` | `contour-integration.{ggb,pdf}` | `contour2.pdf` |

## Command-line export with GeoGebra Classic 5

GeoGebra's command-line manual documents `--export=<file>` for the Classic 5
offline installer. The output extension selects PDF, and `--dpi` sets the
resolution for formats that use it. On macOS, set `GEOGEBRA5` to the Classic 5
launcher or to a wrapper around its bundled Java command. Then run each source
from the repository root:

```bash
GEOGEBRA5=/path/to/geogebra-classic-5
RUN_ID=geogebra-paper-20260823

mkdir -p Code/results/runs/illustrations/convexity/$RUN_ID/figures
"$GEOGEBRA5" --export=Code/results/runs/illustrations/convexity/$RUN_ID/figures/convexity-half-cauchy-student-t.pdf --dpi=300 Code/geogebra/convexity-half-cauchy-student-t.ggb
"$GEOGEBRA5" --export=Code/results/runs/illustrations/convexity/$RUN_ID/figures/convexity-cauchy-student-t.pdf --dpi=300 Code/geogebra/convexity-cauchy-student-t.ggb
"$GEOGEBRA5" --export=Code/results/runs/illustrations/convexity/$RUN_ID/figures/convexity-fisher-normal.pdf --dpi=300 Code/geogebra/convexity-fisher-normal.ggb
"$GEOGEBRA5" --export=Code/results/runs/illustrations/convexity/$RUN_ID/figures/convexity-fisher-student-t.pdf --dpi=300 Code/geogebra/convexity-fisher-student-t.ggb

mkdir -p Code/results/runs/illustrations/distribution-densities/$RUN_ID/figures
"$GEOGEBRA5" --export=Code/results/runs/illustrations/distribution-densities/$RUN_ID/figures/density-cauchy-score-transform.pdf --dpi=300 Code/geogebra/density-cauchy-score-transform.ggb
"$GEOGEBRA5" --export=Code/results/runs/illustrations/distribution-densities/$RUN_ID/figures/density-half-cauchy-score-transform.pdf --dpi=300 Code/geogebra/density-half-cauchy-score-transform.ggb

mkdir -p Code/results/runs/illustrations/simultaneous-projection/$RUN_ID/figures
"$GEOGEBRA5" --export=Code/results/runs/illustrations/simultaneous-projection/$RUN_ID/figures/simultaneous-interval-projection.pdf --dpi=300 Code/geogebra/simultaneous-interval-projection.ggb

mkdir -p Code/results/runs/illustrations/contour-integration/$RUN_ID/figures
"$GEOGEBRA5" --export=Code/results/runs/illustrations/contour-integration/$RUN_ID/figures/contour-integration.pdf --dpi=300 Code/geogebra/contour-integration.ggb
```

The current machine has GeoGebra Classic 6.0.927, not Classic 5. A direct
non-GUI probe confirms that Classic 6 rejects `--export` as an unrecognized
option; its help lists only help, version, logging, and GPU options. The Classic
5 commands above were therefore documented but not executed here.

## Manual Classic 6 fallback

1. Pick a lowercase run ID and create
   `Code/results/runs/<reproduction-id>/<run-id>/figures/`.
2. Validate the sources before opening them:

   ```bash
   cd Code/geogebra
   shasum -a 256 -c SOURCES.sha256
   ```

3. Open one mapped `.ggb` in GeoGebra Classic 6. Do not alter its visible
   objects, axes, labels, colors, line widths, or saved Graphics View bounds.
4. From GeoGebra's main menu, download/export the Graphics View as PDF. In menu
   layouts that first open an export dialog, choose PDF rather than PNG. The
   retained project artifact must be a vector PDF.
5. Save it under the corresponding experiment run directory using exactly the
   source stem shown in the mapping table. Refuse to overwrite an existing run.
6. Record the GeoGebra version, source SHA-256, export date, and output path in
   a `run.json` beside the run's `figures/` directory.
7. Visually compare the exported PDF with the current paper PDF and check for
   embedded raster images before approving it. Do not export into `Paper/`.

## Removed unmapped constructions

The following sources were removed from the active tree on 2026-08-23 after
thumbnail, construction-XML, and project-reference audits. None generated a
current paper artifact, and each mapped source above is self-contained.

| Removed source | SHA-256 | Reason |
|---|---|---|
| `convexity.ggb` | `dfeed43ca6948bf04b800b553cad12ff1db9cd558dcc63f17221b6572415b6b2` | Early single-curve convexity exploration. |
| `convexity_full.ggb` | `280d364e636e1044370e71d2ac5094c16aa1411040889ac06f913b155a5a9504` | Redundant combined working view; the four final panels are independent sources. |
| `contour.ggb` | `95f34b9bc67235a370d6463120cac13c58bed66572b3532bcb6dff441dc0ed96` | Superseded contour with a different geometry. |
| `func.ggb` | `62a64f439262a0e2d7b03170a406cbe7f6444507adc108329b58e1486e6123bf` | Unrelated radius/angle construction. |
| `simultaneous-0.ggb` | `c3e11de03e38ef6950461fccb81a4a96339832722399a884e2e11defac8f7759` | Earlier projection layout, superseded by the mapped final construction. |
