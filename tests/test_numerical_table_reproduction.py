"""Bounded checks for the precision tables and all non-CAtr p-value columns.

Full-grid/export tests mock finite-law evaluations; only the small smoke grid
and six short p-value examples are evaluated numerically. No retained run is
created or modified. The Landau oracle independently integrates the paper's
characteristic function instead of calling the production approximation.
"""

from __future__ import annotations

import csv
import io
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from heavily_right.combination import CombinationTest
from heavily_right.null_laws import Landau, euler_gamma
from reproduction.artifacts import ArtifactStore
from reproduction.numerical import distribution_precision as precision
from reproduction.numerical import pvalue_examples


def _mock_precision_rows(profile: str = "paper") -> list[precision.PrecisionRow]:
  with ExitStack() as stack:
    stack.enter_context(patch.dict(os.environ, {"HCCT_PROFILE": profile}))
    stack.enter_context(redirect_stdout(io.StringIO()))
    for table in precision.TABLES:
      for quantity in ("pdf", "cdf"):
        stack.enter_context(patch.object(table.distribution, quantity,
                                         return_value=(0.25, 2e-10)))
    stack.enter_context(patch.object(precision, "perf_counter",
                                     side_effect=(index / 1000 for index in range(120))))
    return precision.evaluate()


def _landau_fourier(
  x: float, location: float, scale: float,
) -> tuple[float, float, float, float]:
  def phase(t: float) -> float:
    return ((x - location) * t / scale + 2 / math.pi * t * math.log(t / scale)
            if t else 0.0)

  pdf, pdf_error = quad(
    lambda t: math.exp(-t) * math.cos(phase(t)) / (scale * math.pi),
    0, 50, epsabs=1e-12, epsrel=1e-12, limit=300,
  )
  cdf_integral, cdf_error = quad(
    lambda t: math.exp(-t) * math.sin(phase(t)) / (math.pi * t) if t else 0.0,
    0, 50, epsabs=1e-12, epsrel=1e-12, limit=300,
  )
  return pdf, 0.5 + cdf_integral, pdf_error, cdf_error


class NumericalTableReproductionTests(unittest.TestCase):
  def test_locations_follow_paper_and_canonical_asymptotic_calibration(self) -> None:
    for m in (2, 10, 100, 1000):
      entropy_location = math.log(m) + 1 - euler_gamma
      for distribution, expected_location, expected_scale in (
        ("half-cauchy-mean", 2 / math.pi * entropy_location, 1.0),
        ("harmonic-mean", entropy_location, math.pi / 2),
      ):
        with self.subTest(distribution=distribution, m=m):
          location, scale = precision.landau_parameters(m, distribution)
          self.assertAlmostEqual(location, expected_location, places=14)
          self.assertEqual(scale, expected_scale)
    # m>1000 selects the existing canonical approximation without finite-m roots.
    for distribution, method in (("half-cauchy-mean", "HCauchy"), ("harmonic-mean", "EHMP")):
      location, scale = precision.landau_parameters(1001, distribution)
      expected = float(Landau.ppf(0.95, location, scale))
      self.assertAlmostEqual(CombinationTest(1001, method=method).threshold, expected, places=12)

  def test_landau_parameter_input_checks(self) -> None:
    for m in (0, -1, True, 1.5):
      with self.subTest(m=m), self.assertRaises(ValueError):
        precision.landau_parameters(m, "half-cauchy-mean")
    with self.assertRaises(ValueError):
      precision.landau_parameters(2, "unknown")

  def test_corrected_landau_values_match_printed_manuscript_rounding(self) -> None:
    references = (
      ("half-cauchy-mean", 2, 0.2, 0.282722127, 0.223733981),
      ("half-cauchy-mean", 10, 1.0, 0.267219180, 0.161603641),
      ("harmonic-mean", 100, 2.0, 0.000554016, 0.000068807),
      ("harmonic-mean", 1000, 4.0, 0.000043914, 0.000004086),
    )
    for distribution, m, x, pdf, cdf in references:
      location, scale = precision.landau_parameters(m, distribution)
      with self.subTest(distribution=distribution, m=m, x=x):
        self.assertAlmostEqual(float(Landau.pdf(x, location, scale)), pdf, delta=5e-10)
        self.assertAlmostEqual(float(Landau.cdf(x, location, scale)), cdf, delta=5e-10)

  def test_landau_parameterization_matches_independent_fourier_integrals(self) -> None:
    cases = (("half-cauchy-mean", 2, 0.2), ("half-cauchy-mean", 10, 1.0),
             ("harmonic-mean", 100, 2.0), ("harmonic-mean", 1000, 4.0))
    for distribution, m, x in cases:
      location, scale = precision.landau_parameters(m, distribution)
      pdf, cdf, pdf_error, cdf_error = _landau_fourier(x, location, scale)
      with self.subTest(distribution=distribution, m=m, x=x):
        self.assertLess(pdf_error, 1e-10)
        self.assertLess(cdf_error, 1e-10)
        # Allow the existing piecewise Landau approximation's accuracy;
        # its CDF is not the high-precision quadrature used as this oracle.
        self.assertAlmostEqual(float(Landau.pdf(x, location, scale)), pdf, delta=1e-9)
        self.assertAlmostEqual(float(Landau.cdf(x, location, scale)), cdf, delta=3e-8)

  def test_full_grid_and_wide_tables_match_paper_points(self) -> None:
    rows = _mock_precision_rows()
    self.assertEqual(len(rows), 60)
    self.assertEqual(len(precision.POINTS), 16)
    self.assertEqual(len(precision.EHMP_POINTS), 14)
    self.assertNotIn((2, 0.2), precision.EHMP_POINTS)
    self.assertNotIn((10, 1), precision.EHMP_POINTS)
    for table in precision.TABLES:
      wide = precision.table_rows(rows, table.name)
      self.assertEqual([(r["sample_size"], r["x"]) for r in wide], list(table.points))
      for row in wide:
        with self.subTest(distribution=table.name, m=row["sample_size"], x=row["x"]):
          self.assertEqual(row["paper_label"], table.label)
          for quantity in ("pdf", "cdf"):
            self.assertAlmostEqual(row[f"{quantity}_elapsed_s"], 0.001, places=14)
            self.assertEqual(row[f"{quantity}_error_estimate"], 2e-10)
            self.assertEqual(row[f"{quantity}_landau_minus_exact"],
                             row[f"{quantity}_landau"] - row[quantity])

  def test_table_assembly_rejects_missing_duplicate_or_inconsistent_measurements(self) -> None:
    rows = _mock_precision_rows("smoke")
    for invalid in (rows[:1], rows[:2] + rows[:1],
                    [rows[0], {**rows[1], "landau_scale": 123.0}],
                    [{**rows[0], "quantity": "sf"}, rows[1]]):
      with self.subTest(rows=invalid), self.assertRaises(ValueError):
        precision.table_rows(invalid, "half-cauchy-mean")
    with self.assertRaises(ValueError):
      precision.table_rows([], "half-cauchy-mean")

  def test_smoke_measurements_use_current_finite_laws_and_finite_diagnostics(self) -> None:
    with patch.dict(os.environ, {"HCCT_PROFILE": "smoke"}), redirect_stdout(io.StringIO()):
      rows = precision.evaluate()
    self.assertEqual(len(rows), 8)
    first = rows[0]
    self.assertEqual((first["distribution"], first["quantity"], first["sample_size"], first["x"]),
                     ("half-cauchy-mean", "pdf", 2, 0.2))
    self.assertAlmostEqual(first["exact"], 0.2928791657851436, delta=2e-9)
    for row in rows:
      with self.subTest(row=row):
        self.assertTrue(all(math.isfinite(row[key]) for key in (
          "exact", "error_estimate", "elapsed_ms", "landau_location", "landau_scale",
          "landau_approximation", "landau_minus_exact")))
        self.assertGreaterEqual(row["error_estimate"], 0)
        self.assertGreaterEqual(row["elapsed_ms"], 0)

  def test_precision_exports_all_three_named_csvs_without_overwriting(self) -> None:
    rows = _mock_precision_rows("smoke")
    with tempfile.TemporaryDirectory() as directory:
      store = ArtifactStore(precision.EXPERIMENT_ID, "test", Path(directory))
      with patch.object(precision.ArtifactStore, "for_experiment", return_value=store), \
           patch.object(precision, "evaluate", return_value=rows), redirect_stdout(io.StringIO()):
        precision.main()
        for name, expected_rows in ((precision.OUTPUT, 8), (precision.HCCT_OUTPUT, 2),
                                    (precision.EHMP_OUTPUT, 2)):
          with (store.run_dir / "data" / name).open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), expected_rows)
        with self.assertRaises(FileExistsError):
          precision.main()

  def test_pvalue_examples_emit_every_non_catr_column_and_preserve_rounding(self) -> None:
    expected = (
      (0.021, 0.104, 0.060, 0.051, 0.039, 0.039),
      (0.021, 0.139, 0.060, 0.088, 0.039, 0.039),
      (0.021, 0.177, 0.060, 0.837, 0.039, 0.039),
      (0.192, 0.691, 0.045, 0.091, 0.050, 0.049),
      (0.040, 0.272, 0.080, 0.086, 0.045, 0.045),
      (0.040, 0.166, 0.050, 0.197, 0.046, 0.046),
    )
    with redirect_stdout(io.StringIO()):
      rows = pvalue_examples.evaluate()
    self.assertEqual(len(rows), 6)
    self.assertEqual(pvalue_examples.METHODS,
                     ("Fisher", "Stouffer", "Bonferroni", "Cauchy", "HCauchy", "EHMP"))
    for row, pvalues, reference in zip(rows, pvalue_examples.EXAMPLES, expected, strict=True):
      self.assertNotIn("catr", row)
      self.assertEqual(row["bonferroni"], min(1.0, len(pvalues) * min(pvalues)))
      for method, value in zip(pvalue_examples.METHODS, reference, strict=True):
        with self.subTest(pvalues=pvalues, method=method):
          self.assertAlmostEqual(row[method.lower()], value, delta=0.0005)
          self.assertTrue(np.isfinite(row[method.lower()]))

  def test_pvalue_csv_writes_bonferroni_without_a_catr_column(self) -> None:
    row = {"p_values": "0.02;0.03;0.96", "num_tests": 3, "bonferroni": 0.06}
    with tempfile.TemporaryDirectory() as directory:
      store = ArtifactStore(pvalue_examples.EXPERIMENT_ID, "test", Path(directory))
      with patch.object(pvalue_examples.ArtifactStore, "for_experiment", return_value=store), \
           patch.object(pvalue_examples, "evaluate", return_value=[row]), \
           redirect_stdout(io.StringIO()):
        pvalue_examples.main()
      with (store.run_dir / "data" / pvalue_examples.OUTPUT).open() as handle:
        saved = list(csv.DictReader(handle))
      self.assertEqual(saved[0]["bonferroni"], "0.06")
      self.assertNotIn("catr", saved[0])


if __name__ == "__main__":
  unittest.main()
