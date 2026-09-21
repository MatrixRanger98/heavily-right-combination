"""Historical, manually adjusted NMA widths; never a new simulation estimate.

These fixed arrays reproduce the benchmark drawn in manuscript Figure 10.
They are specific to the Senn2013 simulation with n=100 and the original rho
grid. The empirical oracle remains available for configurations outside it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

WidthPolicy = Literal["historical-manual", "empirical"]
HISTORICAL_SOURCE = "reproduction/network_meta_analysis/manual_benchmark.py"
HISTORICAL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HISTORICAL_WIDTHS = (
  (0.30123266, 0.34405365, 0.39331598, 0.42974317, 0.47478013,
   0.518090934, 0.554759427, 0.59404881, 0.62516734, 0.64964069),
  (0.39257832, 0.44821173, 0.51174065, 0.55973089, 0.61830234,
   0.674804275, 0.722225097, 0.77327019, 0.81336679, 0.84543146),
)


def select_wls_ma_widths(
  results: dict[str, np.ndarray], *, profile: str,
  policy: WidthPolicy = "historical-manual",
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, object]]:
  """Select plotting widths without overwriting saved empirical-oracle arrays.

Only the complete fixed paper-profile grid has a matching historical table.
Other profiles/grids use the saved empirical oracle, with an explicit reason.
Historical values are not interpolated/extrapolated to other configurations.
"""
  if policy not in {"historical-manual", "empirical"}:
    raise ValueError(f"unsupported WLS-MA width policy: {policy!r}")
  if profile not in {"paper", "smoke"}:
    raise ValueError(f"unsupported NMA profile: {profile!r}")
  rhos = np.asarray(results["rhos"], dtype=float)
  if rhos.ndim != 1 or not rhos.size or not np.isfinite(rhos).all():
    raise ValueError("NMA plot correlations must be a finite nonempty vector")
  grid = np.arange(0, 1, .1)
  compatible = profile == "paper" and rhos.shape == grid.shape and np.allclose(
    rhos, grid, rtol=0, atol=1e-12,
  )
  historical = policy == "historical-manual" and compatible
  if historical:
    widths = tuple(np.array(row) for row in HISTORICAL_WIDTHS)
    reason = None
  else:
    widths = tuple(np.array(results[f"width_wls_ma_{index}"], dtype=float, copy=True)
                   for index in (1, 2))
    reason = (
      "Historical benchmark is available only for the fixed n=100 paper-profile "
      "Senn2013 experiment on the complete rho=0,0.1,...,0.9 grid."
      if policy == "historical-manual" else None
    )
  if any(row.shape != rhos.shape or not np.isfinite(row).all() or np.any(row < 0)
         for row in widths):
    raise ValueError("WLS-MA plotting widths must be aligned, finite and nonnegative")
  return (widths[0], widths[1]), {
    "schema_version": 1, "requested_width_policy": policy,
    "selected_width_policy": "historical-manual" if historical else "empirical",
    "fallback_reason": reason,
    "historical_source": HISTORICAL_SOURCE if historical else None,
    "historical_source_sha256": HISTORICAL_SOURCE_SHA256 if historical else None,
    "historical_arrays": ["z1", "z2"] if historical else None,
    "width_qualification": (
      "Stored historical manual-adjustment benchmark; not estimated from these "
      "saved replications. Original manuscript describes 500 replications; exact "
      "historical sample/calibration provenance is not independently recovered."
      if historical else
      "In-sample empirical oracle from the saved replicates; not held-out coverage."
    ),
    "coverage_policy": "unchanged saved empirical oracle, not historical coverage",
    "raw_empirical_summary_preserved": True,
    "replicates_per_rho": np.asarray(results["replicates_per_rho"]).tolist(),
    "rhos": rhos.tolist(),
  }
