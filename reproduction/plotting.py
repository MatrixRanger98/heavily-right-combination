"""Plotting helpers shared by paper-reproduction scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from reproduction.artifacts import ArtifactStore


def save_pdf(
  figure: Figure,
  artifacts: ArtifactStore,
  filename: str,
  *,
  tight: bool = False,
) -> Path:
  """Save and close one vector figure, returning its immutable path."""
  path = artifacts.figure(filename)
  options = {"bbox_inches": "tight"} if tight else {}
  figure.savefig(path, format="pdf", **options)
  plt.close(figure)
  print(f"[artifact] {path}", flush=True)
  return path
