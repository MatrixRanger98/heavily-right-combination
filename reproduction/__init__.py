"""Reproduction orchestration for the paper's numerical results.

Reusable statistical code belongs in :mod:`src`.  This package contains only
experiment metadata, execution helpers, and artifact-management utilities.
"""

from reproduction.artifacts import ArtifactStore
from reproduction.manifest import EXPERIMENTS, Experiment, get_experiment

__all__ = ["EXPERIMENTS", "ArtifactStore", "Experiment", "get_experiment"]
