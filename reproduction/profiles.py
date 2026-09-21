"""Shared paper and smoke settings for reproduction scripts."""

from __future__ import annotations

import os
from typing import TypeVar

T = TypeVar("T")

PAPER_PROFILE = "paper"
SMOKE_PROFILE = "smoke"
PROFILES = (PAPER_PROFILE, SMOKE_PROFILE)


def current_profile() -> str:
  """Return the runner-selected workload profile."""
  profile = os.environ.get("HCCT_PROFILE", PAPER_PROFILE)
  if profile not in PROFILES:
    choices = ", ".join(PROFILES)
    raise ValueError(f"unknown HCCT_PROFILE {profile!r}; choose one of: {choices}")
  return profile


def profile_value(*, paper: T, smoke: T) -> T:
  """Select a value without changing the paper-reproduction default."""
  return paper if current_profile() == PAPER_PROFILE else smoke
