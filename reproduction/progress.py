"""Small, dependency-free progress reporting for long experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Progress:
  """Report completed work units and elapsed wall time."""

  label: str
  total: int
  _started: float = field(default_factory=perf_counter, init=False)

  def update(self, completed: int) -> None:
    if not 0 <= completed <= self.total:
      raise ValueError("completed work must be between zero and total")
    elapsed = perf_counter() - self._started
    percent = 100 * completed / self.total if self.total else 100
    print(
      f"[progress] {self.label}: {completed}/{self.total} "
      f"({percent:.1f}%) in {elapsed:.1f}s",
      flush=True,
    )
