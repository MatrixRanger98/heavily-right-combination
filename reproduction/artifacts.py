"""Safe, working-directory-independent paths for reproduction artifacts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKOUT = (CODE_ROOT / "pyproject.toml").is_file()
DEFAULT_RESULTS_ROOT = (
  CODE_ROOT / "results" / "runs"
  if SOURCE_CHECKOUT
  else Path.cwd() / "heavily-right-results" / "runs"
)

_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_relative_path(value: str, *, label: str) -> Path:
  path = Path(value)
  if path.is_absolute() or not path.parts:
    raise ValueError(f"{label} must be a nonempty relative path")
  if any(part in {".", ".."} or not _COMPONENT.fullmatch(part) for part in path.parts):
    raise ValueError(
      f"{label} may contain lowercase letters, digits, '.', '_' and '-' only: {value!r}"
    )
  return path


def _new_run_id() -> str:
  return datetime.now(UTC).strftime("%Y%m%dt%H%M%S.%fz")


def _validate_run_id(value: str) -> str:
  path = _validate_relative_path(value, label="run_id")
  if len(path.parts) != 1:
    raise ValueError("run_id must not contain subdirectories")
  return path.as_posix()


def _validate_results_root(root: Path) -> Path:
  resolved = root.resolve()
  # Discover a source project from the output path as well as this installation;
  # an installed wheel must enforce the same protection as a source checkout.
  candidates = {CODE_ROOT.parent, *resolved.parents, resolved}
  for project_root in candidates:
    code_root = project_root / "Code"
    if not (code_root / "reproduction" / "manifest.py").is_file():
      continue
    protected_roots = (project_root / "Paper", project_root / "Archived")
    if any(resolved == path or resolved.is_relative_to(path) for path in protected_roots):
      raise ValueError(f"results_root must not be inside a protected project tree: {resolved}")
    legacy_roots = tuple(code_root / name for name in (
      "data", "fig", "tmp", "legacy", "results/historical", "results/legacy",
    ))
    if any(resolved == path or resolved.is_relative_to(path) for path in legacy_roots):
      raise ValueError(f"results_root must not be inside a legacy output tree: {resolved}")
  return resolved


@dataclass(frozen=True)
class ArtifactStore:
  """Paths for one immutable experiment run.

  Construct one store per script and reuse it for every saved artifact.  Path
  methods create only their containing directory and reject an existing target,
  preventing an accidental overwrite when a run ID is reused.
  """

  experiment_id: str
  run_id: str
  results_root: Path

  @classmethod
  def for_experiment(cls, experiment_id: str) -> ArtifactStore:
    configured_id = os.environ.get("HCCT_EXPERIMENT_ID")
    if configured_id is not None and configured_id != experiment_id:
      raise ValueError(
        f"runner selected {configured_id!r}, but script declared {experiment_id!r}"
      )
    validated_id = _validate_relative_path(experiment_id, label="experiment_id")
    run_id = os.environ.get("HCCT_RUN_ID", _new_run_id())
    validated_run_id = _validate_run_id(run_id)
    root = _validate_results_root(Path(os.environ.get("HCCT_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)))
    return cls(
      experiment_id=validated_id.as_posix(),
      run_id=validated_run_id,
      results_root=root,
    )

  @property
  def run_dir(self) -> Path:
    return self.results_root / self.experiment_id / self.run_id

  def data(self, filename: str) -> Path:
    """Reserve a data-product path for this run."""
    return self._reserve("data", filename)

  def figure(self, filename: str) -> Path:
    """Reserve a vector-PDF figure path for this run."""
    if Path(filename).suffix.lower() != ".pdf":
      raise ValueError("reproduction figures must use the .pdf extension")
    return self._reserve("figures", filename)

  def diagnostic(self, filename: str) -> Path:
    """Reserve a path for debugging or failed-replication state."""
    return self._reserve("diagnostics", filename)

  def _reserve(self, kind: str, filename: str) -> Path:
    relative = _validate_relative_path(filename, label="filename")
    if len(relative.parts) != 1:
      raise ValueError("filename must not contain subdirectories")
    target = self.run_dir / kind / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
      raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    return target
