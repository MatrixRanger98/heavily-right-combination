"""Package boundaries and installation metadata."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import heavily_right

CODE_ROOT = Path(__file__).resolve().parents[1]


class PackageAPITests(unittest.TestCase):
  def test_version_and_runtime_dependencies_match_metadata(self) -> None:
    metadata = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text())
    self.assertEqual(metadata["project"]["version"], heavily_right.__version__)
    self.assertEqual(metadata["project"]["dependencies"], ["numpy>=1.26", "scipy>=1.11"])
    self.assertTrue((CODE_ROOT / "heavily_right/py.typed").is_file())

  def test_only_canonical_namespaces_are_distributed(self) -> None:
    metadata = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text())
    discovery = metadata["tool"]["setuptools"]["packages"]["find"]
    self.assertEqual(discovery["include"], ["heavily_right*", "reproduction*"])
    self.assertIn("src*", discovery["exclude"])

  def test_numerical_api_import_does_not_load_optional_dependencies(self) -> None:
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'pandas', 'seaborn'}:
            raise ImportError('optional dependency unexpectedly loaded: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import heavily_right
assert heavily_right.CombinationTest(1).get_global_p(0.2) == 0.2
"""
    environment = {**os.environ, "PYTHONPATH": str(CODE_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False)
    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
  unittest.main()
