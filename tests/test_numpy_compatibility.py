"""Regression tests for NumPy 2 import compatibility."""

import unittest
from pathlib import Path


class NumPyCompatibilityTest(unittest.TestCase):
  def test_active_sources_do_not_use_removed_numpy_float_alias(self):
    code_root = Path(__file__).resolve().parents[1]
    paths = [path for name in ("heavily_right", "reproduction")
             for path in (code_root / name).rglob("*.py")]
    self.assertTrue(paths, "The source scan must not pass vacuously")
    for path in paths:
      with self.subTest(path=path.relative_to(code_root)):
        self.assertNotIn("np.float_", path.read_text())


if __name__ == "__main__":
  unittest.main()
