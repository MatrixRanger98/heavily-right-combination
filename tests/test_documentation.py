"""Execute installed-help examples and independent Markdown examples."""

from __future__ import annotations

import contextlib
import doctest
import importlib
import io
import re
import tempfile
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DOCS = CODE_ROOT / "docs"
CHECKOUT_ONLY_README_LINKS = (
  "data/", "geogebra/", "wolfram/", "results/", "legacy/",
  "heavilyrightR/", "../Note/",
)
HELP_MODULES = (
  "calibration", "combination", "_finite_null_laws", "univariate", "multivariate",
  "empty_sets", "workflows", "network", "dac",
)


class DocumentationTests(unittest.TestCase):
  def test_installed_help_examples(self) -> None:
    finder = doctest.DocTestFinder()
    for name in HELP_MODULES:
      module = importlib.import_module(f"heavily_right.{name}")
      with self.subTest(module=name):
        examples = [example for example in finder.find(module) if example.examples]
        self.assertTrue(examples, f"No executable help examples in {name}")
        failures = io.StringIO()
        runner = doctest.DocTestRunner(verbose=False)
        with tempfile.TemporaryDirectory(prefix="heavily-right-help-") as directory:
          with contextlib.chdir(directory):
            for example in examples:
              runner.run(example, out=failures.write)
        self.assertEqual(runner.failures, 0, failures.getvalue())

  def test_reference_and_guide_examples_are_independent(self) -> None:
    pages = [*sorted((DOCS / "reference").glob("*.md")),
             DOCS / "METHODS.md", DOCS / "TROUBLESHOOTING.md",
             DOCS / "NULL_LAW_IMPLEMENTATION.md", DOCS / "OPTIMIZATION_IMPLEMENTATION.md",
             DOCS / "README.md", DOCS / "RECENT_FIXES.md"]
    count = 0
    for path in pages:
      blocks = re.findall(r"^```python[ \t]*\n(.*?)^```[ \t]*$", path.read_text(), re.M | re.S)
      for index, source in enumerate(blocks, 1):
        count += 1
        with self.subTest(page=path.name, example=index):
          output = io.StringIO()
          with tempfile.TemporaryDirectory(prefix="heavily-right-doc-") as directory:
            with contextlib.chdir(directory), contextlib.redirect_stdout(output):
              exec(compile(source, f"{path}:example-{index}", "exec"),
                   {"__name__": "__documentation_example__"})
            self.assertEqual(list(Path(directory).iterdir()), [],
                             "Reference examples must not write files")
    self.assertGreaterEqual(count, 15)

  def test_new_documentation_local_links_exist(self) -> None:
    pages = [*sorted((DOCS / "reference").glob("*.md")), DOCS / "METHODS.md",
             DOCS / "TROUBLESHOOTING.md", CODE_ROOT / "README.md", DOCS / "API.md",
             DOCS / "NULL_LAW_IMPLEMENTATION.md", DOCS / "OPTIMIZATION_IMPLEMENTATION.md",
             DOCS / "README.md", DOCS / "RECENT_FIXES.md"]
    for path in pages:
      for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith(("#", "mailto:")):
          continue
        with self.subTest(page=path.name, target=target):
          destination = target.split("#", 1)[0]
          # The top-level folder tour also documents non-distributed project
          # artifacts. Check all links in the checkout, but do not require R
          # packages, historical outputs, or audit notes in the Python sdist.
          if (path == CODE_ROOT / "README.md"
              and not (CODE_ROOT / "results").exists()
              and destination.startswith(CHECKOUT_ONLY_README_LINKS)):
            continue
          if (not (CODE_ROOT.parent / "Note").exists()
              and destination.startswith("../../Note/")):
            # Dated project audit records are linked by the implementation
            # guides but intentionally do not ship in the Python sdist.
            continue
          self.assertTrue((path.parent / destination).exists())


if __name__ == "__main__":
  unittest.main()
