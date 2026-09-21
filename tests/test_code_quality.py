"""Lightweight structural checks for the maintained Python tree."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_DIRECTORIES = (CODE_ROOT / "heavily_right", CODE_ROOT / "reproduction")


def _missing_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
  missing: list[str] = []
  arguments = (
    function.args.posonlyargs
    + function.args.args
    + function.args.kwonlyargs
  )
  for argument in arguments:
    if argument.arg not in {"self", "cls"} and argument.annotation is None:
      missing.append(argument.arg)
  if function.args.vararg is not None and function.args.vararg.annotation is None:
    missing.append(f"*{function.args.vararg.arg}")
  if function.args.kwarg is not None and function.args.kwarg.annotation is None:
    missing.append(f"**{function.args.kwarg.arg}")
  if function.returns is None:
    missing.append("return")
  return missing


class MaintainedTreeStructureTests(unittest.TestCase):
  def test_modules_have_docstrings(self) -> None:
    missing: list[str] = []
    for directory in MAINTAINED_DIRECTORIES:
      for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text())
        if ast.get_docstring(tree) is None:
          missing.append(str(path.relative_to(CODE_ROOT)))
    self.assertEqual(missing, [])

  def test_public_callables_are_annotated(self) -> None:
    missing: list[str] = []
    for directory in MAINTAINED_DIRECTORIES:
      for path in sorted(directory.rglob("*.py")):
        tree = ast.parse(path.read_text())
        relative = path.relative_to(CODE_ROOT)
        for node in tree.body:
          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
              continue
            absent = _missing_annotations(node)
            if absent:
              missing.append(f"{relative}:{node.lineno} {node.name}: {absent}")
          elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for item in node.body:
              if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
              if item.name.startswith("_") and item.name != "__init__":
                continue
              absent = _missing_annotations(item)
              if absent:
                missing.append(
                  f"{relative}:{item.lineno} {node.name}.{item.name}: {absent}"
                )
    self.assertEqual(missing, [])


if __name__ == "__main__":
  unittest.main()
