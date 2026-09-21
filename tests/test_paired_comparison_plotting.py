"""Coincident paired method curves remain visible without changing their data."""

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from reproduction.one_dimensional import false_positive_rate, power_comparison
from reproduction.one_dimensional.common import build_method_figure


class PairedComparisonPlottingTests(unittest.TestCase):
  def setUp(self):
    self.original_figures = set(plt.get_fignums())
    self.rhos = np.array([0., .3, .6, .9])
    self.values = np.linspace(.01, .9, 9 * 4 * 2).reshape(9, 4, 2)
    # Exercise exact overlap, the case that hid HCauchy before this style change.
    self.values[1] = self.values[0]

  def tearDown(self):
    for number in set(plt.get_fignums()) - self.original_figures:
      plt.close(number)

  def render_figures(self, module):
    with patch.object(module, "save_pdf") as save:
      module.render(self.values, self.rhos, object())
    self.assertEqual(save.call_count, 2)
    return [call.args[0] for call in save.call_args_list]

  def test_hcauchy_underlay_and_ehmp_dashes_are_visible_in_all_panels_and_legends(self):
    for module in (false_positive_rate, power_comparison):
      for figure in self.render_figures(module):
        with self.subTest(module=module.__name__, ylabel=figure.axes[0].get_ylabel()):
          axis = figure.axes[0]
          lines = {line.get_label(): line for line in axis.lines}
          hcauchy, ehmp = lines["HCauchy"], lines["EHMP"]
          self.assertEqual(hcauchy.get_linewidth(), 2.5)
          self.assertEqual(ehmp.get_linewidth(), 1.5)
          self.assertEqual(hcauchy.get_linestyle(), "-")
          self.assertEqual(ehmp.get_linestyle(), "--")
          self.assertNotEqual(hcauchy.get_color(), ehmp.get_color())
          self.assertLess(hcauchy.get_zorder(), ehmp.get_zorder())
          self.assertTrue(all(
            line.get_zorder() < hcauchy.get_zorder()
            for label, line in lines.items() if label not in {"HCauchy", "EHMP"}
          ))
          np.testing.assert_array_equal(hcauchy.get_ydata(), ehmp.get_ydata())
          legend = axis.get_legend()
          handles = dict(zip(
            [label.get_text() for label in legend.get_texts()],
            legend.get_lines(), strict=True,
          ))
          for method in ("HCauchy", "EHMP"):
            self.assertEqual(handles[method].get_linewidth(), lines[method].get_linewidth())
            self.assertEqual(handles[method].get_linestyle(), lines[method].get_linestyle())
            self.assertEqual(handles[method].get_color(), lines[method].get_color())

  def test_style_update_preserves_every_plotted_coordinate_and_input(self):
    values_before, rhos_before = self.values.copy(), self.rhos.copy()
    for module in (false_positive_rate, power_comparison):
      for index, figure in enumerate(self.render_figures(module)):
        options = {"ylabel": "power"}
        if module is false_positive_rate:
          options = {
            "ylabel": "false positive rate", "reference": .05,
            "ylim": ((0, .15), (0, .35))[index],
          }
        reference = build_method_figure(
          self.values, self.rhos, module.METHODS, series=index, **options,
        )
        actual_axis, expected_axis = figure.axes[0], reference.axes[0]
        self.assertEqual(len(actual_axis.lines), len(expected_axis.lines))
        for actual, expected in zip(actual_axis.lines, expected_axis.lines, strict=True):
          self.assertEqual(actual.get_label(), expected.get_label())
          self.assertEqual(actual.get_color(), expected.get_color())
          np.testing.assert_array_equal(actual.get_xdata(), expected.get_xdata())
          np.testing.assert_array_equal(actual.get_ydata(), expected.get_ydata())
        self.assertEqual(actual_axis.get_xlim(), expected_axis.get_xlim())
        self.assertEqual(actual_axis.get_ylim(), expected_axis.get_ylim())
        self.assertEqual(actual_axis.get_xlabel(), expected_axis.get_xlabel())
        self.assertEqual(actual_axis.get_ylabel(), expected_axis.get_ylabel())
    np.testing.assert_array_equal(self.values, values_before)
    np.testing.assert_array_equal(self.rhos, rhos_before)

  def test_render_does_not_simulate_or_consume_random_draws(self):
    before = np.random.get_state()
    for module in (false_positive_rate, power_comparison):
      with patch.object(module, "simulate", side_effect=AssertionError("must not simulate")):
        self.render_figures(module)
    after = np.random.get_state()
    self.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
  unittest.main()
