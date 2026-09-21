"""Both width panels must have a local, order-independent white-grid style."""

import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba

from reproduction.one_dimensional.interval_width import RHOS, build_figure


def axis_style(axis):
  """Capture visible axes styling, excluding their intentionally different limits."""
  return {
    "background": axis.get_facecolor(),
    "spines": {
      name: (spine.get_visible(), spine.get_edgecolor(), spine.get_linewidth())
      for name, spine in axis.spines.items()
    },
    "x_grid": {
      (line.get_visible(), line.get_color(), line.get_linestyle(), line.get_linewidth())
      for line in axis.get_xgridlines()
    },
    "y_grid": {
      (line.get_visible(), line.get_color(), line.get_linestyle(), line.get_linewidth())
      for line in axis.get_ygridlines()
    },
    "axisbelow": axis.get_axisbelow(),
  }


class IntervalWidthPlottingTests(unittest.TestCase):
  def setUp(self):
    self.style_context = mpl.rc_context()
    self.style_context.__enter__()
    self.original_figures = set(plt.get_fignums())
    # Positive, distinct, deterministic samples avoid simulation and singular KDEs.
    support = np.linspace(-2, 2, 65)
    self.widths = np.array([
      [0.15 + np.exp(.1 * case + .07 * rho + .18 * support) for rho in RHOS]
      for case in range(2)
    ])

  def tearDown(self):
    for number in set(plt.get_fignums()) - self.original_figures:
      plt.close(number)
    self.style_context.__exit__(None, None, None)

  def test_both_panels_have_white_background_grid_and_consistent_spines(self):
    sns.set_style("dark")
    expected = sns.axes_style("whitegrid")
    axes = [build_figure(self.widths, dependence_index=index).axes[0] for index in (0, 1)]
    for axis in axes:
      self.assertEqual(axis.get_facecolor(), to_rgba(expected["axes.facecolor"]))
      for lines in (axis.get_xgridlines(), axis.get_ygridlines()):
        self.assertTrue(lines)
        self.assertTrue(all(line.get_visible() for line in lines))
        self.assertTrue(all(to_rgba(line.get_color()) == to_rgba(expected["grid.color"]) for line in lines))
      for side, spine in axis.spines.items():
        self.assertEqual(spine.get_visible(), expected[f"axes.spines.{side}"])
        self.assertEqual(spine.get_edgecolor(), to_rgba(expected["axes.edgecolor"]))
    self.assertEqual(axis_style(axes[0]), axis_style(axes[1]))
    self.assertEqual(axes[0].get_xlim(), (.2, 3.))
    self.assertEqual(axes[1].get_xlim(), (.4, 4.1))

  def test_panel_style_is_independent_of_call_order_and_incoming_style(self):
    expected = {}
    for initial_style in ("dark", "ticks"):
      for order in ((0, 1), (1, 0)):
        with self.subTest(initial_style=initial_style, order=order), mpl.rc_context():
          sns.set_style(initial_style)
          for index in order:
            figure = build_figure(self.widths, dependence_index=index)
            actual = axis_style(figure.axes[0])
            if index in expected:
              self.assertEqual(actual, expected[index])
            else:
              expected[index] = actual
            plt.close(figure)

  def test_building_each_panel_does_not_mutate_global_rc_parameters(self):
    mpl.rcParams.update({
      "axes.facecolor": "#112233",
      "axes.edgecolor": "#abcdef",
      "axes.grid": False,
      "grid.color": "#ff0000",
      "xtick.bottom": True,
      "ytick.left": True,
    })
    before = dict(mpl.rcParams)
    for index in (0, 1):
      with self.subTest(dependence_index=index):
        figure = build_figure(self.widths, dependence_index=index)
        self.assertEqual(dict(mpl.rcParams), before)
        plt.close(figure)

  def test_style_fix_preserves_kde_curve_coordinates(self):
    for index in (0, 1):
      with self.subTest(dependence_index=index):
        # Freeze the pre-fix drawing order in a context so its global style
        # mutation cannot escape this numerical-curve reference calculation.
        with mpl.rc_context():
          sns.set_style("dark")
          reference, axis = plt.subplots()
          sns.set_style("whitegrid")
          data = pd.DataFrame({
            rf"$\rho={rho:g}$": self.widths[index, rho_index]
            for rho_index, rho in enumerate(RHOS)
          })
          sns.kdeplot(data, ax=axis)
          expected = [(line.get_xdata().copy(), line.get_ydata().copy()) for line in axis.lines]
          plt.close(reference)

        actual = build_figure(self.widths, dependence_index=index)
        lines = actual.axes[0].lines
        self.assertEqual(len(lines), len(RHOS))
        self.assertEqual(len(lines), len(expected))
        for line, (x, y) in zip(lines, expected, strict=True):
          np.testing.assert_array_equal(line.get_xdata(), x)
          np.testing.assert_array_equal(line.get_ydata(), y)
        plt.close(actual)


if __name__ == "__main__":
  unittest.main()
