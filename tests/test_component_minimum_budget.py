"""Aggregate minimum certificates, including the saved NMA rho=0 replicate114 failure."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from heavily_right import MetaAnalysisMD
from heavily_right.minimum import minimize_score_region
from heavily_right.projected_region import ProjectedScoreRegion

# Exact float inputs from full-reproduction-20260921, rho=0, replicate114.
# Keep the regression independent of generated results and reproduction imports.
MEANS = np.array([
  -.9235547877177038, -1.2123716842226262, -1.0413815245098765, -.9081055261327049,
  -1.2430190732874253, -.45515077333688586, -1.231551357005721, .4856965856580656,
  .4992083447201604, -.815073318927908, -1.0385136997996947, -1.0545400404054852,
  .15124543115137423, -.5284622211531385, .421072549309604, .03451511797749109,
  .07768668221117091, -1.0424529170847947, -.5786284461615806, .029457035007630285,
  -.09579944443419365, .1534489704627513, .0630815287912311, -.8767096824440432,
  -.6891993921854633, -.5452829920777958, -.9424942925151076, -.09113978694325173,
])
VARIANCES = np.array([
  .009541104589970598, .011045536097184207, .010409111449021359, .008178959933424617,
  .011681371935128821, .01106917799440211, .006860444299186636, .00909572543577906,
  .011171912157005949, .011663689347595005, .010194828177729019, .009880066093024184,
  .00958081684822472, .009091926501806357, .008192868312363313, .009528547184720803,
  .009213917871079181, .011592352564315055, .008999905273741563, .007956370755338506,
  .010755855039327763, .010075966598386917, .010101704225241263, .011073652710447435,
  .013589406855185077, .008527668044300675, .010472112555212798, .007617548267542314,
])
DESIGN = np.array([
  [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [-1,0,1,0,0,0,0,0,0],
  [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,1,0,0,0,0],
  [0,0,0,0,0,1,0,0,0], [0,0,-1,0,1,0,0,0,0], [0,0,0,0,1,-1,0,0,0],
  [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0],
  [0,0,-1,0,0,1,0,0,0], [0,0,0,0,0,1,0,-1,0], [1,0,0,0,0,0,0,-1,0],
  [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,0,0,1],
  [0,0,1,0,0,0,0,-1,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0],
  [0,0,-1,0,0,1,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,1,0,0,0,0,0,0],
  [0,1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0],
  [1,0,0,0,0,0,0,0,0],
], dtype=float)
POWELL_START = np.array([
  -.025113872867228458, .0027977559286125543, -.05939799449391517,
  .017089096576109575, -.10698392017788214, -.1962841629718903,
  0., -.031977669134104644, 0.,
])
CUTOFF = 16.24181581228568


def _two_components(
  mass: float = .1, *, compatible: bool = False, method: str = "HCauchy",
) -> ProjectedScoreRegion:
  return ProjectedScoreRegion(
    dimension=2, weights=np.array([(1-mass)/2]*2 + [mass/2]*2), method=method,
    xi_hat=np.zeros((4, 1)) if compatible else np.array([[-.2], [.2], [-.1], [.1]]),
    sigma=np.ones((4, 1, 1)), sub_dim=np.array([[0], [0], [1], [1]]),
    degrees_freedom=np.full(4, 99.), projections=np.eye(2), equal_sub_dim=True,
  )


class ComponentMinimumBudgetTests(unittest.TestCase):
  def test_saved_nma_minimum_meets_unchanged_parent_gap_after_refinement(self) -> None:
    region = ProjectedScoreRegion(
      dimension=9, weights=np.full(28, 1/28), method="HCauchy",
      xi_hat=MEANS[:, None], sigma=VARIANCES[:, None, None], sub_dim=np.arange(28)[:, None],
      degrees_freedom=np.full(28, 99.), projections=DESIGN, equal_sub_dim=True,
    )
    result = minimize_score_region(region, POWELL_START, cutoff=CUTOFF)
    self.assertTrue(result.success, result.diagnostics.message)
    diagnostics = result.diagnostics
    self.assertEqual(diagnostics.optimality_gap_tolerance, 2e-6*CUTOFF)
    self.assertLessEqual(diagnostics.optimality_gap_bound, diagnostics.optimality_gap_tolerance)
    self.assertGreaterEqual(diagnostics.component_refinements, 1)
    self.assertFalse(diagnostics.empty_certified)
    self.assertAlmostEqual(result.score, 3.213019157430995, delta=1e-9)
    self.assertLess(diagnostics.component_diagnostics[1].optimality_gap_tolerance, .000454770842743999)

  def test_saved_nma_public_support_and_reused_minimum_validate_both_directions(self) -> None:
    analysis = MetaAnalysisMD(28, dim=9, method="HCauchy", level=.05)
    self.assertAlmostEqual(analysis.threshold, CUTOFF, delta=1e-12)
    first = analysis.support_interval(
      np.eye(9)[0], MEANS, VARIANCES, sub_dim=1, df=99, projs=DESIGN, method="Powell",
    )
    self.assertTrue(first.success)
    self.assertAlmostEqual(first.lower, -.2753011120137878, delta=2e-7)
    self.assertAlmostEqual(first.upper, .18785467543640175, delta=2e-7)
    second = analysis.support_interval(
      np.eye(9)[1], MEANS, VARIANCES, sub_dim=1, df=99, projs=DESIGN, method="Powell",
      x0=first.minimum_point, find_min=False,
    )
    for result in (first, second):
      self.assertTrue(result.success)
      self.assertLessEqual(result.minimum_diagnostics.optimality_gap_bound,
                           result.minimum_diagnostics.optimality_gap_tolerance)
      for endpoint in (result.lower_diagnostics, result.upper_diagnostics):
        self.assertTrue(endpoint.success)
        self.assertLessEqual(abs(endpoint.boundary_error) / endpoint.score_scale, 2e-8)
        self.assertLessEqual(endpoint.kkt_residual, 2e-6)
        self.assertGreater(endpoint.eta, 0)
      self.assertTrue(result.empty_set_diagnostics is None or result.empty_set_diagnostics.removals == ())

  def test_already_valid_aggregate_performs_no_refinement(self) -> None:
    with patch("heavily_right.minimum.minimize", side_effect=AssertionError("unnecessary optimizer")):
      result = minimize_score_region(_two_components(compatible=True), cutoff=14.)
    self.assertTrue(result.success)
    self.assertEqual(result.diagnostics.component_refinements, 0)
    self.assertEqual(result.diagnostics.iterations, 0)

  def _loose_components(self, mass: float, *, reject_refinement: bool = False,
                        exhausted: bool = False, method: str = "HCauchy"):
    region = _two_components(mass, method=method)
    observed_budgets = []

    def solve_child(child, *args, **kwargs):
      result = minimize_score_region(child, *args, **kwargs)
      budget = kwargs.get("_maximum_gap")
      if budget is not None:
        observed_budgets.append((budget, kwargs["maxiter"]))
        return replace(result, success=False) if reject_refinement else result
      # Deliberately weaken a VALID convex lower bound, without inventing an
      # overconfident certificate. Each child still passes its own tolerance.
      gap = .9*result.diagnostics.optimality_gap_tolerance
      self.assertGreater(gap, result.diagnostics.optimality_gap_bound)
      diagnostics = replace(
        result.diagnostics, minimum_value_lower_bound=result.score-gap,
        optimality_gap_bound=gap, iterations=20 if exhausted else 3,
      )
      return replace(result, diagnostics=diagnostics)

    with patch("heavily_right.minimum.minimize_score_region", side_effect=solve_child):
      result = minimize_score_region(region, cutoff=14., maxiter=20)
    return result, observed_budgets

  def test_loose_component_certificates_are_refined_in_parent_units_even_for_weak_mass(self) -> None:
    for method in ("HCauchy", "EHMP"):
      for mass in (.1, 1e-8):
        with self.subTest(mass=mass, method=method):
          result, budgets = self._loose_components(mass, method=method)
          self.assertTrue(result.success, result.diagnostics.message)
          self.assertEqual(result.diagnostics.optimality_gap_tolerance, 2e-6*14)
          self.assertEqual(result.diagnostics.component_refinements, 2)
          self.assertEqual(len(budgets), 2)
          self.assertTrue(all(remaining == 17 for _, remaining in budgets))
          self.assertLessEqual(result.diagnostics.optimality_gap_bound, 2e-6*14)
          self.assertTrue(all(child.iterations <= 20 for child in result.diagnostics.component_diagnostics))

  def test_iteration_exhaustion_keeps_aggregate_failure_distinct_from_child_failure(self) -> None:
    result, budgets = self._loose_components(.1, exhausted=True)
    self.assertFalse(result.success)
    self.assertFalse(result.diagnostics.empty_certified)
    self.assertEqual(result.diagnostics.component_refinements, 0)
    self.assertEqual(budgets, [])
    self.assertIn("component minima validated but aggregate gap exceeds tolerance", result.diagnostics.message)
    self.assertGreater(result.diagnostics.optimality_gap_bound, result.diagnostics.optimality_gap_tolerance)

  def test_failed_refinement_cannot_promote_child_or_parent_to_success(self) -> None:
    result, budgets = self._loose_components(.1, reject_refinement=True)
    self.assertFalse(result.success)
    self.assertFalse(result.diagnostics.empty_certified)
    self.assertEqual(len(budgets), 2)
    self.assertIn("parent gap-budget refinement not accepted", result.diagnostics.message)


if __name__ == "__main__":
  unittest.main()
