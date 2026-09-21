"""Reproduction failures retain the attempted NMA support certificate."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from heavily_right.multivariate import SupportIntervalResult
from heavily_right.support import SupportDiagnostics
from reproduction.artifacts import ArtifactStore
from reproduction.network_meta_analysis.real_data import hcct_width_matrix


class NmaDiagnosticsTests(unittest.TestCase):
  def test_failed_endpoint_is_retained_before_exception(self):
    diagnostic = SupportDiagnostics(
      success=False, solver="slsqp", score=np.nan, cutoff=1.0,
      boundary_error=np.nan, kkt_residual=np.inf, eta=0.0,
      iterations=10, evaluations=20, line_search_steps=0,
      message="forced diagnostic regression",
    )
    result = SupportIntervalResult(
      lower=-1.0, upper=1.0, lower_point=np.array([-1.0]),
      upper_point=np.array([1.0]), lower_diagnostics=diagnostic,
      upper_diagnostics=diagnostic, central_symmetry_used=False,
      active_dimension=1, minimum_point=np.zeros(1), minimum_score=0.0,
    )
    analysis = Mock()
    analysis.support_interval.return_value = result
    with tempfile.TemporaryDirectory() as directory:
      artifacts = ArtifactStore("network-meta-analysis/real-data", "test", Path(directory))
      with self.assertRaisesRegex(RuntimeError, "uncertified NMA support direction 0"):
        hcct_width_matrix(
          analysis, np.zeros(1), np.ones(1), np.ones((1, 1)), artifacts=artifacts,
        )
      path = artifacts.run_dir / "diagnostics/nma-real-data-support-certificates.json"
      payload = json.loads(path.read_text())
    self.assertEqual(payload["status"], "failed")
    self.assertEqual(payload["completed_directions"], 0)
    self.assertEqual(payload["failure"]["index"], 0)
    self.assertEqual(payload["directions"][0]["result"]["lower_diagnostics"]["score"], "nan")
    self.assertEqual(payload["directions"][0]["result"]["lower_diagnostics"]["iterations"], 10)


if __name__ == "__main__":
  unittest.main()
