"""Heavily-right combination tests and confidence-region inference.

The numerical API requires NumPy and SciPy only. Plotting and experiment
dependencies are optional; reproduction workflows live in ``reproduction``.
"""

from .calibration import covariance_of_mean, standard_error_of_mean
from .combination import CombinationMethod, CombinationTest
from .dac import DacFit, dac_studies, fit_dac
from .empty_sets import (
  EmptySetDiagnostics,
  EmptySetNumericalError,
  EmptySetPolicy,
  EmptySetResolutionError,
)
from .meta_analysis import (
  ConfidenceIntervalResult,
  MetaAnalysis1D,
  MetaAnalysisMD,
  SupportIntervalResult,
)
from .minimum import MinimumDiagnostics
from .network import (
  NetworkData,
  NetworkFit,
  fit_nma,
  network_arm_studies,
  network_studies,
)
from .null_laws import (
  HalfCauchyMean,
  HarmonicMean,
  NumericalCalibrationError,
  NumericalIntegrationError,
  NumericalRootError,
)
from .support import SupportDiagnostics
from .workflows import (
  Study,
  StudyCollection,
  WorkflowFit,
  WorkflowFitError,
  WorkflowInterval,
  WorkflowQueryError,
  fit_studies,
)

__version__ = "0.1.0"

__all__ = [
  "CombinationMethod",
  "CombinationTest",
  "ConfidenceIntervalResult",
  "DacFit",
  "EmptySetDiagnostics",
  "EmptySetNumericalError",
  "EmptySetPolicy",
  "EmptySetResolutionError",
  "HalfCauchyMean",
  "HarmonicMean",
  "MetaAnalysis1D",
  "MetaAnalysisMD",
  "MinimumDiagnostics",
  "NetworkData",
  "NetworkFit",
  "NumericalCalibrationError",
  "NumericalIntegrationError",
  "NumericalRootError",
  "SupportDiagnostics",
  "SupportIntervalResult",
  "Study",
  "StudyCollection",
  "WorkflowFit",
  "WorkflowFitError",
  "WorkflowInterval",
  "WorkflowQueryError",
  "covariance_of_mean",
  "dac_studies",
  "fit_dac",
  "fit_nma",
  "fit_studies",
  "network_arm_studies",
  "network_studies",
  "standard_error_of_mean",
]
