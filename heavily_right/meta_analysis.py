"""Compatibility façade for one- and multidimensional meta-analysis.

New implementations live in focused univariate and multivariate modules. The
historical lowercase names remain exact aliases for existing user code.
"""

from .combination import CombinationTest, combination_test, cot
from .empty_sets import (
  EmptySetDiagnostics,
  EmptySetPolicy,
  EmptySetResolutionError,
  StudyRemoval,
)
from .multivariate import MetaAnalysisMD, SupportIntervalResult
from .univariate import ConfidenceIntervalResult, MetaAnalysis1D

meta_analysis_1d = MetaAnalysis1D
meta_analysis_md = MetaAnalysisMD


__all__ = [
  "CombinationTest",
  "ConfidenceIntervalResult",
  "EmptySetDiagnostics",
  "EmptySetPolicy",
  "EmptySetResolutionError",
  "MetaAnalysis1D",
  "MetaAnalysisMD",
  "StudyRemoval",
  "SupportIntervalResult",
  "combination_test",
  "cot",
  "meta_analysis_1d",
  "meta_analysis_md",
]
