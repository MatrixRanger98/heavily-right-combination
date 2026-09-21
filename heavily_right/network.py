"""Network meta-analysis from contrast or independent-arm summaries.

Every contrast has the sign ``treat1 - treat2``. Parameters are treatment
effects relative to the declared reference. Study blocks, including supplied
within-study covariance, remain intact when fitting or removing studies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .empty_sets import EmptySetDiagnostics, EmptySetPolicy
from .workflows import Study, StudyCollection, WorkflowFit, WorkflowInterval, fit_studies

FloatArray = NDArray[np.float64]


def _numeric_vector(value: ArrayLike, name: str) -> FloatArray:
  raw = np.asarray(value)
  if raw.dtype.kind not in "iuf":
    raise ValueError(f"{name} must contain real numeric values")
  result = np.asarray(raw, dtype=float)
  if result.ndim != 1 or not result.size or not np.isfinite(result).all():
    raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
  return result.copy()


def _labels(values: Sequence[Any], name: str, count: int | None = None) -> tuple[str, ...]:
  if isinstance(values, (str, bytes)):
    raise ValueError(f"{name} must be a sequence of labels")
  try:
    raw = tuple(values)
  except TypeError as error:
    raise ValueError(f"{name} must be a sequence of labels") from error
  if not raw or (count is not None and len(raw) != count):
    raise ValueError(f"{name} must contain {count if count is not None else 'at least one'} labels")
  if any(value is None or not np.isscalar(value) or
         (isinstance(value, (float, np.floating)) and not np.isfinite(value)) or
         not str(value).strip() for value in raw):
    raise ValueError(f"{name} labels must be nonmissing and nonempty")
  return tuple(str(value) for value in raw)


def _treatment_order(
  observed: Sequence[str], treatments: Sequence[str] | None, reference: str | None,
) -> tuple[tuple[str, ...], str]:
  ordered = tuple(dict.fromkeys(observed)) if treatments is None else _labels(treatments, "treatments")
  if len(ordered) < 2 or len(set(ordered)) != len(ordered):
    raise ValueError("treatments must contain at least two distinct unique labels")
  if set(observed) != set(ordered):
    raise ValueError("treatments must list exactly the observed treatment labels")
  reference = ordered[0] if reference is None else str(reference)
  if reference not in ordered:
    raise ValueError("reference must be an observed treatment")
  return ordered, reference


def _groups(ids: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[NDArray[np.int64], ...]]:
  labels = tuple(dict.fromkeys(ids))
  return labels, tuple(np.array([i for i, label in enumerate(ids) if label == current]) for current in labels)


def _degrees(df: Any, labels: tuple[str, ...]) -> tuple[float | None, ...]:
  if df is None:
    return (None,) * len(labels)
  if isinstance(df, Mapping):
    normalized = {str(key): value for key, value in df.items()}
    if len(normalized) != len(df) or set(normalized) != set(labels):
      raise ValueError("df mapping must provide exactly one entry per study ID")
    values = tuple(normalized[label] for label in labels)
  elif np.ndim(df) == 0:
    values = (df,) * len(labels)
  else:
    values = tuple(df)
    if len(values) != len(labels):
      raise ValueError("df must have one entry per study block, in first-seen order")
  if any(value is None for value in values):
    raise ValueError("mixing normal calibration (df=None) and finite df across studies is not supported")
  return values


def _covariances(
  covariance: Any, labels: tuple[str, ...], groups: tuple[NDArray[np.int64], ...],
  se: FloatArray,
) -> tuple[FloatArray, ...]:
  if covariance is None:
    return tuple(np.diag(se[index] ** 2) for index in groups)
  if isinstance(covariance, Mapping):
    normalized = {str(key): value for key, value in covariance.items()}
    if len(normalized) != len(covariance) or set(normalized) != set(labels):
      raise ValueError("covariance mapping must provide exactly one matrix per study ID")
    return tuple(np.asarray(normalized[label]) for label in labels)
  try:
    matrix = np.asarray(covariance)
  except (TypeError, ValueError):
    matrix = None
  if matrix is not None and matrix.shape == (len(se), len(se)):
    if matrix.dtype.kind not in "iuf":
      raise ValueError("covariance must contain real numeric values")
    within = np.zeros(matrix.shape, dtype=bool)
    for index in groups:
      within[np.ix_(index, index)] = True
    if np.any(matrix[~within] != 0):
      raise ValueError("covariance between different study IDs is not supported; supply independent study blocks")
    return tuple(matrix[np.ix_(index, index)] for index in groups)
  try:
    blocks = tuple(covariance)
  except TypeError as error:
    raise ValueError("covariance must be a full row matrix, a mapping, or a sequence of study matrices") from error
  if len(blocks) != len(labels):
    raise ValueError("covariance must have one matrix per study block, in first-seen order")
  return tuple(np.asarray(block) for block in blocks)


@dataclass(frozen=True, eq=False)
class NetworkData:
  """Study summaries and the treatment-to-parameter mapping for one network.

  Prefer :func:`network_studies` or :func:`network_arm_studies` to construct
  these data. ``studies.parameter_names`` must equal the treatment order
  after excluding ``reference``. Study estimates and covariance are copied.

  Examples
  --------
  >>> from heavily_right.network import NetworkData, network_studies
  >>> data = network_studies([0.2], [0.1], ["active"], ["control"],
  ...                        reference="control")
  >>> copied = NetworkData(data.studies, data.treatments, data.reference)
  >>> assert copied.parameter_names == ("active",)
  """

  studies: StudyCollection
  treatments: tuple[str, ...]
  reference: str

  def __post_init__(self) -> None:
    if not isinstance(self.studies, StudyCollection):
      raise TypeError("studies must be a StudyCollection")
    treatments = _labels(self.treatments, "treatments")
    reference = str(self.reference)
    if len(set(treatments)) != len(treatments) or reference not in treatments:
      raise ValueError("treatments must be unique and contain the reference")
    if tuple(treatment for treatment in treatments if treatment != reference) != self.studies.parameter_names:
      raise ValueError("nonreference treatment order must match study parameter names")
    object.__setattr__(self, "treatments", treatments)
    object.__setattr__(self, "reference", reference)

  @property
  def parameter_names(self) -> tuple[str, ...]:
    return self.studies.parameter_names

  @property
  def treatment_index(self) -> dict[str, int]:
    """Indices of nonreference treatment effects; the reference has effect zero."""
    return {treatment: index for index, treatment in enumerate(self.parameter_names)}

  def contrast_direction(self, treat1: str, treat2: str) -> FloatArray:
    """Return the parameter-space direction for ``treat1 - treat2``.

    Both treatments must occur in the network and must differ. The reference
    has effect zero and is absent from the returned parameter vector.

    Examples
    --------
    >>> from heavily_right.network import network_studies
    >>> data = network_studies([0.2], [0.1], ["b"], ["a"], reference="a")
    >>> assert data.contrast_direction("b", "a").tolist() == [1.0]
    >>> assert data.contrast_direction("a", "b").tolist() == [-1.0]
    """
    first, second = str(treat1), str(treat2)
    if first not in self.treatments or second not in self.treatments:
      raise ValueError("contrast treatments must belong to the fitted network")
    if first == second:
      raise ValueError("a treatment contrast must compare distinct treatments")
    indices = self.treatment_index
    direction = np.zeros(len(indices))
    if first != self.reference:
      direction[indices[first]] = 1
    if second != self.reference:
      direction[indices[second]] = -1
    return direction


def network_studies(
  estimate: ArrayLike, se: ArrayLike | None, treat1: Sequence[str], treat2: Sequence[str], *,
  study_id: Sequence[str] | None = None, reference: str | None = None,
  treatments: Sequence[str] | None = None, covariance: Any = None, df: Any = None,
  multiarm: str = "require_covariance",
) -> NetworkData:
  """Build a connected network from contrasts signed ``treat1 - treat2``.

  Repeated study IDs define one correlated block and require covariance.
  Supply covariance as a mapping keyed by study ID, a sequence of matrices
  in first-seen study order, or a full row covariance with zero entries
  between studies. Its diagonal is authoritative, including when corrected
  standard errors differ from ``se``. Pass ``se=None`` when supplying only
  covariance. Without covariance, ``se`` is required and constructs the
  diagonal. ``multiarm='independent'`` explicitly treats each contrast as an
  independent study and cannot be combined with supplied covariance.

  Each correlated block must contain independent contrast rows. For complete
  pairwise reports of a multi-arm trial, provide an independent contrast set
  or use :func:`network_arm_studies`. ``df`` is scalar, a mapping by study ID,
  or one value per study block; normal and finite-df blocks cannot be mixed.

  ``reference`` defaults to the first observed treatment. ``treatments`` may
  set an explicit ordering and must list every observed treatment exactly
  once. The returned :class:`NetworkData` represents a common treatment-effect
  vector; it does not estimate random effects or between-study heterogeneity.
  Invalid labels, disconnected networks, redundant within-study rows, and
  non-positive-definite covariance raise ``ValueError``.

  Examples
  --------
  A supplied covariance contains all uncertainty, so SEs may be omitted.

  >>> from heavily_right.network import network_studies
  >>> data = network_studies([0.2, 0.3], None, ["b", "c"], ["a", "a"],
  ...     study_id=["trial", "trial"], reference="a",
  ...     covariance={"trial": [[0.04, 0.01], [0.01, 0.09]]})
  >>> assert data.parameter_names == ("b", "c")
  >>> assert len(data.studies.studies) == 1
  >>> assert data.studies.studies[0].projection.tolist() == [[1.0, 0.0], [0.0, 1.0]]
  """
  estimates = _numeric_vector(estimate, "estimate")
  if se is None:
    if covariance is None:
      raise ValueError("se is required when covariance is not provided")
    # Covariance supplies all uncertainty; only row count is used below.
    errors = np.ones(len(estimates))
  else:
    errors = _numeric_vector(se, "se")
    if errors.shape != estimates.shape or np.any(errors <= 0):
      raise ValueError("se must match estimate and contain strictly positive values")
  first = _labels(treat1, "treat1", len(estimates))
  second = _labels(treat2, "treat2", len(estimates))
  if any(a == b for a, b in zip(first, second, strict=True)):
    raise ValueError("each contrast must compare distinct treatments")
  observed = tuple(item for pair in zip(first, second, strict=True) for item in pair)
  ordered, reference = _treatment_order(observed, treatments, reference)
  parameters = tuple(treatment for treatment in ordered if treatment != reference)
  parameter_index = {treatment: index for index, treatment in enumerate(parameters)}
  projection = np.zeros((len(estimates), len(parameters)))
  for row, (a, b) in enumerate(zip(first, second, strict=True)):
    if a != reference:
      projection[row, parameter_index[a]] = 1
    if b != reference:
      projection[row, parameter_index[b]] = -1
  if np.linalg.matrix_rank(projection) != len(parameters):
    raise ValueError("the treatment network is disconnected; all treatments must connect to the reference")
  ids = (tuple(f"study-{i + 1}" for i in range(len(estimates))) if study_id is None
         else _labels(study_id, "study_id", len(estimates)))
  labels, groups = _groups(ids)
  degrees = _degrees(df, labels)
  if multiarm not in ("require_covariance", "independent"):
    raise ValueError("multiarm must be 'require_covariance' or 'independent'")
  if multiarm == "independent" and covariance is not None:
    raise ValueError("multiarm='independent' cannot discard supplied covariance")
  repeated = any(len(index) > 1 for index in groups)
  if repeated and multiarm == "independent":
    degrees_by_id = dict(zip(labels, degrees, strict=True))
    degrees = tuple(degrees_by_id[label] for label in ids)
    # Preserve the source ID and row identity without risking label collisions.
    labels = tuple(f"{label} [contrast {i + 1}]" for i, label in enumerate(ids))
    groups = tuple(np.array([i]) for i in range(len(estimates)))
  elif repeated and covariance is None:
    raise ValueError("repeated study IDs require within-study covariance; use arm summaries or explicitly set multiarm='independent'")
  matrices = _covariances(covariance, labels, groups, errors)
  blocks = []
  for label, index, matrix, degrees_freedom in zip(labels, groups, matrices, degrees, strict=True):
    design = projection[index]
    if np.linalg.matrix_rank(design) != len(index):
      raise ValueError(f"study {label!r} has redundant contrasts; provide independent contrasts or arm summaries")
    blocks.append(Study(estimates[index], matrix, design, df=degrees_freedom, label=label))
  return NetworkData(StudyCollection(tuple(blocks), parameters), ordered, reference)


def network_arm_studies(
  mean: ArrayLike, se: ArrayLike, treatment: Sequence[str], study_id: Sequence[str], *,
  reference: str | None = None, treatments: Sequence[str] | None = None, df: Any = None,
) -> NetworkData:
  """Build correlated contrasts from independent arm means and their SEs.

  Each study's first input arm is its baseline. For other arms ``i`` and
  ``j``, the covariance of ``mean_i - mean_baseline`` and
  ``mean_j - mean_baseline`` is ``se_baseline**2``; diagonal entries also
  include the corresponding nonbaseline arm variance. ``se`` means standard
  error of each estimated mean, not its raw-observation standard deviation.
  Degrees of freedom calibrate each resulting contrast block and are never
  inferred from arm SEs.

  All four positional inputs have one entry per arm. Each study needs at
  least two distinct treatment arms. ``reference`` and ``treatments`` select
  the network parameterization and do not change the within-trial baseline.
  The result is :class:`NetworkData`; malformed or disconnected inputs raise
  ``ValueError``. Use :func:`network_studies` with supplied covariance if arm
  estimates are correlated before constructing differences.

  Examples
  --------
  >>> import numpy as np
  >>> from heavily_right.network import network_arm_studies
  >>> data = network_arm_studies([0.0, 1.0, -0.5], [0.1, 0.2, 0.15],
  ...     ["a", "b", "c"], ["trial"] * 3, reference="a")
  >>> block = data.studies.studies[0]
  >>> np.testing.assert_allclose(block.estimate, [1.0, -0.5])
  >>> np.testing.assert_allclose(block.covariance, [[0.05, 0.01], [0.01, 0.0325]])
  """
  means = _numeric_vector(mean, "mean")
  errors = _numeric_vector(se, "se")
  if means.shape != errors.shape or np.any(errors <= 0):
    raise ValueError("se must match mean and contain strictly positive values")
  arms = _labels(treatment, "treatment", len(means))
  ids = _labels(study_id, "study_id", len(means))
  ordered, reference = _treatment_order(arms, treatments, reference)
  labels, groups = _groups(ids)
  degrees = _degrees(df, labels)
  estimates, contrast_se, first, second, contrast_ids = [], [], [], [], []
  covariance = {}
  for label, index in zip(labels, groups, strict=True):
    if len(index) < 2:
      raise ValueError(f"study {label!r} needs at least two treatment arms")
    if len(set(arms[i] for i in index)) != len(index):
      raise ValueError(f"study {label!r} repeats a treatment arm")
    baseline, other = index[0], index[1:]
    baseline_variance = errors[baseline] ** 2
    matrix = np.diag(errors[other] ** 2) + baseline_variance
    covariance[label] = matrix
    estimates.extend(means[other] - means[baseline])
    contrast_se.extend(np.sqrt(np.diag(matrix)))
    first.extend(arms[i] for i in other)
    second.extend([arms[baseline]] * len(other))
    contrast_ids.extend([label] * len(other))
  return network_studies(estimates, contrast_se, first, second, study_id=contrast_ids,
    reference=reference, treatments=ordered, covariance=covariance,
    df=None if df is None else dict(zip(labels, degrees, strict=True)))


@dataclass(frozen=True, eq=False)
class NetworkFit:
  """Fitted joint network region with named treatment contrasts.

  Construct with :func:`fit_nma`. ``estimate`` follows ``parameter_names``;
  ``inputs`` and ``active_inputs`` retain original and retained study blocks.
  Every later contrast uses the same retained set and calibrated threshold.

  Examples
  --------
  >>> from heavily_right.network import NetworkFit, fit_nma, network_studies
  >>> fitted = fit_nma(network_studies([0.2], [0.1], ["b"], ["a"], reference="a"))
  >>> assert isinstance(fitted, NetworkFit)
  >>> assert fitted.parameter_names == ("b",)
  >>> assert fitted.original_indices == (0,)
  """

  data: NetworkData
  fit: WorkflowFit

  @property
  def inputs(self) -> StudyCollection:
    return self.fit.inputs

  @property
  def active_inputs(self) -> StudyCollection:
    return self.fit.active_inputs

  @property
  def original_indices(self) -> tuple[int, ...]:
    return self.fit.original_indices

  @property
  def weights(self) -> FloatArray:
    return self.fit.weights

  @property
  def threshold(self) -> float:
    return self.fit.threshold

  @property
  def estimate(self) -> FloatArray:
    return self.fit.estimate

  @property
  def parameter_names(self) -> tuple[str, ...]:
    return self.data.parameter_names

  @property
  def empty_set_diagnostics(self) -> EmptySetDiagnostics | None:
    return self.fit.empty_set_diagnostics

  def support_interval(self, direction: ArrayLike) -> WorkflowInterval:
    """Project the joint region onto a finite nonzero parameter direction.

    The result is a :class:`~heavily_right.workflows.WorkflowInterval` with
    scalar bounds, endpoint points in parameter coordinates, ``success``,
    and ``native_result``. Always check ``success`` before using the bounds.
    Scalar networks also accept a scalar direction.

    Examples
    --------
    >>> from heavily_right.network import fit_nma, network_studies
    >>> fitted = fit_nma(network_studies([0.2], [0.1], ["b"], ["a"], reference="a"))
    >>> interval = fitted.support_interval([1.0])
    >>> assert interval.success and interval.lower < 0.2 < interval.upper
    """
    return self.fit.support_interval(direction)

  def contrast(self, treat1: str, treat2: str) -> WorkflowInterval:
    """Project the fitted joint region onto the effect ``treat1 - treat2``.

    Unknown or identical treatments raise ``ValueError``. Bounds have the
    input effect units. Endpoint points remain in reference-relative parameter
    coordinates, including when the requested contrast reverses direction.

    Examples
    --------
    >>> import numpy as np
    >>> from heavily_right.network import fit_nma, network_studies
    >>> fitted = fit_nma(network_studies([0.2], [0.1], ["b"], ["a"], reference="a"))
    >>> forward, reverse = fitted.contrast("b", "a"), fitted.contrast("a", "b")
    >>> assert forward.success and reverse.success
    >>> np.testing.assert_allclose([reverse.lower, reverse.upper],
    ...                            [-forward.upper, -forward.lower])
    """
    return self.support_interval(self.data.contrast_direction(treat1, treat2))

  def pairwise_intervals(self) -> dict[tuple[str, str], WorkflowInterval]:
    """Return each unordered treatment pair, signed in treatment-list order.

    Dictionary key ``(a, b)`` denotes ``a - b``. These are projections of the
    same fitted joint region; this method does not refit or remove studies.

    Examples
    --------
    >>> from heavily_right.network import fit_nma, network_arm_studies
    >>> data = network_arm_studies([0.0, 1.0, -0.5], [0.1, 0.2, 0.15],
    ...     ["a", "b", "c"], ["trial"] * 3, reference="a")
    >>> intervals = fit_nma(data).pairwise_intervals()
    >>> assert set(intervals) == {("a", "b"), ("a", "c"), ("b", "c")}
    >>> assert all(interval.success for interval in intervals.values())
    """
    return {(a, b): self.contrast(a, b) for a, b in combinations(self.data.treatments, 2)}

  def score(self, point: ArrayLike) -> float:
    """Evaluate the retained studies' score at a parameter-space point.

    Examples
    --------
    >>> from heavily_right.network import fit_nma, network_studies
    >>> fitted = fit_nma(network_studies([0.2], [0.1], ["b"], ["a"], reference="a"))
    >>> assert fitted.score([0.2]) <= fitted.threshold
    """
    return self.fit.score(point)

  def contains(self, point: ArrayLike) -> bool:
    """Test whether the retained-study score is at most its fitted threshold.

    Examples
    --------
    >>> from heavily_right.network import fit_nma, network_studies
    >>> fitted = fit_nma(network_studies([0.2], [0.1], ["b"], ["a"], reference="a"))
    >>> assert fitted.contains(fitted.estimate)
    """
    return self.fit.contains(point)


def fit_nma(
  data: NetworkData, *, weights: ArrayLike | None = None, method: str = "HCauchy",
  level: float = 0.05, empty_set_policy: EmptySetPolicy | None = None, **solver_options: Any,
) -> NetworkFit:
  """Fit a network region, assigning one combination weight per study block.

  The default preserves all studies. An explicit empty-set policy operates
  on entire study blocks and records its decisions in ``empty_set_diagnostics``.
  The retained set is fixed for all subsequent projections. Data-dependent
  deletion is a sensitivity analysis, without a post-selection coverage
  correction. A numerical failure does not authorize deletion.

  ``weights`` supplies positive block weights summing to one; the default is
  equal weights. ``method`` is ``"HCauchy"`` or ``"EHMP"``, and ``level`` is
  the significance level. Multidimensional solver controls are forwarded to
  :func:`~heavily_right.workflows.fit_studies`; scalar networks use the scalar
  solver and accept no multidimensional controls. The return is
  :class:`NetworkFit`. Unresolved empty regions and uncertified numerical
  results raise errors with diagnostics where available.

  Examples
  --------
  A two-treatment network has one parameter and supports ordinary contrasts.

  >>> from heavily_right.network import fit_nma, network_studies
  >>> data = network_studies([0.2], [0.1], ["b"], ["a"], reference="a")
  >>> fitted = fit_nma(data, level=0.05)
  >>> interval = fitted.contrast("b", "a")
  >>> assert interval.success and interval.lower < 0.2 < interval.upper
  >>> assert fitted.original_indices == (0,)
  """
  if not isinstance(data, NetworkData):
    raise TypeError("data must be constructed by network_studies or network_arm_studies")
  fitted = fit_studies(data.studies, weights=weights, method=method, level=level,
    empty_set_policy=empty_set_policy, **solver_options)
  return NetworkFit(data, fitted)


__all__ = ["NetworkData", "NetworkFit", "network_studies", "network_arm_studies", "fit_nma"]
