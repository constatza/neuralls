"""Composition-layer orchestration models for comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeIs

import numpy as np
import torch

from neuralls.domain.presentation.methods import (
    ExecutedMethod,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
)
from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.presentation.titles import ComparisonTitle
from neuralls.domain.solver.models.benchmark_events import (
    MatrixSetupEvent,
    PeakMemory,
    SolveAttempt,
    SystemTag,
)
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.domain.solver.models.provenance import (
    HeldOutMatrix,
    MatrixProvenance,
    SharedSingleMatrix,
)
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.platform.config.models.comparison import RHS_SOURCE_MODEL_BY_KIND
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import PreconditionerConfig
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey
from neuralls.platform.storage.manifest import DatasetNormalization
from neuralls.shared.digest import Digest, Json, canonical
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import (
    ComparisonRhsSourceKind,
    DatasetId,
    MatrixId,
    PreconditionerName,
    RowKind,
    SystemId,
)

__all__ = [
    "ComparisonCompleted",
    "ComparisonEntryFailed",
    "ComparisonOutcome",
    "ComparisonParams",
    "ComparisonPaths",
    "ComparisonPresentation",
    "ComparisonResult",
    "ComparisonReused",
    "CompletedMethodWorkload",
    "LinearSystem",
    "MatrixRef",
    "MatrixWorkload",
    "MatrixWorkloadMember",
    "PlannedSystem",
    "PreconditionerComparisonEntry",
    "ResolvedComparisonInput",
    "ResolvedMethod",
    "UndescribedComparison",
    "canonical_method_config",
    "is_comparison_outcome",
]


def canonical_method_config(config: PreconditionerConfig) -> dict[str, Json]:
    """Return the stable config payload persisted with a method workload.

    Pydantic's JSON dump contains resolved lease-local checkpoint paths. The
    shared canonicalizer instead applies the config field markers: checkpoint
    inputs become content digests and cosmetic/runtime fields are omitted.
    The derived method name remains useful artifact metadata.
    """
    payload = canonical(config)
    if not isinstance(payload, dict):
        raise TypeError("preconditioner config must canonicalize to an object")
    payload["name"] = config.name
    return _without_none(payload)


def _without_none(value: dict[str, Json]) -> dict[str, Json]:
    """Remove unset optional fields from a canonical JSON object recursively."""
    return {key: _without_none_value(item) for key, item in value.items() if item is not None}


def _without_none_value(value: Json) -> Json:
    if isinstance(value, dict):
        return _without_none(value)
    if isinstance(value, list):
        return [_without_none_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ResolvedMethod:
    """One resolved candidate method configuration."""

    config: PreconditionerConfig
    identity_digest: Digest | None = None

    @property
    def name(self) -> PreconditionerName:
        """The stable result identifier derived by the config model."""
        return PreconditionerName(self.config.name)


@dataclass(frozen=True, slots=True)
class MatrixWorkloadMember:
    """One planned RHS member, already read but not moved to the solver device."""

    tag: SystemTag
    rhs: np.ndarray


@dataclass(frozen=True, slots=True)
class MatrixWorkload:
    """One matrix identity and its ordered planned RHS members."""

    dataset: DatasetId
    matrix_id: MatrixId
    matrix: np.ndarray
    matrix_path: Path
    matrix_index: int
    members: tuple[MatrixWorkloadMember, ...]

    def __post_init__(self) -> None:
        """Keep membership tied to the bundle identity."""
        if not self.members:
            raise ValueError("a matrix workload must contain at least one planned RHS")
        if any(
            member.tag.dataset != self.dataset or member.tag.matrix_id != self.matrix_id
            for member in self.members
        ):
            raise ValueError("matrix workload members must match the bundle identity")


@dataclass(frozen=True, slots=True)
class CompletedMethodWorkload:
    """Canonical raw events and diagnostics produced by one method workload.

    ``presentation`` holds the realized facts captured during execution, so reporting never
    needs the released preconditioner; it has no systems when no setup succeeded. It must
    belong to ``method``.

    ``generation_cost`` is the optional historical generation cost; checkpoint
    and data loading are I/O outside the benchmark cost model. Generation
    never contributes to setup, solve, or workload-online totals.
    """

    method: PreconditionerName
    planned: tuple[SystemTag, ...]
    setup_events: tuple[MatrixSetupEvent, ...]
    solve_attempts: tuple[SolveAttempt, ...]
    diagnostics: tuple[tuple[SystemId, CGComparisonResult], ...]
    generation_cost: StageCost | None
    workload_peak_memory: PeakMemory
    config_digest: Digest
    method_config: dict[str, Json]
    presentation: ExecutedMethod
    workload_identities: tuple[Digest, ...] = ()
    plot_styles: dict[str, PreconditionerPlotStyle] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require the presentation to describe this workload's own method."""
        if self.presentation.prepared.method_id != self.method:
            raise ValueError(
                f"presentation belongs to method {self.presentation.prepared.method_id!r}, "
                f"not {self.method!r}"
            )
        planned_ids = {tag.system_id for tag in self.planned}
        for system in self.presentation.systems:
            if system.system_id not in planned_ids:
                raise ValueError(
                    f"realization of system {system.system_id!r} which this workload did not plan"
                )


@dataclass(frozen=True)
class ComparisonPaths:
    """Resolved paths for a single comparison run.

    Attributes:
        matrix: Path to system matrix file (.txt or .npy).
        rhs: Path to right-hand side vector file (.txt or .npy).
        output: Root output directory for comparison results.
        figures: Directory for diagnostic plots.
    """

    matrix: Path
    rhs: Path
    output: Path
    figures: Path


@dataclass(frozen=True)
class LinearSystem:
    """Loaded and validated linear system (A, b) pair.

    Attributes:
        matrix: System matrix A in Ax=b (shape: n x n).
        rhs: Right-hand side vector b in Ax=b (shape: n,).
    """

    matrix: torch.Tensor
    rhs: torch.Tensor


@dataclass(frozen=True)
class PreconditionerComparisonEntry:
    """Outcome of comparing one preconditioner: solve result and plot metadata.

    Attributes:
        name: Preconditioner config name.
        result: CG solver outcome for this preconditioner.
        presentation: The validated scientific presentation of the method; the plot
            sink renders its legend text.
        family: Plot-style family key (see ``preconditioner_family.preconditioner_family``),
            driving shared linestyle across same-family preconditioners.
        color_key: Optional plot color-axis key (e.g. a POD-2G fit dataset).
            Entries sharing a key share a color; ``None`` falls back to ``family``.
        marker_key: Optional plot marker-axis key (e.g. a POD-2G weighting
            scheme). Entries sharing a key share a marker; ``None`` falls back
            to ``family``.
    """

    name: str
    result: CGComparisonResult
    presentation: PreparedMethodPresentation
    family: PreconditionerFamilyKey
    color_key: str | None = None
    marker_key: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonPresentation:
    """Everything a comparison's figures need to name what they show.

    ``methods`` is the validated cohort built once at preflight; ``title`` names what the
    comparison ran on; ``notation`` is the case's symbol policy. Nothing here is text:
    the plot sinks render it when they draw.
    """

    methods: PreparedComparisonPresentation
    title: ComparisonTitle
    notation: NotationPolicy


@dataclass(frozen=True)
class ResolvedComparisonInput:
    """Resolved comparison system and provenance used for execution and tracking."""

    matrix: np.ndarray
    rhs: np.ndarray
    matrix_dataset_id: str
    matrix_index: int
    matrix_provenance: MatrixProvenance
    """Whether this matrix is the dataset's one shared matrix or a held-out
    matrix of a multi-matrix dataset: the claim the planner proved for the
    system, carried through so no resolved input is ever without one."""
    lhs: np.ndarray | None = None
    rhs_dataset_id: str | None = None
    rhs_sample_index: int | None = None
    row_kind: RowKind | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    rhs_source_params: dict[str, object] | None = None
    matrix_normalization: DatasetNormalization | None = None
    """Persisted normalization metadata for the dataset the matrix was loaded
    from, or None when the matrix came from a raw/external file with no
    manifest. Drives whether comparison-time code treats the matrix as
    already-normalized (skip/derive) or genuinely raw (compute fresh)."""

    @property
    def rhs_source_type(self) -> str:
        match self.rhs_source_kind:
            case None:
                return "unknown"
            case kind:
                return "generated" if RHS_SOURCE_MODEL_BY_KIND[kind].generated else str(kind)


@dataclass(frozen=True)
class ComparisonParams:
    """Runtime parameters for comparison execution.

    Attributes:
        force: Rerun every comparison even if a matching one (same derived
            identity: config, data, preconditioners and checkpoint content)
            already completed successfully.
    """

    force: bool = False


@dataclass(frozen=True, slots=True)
class MatrixRef:
    """One matrix of a dataset: its id and the storage row to read it from.

    `matrix_id` is the sample id the generator bound rows to; `row` is the
    physical row passed to matrix and parameter-array readers. The two only
    coincide by accident, so neither is derived from the other.
    """

    matrix_id: MatrixId
    row: int


@dataclass(frozen=True, slots=True)
class PlannedSystem:
    """One concrete linear system a comparison will solve, fully bound to its data.

    `entry` is a concrete registry entry with a unique id and the RHS seed
    derived for this system, so everything downstream that already consumes an
    entry (naming, method, assignments, RHS source) keeps working unchanged.

    `rhs_row` is the held-out dataset row used as RHS for the `dataset` kind
    and `None` for every other kind. `rhs_ordinal` is this system's position
    among the `rhs_count` draws planned on its matrix (0 for a single draw).

    `provenance` is what the planner proved about the system's matrix: the
    dataset's one shared matrix, or a held-out matrix of a multi-matrix dataset.
    """

    entry: ComparisonRegistryEntry
    matrix: MatrixRef
    rhs_row: int | None
    rhs_ordinal: int
    provenance: SharedSingleMatrix | HeldOutMatrix


@dataclass(frozen=True, slots=True)
class UndescribedComparison:
    """A comparison that failed before its dataset and methods were resolved.

    Nothing scientific is known yet, so it has no title; the failure reason says why.
    """


@dataclass(frozen=True, slots=True)
class ComparisonCompleted:
    """A comparison that executed and produced a new result.

    ``run`` is the typed name of its MLflow run; the CLI and the logs render it, so
    they show the text MLflow shows.
    """

    comparison_id: str
    run: NamedRun
    payload: ComparisonResult
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonReused:
    """A comparison satisfied by an already-complete reusable run."""

    comparison_id: str
    run: NamedRun
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonEntryFailed:
    """One comparison entry that failed without blocking sibling entries."""

    comparison_id: str
    run: NamedRun | UndescribedComparison
    reason: str
    warnings: tuple[str, ...] = ()


type ComparisonOutcome = ComparisonCompleted | ComparisonReused | ComparisonEntryFailed
"""The closed set of per-entry comparison outcomes."""


def is_comparison_outcome(value: object) -> TypeIs[ComparisonOutcome]:
    """Narrow an orchestration item to one of the closed terminal outcomes."""
    return isinstance(value, (ComparisonCompleted, ComparisonReused, ComparisonEntryFailed))
