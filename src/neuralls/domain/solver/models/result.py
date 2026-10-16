"""Result dataclasses for solver comparison reporting.

This module defines result containers for comparison workflows. Per-solve
diagnostics (what torchalg calls ``SolverResult``, and the ``IterationContext``
passed to its flexible preconditioners) belong to torchalg, not here.

Design:
    - CGComparisonResult: Extended result for comparison experiments

Theory:
    Results contain both numerical outcomes (solution, residual) and diagnostic
    information (iterations, convergence status, breakdown flags). This separation
    enables post-analysis and debugging.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neuralls.shared.types import CostProvenance, DatasetId, GroupId

if TYPE_CHECKING:
    from neuralls.shared.types import ComparisonRhsSourceKind

    from .config import ComparisonGeneral


@dataclass(frozen=True, slots=True)
class StageCost:
    """One pipeline stage's wall-clock cost, tagged with where the number came from.

    See shared.types.CostProvenance for what MEASURED/HISTORICAL/UNAVAILABLE mean.
    """

    wall_time_seconds: float
    peak_memory_bytes: int | None
    provenance: CostProvenance


@dataclass(slots=True)
class CGComparisonResult:
    """Result container for CG comparison experiments.

    Extended result class for comparison workflows that test multiple
    warm start and preconditioner combinations.

    Attributes:
        x: Final solution vector.
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm at termination.
        residual_abs: Absolute residual norm at termination.
        residual_history_rel: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        warm_start: Warm start strategy name.
        preconditioner: Preconditioner strategy name.
        step_helper: Step helper strategy name (e.g., 'none', 'neural').
        initial_guess: Initial guess vector x_0.
        exact_error: Error ||x - x_exact|| if exact solution known.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        error_history_a_rel: Relative energy-norm errors across iterations.
        error_bound_a_rel: Golub-Meurant lower bound on the relative energy-norm
            error, used when the exact value is unavailable.
        helper_iterations: Iterations where step helper was invoked.
        helper_norms: Residual norms after step helper application.
        error: Error message if solver failed.
        generation_cost: Dataset-generation cost feeding this preconditioner's
            fit/train step, or `None` if generation doesn't apply.
        setup_cost: Wall-clock seconds to construct and `.setup()` this
            preconditioner. Always measured — every `torchalg.Preconditioner`
            requires an explicit `setup(matrix)` call before `apply()`, so
            this cost is deterministic and present for every result.
        solve_time_seconds: Wall time for the CG/PCG/FCG solve itself.
            `None` if never measured.
        solve_peak_memory_bytes: Peak memory during the solve. `None` if
            never measured.

    Example:
        >>> result = CGComparisonResult(
        ...     x=np.zeros(10),
        ...     converged=True,
        ...     iterations=5,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     residual_history_rel=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     residual_history_abs=[1.0, 0.1, 0.01, 0.001, 1e-8],
        ...     preconditioner="jacobi",
        ...     initial_guess=np.zeros(10),
        ...     exact_error=None,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> result.preconditioner
        'jacobi'
    """

    x: np.ndarray
    """Final solution vector."""

    converged: bool
    """Whether convergence criterion satisfied."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm at termination."""

    residual_abs: float
    """Absolute residual norm at termination."""

    residual_history_rel: list[float]
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: list[float]
    """Absolute residual norms ||r_k|| across iterations."""

    preconditioner: str
    """Preconditioner result identifier (e.g., 'identity', 'jacobi', 'neural')."""

    initial_guess: np.ndarray
    """Initial guess vector x_0."""

    exact_error: float | None
    """Error ||x - x_exact|| if exact solution known."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred."""

    error: str | None = None
    """Error message if solver failed."""

    error_history_a_rel: list[float] | None = None
    """Energy-norm errors ||e_k||_A / ||e_0||_A across iterations (None if unavailable)."""

    error_bound_a_rel: list[float] | None = None
    """Golub-Meurant lower bound on ||e_k||_A / ||e_0||_A, used when the exact
    energy error is unavailable (no reference solution). None if neither is available."""

    generation_cost: StageCost | None = None
    """Dataset-generation cost feeding this preconditioner's fit/train step, or None if
    generation doesn't apply (classical/geometric AMG, standard/Jacobi/IC0 preconditioners
    have no training data at all). When present it is always HISTORICAL (read from the
    dataset's manifest) or UNAVAILABLE — a comparison run never generates a dataset itself,
    that's a separate, earlier pipeline stage (see composition/generation's own docs)."""

    setup_cost: float = 0.0
    """Wall-clock seconds to construct and `.setup()` this preconditioner. Always
    measured; 0.0 only for a placeholder result where no build was ever attempted
    (see comparison_run.py::_breakdown_result)."""

    setup_peak_memory_bytes: int | None = None
    """Peak memory while constructing and `.setup()`-ing this preconditioner.
    None only for a placeholder result where no build was ever attempted."""

    solve_time_seconds: float | None = None
    """Wall time for the CG/PCG/FCG solve. None if not measured."""

    solve_peak_memory_bytes: int | None = None
    """Peak memory during the solve. None if not measured."""

    setup_time_samples: tuple[float, ...] = ()
    """Every timed build+setup execution behind `setup_cost` (its statistic), in run
    order. Empty for a cache hit (the cost was copied forward) or a placeholder."""

    solve_time_samples: tuple[float, ...] = ()
    """Every timed solve execution behind `solve_time_seconds` (its statistic), in run
    order. Empty when the solve was not measured."""

    @property
    def total_time_seconds(self) -> float | None:
        """Setup + generation + solve wall time, or None if nothing could be summed.

        ``setup_cost`` is always a real measured number, so it is always
        included. ``generation_cost`` excludes an UNAVAILABLE-provenance
        component — an unavailable cost is a real, unrecorded cost, and
        counting it as 0.0 would silently understate the total.
        """
        parts = [self.setup_cost]
        if (
            self.generation_cost is not None
            and self.generation_cost.provenance is not CostProvenance.UNAVAILABLE
        ):
            parts.append(self.generation_cost.wall_time_seconds)
        if self.solve_time_seconds is not None:
            parts.append(self.solve_time_seconds)
        return sum(parts) if parts else None

    @property
    def has_unavailable_cost(self) -> bool:
        """True when generation cost exists but couldn't be resolved to a real number.

        ``setup_cost`` can no longer be UNAVAILABLE — every preconditioner's
        `.setup()` call is always measured.
        """
        return self.generation_cost is not None and (
            self.generation_cost.provenance is CostProvenance.UNAVAILABLE
        )

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time divided by iteration count — the averaged per-iteration solve cost.

        Does NOT include generation/setup overhead; see total_cost_per_iteration_seconds
        for amortized full cost.
        """
        if self.solve_time_seconds is None or self.iterations <= 0:
            return None
        return self.solve_time_seconds / self.iterations

    @property
    def total_cost_per_iteration_seconds(self) -> float | None:
        """Full pipeline cost amortized per iteration: (generation + setup + solve) / iterations.

        Includes one-time setup and generation costs spread across all iterations,
        so its ratio to `avg_iteration_time_seconds` shows how much one-time cost
        inflates a method's effective iteration cost. A diagnostic, not a ranking
        criterion: it depends on the stopping tolerance through the iteration count.
        Rank by `total_time_seconds` (time-to-solution) instead.
        """
        total = self.total_time_seconds
        if total is None or self.iterations <= 0:
            return None
        return total / self.iterations

    @property
    def peak_memory_bytes(self) -> int | None:
        """Max of setup and solve peak memory, or None if neither was measured."""
        values = [
            v for v in (self.setup_peak_memory_bytes, self.solve_peak_memory_bytes) if v is not None
        ]
        return max(values) if values else None


@dataclass(frozen=True, slots=True)
class CostFieldSpec:
    """One reportable scalar cost, shared by every consumer that reports costs.

    `field_name` is the stable key written to `comparison.toml`; `extractor`
    is the same accessor the cost bar-charts use. Titles and filenames are
    presentation and live with the plots (`composition/comparison/_plots.py`),
    keyed by `field_name` — a test keeps the two in step.
    """

    field_name: str
    extractor: Callable[[CGComparisonResult], float | int | None]


COST_FIELD_REGISTRY: tuple[CostFieldSpec, ...] = (
    CostFieldSpec(
        field_name="generation_time_seconds",
        extractor=lambda r: (
            r.generation_cost.wall_time_seconds if r.generation_cost is not None else None
        ),
    ),
    CostFieldSpec(
        field_name="setup_time_seconds",
        extractor=lambda r: r.setup_cost,
    ),
    CostFieldSpec(
        field_name="solve_time_seconds",
        extractor=lambda r: r.solve_time_seconds,
    ),
    CostFieldSpec(
        field_name="avg_iteration_time_seconds",
        extractor=lambda r: r.avg_iteration_time_seconds,
    ),
    CostFieldSpec(
        field_name="total_cost_per_iteration_seconds",
        extractor=lambda r: r.total_cost_per_iteration_seconds,
    ),
    CostFieldSpec(
        field_name="peak_memory_bytes",
        extractor=lambda r: r.peak_memory_bytes,
    ),
)
"""Every scalar cost reported per preconditioner per comparison run.

Iterated by `_plots.py`'s bar-chart generation and by
`_save_comparison_toml` (`platform/reporting/artifacts.py`) — the one list
both read from.
"""


@dataclass(frozen=True, slots=True)
class SetupSharingPolicy:
    """Whether one preconditioner's setup/generation cost amortizes across sibling systems.

    Two independent facts, not a single state — a preconditioner can share
    one without the other (e.g. a checkpointless POD-2G AMG preconditioner
    in a fixed-matrix sweep: its `AMGPreconditioner.setup(matrix)` is
    shareable because the matrix never changes, but its inline-fit
    generation cost reruns every comparison regardless).

    Attributes:
        share_setup: True iff the preconditioner's `setup()` is provably
            reusable across the sibling systems being aggregated — either
            because it is a `torchalg.StaticSetupPreconditioner` (setup
            never reads the matrix at all) or because every sibling resolves
            to the same matrix (an RHS-only sweep).
        share_generation: True iff `GenerationRepeatability.ONE_TIME` for
            this preconditioner's config — its fit/train step already
            happened once, upstream, and is not being repeated here.
    """

    share_setup: bool
    share_generation: bool


@dataclass(frozen=True, slots=True)
class ProjectedComparisonResult:
    """Scaling-law projection for one preconditioner, aggregated across sibling systems.

    Built from the `n_systems` sibling `CGComparisonResult`s a comparison
    sweep actually ran — never from N systems actually
    executed. `projected_total_seconds(n)`/`projected_avg_per_system_seconds(n)`
    extrapolate the measured constants to an arbitrary N, purely for
    plotting, so the one-time cost's contribution can be shown shrinking
    at scale without ever running N real systems.

    Each of `setup`/`generation` is amortized or repeated independently per
    `setup_sharing` — not a single combined choice, since the two costs can
    have different sharing policies for the same preconditioner (see
    `SetupSharingPolicy`).

    Attributes:
        preconditioner_name: Matches `CGComparisonResult.preconditioner`.
        n_systems: Number of sibling systems actually aggregated.
        setup_sharing: Which cost components amortize vs. repeat per system.
        setup_cost_total_seconds: Sum of `setup_cost` across all `n_systems`
            measured results. If `setup_sharing.share_setup`, every sibling
            already carries the same real one-time value (the caching layer
            copies it forward rather than reporting 0.0 on a cache hit), so
            dividing by `n_systems` recovers that single value exactly.
        generation_cost_total_seconds: Mean generation wall time over the
            results whose cost is available, scaled to `n_systems` (so that
            dividing by `n_systems` recovers that mean), or `None` if
            generation does not apply or is unavailable for every result.
        total_solve_time_seconds: Mean `solve_time_seconds` over the results
            that measured one, scaled to `n_systems` — always repeats per
            system; solving is never a one-time cost.
        total_iterations: Sum of `iterations` across all `n_systems`
            measured results.
        solve_time_std_seconds: Sample standard deviation (ddof=1) of the
            per-system solve time — the spread behind `avg_solve_time_per_system_seconds`,
            as the learned-preconditioner literature reports (mean ± std over
            the test set). `None` for a single system.
        iterations_std: Sample standard deviation of the per-system iteration
            count, or `None` for a single system.
    """

    preconditioner_name: str
    n_systems: int
    setup_sharing: SetupSharingPolicy
    setup_cost_total_seconds: float
    generation_cost_total_seconds: float | None
    total_solve_time_seconds: float
    total_iterations: int
    solve_time_std_seconds: float | None = None
    iterations_std: float | None = None

    @property
    def avg_setup_cost_seconds(self) -> float:
        """Setup cost per measured system — the one-time value itself when shared."""
        return self.setup_cost_total_seconds / self.n_systems

    @property
    def avg_generation_cost_seconds(self) -> float | None:
        """Generation cost per measured system, or `None` if not applicable."""
        if self.generation_cost_total_seconds is None:
            return None
        return self.generation_cost_total_seconds / self.n_systems

    @property
    def avg_solve_time_per_system_seconds(self) -> float:
        """Solve time per measured system — always a recurring cost."""
        return self.total_solve_time_seconds / self.n_systems

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time divided by total iteration count across all measured systems."""
        if self.total_iterations <= 0:
            return None
        return self.total_solve_time_seconds / self.total_iterations

    def projected_total_seconds(self, n: int) -> float:
        """Projected total wall time to solve `n` systems with this preconditioner.

        Each cost component amortizes (contributes its one-time value once,
        regardless of `n`) or repeats (contributes `n *` its per-system
        average) according to its own entry in `setup_sharing` — solve time
        always repeats.

        Args:
            n: Number of systems to project to. Never actually run; this is
                a closed-form extrapolation from the `n_systems` real
                measurements, for plotting only.

        Returns:
            float: Projected total wall-clock seconds.
        """
        setup_term = (
            self.avg_setup_cost_seconds
            if self.setup_sharing.share_setup
            else n * self.avg_setup_cost_seconds
        )
        generation_term = 0.0
        avg_generation = self.avg_generation_cost_seconds
        if avg_generation is not None:
            generation_term = (
                avg_generation if self.setup_sharing.share_generation else n * avg_generation
            )
        return setup_term + generation_term + n * self.avg_solve_time_per_system_seconds

    def projected_avg_per_system_seconds(self, n: int) -> float:
        """Projected total cost amortized per system — flattens toward the solve floor as `n` grows.

        Args:
            n: Number of systems to project to (must be positive).

        Returns:
            float: `projected_total_seconds(n) / n`.
        """
        return self.projected_total_seconds(n) / n


@dataclass(frozen=True, slots=True)
class FineLabel:
    """One dataset and one RHS recipe: a homogeneous problem family."""

    dataset: DatasetId
    recipe: str


@dataclass(frozen=True, slots=True)
class MediumLabel:
    """One dataset, pooling every RHS recipe listed in `mixture_of`."""

    dataset: DatasetId
    mixture_of: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GlobalLabel:
    """Every dataset and recipe of the case, pooling the recipes in `mixture_of`."""

    mixture_of: tuple[str, ...]


type GroupLabel = FineLabel | MediumLabel | GlobalLabel
"""Identifies the keys of one report group. Medium and global groups are
mixtures: their spread mostly reflects recipes differing, so titles must say so."""


@dataclass(frozen=True, slots=True)
class PreconditionerPlotStyle:
    """How one preconditioner is drawn: its legend label and style-axis keys.

    Carried from the run that built the preconditioner to every later plot of
    it, so a group report is styled exactly like the per-run convergence plots.

    Attributes:
        label: Scientific legend label built from the constructed preconditioner.
        family: Plot-style family; same-family preconditioners share a linestyle.
        color_key: Optional colour-axis key; ``None`` falls back to ``family``.
        marker_key: Optional marker-axis key; ``None`` falls back to ``family``.
    """

    label: str
    family: StrEnum
    color_key: str | None = None
    marker_key: str | None = None


@dataclass(frozen=True, slots=True)
class PlotStyleMaps:
    """Per-name mappings in the shape the plot functions take.

    Keys absent from ``color_keys``/``marker_keys`` fall back to the family.
    """

    labels: Mapping[str, str]
    families: Mapping[str, Hashable]
    color_keys: Mapping[str, Hashable]
    marker_keys: Mapping[str, Hashable]


def plot_style_maps(styles: Mapping[str, PreconditionerPlotStyle]) -> PlotStyleMaps:
    """Split per-preconditioner styles into the mappings plots consume.

    The one derivation shared by per-run and per-group plots, so a
    preconditioner keeps the same colour, marker and linestyle everywhere.
    """
    return PlotStyleMaps(
        labels={name: style.label for name, style in styles.items()},
        families={name: style.family for name, style in styles.items()},
        color_keys={
            name: style.color_key for name, style in styles.items() if style.color_key is not None
        },
        marker_keys={
            name: style.marker_key for name, style in styles.items() if style.marker_key is not None
        },
    )


@dataclass(frozen=True, slots=True)
class GroupPresentation:
    """Everything a sink needs, besides the numbers, to draw one report group.

    Attributes:
        subject: What the group is, for titles (e.g. ``"ds, gaussian"`` or
            ``"ds, mixture of gaussian, sparse[3 nnz]"``).
        subtitle: Level, keys, recipe and system count, as one plot subtitle.
        plot_styles: Style of each preconditioner of the group, by name.
    """

    subject: str
    subtitle: str
    plot_styles: Mapping[str, PreconditionerPlotStyle]


def render_group_id(label: GroupLabel) -> GroupId:
    """The one stable, level-unique string form of a label (filename stem, metric key segment).

    The level prefix keeps a dataset named like another level's id from
    colliding with it.
    """
    match label:
        case FineLabel(dataset=dataset, recipe=recipe):
            return GroupId(f"fine/{dataset}/{_slug(recipe)}")
        case MediumLabel(dataset=dataset):
            return GroupId(f"medium/{dataset}")
        case GlobalLabel():
            return GroupId("global")


def _slug(text: str) -> str:
    """Filesystem-safe form of a recipe label, keeping parameters (``ω`` becomes ``omega``)."""
    ascii_text = text.replace("ω", "omega")
    return _SLUG_SEPARATORS.sub("_", ascii_text).strip("_")


_SLUG_SEPARATORS = re.compile(r"[^A-Za-z0-9.]+")


@dataclass(frozen=True, slots=True)
class GroupAggregated:
    """One or more comparison group members contributed a real result.

    Attributes:
        group_id: Stable, unique rendering of `label`, used for filenames and
            metric keys.
        label: Which report level and keys the projection belongs to; mixture
            levels name the recipes they pool.
        projected: One `ProjectedComparisonResult` per preconditioner name.
        n_aggregated: Systems actually contributing a result this run.
        n_matrices: Distinct matrices those systems solve on.
        n_in_group: Total planned systems of the group; `n_aggregated < n_in_group`
            means the rest were cache hits (reused from an earlier run,
            carrying no result to aggregate this time) or failed.
        time_to_solution_by_system: For each preconditioner name, each
            contributing system's time-to-solution in seconds, keyed by
            system id in plan order — the individual points behind the
            projection's means. A system whose time is incomparable (an
            unavailable cost) is omitted for that preconditioner.
    """

    group_id: GroupId
    label: GroupLabel
    projected: Mapping[str, ProjectedComparisonResult]
    n_aggregated: int
    n_matrices: int
    n_in_group: int
    time_to_solution_by_system: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroupSkippedAllCacheHit:
    """No system of this report group produced a result this run.

    Every member was a cache hit (or failed). There is no `CGComparisonResult`
    to aggregate — not the same as zero cost, just nothing measured this run.
    """

    group_id: GroupId
    label: GroupLabel
    n_in_group: int


type ComparisonGroupOutcome = GroupAggregated | GroupSkippedAllCacheHit
"""The outcome of attempting to aggregate one comparison group's results.

Two variants because there are exactly two cases that call for different
handling: there is a projection to act on, or there is nothing measured this
run. A group of one system is still a projection (without a spread). Whether every
member contributed (`n_aggregated == n_in_group`) or only some did is a
fact carried as data on `GroupAggregated`, not another variant — both
cases are plotted/logged/uploaded identically, differing only in whether a
warning fires.
"""


@dataclass(frozen=True, slots=True)
class PlotPaths:
    """Paths to generated diagnostic plots from a comparison workflow.

    Args:
        convergence: Convergence curve plot.
        parity: Parity plot (predicted vs true).
        residuals: Residual history plot.
        iterations_barplot: Horizontal bar chart of iteration counts.
        error_convergence: Relative energy-norm error convergence plot.
        generation_time_barplot: Horizontal bar chart of dataset-generation times.
        setup_time_barplot: Horizontal bar chart of preconditioner setup times.
        solve_time_barplot: Horizontal bar chart of CG solve times.
        avg_iteration_time_barplot: Horizontal bar chart of solve time per iteration.
        total_cost_per_iteration_barplot: Horizontal bar chart of total amortized cost
            (generation + setup + solve) per iteration.
        peak_memory_barplot: Horizontal bar chart of peak memory usage.
        time_breakdown_barplot: Stacked horizontal bar chart of generation vs.
            setup vs. solve time per method.
        work_precision: Work-precision diagram (final precision vs. total cost).
        residual_vs_time: Residual vs wall time (generation + setup + solve).
        error_vs_time: Energy-norm error vs wall time (when available).
        residual_vs_solve_time: Residual vs solve time only; every curve starts at t=0.
        error_vs_solve_time: Energy-norm error vs solve time only (when available).
    """

    convergence: Path | None = None
    parity: Path | None = None
    residuals: Path | None = None
    iterations_barplot: Path | None = None
    error_convergence: Path | None = None
    generation_time_barplot: Path | None = None
    setup_time_barplot: Path | None = None
    solve_time_barplot: Path | None = None
    avg_iteration_time_barplot: Path | None = None
    total_cost_per_iteration_barplot: Path | None = None
    peak_memory_barplot: Path | None = None
    time_breakdown_barplot: Path | None = None
    work_precision: Path | None = None
    residual_vs_time: Path | None = None
    error_vs_time: Path | None = None
    residual_vs_solve_time: Path | None = None
    error_vs_solve_time: Path | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Path] | None) -> PlotPaths:
        """Build typed plot paths from a loose mapping.

        Args:
            mapping: Dict of plot type → path.

        Returns:
            PlotPaths with matched fields populated.
        """
        if mapping is None:
            return cls()
        return cls(
            convergence=mapping.get("convergence"),
            parity=mapping.get("parity"),
            residuals=mapping.get("residuals"),
            iterations_barplot=mapping.get("iterations_barplot"),
            error_convergence=mapping.get("error_convergence"),
            generation_time_barplot=mapping.get("generation_time_barplot"),
            setup_time_barplot=mapping.get("setup_time_barplot"),
            solve_time_barplot=mapping.get("solve_time_barplot"),
            avg_iteration_time_barplot=mapping.get("avg_iteration_time_barplot"),
            total_cost_per_iteration_barplot=mapping.get("total_cost_per_iteration_barplot"),
            peak_memory_barplot=mapping.get("peak_memory_barplot"),
            time_breakdown_barplot=mapping.get("time_breakdown_barplot"),
            work_precision=mapping.get("work_precision"),
            residual_vs_time=mapping.get("residual_vs_time"),
            error_vs_time=mapping.get("error_vs_time"),
            residual_vs_solve_time=mapping.get("residual_vs_solve_time"),
            error_vs_solve_time=mapping.get("error_vs_solve_time"),
        )

    def to_mapping(self) -> dict[str, Path]:
        """Return only populated plot paths.

        Returns:
            Dict of plot type → path (omits None entries).
        """
        values = {
            "convergence": self.convergence,
            "parity": self.parity,
            "residuals": self.residuals,
            "iterations_barplot": self.iterations_barplot,
            "error_convergence": self.error_convergence,
            "generation_time_barplot": self.generation_time_barplot,
            "setup_time_barplot": self.setup_time_barplot,
            "solve_time_barplot": self.solve_time_barplot,
            "avg_iteration_time_barplot": self.avg_iteration_time_barplot,
            "total_cost_per_iteration_barplot": self.total_cost_per_iteration_barplot,
            "peak_memory_barplot": self.peak_memory_barplot,
            "time_breakdown_barplot": self.time_breakdown_barplot,
            "work_precision": self.work_precision,
            "residual_vs_time": self.residual_vs_time,
            "error_vs_time": self.error_vs_time,
            "residual_vs_solve_time": self.residual_vs_solve_time,
            "error_vs_solve_time": self.error_vs_solve_time,
        }
        return {key: path for key, path in values.items() if path is not None}


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """A single ranked entry from a solver comparison.

    Args:
        label: Preconditioner/solver label.
        iterations: Iterations to converge.
        residual: Final relative residual.
        residual_abs: Final absolute residual.
        breakdown: Whether breakdown occurred.
        total_time_seconds: Time-to-solution (generation + setup + solve), the
            ranking key; `None` when it cannot be compared fairly (an
            unavailable-provenance cost), which ranks after every timed entry.
    """

    label: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool
    total_time_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ComparisonRecommendations:
    """Ranked recommendations from a comparison run.

    Args:
        ranked: All converged entries sorted by time-to-solution (best first).
        overall_best: The single best entry, or None if none converged.
    """

    ranked: tuple[RankedRecommendation, ...] = ()
    overall_best: RankedRecommendation | None = None


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Typed result from compare_preconditioners workflow.

    Args:
        results: Raw CG comparison results keyed by preconditioner name.
        summary: Formatted text summary of results.
        solver_params: Solver parameters and data context used.
        plot_paths: Paths to generated diagnostic plots.
        preconditioners: Preconditioner names tested.
        recommendations: Ranked recommendations.
        output_dir: Root output directory for comparison artifacts.
        matrix_shape: Shape of the system matrix A, e.g. ``(n, n)``.
        rhs_shape: Shape of the right-hand side vector b, e.g. ``(n,)``.
        rhs_source_kind: Provenance of the RHS (e.g. gaussian, sparse, raw_lhs,
            raw_rhs, dataset) — a matrix may be paired with several different
            RHS sources across sibling ``[[comparisons]]`` entries, each its
            own MLflow run, so this disambiguates which one a given run used.
        plot_styles: How each preconditioner is drawn, by name, so later
            group-level plots reuse the per-run styling.
    """

    results: dict[str, Any]
    summary: str
    solver_params: ComparisonGeneral
    plot_paths: PlotPaths = field(default_factory=PlotPaths)
    preconditioners: tuple[str, ...] = ()
    recommendations: ComparisonRecommendations = field(default_factory=ComparisonRecommendations)
    output_dir: Path | None = None
    matrix_shape: tuple[int, int] | None = None
    rhs_shape: tuple[int, ...] | None = None
    rhs_source_kind: ComparisonRhsSourceKind | None = None
    plot_styles: Mapping[str, PreconditionerPlotStyle] = field(default_factory=dict)
