"""MLflow setup helpers for comparison workflows."""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import assert_never

import mlflow
import polars as pl
from mlflow.tracking import MlflowClient

from neuralls.domain.benchmarking.costs import amortized_online_seconds, solve_seconds
from neuralls.domain.benchmarking.selection import (
    ConfiguredReference,
    NoReferenceConfigured,
    ReferenceSelection,
)
from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
    PreparedMethodPresentation,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import (
    MeasuredPeakMemory,
    SetupFailed,
    SolveFailed,
)
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.figure_scope import (
    AllDatasetsScope,
    DatasetScope,
    MethodScope,
    RecipeScope,
)
from neuralls.domain.solver.models.group_outcomes import (
    RecipeProfileReport,
    WorkloadMixtureReport,
)
from neuralls.domain.solver.models.group_presentation import GroupPresentation, SolverTimingContext
from neuralls.domain.solver.models.plot_style import plot_style_maps
from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    ReportScope,
    System,
    parent_scope,
    scope_of_label,
)
from neuralls.domain.solver.models.statistics import MethodComparison, PairedComparison
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.domain.solver.reporting.report_levels import (
    ReportLevelName,
    level_name,
    level_of_scope,
)
from neuralls.platform.config.models.experiments import ExperimentNamesConfig
from neuralls.platform.config.resolution import MlflowPaths
from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    MethodBenchmarkArtifacts,
    write_method_benchmark_artifacts,
)
from neuralls.platform.reporting.figure_card import (
    FigureCard,
    build_figure_card,
    card_name,
    write_figure_card,
)
from neuralls.platform.reporting.figure_index import (
    FIGURE_INDEX_FILE,
    ArtifactLocation,
    index_of,
    verify_figure_records,
    write_figure_index,
)
from neuralls.platform.reporting.figure_metadata import (
    FailureTreatment,
    FigureKind,
    FigureMetadata,
    FigureMethod,
    FigurePlan,
    FigureSlugs,
    NoTable,
    SiblingTable,
    StagedFigure,
    TimingScenario,
    collision_key,
    make_plan,
    method_source,
)
from neuralls.platform.reporting.figure_table import build_group_table, write_group_table
from neuralls.platform.reporting.figure_text import (
    EditorialTitle,
    Legend,
    ScopeSubtitle,
    method_legend,
    scope_label,
    series_legend,
)
from neuralls.platform.reporting.outcome_plots import (
    GroupStyle,
    plot_aggregate_bars,
    plot_ecdf_profile,
    plot_online_time_bars,
    plot_paired_time_scatter,
    plot_performance_profile,
    plot_speed_ratio_cdf,
)
from neuralls.platform.reporting.plots import (
    plot_convergence_comparison,
    plot_error_convergence_comparison,
)
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.mlflow import ensure_experiment, sanitize_metric_key_segment
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.platform.tracking.session_artifacts import PARENT_RUN_TAG
from neuralls.shared.rhs_recipe import recipe_key
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SystemId


def setup_comparison_tracking(
    tracking_uri: str,
    artifact_location: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking for comparison runs."""
    resolved_experiment_name = experiment_name or ExperimentNamesConfig().comparison
    ensure_experiment(
        resolved_experiment_name,
        MlflowPaths(tracking_uri=tracking_uri, artifact_uri=artifact_location),
    )
    mlflow.set_experiment(resolved_experiment_name)


def parent_run_metric_key(metric: str, preconditioner_name: str) -> str:
    """Build the parent-run key for one preconditioner's metric.

    MLflow's metric namespace is flat strings only — no structured/tuple
    keys — so every per-preconditioner metric on the parent run is named
    ``"<metric>/<preconditioner>"`` (MLflow's own UI renders ``"/"`` as a
    tree, which is why this convention exists at all). This is the single
    place that builds such a key: callers (including tests asserting on
    logged metric names) should use this instead of hand-reconstructing the
    separator themselves, so the convention can't silently drift between
    production code and its tests.

    Args:
        metric: Bare metric name, e.g. ``"iterations"``, ``"setup_time_s"``.
        preconditioner_name: Preconditioner's result key (sanitized here).

    Returns:
        The full parent-run metric key, e.g. ``"iterations/jacobi"``.
    """
    return f"{metric}/{sanitize_metric_key_segment(preconditioner_name)}"


def log_convergence_metrics(convergence: Mapping[PreconditionerName, ConvergenceSummary]) -> None:
    """Log explicit planned, attempted, and converged counts."""
    for name, summary in convergence.items():
        mlflow.log_metric(parent_run_metric_key("n_planned_rhs", name), summary.n_planned_rhs)
        mlflow.log_metric(parent_run_metric_key("n_attempted_rhs", name), summary.n_attempted_rhs)
        mlflow.log_metric(parent_run_metric_key("n_converged", name), summary.n_converged)
        mlflow.log_metric(parent_run_metric_key("success_rate", name), summary.success_rate)
        if summary.iterations is not None:
            _log_optional_metrics(
                name,
                {
                    "iterations_mean": summary.iterations.mean,
                    "iterations_median": summary.iterations.median,
                    "iterations_std": summary.iterations.std,
                },
            )


def _log_optional_metrics(preconditioner_name: str, metrics: Mapping[str, float | None]) -> None:
    """Log each defined (non-`None`) metric under the method-namespaced key."""
    for metric, value in metrics.items():
        if value is not None:
            mlflow.log_metric(parent_run_metric_key(metric, preconditioner_name), value)


def _paired_metrics(paired: PairedComparison) -> dict[str, float | None]:
    """Failure counts, median ratio, and matrix-cluster interval."""
    interval = paired.median_speed_ratio_ci
    return {
        "n_both_solved": float(paired.n_both_solved),
        "n_reference_only": float(paired.n_reference_only),
        "n_method_only": float(paired.n_method_only),
        "speed_ratio_median": paired.median_speed_ratio,
        "speed_ratio_ci_low": interval.low if interval is not None else None,
        "speed_ratio_ci_high": interval.high if interval is not None else None,
    }


def log_comparison_metrics(comparisons: Mapping[PreconditionerName, MethodComparison]) -> None:
    """Log each method's failure-aware reference/method speed ratios."""
    for name, comparison in comparisons.items():
        _log_optional_metrics(name, _paired_metrics(comparison.paired))


class DuplicateFigureNameError(ValueError):
    """Two figures of one run directory would share a file name once made filesystem-safe."""


class UnknownStoredMethodError(LookupError):
    """A figure names a method the report has no stored run for."""


@dataclass(slots=True)
class MLflowReportSink:
    """Records a report as a tree of nested MLflow runs, one per aggregation scope.

    The run active at the first scope is the root (all matrix sets); every other scope gets
    a child run of its parent scope, created on first use, so parents always exist before
    children. ``notation`` is the case's symbol policy and ``datasets`` presents each
    dataset a scope names; run names and figures render both. ``session_run_id`` and
    ``methods`` identify the compare session and the stored method runs every figure is
    computed from.

    Every figure is uploaded with its card (same directory, same stem) into the run of its
    own scope; file stems only need to be unique inside one run directory. The root run
    also holds the method figures and, once finalized, the index of all figures.
    """

    notation: NotationPolicy
    session_run_id: str
    methods: Sequence[StoredMethod] = ()
    datasets: Mapping[DatasetId, DatasetPresentation] = field(default_factory=dict)
    _staging: tempfile.TemporaryDirectory[str] = field(
        default_factory=lambda: tempfile.TemporaryDirectory(prefix="neuralls-figures-")
    )
    _slugs: FigureSlugs = field(default_factory=FigureSlugs)
    _finalized: bool = False
    _run_ids: dict[ReportScope, str] = field(default_factory=dict)
    _taken: dict[tuple[str, str | None], set[str]] = field(default_factory=dict)
    _uploaded: list[ArtifactLocation] = field(default_factory=list)
    _cards: dict[ArtifactLocation, FigureCard] = field(default_factory=dict)

    def _scope_run_id(self, scope: ReportScope) -> str:
        """The run of ``scope``, creating it (and its ancestors) under its parent first."""
        known = self._run_ids.get(scope)
        if known is not None:
            return known
        parent = parent_scope(scope)
        active = mlflow.active_run()
        if parent is None or active is None:
            raise RuntimeError("a report needs its root run active")
        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        parent_id = (
            active.info.run_id if isinstance(parent, AllMatrixSets) else self._scope_run_id(parent)
        )
        run = client.create_run(
            active.info.experiment_id,
            tags={
                PARENT_RUN_TAG: parent_id,
                "phase": REPORT_SCOPE_PHASE,
                "report_level": level_of_scope(scope).value,
            },
            run_name=self._scope_name(scope),
        )
        for key, value in scope_params(scope).items():
            client.log_param(run.info.run_id, key, value)
        client.set_terminated(run.info.run_id, status="FINISHED")
        self._run_ids[scope] = run.info.run_id
        return run.info.run_id

    def _presentation_of(self, dataset: DatasetId) -> DatasetPresentation:
        found = self.datasets.get(dataset)
        if found is None:
            raise MissingMethodPresentationError(str(dataset))
        return found

    def _scope_name(self, scope: ReportScope) -> str:
        """The scientific text naming the run of ``scope``."""
        match scope:
            case AllMatrixSets():
                return scope_label(AllDatasetsScope(), self.notation)
            case MatrixSet(dataset=dataset):
                return scope_label(DatasetScope(self._presentation_of(dataset)), self.notation)
            case MatrixSetRhsType(dataset=dataset, recipe=recipe):
                return scope_label(
                    RecipeScope(self._presentation_of(dataset), recipe), self.notation
                )
            case System(matrix_id=matrix_id, rhs_ordinal=rhs_ordinal):
                return f"matrix {int(matrix_id)}, RHS {rhs_ordinal}"
            case unreachable:
                assert_never(unreachable)

    @contextmanager
    def _in_scope(self, scope: ReportScope) -> Generator[str]:
        """Log into the run of ``scope`` and yield its id; the root run is already active."""
        if isinstance(scope, AllMatrixSets):
            active = mlflow.active_run()
            if active is None:
                raise RuntimeError("a report needs its root run active")
            yield active.info.run_id
            return
        run_id = self._scope_run_id(scope)
        with mlflow.start_run(run_id=run_id, nested=True):
            yield run_id

    def _stored(self, name: PreconditionerName) -> StoredMethod:
        found = next((method for method in self.methods if method.name == name), None)
        if found is None:
            raise UnknownStoredMethodError(f"no stored run for method {name}")
        return found

    def _check_free(self, run_id: str, directory: str | None, filenames: Iterable[str]) -> None:
        """Reject names that collide with each other or with a figure already in the directory."""
        seen = set(self._taken.get((run_id, directory), ()))
        for filename in filenames:
            key = collision_key(filename)
            if key in seen:
                raise DuplicateFigureNameError(
                    f"duplicate or colliding figure filename: {filename}"
                )
            seen.add(key)

    def _upload_figure(
        self,
        run_id: str,
        directory: str | None,
        staged: StagedFigure,
        table: pl.DataFrame | None = None,
    ) -> None:
        """Upload a figure, its card and its plotted table side by side into the active run.

        ``table`` is required exactly for the kinds whose card names one.
        """
        self._check_free(run_id, directory, [staged.plan.filename])
        card = build_figure_card(
            staged.plan.filename,
            staged.plan.title,
            staged.plan.metadata,
            session_run_id=self.session_run_id,
            notation=self.notation,
        )
        card_path = staged.path.with_name(card_name(staged.plan.filename))
        write_figure_card(card_path, card)
        mlflow.log_artifact(str(staged.path), artifact_path=directory)
        mlflow.log_artifact(str(card_path), artifact_path=directory)
        prefix = f"{directory}/" if directory else ""
        match card.table:
            case SiblingTable(name=name):
                if table is None:
                    raise ValueError(f"figure {staged.plan.filename} needs its plotted table")
                table_path = staged.path.with_name(name)
                write_group_table(table_path, table)
                mlflow.log_artifact(str(table_path), artifact_path=directory)
                self._uploaded.append(ArtifactLocation(run_id, f"{prefix}{name}"))
            case NoTable():
                if table is not None:
                    raise ValueError(f"figure {staged.plan.filename} has no plotted table")
            case unreachable:
                assert_never(unreachable)
        self._taken.setdefault((run_id, directory), set()).add(collision_key(staged.plan.filename))
        figure_location = ArtifactLocation(run_id, f"{prefix}{staged.plan.filename}")
        card_location = ArtifactLocation(run_id, f"{prefix}{card_path.name}")
        self._uploaded.extend((figure_location, card_location))
        self._cards[card_location] = card

    def register_method_figures(self, *, datasets: Mapping[DatasetId, DatasetPresentation]) -> None:
        """Draw each method's convergence histories from its stored table and upload them."""
        with self._in_scope(AllMatrixSets()) as root_id:
            for method in self.methods:
                self._register_method(root_id, method, datasets)

    def _register_method(
        self, root_id: str, method: StoredMethod, datasets: Mapping[DatasetId, DatasetPresentation]
    ) -> None:
        tags = tuple(system.tag for system in method.systems)
        dataset_ids = tuple(dict.fromkeys(str(tag.dataset) for tag in tags))
        recipes = tuple(dict.fromkeys(tag.recipe for tag in tags))
        counts = tuple(sum(tag.recipe == recipe for tag in tags) for recipe in recipes)
        contexts = tuple(dict.fromkeys(system.solver for system in method.systems))
        name = str(method.name)
        scope = MethodScope(method.presentation.prepared.detailed)
        method_slug = self._slugs.slug(f"method:{name}", scope_label(scope, self.notation))
        n_matrices = len({(tag.dataset, tag.matrix_id) for tag in tags})
        recipe_weights = tuple(count / len(tags) for count in counts)
        artifacts = MethodBenchmarkArtifacts()
        figure_method = FigureMethod(method.name, scope.method, method.config_digest)
        figures_dir = Path(self._staging.name) / METHOD_FIGURE_ARTIFACT_DIR
        figures_dir.mkdir(exist_ok=True)
        for kind, filename, metric in (
            (
                FigureKind.RESIDUAL_HISTORY,
                f"residual_history_{method_slug}.png",
                "relative residual by iteration",
            ),
            (
                FigureKind.ENERGY_HISTORY,
                f"energy_history_{method_slug}.png",
                "relative energy error by iteration",
            ),
        ):
            metadata = FigureMetadata(
                kind=kind,
                metric_definition=metric,
                scenario=None,
                report_level=ReportLevelName.METHOD,
                scope=scope,
                dataset_ids=dataset_ids,
                recipes=recipes,
                n_matrices=n_matrices,
                n_planned_rhs=len(tags),
                reference_selection=NoReferenceConfigured(),
                failure_treatment=FailureTreatment.NONE,
                timing_policy="diagnostic histories; excluded from online timing",
                methods=(figure_method,),
                sources=(
                    method_source(method.run_id, artifacts.convergence, artifacts.solve_attempts),
                ),
                recipe_counts=counts,
                aggregation_rule="one diagnostic history per attempted RHS",
                recipe_weights=recipe_weights,
                solver_contexts=contexts,
            )
            plan = replace(make_plan(kind, metadata, group_slug=method_slug), filename=filename)
            self._check_free(root_id, METHOD_FIGURE_ARTIFACT_DIR, [filename])
            path = figures_dir / filename
            _draw_method_figure(path, kind, method, self.notation, datasets)
            self._upload_figure(root_id, METHOD_FIGURE_ARTIFACT_DIR, StagedFigure(plan, path))

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Log explicit counts and the supported failure-aware plots into the group's run."""
        if self._finalized:
            raise RuntimeError("cannot report after finalization")
        group_id = outcome.group_id
        scope = scope_of_label(outcome.label)
        with self._in_scope(scope) as run_id, tempfile.TemporaryDirectory() as tmp:
            log_convergence_metrics(outcome.convergence)
            if isinstance(outcome, RecipeProfileReport):
                log_comparison_metrics(outcome.comparisons)
            file_key = self._slugs.slug(
                f"group:{group_id}", scope_label(presentation.scope, self.notation)
            )
            plans = _figure_plans(
                outcome, presentation, file_key, self._group_methods(presentation)
            )
            self._check_free(run_id, GROUP_REPORT_ARTIFACT_DIR, [plan.filename for plan in plans])
            outcome_plots = _write_outcome_plots(
                outcome, _GroupStyler(presentation, self.notation), Path(tmp), plans
            )
            table = build_group_table(outcome.summaries)
            for artifact in outcome_plots:
                plan = next(plan for plan in plans if plan.filename == artifact.name)
                self._upload_figure(
                    run_id, GROUP_REPORT_ARTIFACT_DIR, StagedFigure(plan, artifact), table
                )

    def _group_methods(
        self, presentation: GroupPresentation
    ) -> dict[PreconditionerName, StoredMethod]:
        """The stored run of every method the group presents."""
        return {name: self._stored(name) for name in presentation.methods}

    def report_system(
        self, scope: System, metrics: Mapping[str, float], figures: Sequence[StagedFigure]
    ) -> None:
        """Log one system's per-method metrics and figures, with their cards, into its own run."""
        if self._finalized:
            raise RuntimeError("cannot report after finalization")
        with self._in_scope(scope) as run_id:
            mlflow.log_metrics(dict(metrics))
            self._check_free(run_id, None, [staged.plan.filename for staged in figures])
            for staged in figures:
                self._upload_figure(run_id, None, staged)

    def finalize(self) -> None:
        """Check every figure has its card, then upload the figure index to the root run."""
        if self._finalized:
            raise RuntimeError("figure index already finalized")
        index = index_of(self._cards)
        verify_figure_records(self._uploaded, self._cards, index)
        path = Path(self._staging.name) / FIGURE_INDEX_FILE
        write_figure_index(path, index)
        mlflow.log_artifact(str(path))
        self._finalized = True
        self._staging.cleanup()


@dataclass(frozen=True, slots=True)
class _FigureSpec:
    filename: str
    kind: FigureKind
    scenario: TimingScenario | None = None


def _figure_plans(
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    presentation: GroupPresentation,
    file_key: str,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> tuple[FigurePlan, ...]:
    """Build every figure's scientific context before drawing.

    ``stored`` holds the method run of each method ``presentation`` lists.
    """
    specs = _planned_figure_specs(outcome, file_key)
    plans: list[FigurePlan] = []
    for spec in specs:
        plan = make_plan(
            spec.kind,
            _figure_metadata(spec.kind, spec.scenario, outcome, presentation, stored),
            group_slug=file_key,
        )
        plans.append(replace(plan, filename=spec.filename))
    return tuple(plans)


def _planned_figure_specs(
    outcome: RecipeProfileReport | WorkloadMixtureReport, file_key: str
) -> tuple[_FigureSpec, ...]:
    aggregate = _FigureSpec(f"aggregate_bars_{file_key}.png", FigureKind.AGGREGATE_BARS)
    solve_bar = _FigureSpec(
        f"solve_time_bars_{file_key}.png", FigureKind.SOLVE_TIME, TimingScenario.SOLVE
    )
    online_bar = _FigureSpec(
        f"online_time_bars_{file_key}.png", FigureKind.ONLINE_TIME, TimingScenario.ONLINE
    )
    if isinstance(outcome, RecipeProfileReport):
        return (
            aggregate,
            solve_bar,
            online_bar,
            _FigureSpec(
                f"paired_time_scatter_{file_key}.png",
                FigureKind.PAIRED_TIME,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(
                f"speed_ratio_cdf_{file_key}.png",
                FigureKind.SPEED_RATIO,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(
                f"performance_profile_solve_{file_key}.png",
                FigureKind.PERFORMANCE_PROFILE,
                TimingScenario.SOLVE,
            ),
            _FigureSpec(
                f"performance_profile_online_{file_key}.png",
                FigureKind.PERFORMANCE_PROFILE,
                TimingScenario.ONLINE,
            ),
            _FigureSpec(f"ecdf_iterations_{file_key}.png", FigureKind.ECDF_ITERATIONS),
            _FigureSpec(
                f"ecdf_time_solve_{file_key}.png",
                FigureKind.ECDF_TIME,
                TimingScenario.SOLVE,
            ),
            _FigureSpec(
                f"ecdf_time_online_{file_key}.png",
                FigureKind.ECDF_TIME,
                TimingScenario.ONLINE,
            ),
        )
    return (aggregate, solve_bar, online_bar)


def _figure_metadata(
    kind: FigureKind,
    scenario: TimingScenario | None,
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    presentation: GroupPresentation,
    stored: Mapping[PreconditionerName, StoredMethod],
) -> FigureMetadata:
    dataset_ids = tuple(str(dataset) for dataset in outcome.dataset_membership)
    recipes = outcome.recipe_membership
    weights = tuple(count / outcome.n_planned_rhs for count in outcome.recipe_counts)
    reference = outcome.reference_selection
    names = sorted(presentation.methods)
    artifacts = MethodBenchmarkArtifacts()
    methods = tuple(
        FigureMethod(name, presentation.methods[name].detailed, stored[name].config_digest)
        for name in names
    )
    sources = tuple(
        method_source(
            stored[name].run_id, artifacts.setup_events, artifacts.solve_attempts, artifacts.summary
        )
        for name in names
    )
    metric = {
        FigureKind.AGGREGATE_BARS: "converged systems",
        FigureKind.SOLVE_TIME: "solve seconds (workload total)",
        FigureKind.ONLINE_TIME: "workload online seconds (setup + solve total)",
        FigureKind.PAIRED_TIME: (
            "amortized online seconds per paired RHS; asymmetric failures axis-censored; "
            "both-failed pairs omitted"
        ),
        FigureKind.PERFORMANCE_PROFILE: (
            "solve-seconds performance ratio"
            if scenario is TimingScenario.SOLVE
            else "amortized-online-seconds performance ratio"
        ),
        FigureKind.ECDF_ITERATIONS: "iterations",
        FigureKind.ECDF_TIME: (
            "solve seconds per planned RHS"
            if scenario is TimingScenario.SOLVE
            else "amortized online seconds per planned RHS"
        ),
        FigureKind.SPEED_RATIO: "reference/method amortized-online speed ratio",
    }.get(kind, kind.value)
    failure_treatment = (
        FailureTreatment.NONE
        if kind in {FigureKind.SOLVE_TIME, FigureKind.ONLINE_TIME}
        else FailureTreatment.OMITTED
        if kind is FigureKind.SPEED_RATIO
        else FailureTreatment.CENSORED
    )
    aggregation_rule = (
        "sum observed setup and solve event costs; recipe contributions follow planned RHS membership"
        if isinstance(outcome, WorkloadMixtureReport)
        else "one observation per planned (matrix, RHS) system"
    )
    return FigureMetadata(
        kind=kind,
        metric_definition=metric,
        scenario=scenario,
        report_level=level_name(outcome.label),
        scope=presentation.scope,
        dataset_ids=dataset_ids,
        recipes=recipes,
        n_matrices=outcome.n_matrices,
        n_planned_rhs=outcome.n_planned_rhs,
        reference_selection=reference,
        failure_treatment=failure_treatment,
        timing_policy="setup once per dataset/matrix + solve per planned RHS",
        recipe_counts=outcome.recipe_counts,
        aggregation_rule=aggregation_rule,
        recipe_weights=weights,
        solver_contexts=presentation.solver_contexts,
        methods=methods,
        sources=sources,
    )


REPORT_SCOPE_PHASE = "report_scope"
"""Phase tag of every non-root run of a report tree; the root carries ``report``."""

METHOD_FIGURE_ARTIFACT_DIR = "method_figures"
"""Report-run artifact directory holding every method's convergence figures."""


@dataclass(frozen=True, slots=True)
class MLflowBenchmarkMethodSink:
    """Write one reduced method workload into the already-active child run.

    ``datasets`` presents each dataset a workload ran on. ``solver_contexts`` and
    ``references`` give the settings and reference state each planned system was timed
    under; they are persisted with the raw events.

    Run ownership deliberately stays in ``comparison_batch``.  Calling this
    sink while no run is active is a caller error, just like calling
    ``mlflow.log_metric`` directly; the sink never silently creates a nested
    system or preconditioner run.
    """

    datasets: Mapping[DatasetId, DatasetPresentation]
    solver_contexts: Mapping[SystemId, SolverTimingContext]
    references: Mapping[SystemId, ReferenceSelection]

    def record_method(
        self,
        workload: MethodWorkload,
        summary: MethodWorkloadSummary,
        *,
        root: Path | None = None,
    ) -> None:
        """Persist canonical event artifacts and one copy of each selected scalar."""
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            temporary = tempfile.TemporaryDirectory(prefix="neuralls-method-")
            root = Path(temporary.name)
        try:
            systems = [
                SystemProvenance(
                    tag, self.solver_contexts[tag.system_id], self.references[tag.system_id]
                )
                for tag in workload.planned
            ]
            artifacts = write_method_benchmark_artifacts(
                workload, summary, root, systems=systems, datasets=self.datasets
            )
            self._log_method_metadata(workload, summary)
            for relative in artifacts.paths():
                mlflow.log_artifact(str(root / relative), artifact_path=METHOD_ARTIFACT_DIR)
        finally:
            if temporary is not None:
                temporary.cleanup()

    @staticmethod
    def _log_method_metadata(workload: MethodWorkload, summary: MethodWorkloadSummary) -> None:
        """Log only reducer-owned searchable scalars, never event-loop aggregates."""
        mlflow.log_param("method_id", str(workload.method))
        mlflow.log_param("method_config_digest", str(workload.config_digest))
        n_setup_failed = sum(
            isinstance(event.outcome, SetupFailed) for event in workload.setup_events
        )
        n_solve_failed = sum(
            isinstance(attempt.outcome, SolveFailed) for attempt in workload.solve_attempts
        )
        metrics = {
            "n_planned_rhs": float(summary.n_planned_rhs),
            "n_attempted_rhs": float(summary.n_attempted_rhs),
            "n_converged": float(summary.n_converged),
            "success_rate": (
                summary.n_converged / summary.n_planned_rhs if summary.n_planned_rhs else 0.0
            ),
            "setup_total_s": summary.setup_total_s,
            "solve_total_s": summary.solve_total_s,
            "workload_online_s": summary.workload_online_s,
            "n_setup_failed": float(n_setup_failed),
            "n_solve_failed": float(n_solve_failed),
            "iterations_total": float(
                sum(attempt.outcome.iterations for attempt in workload.solve_attempts)
            ),
        }
        mlflow.set_tag(
            "workload_outcome",
            "completed_with_failures" if n_setup_failed or n_solve_failed else "completed",
        )
        if summary.online_s_per_attempted_rhs is not None:
            metrics["online_s_per_attempted_rhs"] = summary.online_s_per_attempted_rhs
        if isinstance(summary.workload_peak_memory, MeasuredPeakMemory):
            metrics["workload_peak_memory_bytes"] = float(summary.workload_peak_memory.bytes)
        if workload.generation_cost is not None:
            metrics["generation_s"] = workload.generation_cost.wall_time_seconds
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


def _system_legend(
    method: StoredMethod,
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> Legend:
    """Name each system of a stored method by its dataset, recipe, matrix and RHS position."""
    texts: dict[str, str] = {}
    for system in method.systems:
        tag = system.tag
        dataset = datasets.get(tag.dataset)
        if dataset is None:
            raise MissingMethodPresentationError(str(tag.dataset))
        where = ScopeSubtitle(RecipeScope(dataset, tag.recipe), notation).text
        texts[str(tag.system_id)] = f"{where}, matrix {int(tag.matrix_id)}, RHS {tag.rhs_ordinal}"
    return series_legend(texts)


def _draw_method_figure(
    path: Path,
    kind: FigureKind,
    method: StoredMethod,
    notation: NotationPolicy,
    datasets: Mapping[DatasetId, DatasetPresentation],
) -> None:
    """Draw one method's residual or energy histories, one curve per attempted system."""
    converged = frozenset(
        method.solve_attempts.filter(pl.col("solve_status") == "converged")
        .get_column("system_id")
        .to_list()
    )
    histories: dict[str, dict[str, object]] = {}
    for (system_id,), rows in method.convergence.sort("iteration").group_by(
        "system_id", maintain_order=True
    ):
        histories[str(system_id)] = {
            "residual_history_rel": rows.get_column("residual_rel").to_list(),
            "error_history_a_rel": rows.get_column("energy_error_rel").drop_nulls().to_list(),
            "converged": str(system_id) in converged,
        }
    subtitle = ScopeSubtitle(MethodScope(method.presentation.prepared.detailed), notation)
    recipes = {
        str(system.tag.system_id): recipe_key(system.tag.recipe) for system in method.systems
    }
    series_recipes = {name: recipes[name] for name in histories}
    match kind:
        case FigureKind.RESIDUAL_HISTORY:
            plot, title = plot_convergence_comparison, "Residual History"
        case FigureKind.ENERGY_HISTORY:
            plot, title = plot_error_convergence_comparison, "Energy History"
        case _:
            raise ValueError(f"{kind} is not a method figure")
    plot(
        histories,
        save_path=path,
        title=EditorialTitle(title),
        subtitle=subtitle,
        legend=_system_legend(method, notation, datasets),
        families=series_recipes,
        color_keys=series_recipes,
    )


@dataclass(frozen=True, slots=True)
class _GroupStyler:
    """Turns the pre-draw plan of a figure into the style its plot draws with.

    The plan is the only source of a figure's fixed title and scope; the legend names
    every method of the group from its structured presentation.
    """

    presentation: GroupPresentation
    notation: NotationPolicy

    def style(self, plan: FigurePlan) -> GroupStyle:
        """The style for ``plan``: fixed title, rendered scope, legend and series styling."""
        maps = plot_style_maps(self.presentation.plot_styles)
        return GroupStyle(
            title=plan.title,
            scope=plan.metadata.scope,
            notation=self.notation,
            legend=method_legend(
                {name: method.compact for name, method in self.presentation.methods.items()},
                self.notation,
            ),
            families=maps.families,
            color_keys=maps.color_keys,
            marker_keys=maps.marker_keys,
        )


def _plan_for(
    plans: Sequence[FigurePlan], kind: FigureKind, scenario: TimingScenario | None = None
) -> FigurePlan:
    return next(
        plan for plan in plans if plan.metadata.kind is kind and plan.metadata.scenario is scenario
    )


def _paired_by_method(
    comparisons: Mapping[PreconditionerName, MethodComparison],
) -> dict[PreconditionerName, PairedComparison]:
    """The paired comparisons of the methods that had any system solved alongside the reference."""
    return {name: comparison.paired for name, comparison in comparisons.items()}


def _write_profile_plots(
    outcome: RecipeProfileReport,
    styler: _GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> list[Path | None]:
    """Draw fine-only iteration and separate solve/online time profiles."""
    performance_solve = _plan_for(plans, FigureKind.PERFORMANCE_PROFILE, TimingScenario.SOLVE)
    performance_online = _plan_for(plans, FigureKind.PERFORMANCE_PROFILE, TimingScenario.ONLINE)
    ecdf_iterations = _plan_for(plans, FigureKind.ECDF_ITERATIONS)
    ecdf_solve = _plan_for(plans, FigureKind.ECDF_TIME, TimingScenario.SOLVE)
    ecdf_online = _plan_for(plans, FigureKind.ECDF_TIME, TimingScenario.ONLINE)
    return [
        plot_performance_profile(
            outcome.outcomes,
            style=styler.style(performance_solve),
            cost=solve_seconds,
            save_path=directory / performance_solve.filename,
        ),
        plot_performance_profile(
            outcome.outcomes,
            style=styler.style(performance_online),
            cost=amortized_online_seconds,
            save_path=directory / performance_online.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_iterations),
            save_path=directory / ecdf_iterations.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_solve),
            cost=solve_seconds,
            xlabel="Solve time per system [s]",
            save_path=directory / ecdf_solve.filename,
        ),
        plot_ecdf_profile(
            outcome.outcomes,
            style=styler.style(ecdf_online),
            cost=amortized_online_seconds,
            xlabel="Amortized online time per system [s]",
            save_path=directory / ecdf_online.filename,
        ),
    ]


def _write_system_plots(
    outcome: RecipeProfileReport,
    styler: _GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> list[Path | None]:
    """Draw setup-aware timing bars, paired timing, and ratio plots."""
    paired_plan = _plan_for(plans, FigureKind.PAIRED_TIME, TimingScenario.ONLINE)
    solve_plan = _plan_for(plans, FigureKind.SOLVE_TIME, TimingScenario.SOLVE)
    online_plan = _plan_for(plans, FigureKind.ONLINE_TIME, TimingScenario.ONLINE)
    ratio_plan = _plan_for(plans, FigureKind.SPEED_RATIO, TimingScenario.ONLINE)
    paired = (
        plot_paired_time_scatter(
            outcome.outcomes,
            reference=outcome.reference_selection,
            style=styler.style(paired_plan),
            save_path=directory / paired_plan.filename,
        )
        if isinstance(outcome.reference_selection, ConfiguredReference)
        else None
    )
    plots: list[Path | None] = [
        plot_online_time_bars(
            outcome.summaries,
            scenario="solve",
            style=styler.style(solve_plan),
            save_path=directory / solve_plan.filename,
        ),
        plot_online_time_bars(
            outcome.summaries,
            scenario="online",
            style=styler.style(online_plan),
            save_path=directory / online_plan.filename,
        ),
        paired,
        plot_speed_ratio_cdf(
            _paired_by_method(outcome.comparisons),
            style=styler.style(ratio_plan),
            reference_name=outcome.reference_selection.name
            if isinstance(outcome.reference_selection, ConfiguredReference)
            else None,
            save_path=directory / ratio_plan.filename,
        ),
    ]
    return plots


def _write_outcome_plots(
    outcome: RecipeProfileReport | WorkloadMixtureReport,
    styler: _GroupStyler,
    directory: Path,
    plans: Sequence[FigurePlan],
) -> tuple[Path, ...]:
    """Draw plots allowed by the typed report variant."""
    match outcome:
        case RecipeProfileReport():
            aggregate_plan = _plan_for(plans, FigureKind.AGGREGATE_BARS)
            aggregate = plot_aggregate_bars(
                outcome,
                style=styler.style(aggregate_plan),
                save_path=directory / aggregate_plan.filename,
            )
            drawn = [
                aggregate,
                *_write_system_plots(outcome, styler, directory, plans),
                *_write_profile_plots(outcome, styler, directory, plans),
            ]
        case WorkloadMixtureReport():
            aggregate_plan = _plan_for(plans, FigureKind.AGGREGATE_BARS)
            solve_plan = _plan_for(plans, FigureKind.SOLVE_TIME, TimingScenario.SOLVE)
            online_plan = _plan_for(plans, FigureKind.ONLINE_TIME, TimingScenario.ONLINE)
            aggregate = plot_aggregate_bars(
                outcome,
                style=styler.style(aggregate_plan),
                save_path=directory / aggregate_plan.filename,
            )
            drawn = [
                aggregate,
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="solve",
                    style=styler.style(solve_plan),
                    save_path=directory / solve_plan.filename,
                ),
                plot_online_time_bars(
                    outcome.summaries,
                    scenario="online",
                    style=styler.style(online_plan),
                    save_path=directory / online_plan.filename,
                ),
            ]
        case _ as unreachable:
            assert_never(unreachable)
    return tuple(path for path in drawn if path is not None)


GROUP_REPORT_ARTIFACT_DIR = "group_reports"


def log_configured_method_facts(prepared: PreparedMethodPresentation) -> None:
    """Log every configured fact of a method as stable parameters, before it executes."""
    mlflow.log_params(encode_method_parameters(prepared))
