"""MLflow sink recording a report as a tree of nested scope runs."""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import assert_never

import mlflow

from neuralls.domain.benchmarking.selection import NoReferenceConfigured
from neuralls.domain.presentation.methods import MissingMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetPresentation
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
from neuralls.domain.solver.models.group_presentation import GroupPresentation
from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    ReportScope,
    System,
    scope_of_label,
)
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.benchmark_events import MethodBenchmarkArtifacts
from neuralls.platform.reporting.figure_index import FigureIndexEntry
from neuralls.platform.reporting.figure_metadata import (
    FigureMethod,
    FigureSlugs,
    MethodAggregationRule,
    MethodFigureMetadata,
    StagedFigure,
    StagedTabledFigure,
    TimingPolicy,
    method_source,
)
from neuralls.platform.reporting.figure_planning import FIGURE_RECIPES, figure_plans, plan_figure
from neuralls.platform.reporting.figure_table import build_group_table
from neuralls.platform.reporting.figure_text import scope_label
from neuralls.platform.reporting.group_plots import GroupStyler, write_outcome_plots
from neuralls.platform.reporting.method_figures import METHOD_PLOTS, draw_method_figure
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.platform.tracking.figure_uploader import (
    ArtifactSink,
    FigureUploader,
    MlflowArtifactSink,
)
from neuralls.platform.tracking.metrics_logging import (
    log_comparison_metrics,
    log_convergence_metrics,
)
from neuralls.platform.tracking.scope_run_tree import RunTree, ScopeRunTree
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SessionRunId

METHOD_FIGURE_ARTIFACT_DIR = "method_figures"
"""Report-run artifact directory holding every method's convergence figures."""

GROUP_REPORT_ARTIFACT_DIR = "group_reports"
"""Report-run artifact directory holding every group's figures, cards and tables."""


class UnknownStoredMethodError(LookupError):
    """A figure names a method the report has no stored run for."""


class ReportFinalizedError(RuntimeError):
    """A report was written to, or finalized, after its figure index was uploaded."""


def scope_run_name(
    scope: ReportScope, notation: NotationPolicy, datasets: Mapping[DatasetId, DatasetPresentation]
) -> str:
    """The scientific text naming the run of ``scope``."""

    def presented(dataset: DatasetId) -> DatasetPresentation:
        found = datasets.get(dataset)
        if found is None:
            raise MissingMethodPresentationError(str(dataset))
        return found

    match scope:
        case AllMatrixSets():
            return scope_label(AllDatasetsScope(), notation)
        case MatrixSet(dataset=dataset):
            return scope_label(DatasetScope(presented(dataset)), notation)
        case MatrixSetRhsType(dataset=dataset, recipe=recipe):
            return scope_label(RecipeScope(presented(dataset), recipe), notation)
        case System(matrix_id=matrix_id, rhs_ordinal=rhs_ordinal):
            return f"matrix {int(matrix_id)}, RHS {rhs_ordinal}"
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class FinalizedReport:
    """A report whose figure index was uploaded; it accepts no further figures."""

    index: tuple[FigureIndexEntry, ...]


@dataclass(slots=True)
class OpenReport:
    """A report still accepting scopes: the run tree, the uploader and what they draw from.

    The run active at the first scope is the root (all matrix sets); every other scope gets
    a child run of its parent scope, created on first use. Every figure is uploaded with its
    card into the run of its own scope; the root run also holds the method figures and,
    once finalized, the index of all figures.
    """

    notation: NotationPolicy
    methods: Sequence[StoredMethod]
    runs: ScopeRunTree
    uploader: FigureUploader
    slugs: FigureSlugs = field(default_factory=FigureSlugs)

    def _root_run_id(self) -> str:
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError("a report needs its root run active")
        return active.info.run_id

    @contextmanager
    def _in_scope(self, scope: ReportScope) -> Generator[str]:
        """Log into the run of ``scope`` and yield its id; the root run is already active."""
        root_id = self._root_run_id()
        if isinstance(scope, AllMatrixSets):
            yield root_id
            return
        run_id = self.runs.run_id(scope, root_id)
        with mlflow.start_run(run_id=run_id, nested=True):
            yield run_id

    def _stored(self, name: PreconditionerName) -> StoredMethod:
        found = next((method for method in self.methods if method.name == name), None)
        if found is None:
            raise UnknownStoredMethodError(f"no stored run for method {name}")
        return found

    def register_method_figures(self, *, datasets: Mapping[DatasetId, DatasetPresentation]) -> None:
        """Draw each method's convergence histories from its stored table and upload them."""
        with self._in_scope(AllMatrixSets()) as root_id:
            for method in self.methods:
                self._register_method(root_id, method, datasets)

    def _register_method(
        self, root_id: str, method: StoredMethod, datasets: Mapping[DatasetId, DatasetPresentation]
    ) -> None:
        tags = tuple(system.tag for system in method.systems)
        dataset_ids = tuple(dict.fromkeys(tag.dataset for tag in tags))
        recipes = tuple(dict.fromkeys(tag.recipe for tag in tags))
        counts = tuple(sum(tag.recipe == recipe for tag in tags) for recipe in recipes)
        contexts = tuple(dict.fromkeys(system.solver for system in method.systems))
        name = str(method.name)
        scope = MethodScope(method.presentation.prepared.detailed)
        method_slug = self.slugs.slug(f"method:{name}", scope_label(scope, self.notation))
        n_matrices = len({(tag.dataset, tag.matrix_id) for tag in tags})
        recipe_weights = tuple(count / len(tags) for count in counts)
        artifacts = MethodBenchmarkArtifacts()
        figure_method = FigureMethod(method.name, scope.method, method.config_digest)
        figures_dir = self.uploader.staging_dir(METHOD_FIGURE_ARTIFACT_DIR)
        for kind in METHOD_PLOTS:
            variant = FIGURE_RECIPES[kind].variant(None)
            metadata = MethodFigureMetadata(
                kind=kind,
                metric_definition=variant.metric_definition,
                report_level=ReportLevelName.METHOD,
                scope=scope,
                dataset_ids=dataset_ids,
                recipes=recipes,
                n_matrices=n_matrices,
                n_planned_rhs=len(tags),
                reference_selection=NoReferenceConfigured(),
                timing_policy=TimingPolicy.DIAGNOSTIC_EXCLUDED_FROM_ONLINE,
                methods=(figure_method,),
                sources=(
                    method_source(method.run_id, artifacts.convergence, artifacts.solve_attempts),
                ),
                recipe_counts=counts,
                aggregation_rule=MethodAggregationRule.PER_ATTEMPTED_RHS,
                recipe_weights=recipe_weights,
                solver_contexts=contexts,
            )
            plan = plan_figure(variant, metadata, slug=method_slug)
            filename = plan.filename
            self.uploader.check_free(root_id, METHOD_FIGURE_ARTIFACT_DIR, [filename])
            path = figures_dir / filename
            draw_method_figure(path, kind, method, self.notation, datasets)
            self.uploader.upload(root_id, METHOD_FIGURE_ARTIFACT_DIR, StagedFigure(plan, path))

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Log explicit counts and the supported failure-aware plots into the group's run."""
        group_id = outcome.group_id
        scope = scope_of_label(outcome.label)
        with self._in_scope(scope) as run_id, tempfile.TemporaryDirectory() as tmp:
            log_convergence_metrics(outcome.convergence)
            if isinstance(outcome, RecipeProfileReport):
                log_comparison_metrics(outcome.comparisons)
            file_key = self.slugs.slug(
                f"group:{group_id}", scope_label(presentation.scope, self.notation)
            )
            plans = figure_plans(outcome, presentation, file_key, self._group_methods(presentation))
            self.uploader.check_free(
                run_id, GROUP_REPORT_ARTIFACT_DIR, [plan.filename for plan in plans]
            )
            outcome_plots = write_outcome_plots(
                outcome, GroupStyler(presentation, self.notation), Path(tmp), plans
            )
            table = build_group_table(outcome.summaries)
            plan_of = {plan.filename: plan for plan in plans}
            for artifact in outcome_plots:
                staged = StagedFigure(plan_of[artifact.name], artifact)
                self.uploader.upload(
                    run_id, GROUP_REPORT_ARTIFACT_DIR, StagedTabledFigure(staged, table)
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
        with self._in_scope(scope) as run_id:
            mlflow.log_metrics(dict(metrics))
            self.uploader.check_free(run_id, None, [staged.plan.filename for staged in figures])
            for staged in figures:
                self.uploader.upload(run_id, None, staged)

    def finalize(self) -> FinalizedReport:
        """Check every figure has its card, then upload the figure index to the root run."""
        return FinalizedReport(self.uploader.write_index())


@dataclass(slots=True)
class MLflowReportSink:
    """The ``ReportSink`` adapter that writes a report to MLflow until it is finalized.

    ``notation`` is the case's symbol policy and ``datasets`` presents each dataset a scope
    names; run names and figures render both. ``session_run_id`` and ``methods`` identify
    the compare session and the stored method runs every figure is computed from. ``runs``
    creates the nested scope runs and ``artifacts`` uploads into the active run.
    """

    notation: NotationPolicy
    session_run_id: SessionRunId
    runs: RunTree
    methods: Sequence[StoredMethod] = ()
    datasets: Mapping[DatasetId, DatasetPresentation] = field(default_factory=dict)
    artifacts: ArtifactSink = field(default_factory=MlflowArtifactSink)
    _state: OpenReport | FinalizedReport = field(init=False)

    def __post_init__(self) -> None:
        """Open the report: its scope run tree and figure uploader start empty."""
        uploader = FigureUploader(self.session_run_id, self.notation, self.artifacts)
        self._state = OpenReport(
            self.notation,
            self.methods,
            ScopeRunTree(
                self.runs, lambda scope: scope_run_name(scope, self.notation, self.datasets)
            ),
            uploader,
        )

    def _open(self) -> OpenReport:
        match self._state:
            case OpenReport() as open_report:
                return open_report
            case FinalizedReport():
                raise ReportFinalizedError("cannot write to a finalized report")
            case unreachable:
                assert_never(unreachable)

    def register_method_figures(self, *, datasets: Mapping[DatasetId, DatasetPresentation]) -> None:
        """Draw each method's convergence histories from its stored table and upload them."""
        self._open().register_method_figures(datasets=datasets)

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Log explicit counts and the supported failure-aware plots into the group's run."""
        self._open().report(outcome, presentation=presentation)

    def report_system(
        self, scope: System, metrics: Mapping[str, float], figures: Sequence[StagedFigure]
    ) -> None:
        """Log one system's per-method metrics and figures, with their cards, into its own run."""
        self._open().report_system(scope, metrics, figures)

    def finalize(self) -> None:
        """Upload the figure index; the report accepts nothing afterwards."""
        self._state = self._open().finalize()
