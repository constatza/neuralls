"""MLflow sink recording a report as a tree of nested scope runs."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import assert_never

from neuralls.domain.benchmarking.paired import (
    REFERENCE_METRIC,
    reference_among_methods,
    speed_ratio_summary,
)
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
from neuralls.domain.solver.models.split_record import DatasetSplitRecord
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.reporting.benchmark_events import MethodBenchmarkArtifacts
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_index import FigureIndexEntry
from neuralls.platform.reporting.figure_metadata import (
    FigureMethod,
    FigureSlugs,
    MethodAggregationRule,
    MethodFigureMetadata,
    StagedFigure,
    TimingPolicy,
    method_source,
)
from neuralls.platform.reporting.figure_planning import figure_plans, plan_figure
from neuralls.platform.reporting.figure_recipes import history_variant
from neuralls.platform.reporting.figure_summaries import figure_summaries
from neuralls.platform.reporting.figure_text import scope_label
from neuralls.platform.reporting.group_plots import GroupStyler, write_outcome_plots
from neuralls.platform.reporting.method_figures import METHOD_PLOTS, draw_method_figure
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.platform.tracking.figure_uploader import FigureUploader
from neuralls.platform.tracking.metrics_logging import comparison_metrics, convergence_metrics
from neuralls.platform.tracking.run_writer import RunWriter
from neuralls.platform.tracking.scope_run_tree import RunTree, ScopeRunTree
from neuralls.shared.runtime_environment import RuntimeEnvironment
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SessionRunId

METHOD_FIGURE_ARTIFACT_DIR = "method_figures"
"""Report-run artifact directory holding every method's convergence figures."""

GROUP_REPORT_ARTIFACT_DIR = "group_reports"
"""Report-run artifact directory holding every group's figures, cards and tables."""

SYSTEM_ARTIFACT_DIR = "systems"
"""RHS-type report-run artifact directory holding one subdirectory of figures per system."""

SYSTEM_METRIC_PREFIX = "system"
"""First segment of every per-system metric key on an RHS-type report run."""


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
        case unreachable:
            assert_never(unreachable)


def system_key(scope: System) -> str:
    """The segment naming ``scope`` in its RHS-type run's artifact paths and metric keys."""
    return f"matrix-{int(scope.matrix_id)}_rhs-{scope.rhs_ordinal}"


@dataclass(frozen=True, slots=True)
class FinalizedReport:
    """A report whose figure index was uploaded; it accepts no further figures."""

    index: tuple[FigureIndexEntry, ...]


@dataclass(slots=True)
class OpenReport:
    """A report still accepting scopes: the run tree, the uploader and what they draw from.

    ``root_run_id`` is the run of all matrix sets; every other scope gets a child run of
    its parent scope, created on first use. Every figure is uploaded with its
    card into the run of its own scope, except a system's, which go under
    ``systems/<system_key>/`` of its RHS type's run; the root run also holds the method
    figures and, once finalized, the index of all figures.
    """

    notation: NotationPolicy
    figure_style: FigureStyleConfig
    methods: Sequence[StoredMethod]
    runs: ScopeRunTree
    uploader: FigureUploader
    split_records: Sequence[DatasetSplitRecord]
    writer: RunWriter
    root_run_id: str
    slugs: FigureSlugs = field(default_factory=FigureSlugs)

    def _run_of(self, scope: ReportScope) -> str:
        """The id of the run of ``scope``, created on first use."""
        return self.runs.run_id(scope, self.root_run_id)

    def _stored(self, name: PreconditionerName) -> StoredMethod:
        found = next((method for method in self.methods if method.name == name), None)
        if found is None:
            raise UnknownStoredMethodError(f"no stored run for method {name}")
        return found

    def register_method_figures(self, *, datasets: Mapping[DatasetId, DatasetPresentation]) -> None:
        """Draw each method's convergence histories from its stored table and upload them."""
        for method in self.methods:
            self._register_method(self.root_run_id, method, datasets)

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
            variant = history_variant(kind)
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
            self.uploader.check_free(root_id, METHOD_FIGURE_ARTIFACT_DIR, [plan.stem])
            target = FigureTarget(figures_dir, plan.stem)
            draw_method_figure(
                target,
                kind,
                method,
                self.notation,
                datasets,
                show_titles=self.figure_style.show_titles,
            )
            self.uploader.upload(root_id, METHOD_FIGURE_ARTIFACT_DIR, StagedFigure(plan, target))

    def report(
        self,
        outcome: RecipeProfileReport | WorkloadMixtureReport,
        *,
        presentation: GroupPresentation,
    ) -> None:
        """Log explicit counts and the supported failure-aware plots into the group's run.

        Counts, success rate and iteration statistics are logged only for a class (one matrix
        set and RHS type): pooled over a mixture they would depend on its arbitrary weights.
        """
        group_id = outcome.group_id
        scope = scope_of_label(outcome.label)
        run_id = self._run_of(scope)
        with tempfile.TemporaryDirectory() as tmp:
            if isinstance(outcome, RecipeProfileReport):
                metrics = convergence_metrics(outcome.convergence)
                reference = reference_among_methods(
                    outcome.observations, outcome.reference_selection
                )
                if reference is not None:
                    metrics |= comparison_metrics(
                        speed_ratio_summary(outcome.observations, reference, REFERENCE_METRIC)
                    )
                self.writer.log_metrics(run_id, metrics)
            file_key = self.slugs.slug(
                f"group:{group_id}", scope_label(presentation.scope, self.notation)
            )
            stored = self._group_methods(presentation)
            plans = figure_plans(outcome, presentation, file_key, stored)
            self.uploader.check_free(
                run_id, GROUP_REPORT_ARTIFACT_DIR, [plan.stem for plan in plans]
            )
            figures = write_outcome_plots(
                outcome,
                figure_summaries(
                    outcome, {name: method.offline for name, method in stored.items()}
                ),
                GroupStyler(presentation, self.notation, self.figure_style),
                Path(tmp),
                plans,
            )
            for figure in figures:
                self.uploader.upload(run_id, GROUP_REPORT_ARTIFACT_DIR, figure)

    def _group_methods(
        self, presentation: GroupPresentation
    ) -> dict[PreconditionerName, StoredMethod]:
        """The stored run of every method the group presents."""
        return {name: self._stored(name) for name in presentation.methods}

    def report_system(
        self, scope: System, metrics: Mapping[str, float], figures: Sequence[StagedFigure]
    ) -> None:
        """Log one system's per-method metrics and figures, with their cards, into its RHS type's run.

        A system has no run: its metrics are keyed ``system/<system_key>/<metric>/<method>``
        and its figures sit in ``systems/<system_key>/`` of that run.
        """
        run_id = self._run_of(scope.rhs_type)
        key = system_key(scope)
        directory = f"{SYSTEM_ARTIFACT_DIR}/{key}"
        self.writer.log_metrics(
            run_id,
            {f"{SYSTEM_METRIC_PREFIX}/{key}/{name}": value for name, value in metrics.items()},
        )
        self.uploader.check_free(run_id, directory, [staged.plan.stem for staged in figures])
        for staged in figures:
            self.uploader.upload(run_id, directory, staged)

    def finalize(self) -> FinalizedReport:
        """Check every figure has its card, then upload the figure index to the root run."""
        return FinalizedReport(self.uploader.write_index(self.split_records))


@dataclass(slots=True)
class MLflowReportSink:
    """The ``ReportSink`` adapter that writes a report to MLflow until it is finalized.

    ``notation`` is the case's symbol policy and ``datasets`` presents each dataset a scope
    names; run names and figures render both. ``figure_style`` is the case's figure layout. ``session_run_id`` and ``methods`` identify
    the compare session and the stored method runs every figure is computed from;
    ``split_records`` are the session's held-out records, copied beside the index; ``environment`` is the
    session run's recorded device, CPU, versions and revisions, cited in every figure card. ``runs``
    creates the nested scope runs under ``root_run_id``, the report's root run, and ``writer``
    logs every artifact and metric into the run of its scope.
    """

    notation: NotationPolicy
    session_run_id: SessionRunId
    runs: RunTree
    split_records: Sequence[DatasetSplitRecord]
    environment: RuntimeEnvironment
    writer: RunWriter
    root_run_id: str
    methods: Sequence[StoredMethod] = ()
    datasets: Mapping[DatasetId, DatasetPresentation] = field(default_factory=dict)
    figure_style: FigureStyleConfig = field(default_factory=FigureStyleConfig)
    _state: OpenReport | FinalizedReport = field(init=False)

    def __post_init__(self) -> None:
        """Open the report: its scope run tree and figure uploader start empty."""
        uploader = FigureUploader(
            self.session_run_id, self.notation, self.environment, self.writer, self.root_run_id
        )
        # The namer captures values, not `self`: a cycle back to the sink would keep the
        # uploader's staging directory alive until an arbitrary later garbage collection.
        notation, datasets = self.notation, self.datasets
        self._state = OpenReport(
            self.notation,
            self.figure_style,
            self.methods,
            ScopeRunTree(self.runs, lambda scope: scope_run_name(scope, notation, datasets)),
            uploader,
            self.split_records,
            self.writer,
            self.root_run_id,
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
        """Log one system's per-method metrics and figures, with their cards, into its RHS type's run."""
        self._open().report_system(scope, metrics, figures)

    def finalize(self) -> None:
        """Upload the figure index; the report accepts nothing afterwards."""
        self._state = self._open().finalize()
