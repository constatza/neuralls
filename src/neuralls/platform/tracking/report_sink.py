"""MLflow sink recording a report as a tree of nested scope runs."""

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

from neuralls.domain.benchmarking.selection import (
    NoReferenceConfigured,
)
from neuralls.domain.presentation.methods import (
    MissingMethodPresentationError,
)
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
    parent_scope,
    scope_of_label,
)
from neuralls.domain.solver.reporting.report_levels import (
    ReportLevelName,
    level_of_scope,
)
from neuralls.platform.reporting.benchmark_events import (
    MethodBenchmarkArtifacts,
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
    FigureSlugs,
    NoTable,
    SiblingTable,
    StagedFigure,
    collision_key,
    make_plan,
    method_source,
)
from neuralls.platform.reporting.figure_planning import figure_plans
from neuralls.platform.reporting.figure_table import build_group_table, write_group_table
from neuralls.platform.reporting.figure_text import (
    scope_label,
)
from neuralls.platform.reporting.group_plots import GroupStyler, write_outcome_plots
from neuralls.platform.reporting.method_figures import draw_method_figure
from neuralls.platform.reporting.stored_benchmark import StoredMethod
from neuralls.platform.tracking.metrics_logging import (
    log_comparison_metrics,
    log_convergence_metrics,
)
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.platform.tracking.session_artifacts import PARENT_RUN_TAG
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import DatasetId, PreconditionerName, SessionRunId


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
    session_run_id: SessionRunId
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
            draw_method_figure(path, kind, method, self.notation, datasets)
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
            plans = figure_plans(outcome, presentation, file_key, self._group_methods(presentation))
            self._check_free(run_id, GROUP_REPORT_ARTIFACT_DIR, [plan.filename for plan in plans])
            outcome_plots = write_outcome_plots(
                outcome, GroupStyler(presentation, self.notation), Path(tmp), plans
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


REPORT_SCOPE_PHASE = "report_scope"
"""Phase tag of every non-root run of a report tree; the root carries ``report``."""

METHOD_FIGURE_ARTIFACT_DIR = "method_figures"
"""Report-run artifact directory holding every method's convergence figures."""

GROUP_REPORT_ARTIFACT_DIR = "group_reports"
