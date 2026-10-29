"""Every MLflow run is named from a typed presentation; ids appear only in ``*_id`` tags.

Each test captures the runs a path opens (names and tags) against a mocked MLflow and
checks the contract: no ``method |`` prefix, no id, slug, path or class-name fallback, no
underscore identifier blob, and sibling systems stay distinguishable.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import (
    PreparedComparisonExecution,
    ResolvedComparisonSpecs,
    _build_child_runs,
    _ComparisonResolutionContext,
    _execute_prepared_comparison,
    run_comparison_batch,
)
from neuralls.composition.assignments.presentation_preflight import (
    describe_method,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.models import ComparisonParams, MatrixRef, PlannedSystem
from neuralls.composition.tracking import run_specs
from neuralls.composition.tracking.run_specs import build_comparison_run_spec
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import MethodRunName, NamedRun
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.domain.solver.models.provenance import HeldOutMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.run_names import render_run_name, run_name_text
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import MatrixId
from tests.presentation_support import NO_DATASET_READS, datasets_by_dir, pod_spec
from tests.workflows.comparison_support import MLFLOW_MODULE, mock_cfg, write_experiments_config

pytestmark = pytest.mark.deferred_io("config-loading")

PLAIN = NotationPolicy.default()
FIT_DIR = Path("pod-fit")
STAMP = "2026-01-02T03:04:05"
DATASET_TEXT = "Test problem | Gaussian CG Residuals (k=20, N=100)"


def _execution(
    cfg: object,
    planned: PlannedSystem,
    context: _ComparisonResolutionContext,
    presentation: PreparedComparisonPresentation,
    title: ComparisonTitle,
) -> PreparedComparisonExecution:
    return PreparedComparisonExecution(
        cfg=cfg,
        planned=planned,
        topology=context.topology,
        cleanup=contextlib.ExitStack(),
        resolved=ResolvedComparisonSpecs(
            specs=[StandardPreconditionerConfig(type=PreconditionerType.JACOBI)],
            warnings=(),
            checkpoint_digests={},
        ),
        identity=StageIdentity.build("comparison", {"entry": canonical_digest(planned.entry.id)}),
        settings=context.settings,
        presentation=presentation,
        title=title,
    )


def _session_run_names(
    tmp_path: Path, presentation: PreparedComparisonPresentation, title: ComparisonTitle
) -> list[str]:
    experiments_config = tmp_path / "case_with_ids.toml"
    write_experiments_config(experiments_config, with_comparisons=True)

    def prepare(
        cfg: object, planned: PlannedSystem, context: _ComparisonResolutionContext, **_: object
    ) -> PreparedComparisonExecution:
        return _execution(cfg, planned, context, presentation, title)

    with (
        patch.object(comparison_batch, "_prepare_comparison_entry", side_effect=prepare),
        patch.object(comparison_batch, "resolve_comparison_config", return_value=mock_cfg()),
        patch.object(comparison_batch, "_has_complete_reusable_session", return_value=False),
        patch.object(comparison_batch, "_execute_all", return_value=()),
        patch.object(comparison_batch, "_outcomes_from_workloads", return_value=[]),
        patch.object(comparison_batch, "report_all_levels"),
        patch.object(comparison_batch, "MLflowGroupReportSink"),
        patch(MLFLOW_MODULE) as mock_mlflow,
        patch.object(run_specs, "iso_timestamp", return_value=STAMP),
    ):
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        mock_mlflow.start_run.return_value.__exit__.return_value = False
        run_comparison_batch(experiments_config, ComparisonParams())
    return [call.kwargs["run_name"] for call in mock_mlflow.start_run.call_args_list]


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_session_run_is_named_by_the_dataset_not_the_case_file(
    tmp_path: Path, prepared_presentation: PreparedComparisonPresentation
) -> None:
    (name,) = _session_run_names(tmp_path, prepared_presentation, prepared_presentation.scope)

    assert name == f"{DATASET_TEXT} | {STAMP}"


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_session_run_is_named_by_the_editorial_title_when_one_is_configured(
    tmp_path: Path, prepared_presentation: PreparedComparisonPresentation
) -> None:
    (name,) = _session_run_names(
        tmp_path, prepared_presentation, EditorialPresentation("Spectral comparison")
    )

    assert name == f"Spectral comparison | {STAMP}"


def _planned(entry: ComparisonRegistryEntry, matrix: int, ordinal: int) -> PlannedSystem:
    return PlannedSystem(
        entry=entry,
        matrix=MatrixRef(matrix_id=MatrixId(matrix), row=0),
        rhs_row=None,
        rhs_ordinal=ordinal,
        provenance=HeldOutMatrix(matrix_id=MatrixId(matrix), raw_matrix_index=0),
    )


def test_comparison_parent_runs_of_a_sweep_have_distinct_names_and_no_ids(
    prepared_presentation: PreparedComparisonPresentation, tmp_path: Path
) -> None:
    context = _ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=tmp_path / "case.toml",
        settings=MagicMock(),
        load_dataset=NO_DATASET_READS,
    )
    started: list[tuple[str, dict[str, str]]] = []
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.side_effect = lambda **kwargs: (
        started.append((kwargs["run_name"], kwargs["tags"])) or MagicMock()
    )
    with (
        patch.object(comparison_batch, "mlflow", fake_mlflow),
        patch.object(comparison_batch, "_run_and_log_comparison", return_value=MagicMock()),
        patch.object(comparison_batch, "log_comparison_run_params"),
        patch.object(comparison_batch, "log_comparison_artifact_uri"),
    ):
        for matrix in (3, 4):
            for ordinal in (0, 1):
                entry = ComparisonRegistryEntry(
                    id=f"sweep_{matrix}_{ordinal}",
                    matrix_dataset="solutions",
                    rhs_source={"kind": "gaussian"},
                    rhs_count=2,
                )
                prepared = _execution(
                    mock_cfg(),
                    _planned(entry, matrix, ordinal),
                    context,
                    prepared_presentation,
                    prepared_presentation.scope,
                )
                _execute_prepared_comparison(prepared)

    assert [name for name, _ in started] == [
        f"{DATASET_TEXT} | matrix {matrix}, RHS {ordinal}"
        for matrix in (3, 4)
        for ordinal in (0, 1)
    ]
    assert [tags["comparison_id"] for _, tags in started] == [
        f"sweep_{matrix}_{ordinal}" for matrix in (3, 4) for ordinal in (0, 1)
    ]
    assert {tags["comparison_label"] for _, tags in started} == {DATASET_TEXT}


def test_child_runs_are_named_after_their_method_presentation_and_carry_ids_in_tags(
    prepared_presentation: PreparedComparisonPresentation, tmp_path: Path
) -> None:
    entry = ComparisonRegistryEntry(
        id="cmp_1", matrix_dataset="solutions", rhs_source={"kind": "gaussian"}
    )
    context = _ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=tmp_path / "case.toml",
        settings=MagicMock(),
        load_dataset=NO_DATASET_READS,
    )
    prepared = _execution(
        mock_cfg(),
        PlannedSystem(
            entry=entry,
            matrix=MatrixRef(matrix_id=MatrixId(0), row=0),
            rhs_row=None,
            rhs_ordinal=0,
            provenance=HeldOutMatrix(matrix_id=MatrixId(0), raw_matrix_index=0),
        ),
        context,
        prepared_presentation,
        prepared_presentation.scope,
    )
    parent, _ = build_comparison_run_spec(
        entry=entry,
        planned=prepared.planned,
        title=prepared.title,
        dataset=prepared.presentation.scope.dataset,
        notation=PLAIN,
        include_timestamp=False,
    )

    children = _build_child_runs(
        SimpleNamespace(results={"jacobi": object()}),  # type: ignore[arg-type]
        prepared=prepared,
        parent=parent,
    )

    child = children["jacobi"]
    assert run_name_text(child.run) == "Jacobi"
    assert child.tags["method_id"] == "jacobi"
    assert child.tags["method_label"] == "Jacobi"
    assert child.tags["parent_run_name"] == run_name_text(parent)


def test_a_method_run_and_a_plot_legend_render_the_same_presentation(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    prepared = prepare_comparison_presentation(
        [pod_spec(FIT_DIR, 10)],
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
    ).methods[0]
    method = prepared.compact
    name = MethodRunName(method)
    run = NamedRun(name, PLAIN)

    assert name.method is method
    plain_legend = method_legend({"pod": method}, PLAIN).label("pod")
    assert plain_legend.replace("\n", " ") == run_name_text(run)
    math_legend = method_legend({"pod": method}, NotationPolicy.default(RenderTarget.MATHTEXT))
    assert math_legend.label("pod") != plain_legend
    assert render_run_name(run.name, PLAIN) == run_name_text(run)


def test_configured_facts_become_stable_parameters_without_rendered_text(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    prepared = prepare_comparison_presentation(
        [pod_spec(FIT_DIR, 10)],
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
    ).methods[0]

    parameters = encode_method_parameters(prepared)

    assert parameters["method.base_name"] == "POD-2G"
    assert parameters["method.rank"] == "10"
    assert parameters == {
        "method.base_name": "POD-2G",
        "method.rank": "10",
        "method.pre_smoothing_steps": "2",
        "method.post_smoothing_steps": "2",
        "method.smoother_damping": "auto",
        "training_dataset.source": "Atomic test",
        "training_dataset.strategy_0.kind": "gaussian_forward",
        "training_dataset.strategy_0.mean": "0.0",
        "training_dataset.strategy_0.standard_deviation": "1.0",
        "training_dataset.strategy_0.sample_count": "8",
    }


def test_describing_a_method_yields_a_presentation_with_no_assignment_id(
    pod_fit_presentation: DatasetPresentation,
) -> None:
    load = datasets_by_dir({FIT_DIR: pod_fit_presentation})
    method = describe_method(pod_spec(FIT_DIR, 10, assignment="fit-job"), load)
    assert method == describe_method(pod_spec(FIT_DIR, 10, assignment="another-job"), load)
