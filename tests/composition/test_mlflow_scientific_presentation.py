"""Every MLflow run is named from a typed presentation; ids appear only in ``*_id`` tags.

Each test captures the runs a path opens (names and tags) against a mocked MLflow and
checks the contract: no ``method |`` prefix, no id, slug, path or class-name fallback, no
underscore identifier blob, and sibling systems stay distinguishable.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralls.composition.assignments import comparison_batch, comparison_prepare
from neuralls.composition.assignments.comparison_batch import (
    run_comparison_batch,
)
from neuralls.composition.assignments.comparison_prepared import (
    PreparedComparisonExecution,
)
from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    ResolvedComparisonSpecs,
)
from neuralls.composition.assignments.presentation_preflight import (
    describe_method,
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.models import ComparisonParams, PlannedSystem
from neuralls.composition.tracking import run_specs
from neuralls.domain.identity import StageIdentity
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import MethodRunName, NamedRun
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.run_names import render_run_name, run_name_text
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from tests.presentation_support import datasets_by_dir, pod_spec
from tests.workflows.comparison_support import MLFLOW_MODULE, mock_cfg, write_experiments_config

pytestmark = pytest.mark.deferred_io("config-loading")

PLAIN = NotationPolicy.default()
FIT_DIR = Path("pod-fit")
STAMP = "2026-01-02T03:04:05"
DATASET_TEXT = "Test problem | Gaussian CG Residuals (k=20, N=100)"


def _execution(
    cfg: object,
    planned: PlannedSystem,
    context: ComparisonResolutionContext,
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
        cfg: object, planned: PlannedSystem, context: ComparisonResolutionContext, **_: object
    ) -> PreparedComparisonExecution:
        return _execution(cfg, planned, context, presentation, title)

    with (
        patch.object(comparison_prepare, "prepare_comparison_entry", side_effect=prepare),
        patch.object(comparison_prepare, "resolve_comparison_config", return_value=mock_cfg()),
        patch.object(comparison_batch, "has_complete_reusable_session", return_value=False),
        patch.object(comparison_batch, "execute_all", return_value=()),
        patch.object(comparison_batch, "outcomes_from_workloads", return_value=[]),
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

    assert name == f"{DATASET_TEXT} | 2 systems (1 RHS type x 1 matrix) | {STAMP}"


@pytest.mark.usefixtures("stub_input_gate", "stub_planning")
def test_session_run_is_named_by_the_editorial_title_when_one_is_configured(
    tmp_path: Path, prepared_presentation: PreparedComparisonPresentation
) -> None:
    (name,) = _session_run_names(
        tmp_path, prepared_presentation, EditorialPresentation("Spectral comparison")
    )

    assert name == f"Spectral comparison | 2 systems (1 RHS type x 1 matrix) | {STAMP}"


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
