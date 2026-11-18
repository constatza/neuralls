"""Every MLflow run is named from a typed presentation; ids appear only in ``*_id`` tags.

Each test captures the runs a path opens (names and tags) through recorded tracking and
checks the contract: no ``method |`` prefix, no id, slug, path or class-name fallback, no
underscore identifier blob, and sibling systems stay distinguishable.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from neuralls.composition.assignments import comparison_batch, comparison_prepare
from neuralls.composition.assignments.comparison_batch import (
    run_loaded_comparison_batch,
)
from neuralls.composition.assignments.comparison_prepared import (
    PreparedComparisonExecution,
)
from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    ResolvedComparisonSpecs,
)
from neuralls.composition.assignments.presentation_preflight import (
    prepare_comparison_presentation,
)
from neuralls.composition.comparison.method_data_origin import InlineFit, NoFitData
from neuralls.composition.comparison.models import ComparisonParams, PlannedSystem
from neuralls.domain.identity import IdentityTag, StageIdentity
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.render import render_title
from neuralls.domain.presentation.run_names import MethodRunName, NamedRun
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.platform.storage.facts_codec import encode_method_parameters
from neuralls.platform.tracking.run_names import render_run_name, run_name_text
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.policy import (
    LEGEND_PROFILE,
    NotationPolicy,
    RenderTarget,
)
from tests.fakes.comparison_runs import RecordingTrackingServices
from tests.fakes.mlflow_client import FakeMlflowClient
from tests.fakes.split_store import InMemorySplitStore
from tests.presentation_support import datasets_by_dir, pod_spec, train_splits
from tests.workflows.comparison_support import (
    VIRTUAL_CASE,
    in_memory_case,
    mock_cfg,
    virtual_settings,
)

PLAIN = NotationPolicy.default()
FIT_DIR = Path("pod-fit")
DATASET_TEXT = "Test problem | CG Errors (k=20, N=100)"


class _StopAfterProvenance(Exception):
    """Raised by the patched presentation step to stop the batch once provenance was checked."""


@pytest.fixture
def mismatched_checkpoint_spec() -> NeuralPreconditionerConfig:
    """A checkpoint whose training run recorded another dataset digest than the current one."""
    return NeuralPreconditionerConfig(
        architecture="Net",
        checkpoint_path=Path("network.pt"),
        resolved_run_id="training-run",
        data_config_path=Path("training.toml"),
    )


@pytest.mark.usefixtures(
    "stub_input_gate", "stub_planning", "stub_matrix_scope", "stub_setup_tracking"
)
def test_recorded_dataset_digest_mismatch_blocks_before_presentation_reuse_or_run(
    mismatched_checkpoint_spec: NeuralPreconditionerConfig,
    fake_tracking: RecordingTrackingServices,
) -> None:
    case = in_memory_case(with_comparisons=True)
    config = mock_cfg(preconditioners=[mismatched_checkpoint_spec])
    recorded = {IdentityTag.COMPONENTS: json.dumps({"dataset": "sha256:" + "a" * 16})}
    with (
        patch.object(comparison_prepare, "resolve_comparison_config", return_value=config),
        patch.object(comparison_prepare, "validate_comparison_sources"),
        patch.object(
            comparison_prepare,
            "resolve_comparison_specs",
            return_value=ResolvedComparisonSpecs(
                specs=[mismatched_checkpoint_spec], warnings=(), checkpoint_digests={}
            ),
        ),
        patch.object(
            FakeMlflowClient, "get_run", return_value=MagicMock(data=MagicMock(tags=recorded))
        ),
        patch.object(
            comparison_prepare, "assignment_dataset_dir", return_value=Path("training-dataset")
        ),
        patch.object(
            comparison_prepare, "current_dataset_digest", return_value="sha256:" + "b" * 64
        ),
        patch.object(comparison_prepare, "prepare_comparison_presentation") as preflight,
        pytest.raises(ValueError, match="training dataset digest mismatch"),
    ):
        run_loaded_comparison_batch(
            case, VIRTUAL_CASE, ComparisonParams(), virtual_settings(), tracking=fake_tracking
        )
    preflight.assert_not_called()
    assert fake_tracking.runs.sessions == []


@pytest.mark.usefixtures(
    "stub_input_gate", "stub_planning", "stub_matrix_scope", "stub_setup_tracking"
)
def test_unavailable_checkpoint_provenance_emits_exact_warning(
    mismatched_checkpoint_spec: NeuralPreconditionerConfig,
    fake_tracking: RecordingTrackingServices,
) -> None:
    case = in_memory_case(with_comparisons=True)
    config = mock_cfg(preconditioners=[mismatched_checkpoint_spec])
    warnings: list[str] = []
    sink = logger.add(
        lambda message: (
            warnings.append(str(message).strip())
            if message.record["level"].name == "WARNING"
            else None
        ),
        format="{message}",
    )
    try:
        with (
            patch.object(comparison_prepare, "resolve_comparison_config", return_value=config),
            patch.object(comparison_prepare, "validate_comparison_sources"),
            patch.object(
                comparison_prepare,
                "resolve_comparison_specs",
                return_value=ResolvedComparisonSpecs(
                    specs=[mismatched_checkpoint_spec], warnings=(), checkpoint_digests={}
                ),
            ),
            patch.object(
                FakeMlflowClient, "get_run", return_value=MagicMock(data=MagicMock(tags={}))
            ),
            patch.object(
                comparison_prepare,
                "prepare_comparison_presentation",
                side_effect=_StopAfterProvenance,
            ),
            pytest.raises(_StopAfterProvenance),
        ):
            run_loaded_comparison_batch(
                case,
                VIRTUAL_CASE,
                ComparisonParams(),
                virtual_settings(),
                tracking=fake_tracking,
            )
    finally:
        logger.remove(sink)
    assert warnings == [
        (
            f"Method '{mismatched_checkpoint_spec.name}': training data provenance unavailable: "
            "training run has no dataset identity"
        )
    ]


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
        origins_by_method={"jacobi": NoFitData()},
        training_run_costs={},
        identity=StageIdentity.build("comparison", {"entry": canonical_digest(planned.entry.id)}),
        settings=context.settings,
        presentation=presentation,
        title=title,
    )


def _session_titles(
    presentation: PreparedComparisonPresentation,
    title: ComparisonTitle,
    fake_tracking: RecordingTrackingServices,
) -> list[str]:
    case = in_memory_case(with_comparisons=True)

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
    ):
        run_loaded_comparison_batch(
            case,
            VIRTUAL_CASE,
            ComparisonParams(),
            virtual_settings(),
            tracking=fake_tracking,
        )
    return [session.tags["title"] for session in fake_tracking.runs.sessions]


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_setup_tracking")
def test_session_run_is_titled_by_the_dataset_not_the_case_file(
    prepared_presentation: PreparedComparisonPresentation,
    fake_tracking: RecordingTrackingServices,
) -> None:
    (title,) = _session_titles(prepared_presentation, prepared_presentation.scope, fake_tracking)

    assert title == f"Comparison matrix set: {DATASET_TEXT} | 2 systems (1 RHS type x 1 matrix)"


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_setup_tracking")
def test_session_run_is_titled_by_the_editorial_title_when_one_is_configured(
    prepared_presentation: PreparedComparisonPresentation,
    fake_tracking: RecordingTrackingServices,
) -> None:
    (title,) = _session_titles(
        prepared_presentation, EditorialPresentation("Spectral comparison"), fake_tracking
    )

    assert title == "Spectral comparison | 2 systems (1 RHS type x 1 matrix)"


def test_a_method_run_and_a_plot_legend_render_the_same_presentation(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    prepared = prepare_comparison_presentation(
        [pod_spec(FIT_DIR, 10)],
        {pod_spec(FIT_DIR, 10).name: InlineFit(PODCoarseningConfig(dataset_dir=FIT_DIR))},
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        train_splits(FIT_DIR),
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


@pytest.mark.parametrize(
    ("kind", "expected"),
    [(PreconditionerType.IDENTITY, "Identity"), (PreconditionerType.JACOBI, "Jacobi")],
)
def test_bare_method_legend_and_run_name_keep_comparison_scope_separate(
    kind: PreconditionerType,
    expected: str,
    dataset_presentation: DatasetPresentation,
) -> None:
    cfg = StandardPreconditionerConfig(type=kind)
    prepared = prepare_comparison_presentation(
        [cfg],
        {cfg.name: NoFitData()},
        ComparisonScopePresentation(dataset_presentation),
        lambda path: pytest.fail("unexpected fit dataset read"),
        InMemorySplitStore(),
    )
    method = prepared.methods[0].compact
    assert method_legend({cfg.name: method}, PLAIN).label(cfg.name) == expected
    assert run_name_text(NamedRun(MethodRunName(method), PLAIN)) == expected
    assert render_title(prepared.scope, LEGEND_PROFILE, PLAIN).text == DATASET_TEXT


def test_configured_facts_become_stable_parameters_without_rendered_text(
    pod_fit_presentation: DatasetPresentation, dataset_presentation: DatasetPresentation
) -> None:
    prepared = prepare_comparison_presentation(
        [pod_spec(FIT_DIR, 10)],
        {pod_spec(FIT_DIR, 10).name: InlineFit(PODCoarseningConfig(dataset_dir=FIT_DIR))},
        ComparisonScopePresentation(dataset_presentation),
        datasets_by_dir({FIT_DIR: pod_fit_presentation}),
        train_splits(FIT_DIR),
    ).methods[0]

    parameters = encode_method_parameters(prepared)

    assert parameters["method.base_name"] == "POD"
    assert parameters["method.rank"] == "10"
    assert parameters == {
        "method.base_name": "POD",
        "method.rank": "10",
        "training_dataset.source": "Atomic test",
        "training_dataset.strategy_0.kind": "gaussian_forward",
        "training_dataset.strategy_0.mean": "0.0",
        "training_dataset.strategy_0.standard_deviation": "1.0",
        "training_dataset.strategy_0.sample_count": "8",
    }
