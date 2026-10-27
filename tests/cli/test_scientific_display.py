"""What the commands print about a comparison, a training and a dataset is the plain text of
the same typed presentation MLflow names its runs with, carries the actual parameters, and never
an id.

Execution is mocked; the results are built from the shared presentation fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger
from typer.testing import CliRunner

from neuralls.application.models import AssignmentResult
from neuralls.cli.main import app
from neuralls.composition.comparison.models import (
    ComparisonCompleted,
    ComparisonEntryFailed,
    ComparisonResult,
    ComparisonReused,
    UndescribedComparison,
)
from neuralls.composition.display import UNDESCRIBED_COMPARISON_TEXT
from neuralls.composition.generation.multi_generation import GenerationResult
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.presentation.run_names import (
    ComparisonRunName,
    EditorialTrainingSubject,
    HeldOutMatrixPart,
    NamedRun,
    RhsDrawPart,
    SystemQualifier,
    TrainingPresentation,
    Untimed,
)
from neuralls.domain.presentation.titles import (
    ComparisonScopePresentation,
    ComparisonTitle,
    EditorialPresentation,
)
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.recommendations import ComparisonRecommendations
from neuralls.shared.scientific_description.policy import NotationPolicy
from neuralls.shared.types import MatrixId

runner = CliRunner()

DATASET_TEXT = "Test problem | Gaussian CG Residuals (k=20, N=100)"
REGISTRY_ID = "cmp-secret-17"
EDITORIAL = "Spectral 3D comparison"


@pytest.fixture
def logged() -> Iterator[list[str]]:
    """Every message the commands log, captured from loguru."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def case_config(tmp_path: Path) -> Path:
    config = tmp_path / "case.toml"
    config.write_text("", encoding="utf-8")
    return config


@pytest.fixture
def comparison_run_on() -> Callable[[ComparisonTitle], NamedRun]:
    """The run of the system on matrix 3, right-hand side 1, under the given title."""

    def build(title: ComparisonTitle) -> NamedRun:
        qualifier = SystemQualifier((HeldOutMatrixPart(MatrixId(3)), RhsDrawPart(1)))
        return NamedRun(ComparisonRunName(title, qualifier, Untimed()), NotationPolicy.default())

    return build


@pytest.fixture
def comparison_payload(tmp_path: Path) -> ComparisonResult:
    general = ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=5,
        ),
        data=ComparisonData(
            matrix_path=tmp_path / "matrix.npy",
            rhs_path=tmp_path / "rhs.npy",
        ),
    )
    return ComparisonResult(
        results={},
        summary="CG comparison results:\n- Jacobi  status=ok",
        solver_params=general,
        preconditioners=("jacobi",),
        recommendations=ComparisonRecommendations(),
    )


def _invoke_compare(config: Path, outcomes: list[object]) -> list[str]:
    with (
        patch("neuralls.cli.compare.run_comparison_batch", return_value=outcomes),
        patch("neuralls.cli.compare.load_case_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["compare", str(config)])
    return [result.output, str(result.exit_code)]


def test_compare_logs_the_plain_run_name_with_the_actual_parameters(
    case_config: Path,
    logged: list[str],
    dataset_presentation: DatasetPresentation,
    comparison_run_on: Callable[[ComparisonTitle], NamedRun],
    comparison_payload: ComparisonResult,
) -> None:
    run = comparison_run_on(ComparisonScopePresentation(dataset_presentation))
    outcome = ComparisonCompleted(comparison_id=REGISTRY_ID, run=run, payload=comparison_payload)

    _invoke_compare(case_config, [outcome])

    assert f"{DATASET_TEXT} | matrix 3, RHS 1" in logged
    assert not any(REGISTRY_ID in message for message in logged)
    assert not any("jacobi" in message for message in logged)


def test_compare_shows_an_editorial_title_as_the_case_title(
    case_config: Path,
    logged: list[str],
    comparison_run_on: Callable[[ComparisonTitle], NamedRun],
    comparison_payload: ComparisonResult,
) -> None:
    run = comparison_run_on(EditorialPresentation(EDITORIAL))
    outcome = ComparisonCompleted(comparison_id=REGISTRY_ID, run=run, payload=comparison_payload)

    _invoke_compare(case_config, [outcome])

    assert f"{EDITORIAL} | matrix 3, RHS 1" in logged
    assert "CG comparison results:\n- Jacobi  status=ok" in logged


def test_compare_names_a_failure_by_its_description_and_states_when_it_has_none(
    case_config: Path,
    logged: list[str],
    dataset_presentation: DatasetPresentation,
    comparison_run_on: Callable[[ComparisonTitle], NamedRun],
) -> None:
    described = comparison_run_on(ComparisonScopePresentation(dataset_presentation))
    outcomes = [
        ComparisonEntryFailed(comparison_id=REGISTRY_ID, run=described, reason="solver diverged"),
        ComparisonEntryFailed(
            comparison_id="cmp-other-9", run=UndescribedComparison(), reason="no checkpoint"
        ),
        ComparisonReused(comparison_id="cmp-third-3", run=described),
    ]

    _invoke_compare(case_config, outcomes)

    assert f"  x {DATASET_TEXT} | matrix 3, RHS 1: solver diverged" in logged
    assert f"  x {UNDESCRIBED_COMPARISON_TEXT}: no checkpoint" in logged
    assert not any(registry in message for message in logged for registry in (REGISTRY_ID, "cmp-"))


def test_run_summary_names_trainings_and_comparisons_by_their_descriptions(
    case_config: Path,
    dataset_presentation: DatasetPresentation,
    training_presentation_of: Callable[[str], TrainingPresentation],
    comparison_run_on: Callable[[ComparisonTitle], NamedRun],
) -> None:
    editorial = TrainingPresentation(
        EditorialTrainingSubject(EditorialPresentation(EDITORIAL), dataset_presentation),
        NotationPolicy.default(),
    )
    results = [
        AssignmentResult("asg-1", training_presentation_of("POD-2G"), status="Success"),
        AssignmentResult("asg-2", editorial, status="Failed", error="out of memory"),
    ]
    outcomes = [
        ComparisonReused(
            comparison_id=REGISTRY_ID,
            run=comparison_run_on(ComparisonScopePresentation(dataset_presentation)),
        )
    ]
    with (
        patch("neuralls.cli.run.run_case_pipeline", return_value=(results, outcomes)),
        patch("neuralls.cli.run.load_case_settings", return_value=MagicMock()),
    ):
        result = runner.invoke(app, ["run", str(case_config)])

    assert "  SUCCESS POD-2G | Gaussian CG Residuals (k=20, N=100)\n" in result.output
    assert f"  FAILED {EDITORIAL} | {DATASET_TEXT} (out of memory)\n" in result.output
    assert f"  SUCCESS {DATASET_TEXT} | matrix 3, RHS 1\n" in result.output
    assert "asg-" not in result.output
    assert REGISTRY_ID not in result.output


def test_generate_prints_the_dataset_description_beside_where_it_was_written(
    case_config: Path, tmp_path: Path, dataset_presentation: DatasetPresentation
) -> None:
    generated = GenerationResult(
        dataset_id="residuals-registry-id",
        config_path=tmp_path / "residuals.toml",
        output_dir=tmp_path / "out",
        dataset=dataset_presentation,
        notation=NotationPolicy.default(),
    )
    with (
        patch("neuralls.cli.generate.generate_batch", return_value=[generated]),
        patch("neuralls.cli.generate.load_validated_case_config") as load_case,
        patch("neuralls.cli.generate.load_case_settings", return_value=MagicMock()),
    ):
        load_case.return_value = (MagicMock(), MagicMock())
        result = runner.invoke(app, ["generate", str(case_config)])

    assert result.exit_code == 0
    assert f"{DATASET_TEXT} -> {tmp_path / 'out'}" in result.output
    assert "residuals-registry-id" not in result.output
