"""Tests for run_case_pipeline's composition of the generate, sweep, and compare stages."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuralls.application.models import AssignmentSweepResult
from neuralls.composition.assignments.case_pipeline import run_case_pipeline
from neuralls.composition.comparison.models import ComparisonParams


@dataclass
class _PipelineMocks:
    generate: MagicMock
    sweep: MagicMock
    load_cfg: MagicMock
    compare: MagicMock


def _patch_pipeline(stack: ExitStack, *, has_comparisons: bool) -> _PipelineMocks:
    cfg = MagicMock(comparisons=(MagicMock(),) if has_comparisons else ())
    stack.enter_context(
        patch(
            "neuralls.composition.assignments.case_pipeline.require_settings",
            side_effect=lambda settings, **_: settings,
        )
    )
    generate = stack.enter_context(
        patch("neuralls.composition.assignments.case_pipeline.generate_batch")
    )
    sweep = stack.enter_context(
        patch(
            "neuralls.composition.assignments.case_pipeline.run_assignment_sweep",
            return_value=AssignmentSweepResult(
                results=["assignment-result"], tracking_uri=None, parent_run_id=None
            ),
        )
    )
    load_cfg = stack.enter_context(
        patch(
            "neuralls.composition.assignments.case_pipeline.load_validated_case_config",
            return_value=(cfg, MagicMock()),
        )
    )
    compare = stack.enter_context(
        patch(
            "neuralls.composition.assignments.case_pipeline.run_comparison_batch",
            return_value=["comparison-outcome"],
        )
    )
    return _PipelineMocks(generate=generate, sweep=sweep, load_cfg=load_cfg, compare=compare)


def test_skips_comparison_batch_when_case_config_declares_none(tmp_path: Path) -> None:
    case_config_path = tmp_path / "case.toml"
    settings = MagicMock()
    with ExitStack() as stack:
        mocks = _patch_pipeline(stack, has_comparisons=False)
        assignment_results, comparison_outcomes = run_case_pipeline(case_config_path, settings)

    mocks.generate.assert_called_once()
    mocks.sweep.assert_called_once()
    mocks.load_cfg.assert_called_once()
    mocks.compare.assert_not_called()
    assert assignment_results == ["assignment-result"]
    assert comparison_outcomes == []


def test_runs_comparison_batch_when_case_config_declares_comparisons(tmp_path: Path) -> None:
    case_config_path = tmp_path / "case.toml"
    settings = MagicMock()
    with ExitStack() as stack:
        mocks = _patch_pipeline(stack, has_comparisons=True)
        assignment_results, comparison_outcomes = run_case_pipeline(case_config_path, settings)

    mocks.sweep.assert_called_once()
    mocks.compare.assert_called_once()
    assert assignment_results == ["assignment-result"]
    assert comparison_outcomes == ["comparison-outcome"]


def test_threads_force_flags_to_each_stage(tmp_path: Path) -> None:
    case_config_path = tmp_path / "case.toml"
    settings = MagicMock()
    with ExitStack() as stack:
        mocks = _patch_pipeline(stack, has_comparisons=True)
        run_case_pipeline(
            case_config_path,
            settings,
            force_generate=True,
            force_train=True,
            force_compare=True,
        )

    assert mocks.generate.call_args.kwargs["force"] is True

    sweep_kwargs = mocks.sweep.call_args.kwargs
    assert sweep_kwargs["force"] is True
    assert "force_generate" not in sweep_kwargs  # training never generates

    compare_args = mocks.compare.call_args.args
    compare_params = next(arg for arg in compare_args if isinstance(arg, ComparisonParams))
    assert compare_params.force is True


def test_run_case_pipeline_calls_generate_batch_as_an_explicit_stage(
    tmp_path: Path,
) -> None:
    """Dataset generation must be run_case_pipeline's own explicit stage-1 call
    — not something delegated back into the training stage — so `run` genuinely
    reuses the same generate-stage function `neuralls generate` uses, instead of
    a second, independent generation trigger buried inside the training step.
    """
    case_config_path = tmp_path / "case.toml"
    settings = MagicMock()
    call_order: list[str] = []
    with (
        patch(
            "neuralls.composition.assignments.case_pipeline.require_settings",
            side_effect=lambda settings, **_: settings,
        ),
        patch(
            "neuralls.composition.assignments.case_pipeline.load_validated_case_config",
            return_value=(MagicMock(comparisons=()), tmp_path),
        ),
        patch(
            "neuralls.composition.assignments.case_pipeline.generate_batch",
            side_effect=lambda *a, **k: call_order.append("generate"),
        ) as mock_generate,
        patch(
            "neuralls.composition.assignments.case_pipeline.run_assignment_sweep",
            side_effect=lambda *a, **k: (
                call_order.append("sweep")
                or AssignmentSweepResult(results=[], tracking_uri=None, parent_run_id=None)
            ),
        ),
    ):
        run_case_pipeline(case_config_path, settings, force_generate=True)

    assert call_order == ["generate", "sweep"]
    assert mock_generate.call_args.kwargs.get("force") is True


def test_sweep_runs_before_comparison_batch(tmp_path: Path) -> None:
    """Comparisons must resolve checkpoints from a sweep that already ran."""
    case_config_path = tmp_path / "case.toml"
    settings = MagicMock()
    call_order: list[str] = []
    with (
        patch(
            "neuralls.composition.assignments.case_pipeline.require_settings",
            side_effect=lambda settings, **_: settings,
        ),
        patch("neuralls.composition.assignments.case_pipeline.generate_batch"),
        patch(
            "neuralls.composition.assignments.case_pipeline.run_assignment_sweep",
            side_effect=lambda *a, **k: (
                call_order.append("sweep")
                or AssignmentSweepResult(results=[], tracking_uri=None, parent_run_id=None)
            ),
        ),
        patch(
            "neuralls.composition.assignments.case_pipeline.load_validated_case_config",
            return_value=(MagicMock(comparisons=(MagicMock(),)), MagicMock()),
        ),
        patch(
            "neuralls.composition.assignments.case_pipeline.run_comparison_batch",
            side_effect=lambda *a, **k: call_order.append("compare") or [],
        ),
    ):
        run_case_pipeline(case_config_path, settings)

    assert call_order == ["sweep", "compare"]
