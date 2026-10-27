"""A job registry entry names its neural architecture explicitly, apart from its editorial title."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.experiments import CaseConfig, JobEntry
from neuralls.platform.config.registry import resolve_assignment_binding
from neuralls.shared.scientific_description.models import ValidatedLabel

TRAINING_TITLE = "Constant Width FFNN Optuna Search (ReLU, no bias)"


def _case(job: dict[str, str]) -> CaseConfig:
    return CaseConfig.model_validate(
        {
            "datasets": [{"id": "d", "path": "datasets/d.toml"}],
            "jobs": [{"id": "j", "path": "jobs/j.toml", **job}],
            "assignments": [{"id": "a", "dataset": "d", "job": "j"}],
        }
    )


def test_a_job_title_is_free_text_and_architecture_is_separate() -> None:
    cfg = _case({"display_name": TRAINING_TITLE, "architecture": "Constant-Width FFNN"})

    job = cfg.jobs[0]
    assert isinstance(job, JobEntry)
    assert job.display_name == TRAINING_TITLE
    assert job.architecture == "Constant-Width FFNN"


def test_a_job_without_architecture_is_valid_until_it_feeds_a_neural_comparison() -> None:
    assert _case({}).jobs[0].architecture is None


@pytest.mark.parametrize("bad", ["Wide FFNN (ReLU)", "ffnn_wide", "two  spaces", ""])
def test_an_invalid_architecture_fails_at_load_naming_the_job(bad: str) -> None:
    with pytest.raises(ValidationError, match="Job 'j' architecture"):
        _case({"architecture": bad})


def test_the_binding_carries_the_parsed_architecture_never_the_title(tmp_path: Path) -> None:
    cfg = _case({"display_name": TRAINING_TITLE, "architecture": "Wide FFNN"})

    binding = resolve_assignment_binding(cfg, tmp_path, cfg.assignments[0])

    assert binding.architecture == ValidatedLabel("Wide FFNN")


def test_the_binding_has_no_architecture_when_the_job_declares_none(tmp_path: Path) -> None:
    cfg = _case({"display_name": "Wide FFNN"})

    assert resolve_assignment_binding(cfg, tmp_path, cfg.assignments[0]).architecture is None
