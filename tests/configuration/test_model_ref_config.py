"""Validation tests for strict model_ref configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.checkpoint_refs import (
    ByAlias,
    ByRunId,
    ByVersion,
    LatestMatching,
    LatestVersion,
    LoggedModelRefConfig,
    RegisteredModelRefConfig,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
)


def test_registered_model_ref_requires_exactly_one_selector() -> None:
    """Registered refs must provide exactly one of alias/version/latest."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            model_ref=RegisteredModelRefConfig.model_validate(
                {"name": "dataset-model", "alias": "candidate", "version": 1}
            ),
        )


def test_logged_model_ref_requires_filter_for_latest() -> None:
    """Logged latest lookup requires at least one filter."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            model_ref=LoggedModelRefConfig.model_validate({"latest": True}),
        )


def test_logged_model_ref_with_run_id_is_valid() -> None:
    """run_id selector is valid for logged model refs."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="abc123")),
    )
    assert cfg.model_ref is not None
    assert cfg.model_ref.source == "run"


def test_assignment_and_registered_model_ref_name_together_is_rejected() -> None:
    """assignment and model_ref.name are competing identity sources; both is invalid."""
    with pytest.raises(ValidationError):
        NeuralPreconditionerConfig(
            type=PreconditionerType.NEURAL,
            assignment="my-assignment-id",
            model_ref=RegisteredModelRefConfig(
                name="unrelated-model", selector=ByAlias(alias="champion")
            ),
        )


def test_assignment_alone_with_registered_model_ref_alias_is_valid() -> None:
    """assignment plus an alias-only model_ref (no name) is the intended single-source case."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        assignment="my-assignment-id",
        model_ref=RegisteredModelRefConfig(selector=ByAlias(alias="champion")),
    )
    assert cfg.assignment == "my-assignment-id"
    assert isinstance(cfg.model_ref, RegisteredModelRefConfig)
    assert cfg.model_ref.name is None


def test_registered_model_ref_name_alone_without_assignment_is_valid() -> None:
    """model_ref.name with no assignment references an unrelated registered model."""
    cfg = NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        model_ref=RegisteredModelRefConfig(
            name="unrelated-model", selector=ByAlias(alias="champion")
        ),
    )
    assert cfg.assignment is None
    assert isinstance(cfg.model_ref, RegisteredModelRefConfig)
    assert cfg.model_ref.name == "unrelated-model"


@pytest.mark.parametrize(
    ("flat", "selector"),
    [
        ({"alias": "champion"}, ByAlias(alias="champion")),
        ({"version": 3}, ByVersion(version=3)),
        ({"latest": True}, LatestVersion()),
    ],
)
def test_registered_flat_toml_keys_fold_into_one_selector(
    flat: dict[str, object], selector: ByAlias | ByVersion | LatestVersion
) -> None:
    ref = RegisteredModelRefConfig.model_validate({"source": "registry", "name": "m", **flat})
    assert ref.selector == selector


@pytest.mark.parametrize(
    ("flat", "selector"),
    [
        ({"run_id": "abc"}, ByRunId(run_id="abc")),
        (
            {"latest": True, "experiment_name": "e", "tags": {"k": "v"}},
            LatestMatching(experiment_name="e", tags={"k": "v"}),
        ),
    ],
)
def test_logged_flat_toml_keys_fold_into_one_selector(
    flat: dict[str, object], selector: ByRunId | LatestMatching
) -> None:
    ref = LoggedModelRefConfig.model_validate({"source": "run", **flat})
    assert ref.selector == selector


def test_logged_run_id_with_filters_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LoggedModelRefConfig.model_validate({"run_id": "abc", "experiment_name": "e"})


@pytest.mark.parametrize(
    ("ref", "slug"),
    [
        (RegisteredModelRefConfig(name="Net", selector=ByAlias(alias="Best")), "registry_net_best"),
        (RegisteredModelRefConfig(name="Net", selector=ByVersion(version=2)), "registry_net_v2"),
        (RegisteredModelRefConfig(selector=LatestVersion()), "registry_assignment_latest"),
        (LoggedModelRefConfig(selector=ByRunId(run_id="Run-1")), "run_run_1"),
        (LoggedModelRefConfig(selector=LatestMatching(model_name="Net")), "run_latest_net"),
    ],
)
def test_model_ref_slug_names_its_selector(
    ref: RegisteredModelRefConfig | LoggedModelRefConfig, slug: str
) -> None:
    assert ref.slug() == slug
