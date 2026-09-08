"""Characterization tests for comparison-batch checkpoint-ref iteration.

Pins the observable behavior of the three helpers that walk every
``(label, ref)`` pair exposed by ``CheckpointRefBearing`` preconditioner
specs — ``_needs_model_resolution``, ``_referenced_assignment_ids`` and
``_existing_assignment_ids`` — plus which coarsening configs
``AMGPreconditionerConfig.checkpoint_refs`` opts into resolution.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from neuralls.composition.assignments.comparison_batch import (
    _existing_assignment_ids,
    _needs_model_resolution,
    _referenced_assignment_ids,
)
from neuralls.platform.config.models.preconditioner import (
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    LoggedModelRefConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
    TargetDimCoarseningConfig,
)

type NeuralSpecFactory = Callable[..., NeuralPreconditionerConfig]
type AmgSpecFactory = Callable[..., AMGPreconditionerConfig]


@pytest.fixture
def identity_spec() -> StandardPreconditionerConfig:
    """A preconditioner that bears no checkpoint ref at all."""
    return StandardPreconditionerConfig(name="none", type=PreconditionerType.IDENTITY)


@pytest.fixture
def make_neural_spec() -> NeuralSpecFactory:
    """Build a neural preconditioner spec with optional assignment/model_ref."""

    def _build(
        *,
        name: str = "neural",
        assignment: str | None = None,
        model_ref: LoggedModelRefConfig | None = None,
    ) -> NeuralPreconditionerConfig:
        return NeuralPreconditionerConfig(
            name=name,
            type=PreconditionerType.NEURAL,
            assignment=assignment,
            model_ref=model_ref,
        )

    return _build


@pytest.fixture
def make_pod_amg_spec(tmp_path: Path) -> AmgSpecFactory:
    """Build an AMG spec whose POD coarsening carries optional checkpoint identity."""

    def _build(
        *,
        name: str = "amg",
        assignment: str | None = None,
        model_ref: LoggedModelRefConfig | None = None,
    ) -> AMGPreconditionerConfig:
        return AMGPreconditionerConfig(
            name=name,
            type=PreconditionerType.AMG,
            coarsening=PODCoarseningConfig(
                dataset_dir=tmp_path / "snapshots",
                rank=4,
                assignment=assignment,
                model_ref=model_ref,
            ),
        )

    return _build


@pytest.fixture
def logged_ref() -> LoggedModelRefConfig:
    """A model ref that triggers checkpoint resolution."""
    return LoggedModelRefConfig(latest=True, tags={"assignment_id": "a"})


def test_no_model_resolution_needed_without_checkpoint_bearing_specs(
    identity_spec: StandardPreconditionerConfig,
) -> None:
    assert _needs_model_resolution((identity_spec,)) is False


def test_no_model_resolution_needed_when_refs_carry_no_model_ref(
    make_neural_spec: NeuralSpecFactory,
) -> None:
    assert _needs_model_resolution((make_neural_spec(assignment="a"),)) is False


def test_model_resolution_needed_for_neural_model_ref(
    make_neural_spec: NeuralSpecFactory, logged_ref: LoggedModelRefConfig
) -> None:
    assert _needs_model_resolution((make_neural_spec(model_ref=logged_ref),)) is True


def test_model_resolution_needed_for_checkpoint_backed_pod_coarsening(
    make_pod_amg_spec: AmgSpecFactory, logged_ref: LoggedModelRefConfig
) -> None:
    """A POD-2G basis fitted ahead of time triggers resolution like a neural spec."""
    assert _needs_model_resolution((make_pod_amg_spec(model_ref=logged_ref),)) is True


def test_referenced_assignment_ids_preserves_first_seen_order(
    make_neural_spec: NeuralSpecFactory,
    make_pod_amg_spec: AmgSpecFactory,
    identity_spec: StandardPreconditionerConfig,
) -> None:
    specs = (
        make_neural_spec(name="n1", assignment="b"),
        identity_spec,
        make_pod_amg_spec(name="amg", assignment="a"),
        make_neural_spec(name="n2", assignment="b"),
    )
    assert _referenced_assignment_ids(specs) == ("b", "a")


def test_referenced_assignment_ids_skips_refs_without_an_assignment(
    make_neural_spec: NeuralSpecFactory,
) -> None:
    assert _referenced_assignment_ids((make_neural_spec(),)) == ()


def test_existing_assignment_ids_collects_claimed_ids_as_a_set(
    make_neural_spec: NeuralSpecFactory,
    make_pod_amg_spec: AmgSpecFactory,
    identity_spec: StandardPreconditionerConfig,
) -> None:
    specs = (
        make_neural_spec(name="n1", assignment="b"),
        identity_spec,
        make_pod_amg_spec(name="amg", assignment="a"),
        make_neural_spec(name="n2", assignment="b"),
    )
    assert _existing_assignment_ids(specs) == {"a", "b"}


def test_existing_assignment_ids_ignores_unassigned_refs(
    make_neural_spec: NeuralSpecFactory,
) -> None:
    assert _existing_assignment_ids((make_neural_spec(),)) == set()


def test_pod_coarsening_without_checkpoint_identity_is_invisible_to_resolution(
    make_pod_amg_spec: AmgSpecFactory,
) -> None:
    """A plain dataset_dir+rank POD config must keep falling through to the inline fit."""
    assert make_pod_amg_spec().checkpoint_refs() == ()


def test_pod_coarsening_with_assignment_exposes_its_ref(
    make_pod_amg_spec: AmgSpecFactory,
) -> None:
    spec = make_pod_amg_spec(assignment="pod2g")
    assert spec.checkpoint_refs() == (("coarsening", spec.coarsening),)


def test_pod_coarsening_with_model_ref_exposes_its_ref(
    make_pod_amg_spec: AmgSpecFactory, logged_ref: LoggedModelRefConfig
) -> None:
    spec = make_pod_amg_spec(model_ref=logged_ref)
    assert spec.checkpoint_refs() == (("coarsening", spec.coarsening),)


def test_pod_coarsening_with_checkpoint_path_exposes_its_ref(tmp_path: Path) -> None:
    spec = AMGPreconditionerConfig(
        name="amg",
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=tmp_path / "snapshots",
            rank=4,
            checkpoint_path=tmp_path / "basis.ckpt",
        ),
    )
    assert spec.checkpoint_refs() == (("coarsening", spec.coarsening),)


def test_neural_pod_coarsening_is_always_checkpoint_backed(tmp_path: Path) -> None:
    """Its snapshot ensemble only exists via a checkpoint, so it needs no identity fields."""
    spec = AMGPreconditionerConfig(
        name="amg",
        type=PreconditionerType.AMG,
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=tmp_path / "params",
            input_names=("params",),
            rank=4,
        ),
    )
    assert spec.checkpoint_refs() == (("coarsening", spec.coarsening),)


def test_aggregation_coarsening_exposes_no_checkpoint_ref() -> None:
    spec = AMGPreconditionerConfig(
        name="amg",
        type=PreconditionerType.AMG,
        coarsening=AggregationCoarseningConfig(),
    )
    assert spec.checkpoint_refs() == ()


def test_target_dim_coarsening_exposes_no_checkpoint_ref() -> None:
    spec = AMGPreconditionerConfig(
        name="amg",
        type=PreconditionerType.AMG,
        coarsening=TargetDimCoarseningConfig(target_coarse_dim=8),
    )
    assert spec.checkpoint_refs() == ()
