"""Tests for _compute_checkpoint_dependency_hash, the hash the comparison
reuse-check uses to notice when a dependency (a trained checkpoint) has changed.
"""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments.comparison_batch import (
    _compute_checkpoint_dependency_hash,
)
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)


def _resolved_neural_spec(
    *, name: str = "a", checkpoint_path: str = "/tmp/x.ckpt", run_id: str | None = "run-1"
) -> NeuralPreconditionerConfig:
    """A NeuralPreconditionerConfig as it looks after model resolution has run."""
    spec = NeuralPreconditionerConfig(name=name, checkpoint_path=Path(checkpoint_path))
    return spec.model_copy(
        update={
            "resolved_checkpoint_path": Path(checkpoint_path),
            "resolved_run_id": run_id,
        }
    )


def test_same_resolved_run_id_produces_the_same_hash() -> None:
    first = _compute_checkpoint_dependency_hash([_resolved_neural_spec(run_id="run-1")])
    second = _compute_checkpoint_dependency_hash([_resolved_neural_spec(run_id="run-1")])
    assert first == second


def test_a_different_resolved_run_id_changes_the_hash() -> None:
    """The exact scenario the reuse-check depends on: a retrain (new run_id)
    must invalidate the cached comparison's dependency hash."""
    before = _compute_checkpoint_dependency_hash([_resolved_neural_spec(run_id="run-1")])
    after = _compute_checkpoint_dependency_hash([_resolved_neural_spec(run_id="run-2")])
    assert before != after


def test_hash_is_independent_of_spec_order() -> None:
    """Comparison entries can list preconditioners in any order — the
    hash must not depend on that order."""
    spec_a = _resolved_neural_spec(name="a", run_id="run-1")
    spec_b = _resolved_neural_spec(name="b", run_id="run-2")
    forward = _compute_checkpoint_dependency_hash([spec_a, spec_b])
    reversed_order = _compute_checkpoint_dependency_hash([spec_b, spec_a])
    assert forward == reversed_order


def test_different_preconditioner_name_changes_the_hash() -> None:
    """Two distinct preconditioners resolved to the same run_id (unlikely but
    possible) must not collide into the same hash."""
    first = _compute_checkpoint_dependency_hash([_resolved_neural_spec(name="a", run_id="run-1")])
    second = _compute_checkpoint_dependency_hash([_resolved_neural_spec(name="b", run_id="run-1")])
    assert first != second


def test_non_checkpoint_specs_do_not_affect_the_hash() -> None:
    """A static preconditioner (identity/jacobi/...) has no checkpoint dependency
    and must not perturb the hash of the checkpoint-bearing specs beside it."""
    neural_spec = _resolved_neural_spec(run_id="run-1")
    static_spec = StandardPreconditionerConfig(name="identity", type=PreconditionerType.IDENTITY)

    without_static = _compute_checkpoint_dependency_hash([neural_spec])
    with_static = _compute_checkpoint_dependency_hash([neural_spec, static_spec])
    assert without_static == with_static


def test_falls_back_to_checkpoint_path_when_run_id_is_unset() -> None:
    """An explicit checkpoint_path (no model_ref/run_id) still yields a stable,
    distinguishing hash rather than colliding on a missing run_id."""
    first = _compute_checkpoint_dependency_hash(
        [_resolved_neural_spec(checkpoint_path="/tmp/a.ckpt", run_id=None)]
    )
    second = _compute_checkpoint_dependency_hash(
        [_resolved_neural_spec(checkpoint_path="/tmp/b.ckpt", run_id=None)]
    )
    assert first != second
