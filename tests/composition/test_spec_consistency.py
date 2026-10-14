"""Tests for `validate_dataset_specs_consistent`.

It only reads `PlannedSystem` values, so the tests build those directly: no
dataset directory or case file is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.composition.comparison.spec_consistency import validate_dataset_specs_consistent
from neuralls.composition.identity.comparison import preconditioner_identities
from neuralls.domain.solver.models.provenance import HeldOutMatrix, SharedSingleMatrix
from neuralls.platform.config.models.experiments import ComparisonRegistryEntry
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.shared.digest import Digest
from neuralls.shared.types import MatrixId


def _entry(entry_id: str, *, matrix_dataset: str, rhs_kind: str) -> ComparisonRegistryEntry:
    rhs_source: dict[str, object] = {"kind": rhs_kind}
    if rhs_kind == "sinusoidal":
        rhs_source["omega"] = 6.283185307179586
    return ComparisonRegistryEntry(
        id=entry_id, matrix_dataset=matrix_dataset, rhs_source=rhs_source
    )


def _single(entry_id: str, *, rhs_kind: str = "gaussian") -> PlannedSystem:
    """A system on the `single` dataset's one shared matrix."""
    return PlannedSystem(
        entry=_entry(entry_id, matrix_dataset="single", rhs_kind=rhs_kind),
        matrix=MatrixRef(matrix_id=MatrixId(7), row=0),
        rhs_row=None,
        rhs_ordinal=0,
        provenance=SharedSingleMatrix(raw_matrix_index=0),
    )


def _multi(entry_id: str, matrix_id: int, *, rhs_kind: str = "gaussian") -> PlannedSystem:
    """A system on a held-out matrix of the `multi` dataset (ids differ from rows)."""
    return PlannedSystem(
        entry=_entry(entry_id, matrix_dataset="multi", rhs_kind=rhs_kind),
        matrix=MatrixRef(matrix_id=MatrixId(matrix_id), row=matrix_id * 2),
        rhs_row=None,
        rhs_ordinal=0,
        provenance=HeldOutMatrix(matrix_id=MatrixId(matrix_id), raw_matrix_index=matrix_id * 2),
    )


def _jacobi(**overrides: int) -> StandardPreconditionerConfig:
    """Jacobi's derived name ignores schedule fields, so `start_iter` yields a same-name conflict."""
    return StandardPreconditionerConfig(type=PreconditionerType.JACOBI, **overrides)


def _identities(spec: PreconditionerConfig, *digests: str) -> dict[str, Digest]:
    """Identity of one spec whose checkpoint (if any) has the given content digests."""
    return preconditioner_identities([spec], {f"0:ckpt{i}": d for i, d in enumerate(digests)})


def _two_entries() -> list[PlannedSystem]:
    return [_single("a"), _single("b")]


def test_dataset_whose_systems_define_one_name_differently_is_rejected() -> None:
    planned = _two_entries()

    with pytest.raises(ValueError, match="configured differently in entries 'a' and 'b'"):
        validate_dataset_specs_consistent(
            planned, {"a": _identities(_jacobi()), "b": _identities(_jacobi(start_iter=5))}
        )


def test_same_neural_name_with_different_checkpoint_content_is_rejected() -> None:
    """Both are `neural_assignment_pod_run`; only the checkpoint bytes differ."""
    planned = _two_entries()
    first = NeuralPreconditionerConfig(assignment="pod-run")
    second = NeuralPreconditionerConfig(assignment="pod-run")
    assert first.name == second.name

    with pytest.raises(ValueError, match="different checkpoint"):
        validate_dataset_specs_consistent(
            planned,
            {"a": _identities(first, "sha256:aaa"), "b": _identities(second, "sha256:bbb")},
        )


def test_same_neural_model_resolved_to_different_local_paths_is_accepted() -> None:
    """Same bytes at two temp paths is one preconditioner — no false conflict."""
    planned = _two_entries()
    at_a = NeuralPreconditionerConfig(
        assignment="pod-run", resolved_checkpoint_path=Path("/tmp/a/model.ckpt")
    )
    at_b = NeuralPreconditionerConfig(
        assignment="pod-run", resolved_checkpoint_path=Path("/tmp/b/model.ckpt")
    )
    assert at_a != at_b  # why `==` was the wrong comparison

    validate_dataset_specs_consistent(
        planned, {"a": _identities(at_a, "sha256:same"), "b": _identities(at_b, "sha256:same")}
    )


def test_identical_specs_or_cache_hit_systems_pass() -> None:
    planned = _two_entries()

    validate_dataset_specs_consistent(planned, {"a": _identities(_jacobi())})


def test_the_same_name_may_differ_across_datasets() -> None:
    """Consistency is per dataset: two datasets never share a build."""
    validate_dataset_specs_consistent(
        [_single("a"), _multi("b", 1)],
        {"a": _identities(_jacobi()), "b": _identities(_jacobi(start_iter=5))},
    )


def test_conflict_message_names_the_dataset() -> None:
    with pytest.raises(ValueError, match="Dataset 'single'"):
        validate_dataset_specs_consistent(
            _two_entries(),
            {"a": _identities(_jacobi()), "b": _identities(_jacobi(start_iter=5))},
        )
