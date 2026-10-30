from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from tests.provenance_support import provenance_for

from neuralls.composition.comparison.models import (
    CompletedMethodWorkload,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.platform.config.models.preconditioner import (
    PreconditionerType,
)
from neuralls.platform.reporting.stored_benchmark import MethodDirectory, read_stored_benchmark
from neuralls.shared.types import DatasetId


def _dir(path: Path) -> MethodDirectory:
    return MethodDirectory(f"run-{path.name}", path)


def test_session_methods_round_trip_into_one_benchmark(
    tmp_path: Path,
    write_method: Callable[..., CompletedMethodWorkload],
    dataset_presentation: DatasetPresentation,
) -> None:
    jacobi = write_method(PreconditionerType.JACOBI, tmp_path / "jacobi")
    identity = write_method(PreconditionerType.IDENTITY, tmp_path / "identity")

    stored = read_stored_benchmark([_dir(tmp_path / "jacobi"), _dir(tmp_path / "identity")])

    assert [method.name for method in stored.methods] == [jacobi.method, identity.method]
    assert [method.config.type for method in stored.methods] == [
        PreconditionerType.JACOBI,
        PreconditionerType.IDENTITY,
    ]
    assert stored.methods[0].systems == provenance_for(jacobi.planned)
    assert stored.methods[0].presentation == jacobi.presentation
    assert [method.run_id for method in stored.methods] == ["run-jacobi", "run-identity"]
    assert [method.config_digest for method in stored.methods] == [
        jacobi.config_digest,
        identity.config_digest,
    ]
    assert stored.datasets == {DatasetId("ds"): dataset_presentation}
    assert stored.methods[0].solve_attempts.get_column("system_id").to_list() == ["jacobi-0"]


def test_an_empty_session_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one method"):
        read_stored_benchmark([])


def test_a_method_stored_twice_is_rejected(
    tmp_path: Path, write_method: Callable[..., CompletedMethodWorkload]
) -> None:
    write_method(PreconditionerType.JACOBI, tmp_path / "a")
    write_method(PreconditionerType.JACOBI, tmp_path / "b")

    with pytest.raises(ValueError, match="stored more than once"):
        read_stored_benchmark([_dir(tmp_path / "a"), _dir(tmp_path / "b")])


def test_a_presentation_of_another_method_is_rejected(
    tmp_path: Path, write_method: Callable[..., CompletedMethodWorkload]
) -> None:
    write_method(PreconditionerType.JACOBI, tmp_path / "jacobi")
    write_method(PreconditionerType.IDENTITY, tmp_path / "identity")
    (tmp_path / "jacobi" / "presentation.toml").write_bytes(
        (tmp_path / "identity" / "presentation.toml").read_bytes()
    )

    with pytest.raises(ValueError, match="does not belong to method"):
        read_stored_benchmark([_dir(tmp_path / "jacobi")])


def test_a_missing_artifact_is_rejected(
    tmp_path: Path, write_method: Callable[..., CompletedMethodWorkload]
) -> None:
    write_method(PreconditionerType.JACOBI, tmp_path / "jacobi")
    (tmp_path / "jacobi" / "convergence.parquet").unlink()

    with pytest.raises(FileNotFoundError):
        read_stored_benchmark([_dir(tmp_path / "jacobi")])


def test_datasets_presented_differently_by_two_methods_are_rejected(
    tmp_path: Path,
    write_method: Callable[..., CompletedMethodWorkload],
    dataset_presentation: DatasetPresentation,
) -> None:
    other = replace(
        dataset_presentation, generation=replace(dataset_presentation.generation, root_seed=7)
    )
    write_method(PreconditionerType.JACOBI, tmp_path / "jacobi")
    write_method(PreconditionerType.IDENTITY, tmp_path / "identity", dataset=other)

    with pytest.raises(ValueError, match="present dataset ds differently"):
        read_stored_benchmark([_dir(tmp_path / "jacobi"), _dir(tmp_path / "identity")])
