"""Identity markers on config models: coverage walk and digest behaviour."""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Annotated, Union, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel

from neuralls.platform.config.models.comparison import (
    ComparisonConfig,
    ComparisonRhsSourceModel,
)
from neuralls.platform.config.models.data_models import DataConfigFile
from neuralls.platform.config.models.preconditioner import (
    CoarseningConfig,
    ConcretePreconditionerConfig,
    NeuralTransferConfig,
)
from neuralls.shared.digest import MARKER_TYPES, canonical_digest
from tests.configuration.conftest import ComparisonFactory, DataConfigFactory

ROOTS: list[object] = [
    pytest.param(ComparisonConfig, id="ComparisonConfig"),
    pytest.param(DataConfigFile, id="DataConfigFile"),
    pytest.param(ConcretePreconditionerConfig, id="preconditioners"),
    pytest.param(CoarseningConfig, id="coarsening"),
    pytest.param(ComparisonRhsSourceModel, id="rhs_sources"),
    pytest.param(NeuralTransferConfig, id="neural_transfer"),
]


def _unwrap(annotation: object) -> object:
    """Resolve ``type`` aliases and strip ``Annotated`` wrappers."""
    while True:
        if isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


def _members(annotation: object) -> list[object]:
    """Flatten unions and generic arguments into their component types."""
    annotation = _unwrap(annotation)
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType) or origin is not None:
        return [m for arg in get_args(annotation) for m in _members(arg)]
    return [annotation]


def _involves_path(annotation: object) -> bool:
    return any(isinstance(m, type) and issubclass(m, Path) for m in _members(annotation))


def _fields(cls: type) -> list[tuple[str, object, tuple[object, ...]]]:
    """Return ``(name, annotation, metadata)`` for each field of a model class."""
    if issubclass(cls, BaseModel):
        return [(n, f.annotation, tuple(f.metadata)) for n, f in cls.model_fields.items()]
    hints = get_type_hints(cls, include_extras=True)
    out = []
    for f in dataclasses.fields(cls):
        hint = hints[f.name]
        meta = get_args(hint)[1:] if get_origin(hint) is Annotated else ()
        out.append((f.name, hint, tuple(meta)))
    return out


def _is_model(member: object) -> bool:
    return isinstance(member, type) and (
        issubclass(member, BaseModel) or dataclasses.is_dataclass(member)
    )


def _reachable_models(root: object) -> dict[type, None]:
    """Collect every model/dataclass type reachable from ``root`` (ordered set)."""
    seen: dict[type, None] = {}
    stack = _members(root)
    while stack:
        member = stack.pop()
        if not _is_model(member) or member in seen:
            continue
        assert isinstance(member, type)
        seen[member] = None
        for _, annotation, _ in _fields(member):
            stack.extend(_members(annotation))
    return seen


@pytest.mark.parametrize("root", ROOTS)
def test_every_path_field_has_exactly_one_marker(root: object) -> None:
    offenders = []
    for cls in _reachable_models(root):
        for name, annotation, metadata in _fields(cls):
            if not _involves_path(annotation):
                continue
            if sum(isinstance(m, MARKER_TYPES) for m in metadata) != 1:
                offenders.append(f"{cls.__qualname__}.{name}")
    assert offenders == []


def test_comparison_digest_stable(make_comparison: ComparisonFactory) -> None:
    assert canonical_digest(make_comparison()) == canonical_digest(make_comparison())


@pytest.mark.parametrize(
    "override",
    [
        {"rtol": 1e-6},
        {"max_iterations": 7},
        {"ic0_threshold": 0.1},
        {"pod_rank": 5},
        {"rhs_std": 2.0},
    ],
    ids=lambda o: next(iter(o)),
)
def test_comparison_digest_changes_on_identity_field(
    make_comparison: ComparisonFactory, override: dict[str, object]
) -> None:
    assert canonical_digest(make_comparison()) != canonical_digest(make_comparison(**override))


def test_comparison_digest_ignores_cosmetic(
    make_comparison: ComparisonFactory, tmp_path: Path
) -> None:
    other_matrix = tmp_path / "other.bin"
    other_matrix.write_bytes(b"x")
    base = canonical_digest(make_comparison())
    assert base == canonical_digest(make_comparison(jacobi_name="renamed"))
    assert base == canonical_digest(make_comparison(matrix_path=other_matrix))


def test_data_config_digest_stable(make_data_config: DataConfigFactory) -> None:
    assert canonical_digest(make_data_config()) == canonical_digest(make_data_config())


@pytest.mark.parametrize("override", [{"seed": 1}, {"samples": 9}], ids=lambda o: next(iter(o)))
def test_data_config_digest_changes_on_identity_field(
    make_data_config: DataConfigFactory, override: dict[str, object]
) -> None:
    assert canonical_digest(make_data_config()) != canonical_digest(make_data_config(**override))


def test_data_config_digest_ignores_cosmetic(
    make_data_config: DataConfigFactory, tmp_path: Path
) -> None:
    base = canonical_digest(make_data_config())
    assert base == canonical_digest(make_data_config(dataset_id="renamed"))
    assert base == canonical_digest(make_data_config(data_dir=tmp_path / "elsewhere"))
