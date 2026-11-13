"""The parse and bind steps of config loading read and write no file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neuralls.composition.comparison.config_assembler import bind_comparison_config
from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
from neuralls.domain.solver.models.provenance import SharedSingleMatrix
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.loaders import parse_case_config, parse_data_config
from neuralls.shared.types import MatrixId
from tests.workflows.comparison_support import (
    VIRTUAL_CASE_PATH,
    in_memory_case,
    virtual_settings,
)

DATASET_CONFIG_PATH = VIRTUAL_CASE_PATH.parent / "datasets" / "solutions.toml"


@pytest.fixture
def ctx() -> ConfigContext:
    return ConfigContext(config_path=VIRTUAL_CASE_PATH, settings=virtual_settings())


def test_a_dataset_entry_without_an_id_takes_the_id_of_its_dataset_config(
    ctx: ConfigContext,
) -> None:
    raw: dict[str, Any] = {"datasets": [{"path": "datasets/solutions.toml"}]}
    read: list[Path] = []

    def read_dataset_config(path: Path) -> dict[str, Any]:
        read.append(path)
        return {"id": "from-the-dataset-config"}

    case = parse_case_config(raw, ctx, read_dataset_config)

    assert [entry.id for entry in case.datasets] == ["from-the-dataset-config"]
    assert read == [DATASET_CONFIG_PATH.resolve()]


def test_generated_dataset_configs_are_materialized_once_sweeps_are_expanded(
    ctx: ConfigContext,
) -> None:
    raw: dict[str, Any] = {"datasets": [{"id": "plain", "path": "datasets/plain.toml"}]}
    seen: list[list[str]] = []

    def materialize(expanded: dict[str, Any], _ctx: ConfigContext) -> None:
        seen.append([entry["id"] for entry in expanded["datasets"]])

    parse_case_config(raw, ctx, materialize_generated=materialize)

    assert seen == [["plain"]]


def test_a_data_config_is_validated_from_its_mapping(ctx: ConfigContext) -> None:
    data_config = parse_data_config({"id": "solutions", "output": {}}, ctx)

    assert data_config.id == "solutions"


def test_binding_uses_the_given_matrix_path_and_the_case_baseline() -> None:
    case = in_memory_case(with_comparisons=True)
    entry, _ = case.comparisons
    planned = PlannedSystem(
        entry=entry,
        matrix=MatrixRef(matrix_id=MatrixId(3), row=5),
        rhs_row=None,
        rhs_ordinal=1,
        provenance=SharedSingleMatrix(raw_matrix_index=5),
    )
    matrix_path = Path("virtual/processed/solutions")

    bound = bind_comparison_config(case, planned, method_override=None, matrix_path=matrix_path)

    assert bound.general.data.matrix_path == matrix_path
    assert bound.general.data.matrix_index == 5
    assert bound.general.params.max_iterations == 10


def test_binding_without_a_baseline_is_refused() -> None:
    case = in_memory_case(with_comparisons=True).model_copy(update={"comparison_baseline": None})
    (entry, *_) = case.comparisons
    planned = PlannedSystem(
        entry=entry,
        matrix=MatrixRef(matrix_id=MatrixId(0), row=0),
        rhs_row=None,
        rhs_ordinal=1,
        provenance=SharedSingleMatrix(raw_matrix_index=0),
    )

    with pytest.raises(ValueError, match="comparison_baseline"):
        bind_comparison_config(case, planned, method_override=None, matrix_path=Path("virtual"))
