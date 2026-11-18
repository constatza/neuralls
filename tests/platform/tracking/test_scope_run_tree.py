"""``ScopeRunTree``: scope runs are created parents first, once, with their scope identity."""

from __future__ import annotations

import pytest
from tests.fakes.run_tree import FakeRunTree
from tests.recipe_support import GAUSSIAN

from neuralls.domain.solver.models.report_scope import (
    AllMatrixSets,
    MatrixSet,
    MatrixSetRhsType,
    ReportScope,
)
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.platform.tracking.scope_run_tree import REPORT_SCOPE_PHASE, ScopeRunTree
from neuralls.shared.types import DatasetId

ROOT = "root-run"


@pytest.fixture
def runs() -> FakeRunTree:
    return FakeRunTree()


@pytest.fixture
def tree(runs: FakeRunTree) -> ScopeRunTree:
    return ScopeRunTree(runs, lambda scope: type(scope).__name__)


@pytest.fixture
def rhs_type() -> MatrixSetRhsType:
    return MatrixSetRhsType(DatasetId("ds"), GAUSSIAN)


def test_the_root_scope_is_the_open_run_and_creates_nothing(
    tree: ScopeRunTree, runs: FakeRunTree
) -> None:
    assert tree.run_id(AllMatrixSets(), ROOT) == ROOT
    assert runs.created == []


def test_an_rhs_type_creates_its_ancestors_first_in_order(
    tree: ScopeRunTree, runs: FakeRunTree, rhs_type: MatrixSetRhsType
) -> None:
    run_id = tree.run_id(rhs_type, ROOT)

    expected: list[ReportScope] = [MatrixSet(rhs_type.dataset), rhs_type]
    assert tree.created_scopes == tuple(expected)
    assert [(run.parent_run_id, run.name) for run in runs.created] == [
        (ROOT, "MatrixSet"),
        ("child-1", "MatrixSetRhsType"),
    ]
    assert run_id == "child-2"


def test_a_scope_is_created_once_and_carries_its_identity(
    tree: ScopeRunTree, runs: FakeRunTree, rhs_type: MatrixSetRhsType
) -> None:
    first = tree.run_id(rhs_type, ROOT)
    again = tree.run_id(rhs_type, ROOT)

    assert first == again
    assert len(runs.created) == 2
    created = runs.created[-1]
    assert dict(created.params) == scope_params(rhs_type)
    assert created.tags["phase"] == REPORT_SCOPE_PHASE
    assert created.tags["report_level"] == scope_params(rhs_type)["report_level"]
