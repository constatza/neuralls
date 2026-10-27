"""Realized facts come from the built hierarchy, and an unknown class fails instead of going blank."""

from __future__ import annotations

import inspect
from types import ModuleType

import pytest
import torch
from torchalg.preconditioners.base import Preconditioner, PreconditionerNotReadyError
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    AMGPreconditioner,
    JacobiSmoother,
    VCycle,
)
from torchalg.sparse.preconditioners.amg.coarsening import (
    AggregationCoarsening as SparseAggregationCoarsening,
)
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg

from neuralls.composition.preconditioners import builders, coarsening
from neuralls.composition.preconditioners.descriptions import (
    UnsupportedRealizedPreconditionerError,
    _coarsening_facts,
    describe_realized_preconditioner,
)
from neuralls.shared.scientific_description.catalog import COARSE_DIMENSION


@pytest.fixture
def tridiagonal() -> torch.Tensor:
    n = 6
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def built_aggregation_amg(tridiagonal: torch.Tensor) -> AMGPreconditioner:
    cycle = VCycle(smoother=JacobiSmoother(omega=0.67), n_pre=1, n_post=1)
    return AMGPreconditioner(
        tridiagonal, coarsening=AggregationCoarsening(omega=0.67), cycle=cycle, n_levels=2
    ).setup(tridiagonal)


def test_capturing_aggregation_facts_does_not_build_the_transfer_again(
    built_aggregation_amg: AMGPreconditioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = int(
        AggregationCoarsening(omega=0.67).build_transfer(built_aggregation_amg.matrix)[0].shape[0]
    )
    calls: list[torch.Tensor] = []

    def spy(self: AggregationCoarsening, matrix: torch.Tensor) -> object:
        calls.append(matrix)
        raise AssertionError("realized capture must read the built hierarchy")

    monkeypatch.setattr(AggregationCoarsening, "build_transfer", spy)

    facts = describe_realized_preconditioner(built_aggregation_amg)

    assert calls == []
    assert [(fact.key, fact.value) for fact in facts] == [(COARSE_DIMENSION, expected)]


def test_sparse_aggregation_amg_reports_its_coarse_dimension(tridiagonal: torch.Tensor) -> None:
    built = sparse_vcycle_amg(tridiagonal.to_sparse_csr(), theta=0.25, n_levels=2).setup(
        tridiagonal.to_sparse_csr()
    )

    facts = describe_realized_preconditioner(built)

    assert [fact.key for fact in facts] == [COARSE_DIMENSION]


def test_describing_an_unbuilt_preconditioner_fails_loudly(tridiagonal: torch.Tensor) -> None:
    cycle = VCycle(smoother=JacobiSmoother(omega=0.67), n_pre=1, n_post=1)
    unbuilt = AMGPreconditioner(
        tridiagonal, coarsening=AggregationCoarsening(omega=0.67), cycle=cycle, n_levels=2
    )

    with pytest.raises(PreconditionerNotReadyError):
        describe_realized_preconditioner(unbuilt)


class _UnknownPreconditioner(Preconditioner):
    def setup(self, matrix: torch.Tensor, context: object = None) -> _UnknownPreconditioner:
        return self

    def apply(self, residual: torch.Tensor, context: object = None) -> torch.Tensor:
        return residual


class _UnknownCoarsening:
    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, object]:
        return A, object()


def test_an_unknown_preconditioner_class_fails_instead_of_describing_nothing() -> None:
    with pytest.raises(UnsupportedRealizedPreconditionerError, match="_UnknownPreconditioner"):
        describe_realized_preconditioner(_UnknownPreconditioner())


def test_an_unknown_coarsening_class_fails_instead_of_describing_nothing(
    built_aggregation_amg: AMGPreconditioner,
) -> None:
    with pytest.raises(UnsupportedRealizedPreconditionerError, match="_UnknownCoarsening"):
        _coarsening_facts(_UnknownCoarsening(), built_aggregation_amg)  # type: ignore[arg-type]


def _classes_defined_outside_neuralls(*modules: ModuleType) -> set[type]:
    return {
        member
        for module in modules
        for _, member in inspect.getmembers(module, inspect.isclass)
        if not member.__module__.startswith("neuralls.")
        and not member.__module__.startswith("torch.")
    }


def test_every_preconditioner_class_the_builders_construct_is_registered() -> None:
    default = describe_realized_preconditioner.registry[object]
    built = {
        cls
        for cls in _classes_defined_outside_neuralls(builders)
        if issubclass(cls, Preconditioner) and cls is not Preconditioner
    }
    built |= {
        cls
        for _, cls in inspect.getmembers(builders, inspect.isclass)
        if cls.__module__.startswith("neuralls.") and issubclass(cls, Preconditioner)
    }

    assert built, "the builders module should import the torchalg preconditioners it builds"
    unregistered = {
        cls.__name__ for cls in built if describe_realized_preconditioner.dispatch(cls) is default
    }
    assert unregistered == set()


def test_every_coarsening_class_the_repo_builds_is_registered() -> None:
    default = _coarsening_facts.registry[object]
    named = {
        cls
        for cls in _classes_defined_outside_neuralls(builders, coarsening)
        if cls.__name__.endswith(("Coarsening", "CoarseningStrategy"))
    }
    named.add(SparseAggregationCoarsening)

    assert len(named) >= 5
    unregistered = {cls.__name__ for cls in named if _coarsening_facts.dispatch(cls) is default}
    assert unregistered == set()
