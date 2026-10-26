"""Explicit comparison-reference configuration and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.models.comparison import parse_comparison_method_override
from neuralls.platform.config.models.experiments import ComparisonBaseline
from neuralls.shared.types import PreconditionerName


def _general() -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1e-6,
            atol=1e-12,
            max_iterations=20,
            stopping_criterion="residual_norm",
            m_max=5,
        ),
        data=ComparisonData(matrix_path="matrix.npy", rhs_path=None),
    )


def test_comparison_general_and_baseline_carry_explicit_reference() -> None:
    baseline = ComparisonBaseline(reference_preconditioner="amg")
    general = ComparisonGeneral(
        params=_general().params,
        data=_general().data,
        reference_preconditioner=PreconditionerName("amg"),
    )

    assert baseline.reference_preconditioner == "amg"
    assert general.reference_preconditioner == "amg"


def test_method_override_reference_is_explicit_and_can_clear_baseline() -> None:
    override = parse_comparison_method_override(
        {"general": {"reference_preconditioner": None}},
        ConfigContext(config_path=Path("override.toml"), settings=None),
    )

    assert "reference_preconditioner" in override.general.model_fields_set
    assert override.general.reference_preconditioner is None


def test_static_reference_validation_names_available_methods() -> None:
    from neuralls.domain.benchmarking.selection import validate_reference_preconditioner

    with pytest.raises(ValueError):
        validate_reference_preconditioner("amg", ("identity", "jacobi"))


def test_runtime_resolution_skip_does_not_fallback_to_observed_method() -> None:
    from neuralls.domain.benchmarking.selection import (
        ReferenceUnavailable,
        resolve_reference_selection,
    )
    from neuralls.domain.solver.models.outcomes import Solved
    from neuralls.shared.types import MatrixId

    selection = resolve_reference_selection(
        "amg",
        ("jacobi",),
        skipped={"amg": "checkpoint not found"},
    )

    assert isinstance(selection, ReferenceUnavailable)
    assert selection.name == PreconditionerName("amg")
    assert Solved(MatrixId(1), 1, 1.0, 1.0)
