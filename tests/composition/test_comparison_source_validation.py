"""Characterization tests for `_validate_comparison_sources`.

Pins exactly which concrete input artifacts each RHS source kind prevalidates
before any MLflow run is opened — the per-kind dispatch that must stay
identical once it is routed through the comparison source-handler registry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from neuralls.composition.assignments.comparison_batch import _validate_comparison_sources
from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.shared.types import ComparisonRhsSourceKind, RowKind

type ConfigFactory = Callable[..., ComparisonConfig]


@pytest.fixture
def solver_params() -> SolverParams:
    """Minimal solver params; irrelevant to source validation but structurally required."""
    return SolverParams(
        rtol=1.0e-6,
        atol=1.0e-14,
        max_iterations=10,
        stopping_criterion="residual_norm",
        m_max=20,
    )


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    """An existing, loadable comparison matrix file."""
    path = tmp_path / "matrix.npy"
    np.save(path, np.eye(2, dtype=np.float64))
    return path


@pytest.fixture
def vector_file(tmp_path: Path) -> Path:
    """An existing, loadable comparison vector file usable as LHS or RHS."""
    path = tmp_path / "vector.npy"
    np.save(path, np.array([[1.0, 0.0]], dtype=np.float64))
    return path


@pytest.fixture
def missing_file(tmp_path: Path) -> Path:
    """A path that deliberately does not exist."""
    return tmp_path / "missing.npy"


@pytest.fixture
def make_config(solver_params: SolverParams) -> ConfigFactory:
    """Build a ComparisonConfig for one matrix path and RHS source."""

    def _build(
        *,
        matrix_path: Path,
        rhs_source_kind: ComparisonRhsSourceKind | None,
        rhs_source_params: dict[str, object] | None = None,
    ) -> ComparisonConfig:
        return ComparisonConfig(
            general=ComparisonGeneral(
                params=solver_params,
                data=ComparisonData(
                    matrix_path=matrix_path,
                    rhs_path=None,
                    rhs_source_kind=rhs_source_kind,
                    rhs_source_params=rhs_source_params,
                ),
            ),
            preconditioners=(),
        )

    return _build


def test_missing_rhs_source_kind_is_rejected(make_config: ConfigFactory, matrix_file: Path) -> None:
    config = make_config(matrix_path=matrix_file, rhs_source_kind=None)
    with pytest.raises(ValueError, match="must define rhs_source"):
        _validate_comparison_sources(config)


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        (ComparisonRhsSourceKind.GAUSSIAN, {"mean": 0.0, "std": 1.0}),
        (ComparisonRhsSourceKind.SPARSE, {"indices": [0], "values": [1.0]}),
    ],
)
def test_generated_sources_validate_only_the_matrix(
    make_config: ConfigFactory,
    matrix_file: Path,
    kind: ComparisonRhsSourceKind,
    params: dict[str, object],
) -> None:
    """Gaussian/sparse RHS is synthesized, so only the matrix input is prevalidated."""
    _validate_comparison_sources(
        make_config(matrix_path=matrix_file, rhs_source_kind=kind, rhs_source_params=params)
    )


@pytest.mark.parametrize("kind", [ComparisonRhsSourceKind.GAUSSIAN, ComparisonRhsSourceKind.SPARSE])
def test_generated_sources_reject_a_missing_matrix(
    make_config: ConfigFactory,
    missing_file: Path,
    kind: ComparisonRhsSourceKind,
) -> None:
    params: dict[str, object] = (
        {"mean": 0.0, "std": 1.0}
        if kind is ComparisonRhsSourceKind.GAUSSIAN
        else {"indices": [0], "values": [1.0]}
    )
    config = make_config(matrix_path=missing_file, rhs_source_kind=kind, rhs_source_params=params)
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_file))):
        _validate_comparison_sources(config)


def test_raw_lhs_validates_both_matrix_and_source_path(
    make_config: ConfigFactory, matrix_file: Path, vector_file: Path
) -> None:
    _validate_comparison_sources(
        make_config(
            matrix_path=matrix_file,
            rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
            rhs_source_params={
                "path": vector_file,
                "sample_index": 0,
                "row_kind": RowKind.STANDARD,
                "scale": 1.0,
            },
        )
    )


def test_raw_lhs_rejects_a_missing_source_path(
    make_config: ConfigFactory, matrix_file: Path, missing_file: Path
) -> None:
    config = make_config(
        matrix_path=matrix_file,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_LHS,
        rhs_source_params={"path": missing_file},
    )
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_file))):
        _validate_comparison_sources(config)


def test_raw_rhs_validates_both_matrix_and_source_path(
    make_config: ConfigFactory, matrix_file: Path, vector_file: Path
) -> None:
    _validate_comparison_sources(
        make_config(
            matrix_path=matrix_file,
            rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
            rhs_source_params={"path": vector_file},
        )
    )


def test_raw_rhs_rejects_a_missing_matrix(
    make_config: ConfigFactory, missing_file: Path, vector_file: Path
) -> None:
    config = make_config(
        matrix_path=missing_file,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
        rhs_source_params={"path": vector_file},
    )
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_file))):
        _validate_comparison_sources(config)


def test_dataset_source_validates_only_the_rhs_source_path(
    make_config: ConfigFactory, missing_file: Path, vector_file: Path
) -> None:
    """The DATASET branch never looks at matrix_path — a missing matrix passes here."""
    _validate_comparison_sources(
        make_config(
            matrix_path=missing_file,
            rhs_source_kind=ComparisonRhsSourceKind.DATASET,
            rhs_source_params={"path": vector_file},
        )
    )


def test_dataset_source_rejects_a_missing_source_path(
    make_config: ConfigFactory, matrix_file: Path, missing_file: Path
) -> None:
    config = make_config(
        matrix_path=matrix_file,
        rhs_source_kind=ComparisonRhsSourceKind.DATASET,
        rhs_source_params={"path": missing_file},
    )
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_file))):
        _validate_comparison_sources(config)


def test_source_params_are_validated_against_the_kinds_schema(
    make_config: ConfigFactory, matrix_file: Path
) -> None:
    """A raw source with no `path` fails schema validation before any file check."""
    config = make_config(
        matrix_path=matrix_file,
        rhs_source_kind=ComparisonRhsSourceKind.RAW_RHS,
        rhs_source_params={},
    )
    with pytest.raises(ValueError):
        _validate_comparison_sources(config)
