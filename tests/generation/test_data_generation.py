from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.linalg import eigh

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.helpers import (
    _generate_eigenvector_combinations,
)
from neuralls.domain.generation.interfaces import (
    ArchiveData,
    ErrorTraceRows,
    GeneratedSamples,
    ResidualTraceRows,
    SolvedRows,
    TracingSolverCallable,
)
from tests.fakes.archive_source import InMemoryArchiveSource

ARCHIVE_SOURCE = InMemoryArchiveSource()


def _pairs(generated: GeneratedSamples) -> tuple[np.ndarray, np.ndarray]:
    """The (rhs, solutions) training pairs one strategy emits, trace rows included."""
    match generated:
        case ErrorTraceRows(error_traces=traces):
            return traces.residuals, traces.errors
        case ResidualTraceRows(residual_traces=traces):
            return traces.residuals, traces.solutions
        case SolvedRows(rhs=rhs, solutions=solutions):
            return rhs, solutions


def _generate(
    matrix: np.ndarray,
    counts: dict[str, int],
    *,
    seed: int = 42,
    overrides: dict[str, dict[str, Any]] | None = None,
    solver_overrides: dict[str, TracingSolverCallable] | None = None,
    archive_solutions: np.ndarray | None = None,
    archive_rhs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run each strategy for its count and stack the rows in mixture order.

    Returns:
        ``(rhs, solutions)``, each of shape (rows, n).
    """
    archive = (
        ArchiveData(lhs=archive_solutions, rhs=archive_rhs)
        if archive_solutions is not None
        else None
    )
    rhs_blocks: list[np.ndarray] = []
    solution_blocks: list[np.ndarray] = []
    for name, count in counts.items():
        if count == 0:
            continue
        cfg = {**(overrides or {}).get(name, {}), "samples": count, "seed": seed}
        generated = run_generation(
            name,
            matrix,
            cfg=cfg,
            solver=(solver_overrides or {}).get(name),
            archive=archive,
            archive_source=ARCHIVE_SOURCE,
        )
        rhs, solutions = _pairs(generated)
        rhs_blocks.append(rhs)
        solution_blocks.append(solutions)
    return np.vstack(rhs_blocks), np.vstack(solution_blocks)


# ========================================================================
# Tests for residuals strategy
# ========================================================================


@pytest.mark.parametrize(
    ("counts", "residuals_options"),
    [
        pytest.param({"residuals": 2}, {"stop": 3, "start": 1}, id="two-rows"),
        pytest.param({"residuals": 3}, {"stop": 2, "start": 1}, id="more-rows-than-stop"),
        pytest.param({"residuals": 2}, {"stop": 4, "start": 1}, id="long-trace"),
        pytest.param({"residuals": 3}, {"stop": 4, "start": 1}, id="trimmed-from-base-systems"),
        pytest.param(
            {"neutral_ones": 1, "residuals": 6}, {"stop": 2, "start": 1}, id="after-forward"
        ),
        pytest.param({"normal": 2, "residuals": 2}, {"stop": 3, "start": 1}, id="after-generated"),
    ],
)
def test_residual_trace_rows_satisfy_the_equation(
    small_spd_matrix: np.ndarray,
    archive_solutions: np.ndarray,
    archive_rhs: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
    counts: dict[str, int],
    residuals_options: dict[str, int],
) -> None:
    """One row per requested sample, mixed strategies concatenated, A @ sol = rhs for every row."""
    rhs, solutions = _generate(
        small_spd_matrix,
        counts,
        overrides={"residuals": residuals_options},
        solver_overrides=solver_overrides,
        seed=test_seed,
        archive_solutions=archive_solutions,
        archive_rhs=archive_rhs,
    )

    assert rhs.shape == (sum(counts.values()), small_spd_matrix.shape[0])
    assert solutions.shape == rhs.shape
    np.testing.assert_allclose((small_spd_matrix @ solutions.T).T, rhs, atol=1e-10)


def test_error_strategy_validation(
    small_spd_matrix: np.ndarray,
    test_seed: int,
    solver_overrides: dict,
) -> None:
    """The strategy rejects an archive with too few solutions for its base systems."""
    insufficient_archive = np.array([[0.5, 0.3]], dtype=np.float64)

    with pytest.raises(ValueError):
        _generate(
            small_spd_matrix,
            {"residuals": 8},
            overrides={"residuals": {"stop": 3, "start": 1}},
            solver_overrides=solver_overrides,
            seed=test_seed,
            archive_solutions=insufficient_archive,
        )


# =============================================================================
# EIGENVECTOR STRATEGY TESTS
# =============================================================================

_SPD_2X2 = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.mark.parametrize(
    ("matrix", "counts", "overrides", "tolerance"),
    [
        pytest.param(_SPD_2X2, {"eigenvector_forward": 2}, {}, 1e-14, id="forward"),
        pytest.param(_SPD_2X2, {"eigenvector_inverse": 2}, {}, 1e-14, id="inverse"),
        pytest.param(
            np.eye(3) * 2.0,
            {"eigenvector_forward": 5},
            {},
            1e-14,
            id="more-samples-than-eigenvectors",
        ),
        pytest.param(
            np.eye(10) * 2.0,
            {"random": 10, "eigenvector_forward": 10},
            {},
            1e-9,
            id="mixed-with-random",
        ),
        pytest.param(
            np.diag([1.0, 2.0, 3.0, 4.0, 5.0]),
            {"eigenvector_forward": 20},
            {
                "eigenvector_forward": {
                    "num_eigenvectors": 3,
                    "which": "smallest",
                    "include_eigenvectors": False,
                }
            },
            1e-14,
            id="forward-combinations-only",
        ),
        pytest.param(
            _SPD_2X2,
            {"eigenvector_inverse": 10},
            {"eigenvector_inverse": {"num_eigenvectors": 2, "include_eigenvectors": False}},
            1e-14,
            id="inverse-combinations-only",
        ),
        pytest.param(
            _SPD_2X2,
            {"eigenvector_forward": 2},
            {"eigenvector_forward": {"num_eigenvectors": 2, "include_eigenvectors": False}},
            1e-14,
            id="forward-rhs-is-a-times-solution",
        ),
        pytest.param(
            np.diag([1.0, 2.0, 3.0]),
            {"eigenvector_forward": 3},
            {"eigenvector_forward": {}},
            1e-14,
            id="default-options",
        ),
    ],
)
def test_eigenvector_samples_satisfy_the_equation(
    matrix: np.ndarray,
    counts: dict[str, int],
    overrides: dict[str, dict[str, Any]],
    tolerance: float,
) -> None:
    """Every row of every eigenvector mixture solves A x = b to the stated relative residual."""
    rhs, solutions = _generate(matrix, counts, overrides=overrides)

    rows = sum(counts.values())
    assert rhs.shape == (rows, matrix.shape[0])
    assert solutions.shape == rhs.shape
    relative = np.linalg.norm(rhs - solutions @ matrix.T, axis=1) / np.linalg.norm(rhs, axis=1)
    assert np.all(relative < tolerance)


def test_eigenvector_requires_symmetric() -> None:
    """Eigenvector strategies reject non-symmetric matrices."""
    A = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)  # Asymmetric

    with pytest.raises(ValueError):
        _generate(A, {"eigenvector_forward": 2})


def test_eigenvector_selection_modes() -> None:
    """Different eigenvalue range selections give different eigenvector subspaces."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])

    _, sols_first = _generate(
        A,
        {"eigenvector_forward": 2},
        overrides={"eigenvector_forward": {"which": "smallest", "num_eigenvectors": 2}},
    )
    _, sols_last = _generate(
        A,
        {"eigenvector_forward": 2},
        overrides={"eigenvector_forward": {"which": "largest", "num_eigenvectors": 2}},
    )

    assert not np.allclose(sols_first, sols_last)


# =============================================================================
# EIGENVECTOR LINEAR COMBINATIONS: Tests for random linear combinations
# =============================================================================


def test_generate_eigenvector_combinations_vectorized() -> None:
    """Vectorized linear combination generation."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    _eigenvalues, eigenvectors = eigh(A)

    # Select first 3 eigenvectors
    selected = eigenvectors[:, :3]  # Shape (4, 3)

    rng = np.random.default_rng(42)
    combinations = _generate_eigenvector_combinations(selected, num_samples=10, rng=rng)

    assert combinations.shape == (10, 4), "Output shape mismatch"

    # Each combination should be in span of selected eigenvectors
    for i in range(10):
        reconstruction = selected @ (selected.T @ combinations[i])
        np.testing.assert_allclose(reconstruction, combinations[i], rtol=1e-10)

    # Combinations should differ from one another
    for i in range(9):
        assert not np.allclose(combinations[i], combinations[i + 1])


def test_eigenvector_forward_include_eigenvectors() -> None:
    """Including original eigenvectors puts them first, then combinations."""
    A = np.diag([1.0, 2.0, 3.0, 4.0])

    _rhs, solutions = _generate(
        A,
        {"eigenvector_forward": 6},  # 2 eigenvectors + 4 combinations
        overrides={
            "eigenvector_forward": {
                "num_eigenvectors": 2,
                "which": "smallest",
                "include_eigenvectors": True,
            }
        },
    )

    assert solutions.shape == (6, 4)

    # First 2 should be eigenvectors (diagonal matrix -> standard basis)
    _eigenvalues, eigenvectors = eigh(A)
    np.testing.assert_allclose(solutions[:2], eigenvectors[:, :2].T, rtol=1e-10)

    # Remaining 4 should be combinations (not any single eigenvector)
    for i in range(2, 6):
        for j in range(4):
            assert not np.allclose(solutions[i], eigenvectors[:, j])


@pytest.mark.parametrize(
    ("dimension", "counts", "options"),
    [
        pytest.param(
            3,
            {"eigenvector_forward": 5},
            {"num_eigenvectors": 5},
            id="more-eigenvectors-than-dim",
        ),
        pytest.param(
            5,
            {"eigenvector_forward": 2},
            {"num_eigenvectors": 3, "include_eigenvectors": True},
            id="include-needs-a-sample-per-eigenvector",
        ),
    ],
)
def test_eigenvector_option_validation(
    dimension: int, counts: dict[str, int], options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        _generate(
            np.eye(dimension, dtype=np.float64) * 2.0,
            counts,
            overrides={"eigenvector_forward": options},
        )
