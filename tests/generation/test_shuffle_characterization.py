"""Characterization tests pinning the sample-shuffle permutation semantics.

Generation shuffles feature/target rows together with any aligned trace
metadata. Traces reference their base system by ``sample_indices``, so a
shuffle must reindex those references through the *inverse* permutation
rather than the permutation itself — getting that backwards silently
mistrains every trace-based strategy.

These tests pin, for a fixed seed:

* the exact permutation applied to the feature/target rows,
* the exact inverse remapping applied to trace ``sample_indices``,
* the per-sample arrays that are permuted (``true_solutions``, row kinds)
  versus the per-row trace arrays that are deliberately left untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralls.domain.generation.orchestration import _shuffle_samples
from neuralls.domain.normalization import ErrorTraceSamples, ResidualTraceSamples

SEED = 0
SAMPLE_COUNT = 4

# Pinned literals for np.random.default_rng(SEED).permutation(SAMPLE_COUNT).
EXPECTED_PERMUTATION = (2, 0, 1, 3)
EXPECTED_INVERSE = (1, 2, 0, 3)


@pytest.fixture
def permutation() -> np.ndarray:
    """The permutation the production call site derives from a seeded generator."""
    return np.random.default_rng(SEED).permutation(SAMPLE_COUNT)


@pytest.fixture
def features() -> np.ndarray:
    """Feature rows, one identifiable value per sample."""
    return np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float64)


@pytest.fixture
def targets() -> np.ndarray:
    """Target rows aligned with ``features``."""
    return np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)


@pytest.fixture
def row_kind_codes() -> np.ndarray:
    """Per-sample row-kind codes aligned with ``features``."""
    return np.array([7, 8, 9, 10], dtype=np.uint8)


@pytest.fixture
def residual_traces() -> ResidualTraceSamples:
    """Residual traces referencing every base sample at least once."""
    return ResidualTraceSamples(
        residuals=np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float64),
        solutions=np.array([[1.1], [1.2], [1.3], [1.4], [1.5]], dtype=np.float64),
        sample_indices=np.array([0, 0, 1, 2, 3], dtype=np.int64),
        iteration_indices=np.array([0, 1, 0, 0, 0], dtype=np.int64),
        search_directions=np.array([[2.1], [2.2], [2.3], [2.4], [2.5]], dtype=np.float64),
        search_direction_products=np.array([3.1, 3.2, 3.3, 3.4, 3.5], dtype=np.float64),
    )


@pytest.fixture
def error_traces(targets: np.ndarray) -> ErrorTraceSamples:
    """Error traces whose ``true_solutions`` are per-sample and must be permuted."""
    return ErrorTraceSamples(
        residuals=np.array([[100.0], [101.0], [200.0], [300.0], [400.0]], dtype=np.float64),
        solutions_current=np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=np.float64),
        errors=np.array([[0.9], [0.8], [1.7], [2.6], [3.5]], dtype=np.float64),
        true_solutions=targets.copy(),
        sample_indices=np.array([0, 0, 1, 2, 3], dtype=np.int64),
        iteration_indices=np.array([0, 1, 0, 0, 0], dtype=np.int64),
    )


def test_seeded_permutation_literals_are_stable(permutation: np.ndarray) -> None:
    """Guards the literals every other test in this module is written against."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(SAMPLE_COUNT)
    assert tuple(int(i) for i in permutation) == EXPECTED_PERMUTATION
    assert tuple(int(i) for i in inverse) == EXPECTED_INVERSE


def test_features_and_targets_follow_the_permutation(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(features, targets, None, None, row_kind_codes, permutation)

    np.testing.assert_allclose(result.rhs, np.array([[30.0], [10.0], [20.0], [40.0]]))
    np.testing.assert_allclose(result.solutions, np.array([[3.0], [1.0], [2.0], [4.0]]))


def test_row_kind_codes_follow_the_permutation(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(features, targets, None, None, row_kind_codes, permutation)

    np.testing.assert_array_equal(result.row_kind_codes, np.array([9, 7, 8, 10], dtype=np.uint8))


def test_residual_trace_sample_indices_are_remapped_through_the_inverse(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    residual_traces: ResidualTraceSamples,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(features, targets, residual_traces, None, row_kind_codes, permutation)

    assert result.residual_traces is not None
    # sample_indices [0, 0, 1, 2, 3] mapped through inverse (1, 2, 0, 3).
    np.testing.assert_array_equal(
        result.residual_traces.sample_indices, np.array([1, 1, 2, 0, 3], dtype=np.int64)
    )


def test_residual_trace_row_arrays_are_left_untouched(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    residual_traces: ResidualTraceSamples,
    permutation: np.ndarray,
) -> None:
    """Trace rows are per-iteration, not per-sample — they must not be permuted."""
    result = _shuffle_samples(features, targets, residual_traces, None, row_kind_codes, permutation)

    assert result.residual_traces is not None
    np.testing.assert_allclose(result.residual_traces.residuals, residual_traces.residuals)
    np.testing.assert_allclose(result.residual_traces.solutions, residual_traces.solutions)
    np.testing.assert_array_equal(
        result.residual_traces.iteration_indices, residual_traces.iteration_indices
    )
    search_directions = result.residual_traces.search_directions
    original_search_directions = residual_traces.search_directions
    assert search_directions is not None
    assert original_search_directions is not None
    np.testing.assert_allclose(search_directions, original_search_directions)

    search_direction_products = result.residual_traces.search_direction_products
    original_search_direction_products = residual_traces.search_direction_products
    assert search_direction_products is not None
    assert original_search_direction_products is not None
    np.testing.assert_allclose(search_direction_products, original_search_direction_products)


def test_error_trace_true_solutions_are_permuted_and_indices_inverted(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    error_traces: ErrorTraceSamples,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(features, targets, None, error_traces, row_kind_codes, permutation)

    assert result.error_traces is not None
    np.testing.assert_allclose(
        result.error_traces.true_solutions, np.array([[3.0], [1.0], [2.0], [4.0]])
    )
    np.testing.assert_array_equal(
        result.error_traces.sample_indices, np.array([1, 1, 2, 0, 3], dtype=np.int64)
    )
    np.testing.assert_allclose(result.error_traces.errors, error_traces.errors)


def test_absent_traces_stay_absent(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(features, targets, None, None, row_kind_codes, permutation)

    assert result.residual_traces is None
    assert result.error_traces is None


def test_both_trace_kinds_can_be_shuffled_together(
    features: np.ndarray,
    targets: np.ndarray,
    row_kind_codes: np.ndarray,
    residual_traces: ResidualTraceSamples,
    error_traces: ErrorTraceSamples,
    permutation: np.ndarray,
) -> None:
    result = _shuffle_samples(
        features, targets, residual_traces, error_traces, row_kind_codes, permutation
    )

    assert result.residual_traces is not None
    assert result.error_traces is not None
    np.testing.assert_array_equal(
        result.residual_traces.sample_indices, result.error_traces.sample_indices
    )
