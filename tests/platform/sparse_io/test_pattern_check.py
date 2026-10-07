"""Sparsity-pattern equality for the shared-pattern stream.

A pattern is a sample's shape, indptr and indices after canonicalisation. Data never takes part.
Fixtures build every matrix; no repo config or network is involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from neuralls.platform.sparse_io.pattern import canonicalise_csr, csr_pattern_matches

_SHAPE = (3, 4)


@pytest.fixture
def base_pattern() -> csr_array:
    """Sorted pattern with one entry per row in rows 0 and 2, row 1 empty."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 2, 2, 3], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=_SHAPE)


@pytest.fixture
def unsorted_same_pattern() -> csr_array:
    """Same stored positions as ``base_pattern`` with row 0 indices reversed."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([2, 0, 1], dtype=np.int64)
    indptr = np.array([0, 2, 2, 3], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=_SHAPE)


@pytest.fixture
def duplicated_same_pattern() -> csr_array:
    """Row 0 splits the base entry (column 0, value 1.0) into 0.25 and 0.75 stored twice."""
    data = np.array([0.25, 0.75, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 3, 3, 4], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=_SHAPE)


@pytest.fixture
def explicit_zero_pattern() -> csr_array:
    """Base pattern with one explicitly stored zero at (1, 3)."""
    data = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float64)
    indices = np.array([0, 2, 3, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3, 4], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=_SHAPE)


@pytest.fixture
def other_shape_pattern() -> csr_array:
    """Same stored entries as the base, but the matrix is one column wider."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 2, 2, 3], dtype=np.int64)
    return csr_array((data, indices, indptr), shape=(3, 5))


@pytest.fixture
def new_values_same_pattern(base_pattern: csr_array) -> csr_array:
    """Identical pattern with different stored values."""
    return csr_array(
        (np.array([-7.0, 0.5, 9.0]), base_pattern.indices.copy(), base_pattern.indptr.copy()),
        shape=_SHAPE,
    )


def test_unsorted_indices_match_sorted_after_canonicalisation(
    base_pattern: csr_array, unsorted_same_pattern: csr_array
) -> None:
    assert csr_pattern_matches(
        canonicalise_csr(base_pattern), canonicalise_csr(unsorted_same_pattern)
    )


def test_duplicate_entries_match_summed_pattern_after_canonicalisation(
    base_pattern: csr_array, duplicated_same_pattern: csr_array
) -> None:
    assert csr_pattern_matches(
        canonicalise_csr(base_pattern), canonicalise_csr(duplicated_same_pattern)
    )


def test_canonicalisation_sums_duplicates_into_the_stored_value(
    duplicated_same_pattern: csr_array,
) -> None:
    canonical = canonicalise_csr(duplicated_same_pattern)
    assert canonical.data[0] == pytest.approx(1.0)


def test_canonicalisation_does_not_modify_the_input(unsorted_same_pattern: csr_array) -> None:
    before = unsorted_same_pattern.indices.copy()
    canonicalise_csr(unsorted_same_pattern)
    assert np.array_equal(unsorted_same_pattern.indices, before)


def test_stored_explicit_zero_makes_patterns_differ(
    base_pattern: csr_array, explicit_zero_pattern: csr_array
) -> None:
    assert not csr_pattern_matches(
        canonicalise_csr(base_pattern), canonicalise_csr(explicit_zero_pattern)
    )


def test_shape_difference_makes_patterns_differ(
    base_pattern: csr_array, other_shape_pattern: csr_array
) -> None:
    assert not csr_pattern_matches(
        canonicalise_csr(base_pattern), canonicalise_csr(other_shape_pattern)
    )


def test_data_only_difference_is_equal(
    base_pattern: csr_array, new_values_same_pattern: csr_array
) -> None:
    assert csr_pattern_matches(
        canonicalise_csr(base_pattern), canonicalise_csr(new_values_same_pattern)
    )


def test_pattern_check_never_densifies(
    monkeypatch: pytest.MonkeyPatch,
    base_pattern: csr_array,
    unsorted_same_pattern: csr_array,
) -> None:
    calls: list[int] = []
    original = csr_array.toarray

    def _counting(self: csr_array) -> np.ndarray:
        calls.append(1)
        return original(self)

    monkeypatch.setattr(csr_array, "toarray", _counting)
    csr_pattern_matches(canonicalise_csr(base_pattern), canonicalise_csr(unsorted_same_pattern))
    assert calls == []
