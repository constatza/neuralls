"""Tests for `StepWindow` — which steps of a bounded trajectory to keep.

Covers first-n, [m, n) range, last-n (including end-relative negative
`start`), stride, the fixed-vs-variable-length equivalence claim, and the
`select`/`resolve_indices`/`select_with_indices` consistency guarantee.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neuralls.domain.generation.step_window import StepWindow

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"stop": 0}, id="stop-must-be-positive"),
        pytest.param({"stop": 5, "step": 0}, id="step-must-be-positive"),
        pytest.param({"stop": 5, "start": 5}, id="nonnegative-start-below-stop"),
        pytest.param({"stop": 5, "start": -7}, id="negative-start-within-trajectory"),
    ],
)
def test_invalid_window_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        StepWindow(**kwargs)


@pytest.mark.parametrize("start", [4, -6], ids=["last-absolute-index", "furthest-negative"])
def test_boundary_starts_are_valid(start: int) -> None:
    StepWindow(stop=5, start=start)


# ---------------------------------------------------------------------------
# Selection semantics: first-n, [m, n) range, last-n, stride, early convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stop", "start", "step", "length", "expected"),
    [
        pytest.param(2, 0, 1, 3, [0, 1, 2], id="first-n"),
        pytest.param(6, 0, 1, 7, [0, 1, 2, 3, 4, 5, 6], id="fixed-length-keeps-the-whole-slice"),
        pytest.param(5, 2, 1, 6, [2, 3, 4, 5], id="range-m-to-n"),
        pytest.param(8, -3, 1, 9, [6, 7, 8], id="last-n-ignores-absolute-stop"),
        pytest.param(8, None, 1, 9, [8], id="default-keeps-only-the-last-row"),
        pytest.param(5, 0, 2, 6, [0, 2, 4], id="stride"),
        pytest.param(50, None, 1, 4, [3], id="last-only-under-early-convergence"),
        pytest.param(50, -3, 1, 4, [1, 2, 3], id="last-n-under-early-convergence"),
    ],
)
def test_select_keeps_the_expected_rows(
    labeled_trajectory: Callable[[int], np.ndarray],
    stop: int,
    start: int | None,
    step: int,
    length: int,
    expected: list[int],
) -> None:
    selected = StepWindow(stop=stop, start=start, step=step).select(labeled_trajectory(length))

    np.testing.assert_array_equal(selected.ravel(), expected)


def test_select_never_squeezes(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """A single-row selection stays 2D, never squeezed to 1D."""
    window = StepWindow(stop=8)
    trajectory = labeled_trajectory(9)
    selected = window.select(trajectory)
    assert selected.ndim == trajectory.ndim
    assert selected.shape == (1, 1)


# ---------------------------------------------------------------------------
# select_with_indices / resolve_indices consistency guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stop", "start", "step", "length"),
    [
        (8, None, 1, 9),  # default last-only, fixed length
        (8, None, 1, 4),  # default last-only, early-converged (shorter) length
        (8, 0, 1, 9),  # full trace
        (8, 0, 2, 9),  # stride
        (8, -3, 1, 9),  # last-n, fixed length
        (8, -3, 1, 4),  # last-n, early-converged length
        (5, 2, 1, 6),  # [m, n) range
    ],
)
def test_select_with_indices_consistency(
    labeled_trajectory: Callable[[int], np.ndarray],
    stop: int,
    start: int | None,
    step: int,
    length: int,
) -> None:
    """Every returned row equals trajectory[index] for its paired resolved index."""
    window = StepWindow(stop=stop, start=start, step=step)
    trajectory = labeled_trajectory(length)

    selected, indices = window.select_with_indices(trajectory)
    indices = list(indices)

    assert selected.shape[0] == len(indices)
    for row, idx in zip(selected, indices):
        np.testing.assert_array_equal(row, trajectory[idx])


def test_resolve_indices_matches_select_row_count(
    labeled_trajectory: Callable[[int], np.ndarray],
) -> None:
    """len(resolve_indices(length)) always equals select(trajectory).shape[0]."""
    window = StepWindow(stop=10, start=3, step=2)
    trajectory = labeled_trajectory(11)

    selected = window.select(trajectory)
    indices = window.resolve_indices(trajectory.shape[0])

    assert selected.shape[0] == len(list(indices))
