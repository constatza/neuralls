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


def test_stop_must_be_positive() -> None:
    with pytest.raises(ValueError, match="stop"):
        StepWindow(stop=0)


def test_step_must_be_positive() -> None:
    with pytest.raises(ValueError, match="step"):
        StepWindow(stop=5, step=0)


def test_nonnegative_start_must_be_less_than_stop() -> None:
    with pytest.raises(ValueError, match="start"):
        StepWindow(stop=5, start=5)


def test_negative_start_cannot_exceed_trajectory_length() -> None:
    with pytest.raises(ValueError, match="start"):
        StepWindow(stop=5, start=-7)


def test_boundary_values_are_valid() -> None:
    """start=stop-1 (last valid absolute index) and start=-(stop+1) (furthest valid negative) both construct."""
    StepWindow(stop=5, start=4)
    StepWindow(stop=5, start=-6)


# ---------------------------------------------------------------------------
# Selection semantics: first-n, [m, n) range, last-n, stride
# ---------------------------------------------------------------------------


def test_first_n(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """start=0, stop=n keeps indices [0, n]."""
    window = StepWindow(stop=2, start=0)
    trajectory = labeled_trajectory(3)  # a fixed-length run: maxiter=stop=2 -> 3 rows
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [0, 1, 2])


def test_range_m_to_n(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """start=m, stop=n keeps indices [m, n]."""
    window = StepWindow(stop=5, start=2)
    trajectory = labeled_trajectory(6)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [2, 3, 4, 5])


def test_last_n_via_negative_start(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """start=-n keeps the last n rows, regardless of the absolute stop value."""
    window = StepWindow(stop=8, start=-3)
    trajectory = labeled_trajectory(9)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [6, 7, 8])


def test_default_start_keeps_only_last_row(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """start unset (None) keeps exactly one row: the true last one."""
    window = StepWindow(stop=8)
    trajectory = labeled_trajectory(9)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [8])


def test_stride(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """step > 1 keeps every step-th row within the selected range."""
    window = StepWindow(stop=5, start=0, step=2)
    trajectory = labeled_trajectory(6)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [0, 2, 4])


def test_select_never_squeezes(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """A single-row selection stays 2D, never squeezed to 1D."""
    window = StepWindow(stop=8)
    trajectory = labeled_trajectory(9)
    selected = window.select(trajectory)
    assert selected.ndim == trajectory.ndim
    assert selected.shape == (1, 1)


# ---------------------------------------------------------------------------
# Fixed-length vs variable-length (early-stopped) trajectories
# ---------------------------------------------------------------------------


def test_last_only_resolves_to_true_end_under_early_convergence(
    labeled_trajectory: Callable[[int], np.ndarray],
) -> None:
    """A trajectory shorter than stop+1 (early convergence) still resolves 'last' correctly."""
    window = StepWindow(stop=50)  # safety cap, not necessarily reached
    trajectory = labeled_trajectory(4)  # converged after 3 steps: 4 rows (0..3)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [3])


def test_last_n_resolves_to_true_end_under_early_convergence(
    labeled_trajectory: Callable[[int], np.ndarray],
) -> None:
    window = StepWindow(stop=50, start=-3)
    trajectory = labeled_trajectory(4)
    selected = window.select(trajectory)
    np.testing.assert_array_equal(selected.ravel(), [1, 2, 3])


def test_fixed_length_equivalence_first_n(labeled_trajectory: Callable[[int], np.ndarray]) -> None:
    """When the trajectory is exactly stop+1 long, results match what an absolute
    slice(start, stop+1, step) over stop+1 rows would give — the design's
    backward-compatibility claim for the fixed-length (unreachable-tolerance) case."""
    stop, start = 6, 0
    window = StepWindow(stop=stop, start=start)
    trajectory = labeled_trajectory(stop + 1)
    selected = window.select(trajectory)
    expected = trajectory[start : stop + 1]
    np.testing.assert_array_equal(selected, expected)


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
