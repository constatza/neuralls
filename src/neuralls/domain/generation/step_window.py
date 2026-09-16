"""Which steps of a bounded iterative trajectory to run and keep.

Both CG-trace strategies (`residuals.py`, `search_directions.py`) and the
weighted-Jacobi smoother-probe strategy (`smoother_probes.py`) harvest
snapshots from a bounded iterative trajectory of a single linear system —
CG iterates in the first case, Jacobi damping sweeps in the second. `krylov.py`
has no trajectory (random basis combinations, not sequential iterates), so it
has no use for this module.

`StepWindow` is the one shared abstraction for "which steps to keep" across
both trajectory kinds, replacing the `cg_iters`/`every_n`/`steps` fields that
used to express this independently (and inconsistently) per strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StepWindow:
    """Which steps of a bounded iterative trajectory to run and keep.

    Mirrors slice/range's own start/stop/step vocabulary. `stop` is required
    (no implicit "run to convergence") and is the hard iteration/sweep cap
    handed to the solver/smoother producing the trajectory — exactly `stop`
    steps are ever attempted, never more. The solver may still stop earlier
    on its own (e.g. a real CG convergence tolerance), producing a
    trajectory shorter than `stop + 1`; selection is always resolved
    against whatever length the trajectory actually turns out to have, via
    ordinary Python slicing — `start=None` (the default, "keep only the
    final step") and any negative `start` ("last n") are relative to the
    true end, not to `stop`. A non-negative `start` is an absolute,
    from-the-beginning index, as usual, and presumes the trajectory reaches
    that far (see `_CgTraceFields`'s tolerance/start cross-validation in
    `strategy_configs.py` for why that combination is otherwise rejected).

    Attributes:
        stop: Iteration/sweep cap. The trajectory has at most `stop + 1`
            rows (index 0 is the untouched initial vector, index `stop` is
            the vector after `stop` steps — reached exactly whenever the
            solver/smoother has no early-stop criterion of its own).
        start: First step kept (inclusive). `None` or a negative value is
            relative to the trajectory's true end (`-1` is the last row,
            same as `None`); a non-negative value is an absolute index from
            the beginning.
        step: Keep every `step`-th row within the selected range.
    """

    stop: int
    start: int | None = None
    step: int = 1

    def __post_init__(self) -> None:
        """Validate bounds directly, per the dataclass's own invariants.

        Raises:
            ValueError: If `stop < 1`, `step < 1`, a non-negative `start`
                is `>= stop`, or a negative `start` is less than
                `-(stop + 1)` (further back than any trajectory could be).
        """
        if self.stop < 1:
            raise ValueError(f"StepWindow.stop must be >= 1, got {self.stop}")
        if self.step < 1:
            raise ValueError(f"StepWindow.step must be >= 1, got {self.step}")
        if self.start is not None:
            if self.start >= 0 and self.start >= self.stop:
                raise ValueError(f"StepWindow.start ({self.start}) must be < stop={self.stop}")
            if self.start < 0 and self.start < -(self.stop + 1):
                raise ValueError(
                    f"StepWindow.start ({self.start}) must be >= -(stop + 1)={-(self.stop + 1)}"
                )

    @property
    def resolved_start(self) -> int:
        """The effective start index — `-1` (the last row) if `start` is unset."""
        return -1 if self.start is None else self.start

    def as_slice(self) -> slice:
        """The equivalent `slice(start, stop, step)` over a trajectory array.

        The upper bound is `None` ("to the actual end of the array"), not
        `stop + 1` — the trajectory is already capped at `stop + 1` rows by
        construction (the solver/smoother is called with `maxiter=stop`),
        so this is a no-op when that cap is exactly reached, and correct by
        construction when the trajectory stopped earlier.
        """
        return slice(self.resolved_start, None, self.step)

    def select(self, trajectory: np.ndarray) -> np.ndarray:
        """Select the kept rows from a trajectory array.

        Args:
            trajectory: Array of shape `(actual_length, ...)`, where
                `actual_length <= stop + 1` — one row per step actually run.

        Returns:
            np.ndarray: The kept rows, always with the same number of
                dimensions as `trajectory` (never squeezed), even when only
                one row is kept.
        """
        return trajectory[self.as_slice()]

    def resolve_indices(self, length: int) -> range:
        """The concrete 0-based step indices kept, for a trajectory of `length` rows.

        `length` is the trajectory's actual row count — not always
        statically `stop + 1` (see the class docstring) — so this takes it
        explicitly rather than assuming. Derived from the same `as_slice()`
        used by `select`, so a caller's row-to-index bookkeeping can never
        drift out of sync with the rows `select` actually returns. Prefer
        `select_with_indices` at call sites so `length` is always read from
        the same array being selected, never a separately-tracked value
        that could disagree with it.
        """
        return range(*self.as_slice().indices(length))

    def select_with_indices(self, trajectory: np.ndarray) -> tuple[np.ndarray, range]:
        """Select rows and their original indices from one array, in one call.

        The two are structurally guaranteed to describe the same
        trajectory, since there is only one array argument to read the
        length from — this is what closes off a caller passing a
        `select(trajectory_a)` alongside a mismatched
        `resolve_indices(len(trajectory_b))`.

        Args:
            trajectory: Array of shape `(actual_length, ...)`.

        Returns:
            The selected rows (as `select`) and their original step
            indices (as `resolve_indices`), guaranteed pairwise consistent.
        """
        return self.select(trajectory), self.resolve_indices(trajectory.shape[0])
