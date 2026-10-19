"""Scaling-law projection of a preconditioner's cost per solved system."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.solver.models.sharing import SetupSharingPolicy


@dataclass(frozen=True, slots=True)
class ProjectedComparisonResult:
    """Scaling-law projection for one preconditioner: cost per *solved* system.

    Built from the `n_systems` sibling `CGComparisonResult`s a comparison
    sweep actually ran, never from N systems actually executed. A run stopped
    by the iteration cap is a failure that still cost time, so every failed
    run's measured cost is charged to the systems that did converge. This is
    the pooled expected running time (ERT) of Hansen et al. 2016, "COCO:
    Performance Assessment" (arXiv:1605.03560): total cost of all runs divided
    by the number of successes. It is a cost per solved system, not the
    expectation of a restart strategy.

    `projected_total_seconds(n)` extrapolates the measured constants to `n`
    solved systems, purely for plotting, as `one_time_seconds +
    n * per_solved_system_seconds`.

    Each of `setup`/`generation` is amortized or repeated independently per
    `setup_sharing` (see `SetupSharingPolicy`). Amortized terms are paid once
    and do not depend on the success rate; repeated terms are paid on every
    attempt, so per solved system they are divided by `success_rate`.

    Attributes:
        preconditioner_name: Matches `CGComparisonResult.preconditioner`.
        n_systems: Number of sibling systems actually aggregated.
        n_converged: Of those, the systems that converged (at least one: a
            preconditioner that never converged has nothing to project).
        setup_sharing: Which cost components amortize vs. repeat per system.
        setup_cost_total_seconds: Sum of `setup_cost` across all `n_systems`
            measured results. If `setup_sharing.share_setup`, every sibling
            already carries the same real one-time value (the caching layer
            copies it forward rather than reporting 0.0 on a cache hit), so
            dividing by `n_systems` recovers that single value exactly.
        generation_cost_total_seconds: Mean generation wall time over the
            results whose cost is available, scaled to `n_systems` (so that
            dividing by `n_systems` recovers that mean), or `None` if
            generation does not apply or is unavailable for every result.
        total_solve_time_seconds: Mean `solve_time_seconds` over all runs that
            measured one (failed runs at their measured cost), scaled to
            `n_systems`. Always repeats per system; solving is never a
            one-time cost.
        total_iterations: Sum of `iterations` across all `n_systems` runs
            (failed runs contribute the iterations they spent).
        solve_time_std_seconds: Sample standard deviation (ddof=1) of the
            solve time over the converged runs only, i.e. the spread of a
            successful solve, as the learned-preconditioner literature reports
            (mean +- std over the test set). `None` below two converged runs.
        iterations_std: Sample standard deviation of the iteration count over
            the converged runs only, or `None` below two converged runs.
    """

    preconditioner_name: str
    n_systems: int
    n_converged: int
    setup_sharing: SetupSharingPolicy
    setup_cost_total_seconds: float
    generation_cost_total_seconds: float | None
    total_solve_time_seconds: float
    total_iterations: int
    solve_time_std_seconds: float | None = None
    iterations_std: float | None = None

    @property
    def avg_setup_cost_seconds(self) -> float:
        """Setup cost per measured system, the one-time value itself when shared."""
        return self.setup_cost_total_seconds / self.n_systems

    @property
    def avg_generation_cost_seconds(self) -> float | None:
        """Generation cost per measured system, or `None` if not applicable."""
        if self.generation_cost_total_seconds is None:
            return None
        return self.generation_cost_total_seconds / self.n_systems

    @property
    def avg_solve_time_per_system_seconds(self) -> float:
        """Solve time per measured system over all runs, failed ones at their measured cost."""
        return self.total_solve_time_seconds / self.n_systems

    @property
    def success_rate(self) -> float:
        """Fraction of the measured systems that converged."""
        return self.n_converged / self.n_systems

    @property
    def avg_solve_time_per_solved_system_seconds(self) -> float:
        """Solve time of all runs divided by the number of converged runs (solve-only ERT)."""
        return self.total_solve_time_seconds / self.n_converged

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time per iteration, both over all runs; `None` without iterations."""
        if self.total_iterations <= 0:
            return None
        return self.total_solve_time_seconds / self.total_iterations

    @property
    def one_time_seconds(self) -> float:
        """Cost paid once whatever the number of systems: the shared setup and generation."""
        return self.setup_sharing.one_time_seconds(
            self.avg_setup_cost_seconds, self.avg_generation_cost_seconds
        )

    @property
    def per_solved_system_seconds(self) -> float:
        """Recurring cost per solved system: repeated setup, repeated generation and solve.

        Every attempt pays the repeated terms and only `success_rate` of the
        attempts solve a system, so their sum is divided by `success_rate`.
        """
        repeated = self.setup_sharing.repeated_seconds(
            self.avg_setup_cost_seconds, self.avg_generation_cost_seconds
        )
        return (repeated + self.avg_solve_time_per_system_seconds) / self.success_rate

    def projected_total_seconds(self, n: int) -> float:
        """Projected total wall time to obtain `n` solved systems with this preconditioner.

        Args:
            n: Number of *solved* systems to project to. Never actually run;
                this is a closed-form extrapolation from the `n_systems` real
                measurements, for plotting only.

        Returns:
            float: `one_time_seconds + n * per_solved_system_seconds`.
        """
        return self.one_time_seconds + n * self.per_solved_system_seconds

    def projected_avg_per_system_seconds(self, n: int) -> float:
        """Projected total cost amortized per solved system, flattening toward the recurring floor.

        Args:
            n: Number of solved systems to project to (must be positive).

        Returns:
            float: `projected_total_seconds(n) / n`.
        """
        return self.projected_total_seconds(n) / n
