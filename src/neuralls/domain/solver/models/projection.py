"""Scaling-law projection of a preconditioner's cost per solved system."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.solver.models.sharing import SetupSharingPolicy


@dataclass(frozen=True, slots=True)
class ProjectedComparisonResult:
    """Scaling-law projection for one preconditioner, aggregated across sibling systems.

    Built from the `n_systems` sibling `CGComparisonResult`s a comparison
    sweep actually ran — never from N systems actually
    executed. `projected_total_seconds(n)`/`projected_avg_per_system_seconds(n)`
    extrapolate the measured constants to an arbitrary N, purely for
    plotting, so the one-time cost's contribution can be shown shrinking
    at scale without ever running N real systems.

    Each of `setup`/`generation` is amortized or repeated independently per
    `setup_sharing` — not a single combined choice, since the two costs can
    have different sharing policies for the same preconditioner (see
    `SetupSharingPolicy`).

    Attributes:
        preconditioner_name: Matches `CGComparisonResult.preconditioner`.
        n_systems: Number of sibling systems actually aggregated.
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
        total_solve_time_seconds: Mean `solve_time_seconds` over the results
            that measured one, scaled to `n_systems` — always repeats per
            system; solving is never a one-time cost.
        total_iterations: Sum of `iterations` across all `n_systems`
            measured results.
        solve_time_std_seconds: Sample standard deviation (ddof=1) of the
            per-system solve time — the spread behind `avg_solve_time_per_system_seconds`,
            as the learned-preconditioner literature reports (mean ± std over
            the test set). `None` for a single system.
        iterations_std: Sample standard deviation of the per-system iteration
            count, or `None` for a single system.
    """

    preconditioner_name: str
    n_systems: int
    setup_sharing: SetupSharingPolicy
    setup_cost_total_seconds: float
    generation_cost_total_seconds: float | None
    total_solve_time_seconds: float
    total_iterations: int
    solve_time_std_seconds: float | None = None
    iterations_std: float | None = None

    @property
    def avg_setup_cost_seconds(self) -> float:
        """Setup cost per measured system — the one-time value itself when shared."""
        return self.setup_cost_total_seconds / self.n_systems

    @property
    def avg_generation_cost_seconds(self) -> float | None:
        """Generation cost per measured system, or `None` if not applicable."""
        if self.generation_cost_total_seconds is None:
            return None
        return self.generation_cost_total_seconds / self.n_systems

    @property
    def avg_solve_time_per_system_seconds(self) -> float:
        """Solve time per measured system — always a recurring cost."""
        return self.total_solve_time_seconds / self.n_systems

    @property
    def avg_iteration_time_seconds(self) -> float | None:
        """Solve time divided by total iteration count across all measured systems."""
        if self.total_iterations <= 0:
            return None
        return self.total_solve_time_seconds / self.total_iterations

    def projected_total_seconds(self, n: int) -> float:
        """Projected total wall time to solve `n` systems with this preconditioner.

        Each cost component amortizes (contributes its one-time value once,
        regardless of `n`) or repeats (contributes `n *` its per-system
        average) according to its own entry in `setup_sharing` — solve time
        always repeats.

        Args:
            n: Number of systems to project to. Never actually run; this is
                a closed-form extrapolation from the `n_systems` real
                measurements, for plotting only.

        Returns:
            float: Projected total wall-clock seconds.
        """
        setup_term = (
            self.avg_setup_cost_seconds
            if self.setup_sharing.share_setup
            else n * self.avg_setup_cost_seconds
        )
        generation_term = 0.0
        avg_generation = self.avg_generation_cost_seconds
        if avg_generation is not None:
            generation_term = (
                avg_generation if self.setup_sharing.share_generation else n * avg_generation
            )
        return setup_term + generation_term + n * self.avg_solve_time_per_system_seconds

    def projected_avg_per_system_seconds(self, n: int) -> float:
        """Projected total cost amortized per system — flattens toward the solve floor as `n` grows.

        Args:
            n: Number of systems to project to (must be positive).

        Returns:
            float: `projected_total_seconds(n) / n`.
        """
        return self.projected_total_seconds(n) / n
