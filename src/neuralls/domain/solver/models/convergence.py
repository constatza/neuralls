"""How often a preconditioner converged over a report group."""

from __future__ import annotations

from dataclasses import dataclass

from neuralls.domain.solver.models.statistics import Interval, IterationStats


@dataclass(frozen=True, slots=True)
class ConvergenceSummary:
    """How often one preconditioner converged over a report group, and at what cost.

    The success rate is reported beside every conditional statistic; ERT is
    the pooled cost per solved system of COCO (Hansen et al. 2016,
    arXiv:1605.03560).

    Attributes:
        n_systems: Systems of the group the preconditioner was run on.
        n_converged: Of those, the runs that reached the tolerance within the
            iteration cap.
        ert_seconds: Expected running time (COCO): the one-time shared cost
            plus the recurring time of every run, failed ones included,
            divided by `n_converged`. `None` when
            nothing converged or a run's time is incomparable.
        iterations: Iteration statistics over the converged runs, `None` when
            nothing converged.
        ert_interval: Bootstrap interval of `ert_seconds` over systems, `None`
            when ERT is undefined or there are too few systems.
    """

    n_systems: int
    n_converged: int
    ert_seconds: float | None
    iterations: IterationStats | None = None
    ert_interval: Interval | None = None

    @property
    def success_rate(self) -> float:
        """Fraction of systems that converged."""
        return self.n_converged / self.n_systems
