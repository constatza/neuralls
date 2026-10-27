"""What a report sink needs, besides the numbers, to draw one group."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.shared.scientific_description.catalog import fact_definition, is_compact_by_policy
from neuralls.shared.scientific_description.models import ScientificFact, require_unique_keys
from neuralls.shared.types import PreconditionerName


@dataclass(frozen=True, slots=True)
class SolverTimingContext:
    """Solver, normalization, and repeat settings needed to interpret a figure."""

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: str
    m_max: int
    reference_precision_margin: float
    timing_repeats: int
    timing_warmup: int
    timing_statistic: str
    normalization: str


@dataclass(frozen=True, slots=True)
class GroupMethodPresentation:
    """One method as a group figure may describe it.

    ``invariant_facts`` are the realized facts whose value is identical on every system
    of the group, so stating them cannot misrepresent any member; facts that vary stay
    in the per-system detail. A method with no realized state has none.
    """

    prepared: PreparedMethodPresentation
    invariant_facts: tuple[ScientificFact, ...]

    def __post_init__(self) -> None:
        """Require each invariant fact to be stated once."""
        require_unique_keys(self.invariant_facts, "GroupMethodPresentation.invariant_facts")

    @property
    def compact_facts(self) -> tuple[ScientificFact, ...]:
        """Compact configured facts plus the invariant facts a compact label may show."""
        return (
            *self.prepared.compact_facts,
            *(
                fact
                for fact in self.invariant_facts
                if is_compact_by_policy(fact_definition(fact.key).visibility_policy)
            ),
        )

    @property
    def detailed_facts(self) -> tuple[ScientificFact, ...]:
        """Every configured fact plus every invariant realized fact."""
        return (*self.prepared.detailed_facts, *self.invariant_facts)


@dataclass(frozen=True, slots=True)
class GroupPresentation:
    """Everything a sink needs, besides the numbers, to draw one report group.

    Attributes:
        subject: What the group is, for titles (e.g. ``"ds, gaussian"`` or
            ``"All datasets"``).
        subtitle: Concise report scope shown below the metric title.
        plot_styles: Style of each preconditioner of the group, by name.
        methods: Structured presentation of each method of the group, by name.
        solver_contexts: Distinct solver/timing configurations represented by
            systems in the group.
    """

    subject: str
    subtitle: str
    plot_styles: Mapping[str, PreconditionerPlotStyle]
    methods: Mapping[PreconditionerName, GroupMethodPresentation]
    solver_contexts: tuple[SolverTimingContext, ...] = ()
