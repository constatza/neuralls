"""What a report sink needs, besides the numbers, to draw one group."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from neuralls.domain.presentation.methods import (
    ConfiguredMethodPresentation,
    PreparedMethodPresentation,
)
from neuralls.domain.solver.models.figure_scope import GroupScope
from neuralls.domain.solver.models.plot_style import PreconditionerPlotStyle
from neuralls.shared.scientific_description.catalog import fact_definition, is_compact_by_policy
from neuralls.shared.scientific_description.models import (
    ScientificDescription,
    ScientificFact,
    require_unique_keys,
)
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

    @property
    def compact(self) -> ConfiguredMethodPresentation:
        """The configured presentation restricted to the compact facts of the group."""
        return self._view(self.compact_facts)

    @property
    def detailed(self) -> ConfiguredMethodPresentation:
        """The configured presentation with every configured and invariant realized fact."""
        return self._view(self.detailed_facts)

    def _view(self, facts: tuple[ScientificFact, ...]) -> ConfiguredMethodPresentation:
        method = self.prepared.configured.method
        return replace(
            self.prepared.configured,
            method=ScientificDescription(base_name=method.base_name, facts=facts),
        )


@dataclass(frozen=True, slots=True)
class GroupPresentation:
    """Everything a sink needs, besides the numbers, to draw one report group.

    Attributes:
        scope: What the group pools, rendered by the sink in its titles.
        plot_styles: Style of each preconditioner of the case, by name.
        methods: Structured presentation of each method of the group, by name. A
            listed method's plot style carries the same prepared presentation.
        solver_contexts: Distinct solver/timing configurations represented by
            systems in the group.
    """

    scope: GroupScope
    plot_styles: Mapping[str, PreconditionerPlotStyle]
    methods: Mapping[PreconditionerName, GroupMethodPresentation]
    solver_contexts: tuple[SolverTimingContext, ...] = ()

    def __post_init__(self) -> None:
        """Require a listed method's plot style, if it has one, to carry its prepared presentation."""
        for name, method in self.methods.items():
            style = self.plot_styles.get(name)
            if style is not None and style.presentation != method.prepared:
                raise ValueError(
                    f"the plot style of method {name!r} carries a different presentation"
                )
