"""Explicit reference selection and common solved-system helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from neuralls.domain.solver.models.outcomes import Outcomes, Solved
from neuralls.shared.types import PreconditionerName, SystemId


@dataclass(frozen=True, slots=True)
class ConfiguredReference:
    """A reference explicitly named by comparison configuration."""

    name: PreconditionerName

    @property
    def reference(self) -> PreconditionerName:
        """Compatibility spelling for consumers that call it a reference."""
        return self.name

    @property
    def preconditioner(self) -> PreconditionerName:
        """The selected method name."""
        return self.name


@dataclass(frozen=True, slots=True)
class NoReferenceConfigured:
    """No reference was requested; paired comparisons are intentionally empty."""


@dataclass(frozen=True, slots=True)
class ReferenceUnavailable:
    """The configured reference was skipped before execution."""

    name: PreconditionerName
    reason: str

    @property
    def reference(self) -> PreconditionerName:
        """Compatibility spelling for consumers that call it a reference."""
        return self.name

    @property
    def preconditioner(self) -> PreconditionerName:
        """The skipped method name."""
        return self.name


type ReferenceSelection = ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable


def validate_reference_preconditioner(
    configured: str | PreconditionerName | None,
    available: Collection[str | PreconditionerName],
) -> None:
    """Validate an explicitly configured reference against final static methods.

    ``available`` must be the final method tuple after assignment filters and
    generated assignment methods have been applied. This check never inspects
    observed solve outcomes.
    """
    if configured is None:
        return
    name = str(configured)
    names = tuple(str(item) for item in available)
    if name not in names:
        rendered = ", ".join(names) if names else "<none>"
        raise ValueError(
            f"Configured reference '{name}' is not among available methods: {rendered}."
        )


def resolve_reference_selection(
    configured: str | PreconditionerName | None,
    available: Collection[str | PreconditionerName],
    *,
    skipped: Mapping[str | PreconditionerName, str] | None = None,
) -> ReferenceSelection:
    """Resolve explicit reference state after static and runtime method resolution.

    A runtime skip is represented as ``ReferenceUnavailable`` and never falls
    back to another method. Static absence remains a focused configuration
    error from :func:`validate_reference_preconditioner`.
    """
    if configured is None:
        return NoReferenceConfigured()
    name = PreconditionerName(str(configured))
    skipped_by_name = {str(key): reason for key, reason in (skipped or {}).items()}
    if str(name) in skipped_by_name:
        return ReferenceUnavailable(name=name, reason=skipped_by_name[str(name)])
    validate_reference_preconditioner(name, available)
    return ConfiguredReference(name=name)


def systems_solved_by_all(outcomes: Outcomes) -> frozenset[SystemId]:
    """Systems on which every method converged.

    Conditional means over a method's own converged systems favour methods that
    only solve easy systems (survivorship bias); comparing on this common
    subset removes it, at the price of a smaller n.
    """
    per_method = [
        {sid for sid, o in per_system.items() if isinstance(o, Solved)}
        for per_system in outcomes.values()
    ]
    return frozenset.intersection(*map(frozenset, per_method)) if per_method else frozenset()
