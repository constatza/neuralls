"""Bind Pydantic config fields to scientific facts, and describe a config by its bindings.

A config field states once what it is: ``Fact(KEY)`` (a scientific fact), ``Nested()`` (a
config model whose own fields are described in its place) or ``Omitted(role)`` (identity,
provenance or operational, never shown). ``describe_model`` walks the fields in declaration
order, so the display order is the declaration order, and refuses any field that states none
of the three, so a new field cannot silently drop out of a description or leak into it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from neuralls.shared.scientific_description.catalog import FactRole, make_fact
from neuralls.shared.scientific_description.models import FieldKey, ScientificFact, ValidatedLabel

__all__ = [
    "AFTER_METHOD_FACTS",
    "Fact",
    "Nested",
    "Omitted",
    "Show",
    "UnboundFieldError",
    "auto_if_none",
    "describe_model",
    "enum_value",
    "on_off",
    "unbound_fields",
]

type FactValue = float | str | ValidatedLabel
type Converter = Callable[[Any], FactValue]
type Condition = Callable[[Any], bool]


AFTER_METHOD_FACTS = 1
"""``Fact.order`` for facts shown after the method's own, such as scheduling controls."""


class Show(StrEnum):
    """When a bound field contributes its fact."""

    ALWAYS = "always"
    NON_DEFAULT = "non_default"
    NOT_NONE = "not_none"


def _as_is(value: Any) -> FactValue:
    return value


def enum_value(value: StrEnum) -> str:
    return value.value


def on_off(value: bool) -> str:
    return "on" if value else "off"


def auto_if_none(value: float | None) -> FactValue:
    return "auto" if value is None else value


def _always(owner: Any) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class Fact:
    """The field is the scientific fact ``key``, converted by ``convert`` for display."""

    key: FieldKey
    convert: Converter = _as_is
    show: Show = Show.ALWAYS
    when: Condition = _always
    """Predicate on the owning model: the fact is meaningful only when it holds."""
    order: int = 0
    """Display rank; equal ranks keep declaration order. Only for a fact that must not
    follow declaration order, such as scheduling controls shown after the method's own."""


@dataclass(frozen=True, slots=True)
class Nested:
    """The field is a config model whose own bound fields are described in its place."""


@dataclass(frozen=True, slots=True)
class Omitted:
    """The field is never a fact: ``role`` says why (identity, provenance or operational)."""

    role: FactRole


type Binding = Fact | Nested | Omitted


class UnboundFieldError(TypeError):
    """A config field declares neither ``Fact``, ``Nested`` nor ``Omitted``."""

    def __init__(self, model: type[BaseModel], names: tuple[str, ...]) -> None:
        self.model = model
        self.names = names
        super().__init__(
            f"{model.__qualname__} fields {list(names)} have no scientific binding; annotate "
            "each with Fact(KEY), Nested() or Omitted(role)"
        )


def _binding(model: type[BaseModel], name: str) -> Binding | None:
    found = [m for m in model.model_fields[name].metadata if isinstance(m, Fact | Nested | Omitted)]
    if len(found) > 1:
        raise TypeError(f"{model.__qualname__}.{name} carries several scientific bindings")
    return found[0] if found else None


def unbound_fields(model: type[BaseModel]) -> tuple[str, ...]:
    """Names of ``model``'s fields that declare no binding."""
    return tuple(name for name in model.model_fields if _binding(model, name) is None)


def _shown(binding: Fact, owner: BaseModel, name: str) -> bool:
    value = getattr(owner, name)
    match binding.show:
        case Show.ALWAYS:
            shown = True
        case Show.NON_DEFAULT:
            shown = value != type(owner).model_fields[name].get_default(call_default_factory=True)
        case Show.NOT_NONE:
            shown = value is not None
    return shown and binding.when(owner)


def describe_model(owner: BaseModel) -> tuple[ScientificFact, ...]:
    """Facts of ``owner`` in display order.

    Raises:
        UnboundFieldError: If a field of ``owner`` or of a nested model declares no binding.
    """
    model = type(owner)
    if missing := unbound_fields(model):
        raise UnboundFieldError(model, missing)
    ranked: list[tuple[int, tuple[ScientificFact, ...]]] = []
    for name in model.model_fields:
        match _binding(model, name):
            case Fact() as fact if _shown(fact, owner, name):
                ranked.append(
                    (fact.order, (make_fact(fact.key, fact.convert(getattr(owner, name))),))
                )
            case Nested():
                ranked.append((0, describe_model(getattr(owner, name))))
            case _:
                pass
    return tuple(fact for _, group in sorted(ranked, key=lambda r: r[0]) for fact in group)
