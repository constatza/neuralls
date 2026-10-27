"""Pure value types for scientific descriptions.

No dependency outside the standard library: this module must stay importable
without strategies, preconditioners, Pydantic, Matplotlib, Loguru, MLflow, or
storage code.

Production code builds facts through ``catalog.make_fact``, which checks the key and
value kind against the catalog; the constructors here only guard what no catalog
entry can change.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import assert_never

_FIELD_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]*[A-Za-z0-9]$|^[A-Za-z0-9]$")


@dataclass(frozen=True, slots=True)
class FieldKey:
    """A validated semantic field identifier, e.g. ``"iteration_count"``.

    Domain packages define their own ``FieldKey`` constants; the renderer
    treats keys generically and never special-cases a concrete key.
    """

    value: str

    def __post_init__(self) -> None:
        """Reject empty or non-snake-case identifiers."""
        if not _FIELD_KEY_PATTERN.fullmatch(self.value):
            raise ValueError(f"FieldKey value must be snake_case, got {self.value!r}")


class FactValueKind(StrEnum):
    """Which fact variant carries a value; the persisted discriminator of a fact."""

    INTEGER = "integer"
    DECIMAL = "decimal"
    TEXT = "text"
    LABEL = "label"


@dataclass(frozen=True, slots=True)
class IntegerFact:
    """An integer-valued scientific fact, e.g. ``iteration_count=15``."""

    key: FieldKey
    value: int

    def __post_init__(self) -> None:
        """Reject ``bool``, which is an ``int`` but never a scientific count."""
        if isinstance(self.value, bool):
            raise TypeError(f"IntegerFact value must be an int, not bool ({self.key.value!r})")


@dataclass(frozen=True, slots=True)
class DecimalFact:
    """A float-valued scientific fact, e.g. ``standard_deviation=0.01``."""

    key: FieldKey
    value: float

    def __post_init__(self) -> None:
        """Reject values with no finite scientific representation."""
        if isinstance(self.value, bool):
            raise TypeError(f"DecimalFact value must be a float, not bool ({self.key.value!r})")
        if not isfinite(self.value):
            raise ValueError(f"DecimalFact value must be finite, got {self.value!r}")


@dataclass(frozen=True, slots=True)
class TextFact:
    """A categorical fact, e.g. ``weighting="uniform"``; the catalog owns the categories."""

    key: FieldKey
    value: str


@dataclass(frozen=True, slots=True)
class ValidatedLabel:
    """A short human-authored name that is safe to show in every render target.

    Letters, digits, single spaces, and hyphens only: no underscores, so an
    identifier or class name cannot pass, and nothing needs escaping in MathText.
    """

    text: str

    def __post_init__(self) -> None:
        """Reject text that is not a plain human-readable name."""
        if not _LABEL_PATTERN.fullmatch(self.text) or "  " in self.text:
            raise ValueError(
                f"ValidatedLabel must be a plain human-readable name, got {self.text!r}"
            )


@dataclass(frozen=True, slots=True)
class DisplayTextFact:
    """A text fact whose value is a validated, human-authored name (e.g. a model name)."""

    key: FieldKey
    value: ValidatedLabel


type ScientificFact = IntegerFact | DecimalFact | TextFact | DisplayTextFact


@dataclass(frozen=True, slots=True)
class FactIdentity:
    """A fact reduced to ``(key, kind, value)``.

    The kind is part of the identity, so ``1``, ``1.0`` and a categorical ``"1"`` of one
    key never compare equal even though Python equates ``1 == 1.0``.
    """

    key: FieldKey
    kind: FactValueKind
    value: int | float | str


def fact_identity(fact: ScientificFact) -> FactIdentity:
    """Return ``fact``'s typed identity."""
    match fact:
        case IntegerFact(key=key, value=value):
            return FactIdentity(key, FactValueKind.INTEGER, value)
        case DecimalFact(key=key, value=value):
            return FactIdentity(key, FactValueKind.DECIMAL, value)
        case TextFact(key=key, value=value):
            return FactIdentity(key, FactValueKind.TEXT, value)
        case DisplayTextFact(key=key, value=label):
            return FactIdentity(key, FactValueKind.LABEL, label.text)
        case _:
            assert_never(fact)


class DuplicateFactKeyError(ValueError):
    """One carrier holds two facts with the same key."""

    def __init__(self, key: FieldKey, owner: str) -> None:
        self.key = key
        super().__init__(f"{owner} holds fact {key.value!r} more than once")


def require_unique_keys(facts: Iterable[ScientificFact], owner: str) -> None:
    """Reject ``facts`` if any key repeats; ``owner`` names the carrier in the message."""
    seen: set[FieldKey] = set()
    for fact in facts:
        if fact.key in seen:
            raise DuplicateFactKeyError(fact.key, owner)
        seen.add(fact.key)


@dataclass(frozen=True, slots=True)
class DescriptionIdentity:
    """A description reduced to its base name and typed fact identities."""

    base_name: str
    facts: tuple[FactIdentity, ...]
    context_facts: tuple[FactIdentity, ...]


@dataclass(frozen=True, slots=True)
class ScientificDescription:
    """A controlled base name plus ordered facts and ordered context facts.

    Contains no rendered string and performs no I/O; ``render()`` is the only
    function that turns this into presentation text. A key may appear only once
    across ``facts`` and ``context_facts``.
    """

    base_name: str
    facts: tuple[ScientificFact, ...] = ()
    context_facts: tuple[ScientificFact, ...] = ()

    def __post_init__(self) -> None:
        """Require a plain human-readable base name and unique fact keys."""
        ValidatedLabel(self.base_name)
        require_unique_keys((*self.facts, *self.context_facts), "ScientificDescription")

    @property
    def identity(self) -> DescriptionIdentity:
        """Structured identity with no rendered text."""
        return DescriptionIdentity(
            base_name=self.base_name,
            facts=tuple(fact_identity(fact) for fact in self.facts),
            context_facts=tuple(fact_identity(fact) for fact in self.context_facts),
        )
