"""Pure value types for scientific descriptions.

No dependency outside the standard library: this module must stay importable
without strategies, preconditioners, Pydantic, Matplotlib, Loguru, MLflow, or
storage code (see ``docs/plan.md``, "Decoupled scientific-description core").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_FIELD_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


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


class FieldImportance(StrEnum):
    """Priority of a fact within ``ScientificDescription.facts``.

    Drives deterministic length-budget reduction: ``VARIANT`` facts are
    dropped before ``ESSENTIAL`` ones, never the other way around.
    """

    ESSENTIAL = "essential"
    VARIANT = "variant"


@dataclass(frozen=True, slots=True)
class IntegerFact:
    """An integer-valued scientific fact, e.g. ``iteration_count=15``."""

    key: FieldKey
    value: int
    importance: FieldImportance


@dataclass(frozen=True, slots=True)
class DecimalFact:
    """A float-valued scientific fact, e.g. ``standard_deviation=0.01``."""

    key: FieldKey
    value: float
    importance: FieldImportance


@dataclass(frozen=True, slots=True)
class TextFact:
    """A text-valued scientific fact, e.g. ``weighting="uniform"``."""

    key: FieldKey
    value: str
    importance: FieldImportance


type ScientificFact = IntegerFact | DecimalFact | TextFact


@dataclass(frozen=True, slots=True)
class ScientificDescription:
    """A controlled base name plus ordered facts and ordered context facts.

    Contains no rendered string and performs no I/O; ``render()`` is the only
    function that turns this into presentation text.
    """

    base_name: str
    facts: tuple[ScientificFact, ...] = ()
    context_facts: tuple[ScientificFact, ...] = ()

    def __post_init__(self) -> None:
        """Reject an empty base name."""
        if not self.base_name:
            raise ValueError("ScientificDescription.base_name must not be empty")
