"""Pure value types for scientific descriptions.

No dependency outside the standard library: this module must stay importable
without strategies, preconditioners, Pydantic, Matplotlib, Loguru, MLflow, or
storage code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

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


class FieldImportance(StrEnum):
    """Semantic selection hint for presentation planning.

    The renderer never uses importance to delete a selected fact.
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

    def __post_init__(self) -> None:
        """Reject values with no finite scientific representation."""
        if not isfinite(self.value):
            raise ValueError(f"DecimalFact value must be finite, got {self.value!r}")


@dataclass(frozen=True, slots=True)
class TextFact:
    """A text-valued scientific fact, e.g. ``weighting="uniform"``."""

    key: FieldKey
    value: str
    importance: FieldImportance


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
    importance: FieldImportance


type ScientificFact = IntegerFact | DecimalFact | TextFact | DisplayTextFact


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
