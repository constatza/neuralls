"""Field bindings: how a config model becomes facts, and what they never change."""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neuralls.shared.digest import canonical_digest
from neuralls.shared.fact_binding import (
    Fact,
    Nested,
    Omitted,
    Show,
    UnboundFieldError,
    describe_model,
    unbound_fields,
)
from neuralls.shared.scientific_description.catalog import (
    DAMPING,
    MULTIGRID_LEVELS,
    THRESHOLD,
    FactRole,
)
from neuralls.shared.scientific_description.models import IntegerFact


class Inner(BaseModel):
    theta: Annotated[float, Fact(THRESHOLD)] = 0.25


class Bound(BaseModel):
    note: Annotated[str, Omitted(FactRole.OPERATIONAL)] = "x"
    omega: Annotated[float | None, Fact(DAMPING, show=Show.NOT_NONE)] = None
    inner: Annotated[Inner, Nested()] = Inner()
    levels: Annotated[int, Fact(MULTIGRID_LEVELS, show=Show.NON_DEFAULT)] = 2


class Unbound(Bound):
    sharpness: float = 1.0


def test_facts_follow_declaration_order_and_hide_what_is_not_shown() -> None:
    assert [f.key for f in describe_model(Bound())] == [THRESHOLD]
    shown = describe_model(Bound(omega=1.5, levels=3))
    assert [f.key for f in shown] == [DAMPING, THRESHOLD, MULTIGRID_LEVELS]
    assert shown[-1] == IntegerFact(MULTIGRID_LEVELS, 3)


def test_a_field_without_a_binding_is_refused() -> None:
    assert unbound_fields(Unbound) == ("sharpness",)
    with pytest.raises(UnboundFieldError, match="sharpness"):
        describe_model(Unbound())


def test_bindings_do_not_enter_the_identity_digest() -> None:
    class Plain(BaseModel):
        omega: float | None = None
        levels: int = 2

    class Annotated_(BaseModel):
        omega: Annotated[float | None, Fact(DAMPING)] = None
        levels: Annotated[int, Omitted(FactRole.OPERATIONAL)] = 2

    assert canonical_digest(Plain()) == canonical_digest(Annotated_())
