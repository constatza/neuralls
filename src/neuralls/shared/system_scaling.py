"""How a linear system is scaled before generation and solving, as a closed sum type.

A system is scaled by one positive scalar per matrix: ``A' = A / alpha``,
``b' = b / alpha``. The solution is unchanged. CG iterates and weighted-Jacobi sweeps
with ``omega = 1 / rho(D^-1 A)`` are unchanged too, so a POD basis fitted to CG errors
or Jacobi probes does not depend on ``alpha``. The scale fixes the magnitude of ``A``,
``b`` and residuals: neural-network inputs, absolute stopping tolerances and energy norms
(``||e||_{A/alpha} = ||e||_A / sqrt(alpha)``).

The type lives in ``shared`` because dataset and case configs parse it and the domain
computes with it; ``domain/normalization.py`` holds the arithmetic.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__ = [
    "DimensionFactor",
    "GershgorinScaling",
    "NoScaling",
    "RhsScaling",
    "SystemScaling",
    "parse_system_scaling",
    "scaling_label",
]


class DimensionFactor(StrEnum):
    """Extra factor of the Gershgorin scale that depends on the dimension ``n``."""

    ONE = "one"
    SQRT_N = "sqrt_n"


class NoScaling(BaseModel):
    """Keep the system as loaded (``alpha = 1``)."""

    kind: Literal["none"] = "none"
    model_config = ConfigDict(extra="forbid", frozen=True)


class GershgorinScaling(BaseModel):
    """``alpha = max(||A||_1, ||A||_inf) * f(n)``, with ``f(n)`` equal to 1 or ``sqrt(n)``.

    ``max(||A||_1, ||A||_inf)`` is the Gershgorin bound on the spectral radius. It is an
    upper bound, not the spectral radius: ``rho(A') <= 1 / f(n)``. ``sqrt_n`` is the
    default because every existing dataset was generated with it.
    """

    kind: Literal["gershgorin"] = "gershgorin"
    dimension_factor: DimensionFactor = DimensionFactor.SQRT_N
    model_config = ConfigDict(extra="forbid", frozen=True)


type SystemScaling = Annotated[NoScaling | GershgorinScaling, Field(discriminator="kind")]

_SYSTEM_SCALING_ADAPTER: TypeAdapter[SystemScaling] = TypeAdapter(SystemScaling)


def scaling_label(scaling: SystemScaling) -> str:
    """Flat text of a scaling for logs, figure cards and event frames: its kind and factor."""
    match scaling:
        case NoScaling():
            return scaling.kind
        case GershgorinScaling(dimension_factor=factor):
            return f"{scaling.kind}({factor.value})"


def parse_system_scaling(raw: object) -> SystemScaling:
    """Validate a stored scaling (its ``model_dump``) back into a ``SystemScaling``.

    Raises:
        pydantic.ValidationError: If ``raw`` is not a known scaling.
    """
    return _SYSTEM_SCALING_ADAPTER.validate_python(raw)


class RhsScaling(StrEnum):
    """Per-system RHS scaling applied after the matrix scaling, at comparison time only.

    ``UNIT`` divides the RHS by its own 2-norm, so every system starts from a unit initial
    residual; a zero RHS is left unchanged. It changes the solution by the same factor and
    leaves the matrix alone.
    """

    NONE = "none"
    UNIT = "unit"
