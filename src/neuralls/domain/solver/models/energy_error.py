"""The energy-norm error curve one solve reports, or why it has none."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExactEnergyError:
    """``||e_k||_A / ||e_0||_A`` per iteration, measured against the reference solution."""

    history: list[float]


@dataclass(frozen=True, slots=True)
class EnergyErrorBound:
    """Golub-Meurant lower bound on ``||e_k||_A / ||e_0||_A``, used without a reference.

    A bound never stands in for the exact error: the two are distinct variants.
    """

    bound: list[float]


@dataclass(frozen=True, slots=True)
class EnergyErrorUnavailable:
    """No energy curve, for an expected reason (no usable iterate, no energy history)."""

    reason: str


@dataclass(frozen=True, slots=True)
class EnergyErrorFailed:
    """The energy-norm metric raised; the solve itself is kept. Operators are told."""

    reason: str


type EnergyError = ExactEnergyError | EnergyErrorBound | EnergyErrorUnavailable | EnergyErrorFailed
"""The energy-norm error of one solve."""


__all__ = [
    "EnergyError",
    "EnergyErrorBound",
    "EnergyErrorFailed",
    "EnergyErrorUnavailable",
    "ExactEnergyError",
]
