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
    """No energy curve: neither the exact error nor its bound could be computed."""

    reason: str


type EnergyError = ExactEnergyError | EnergyErrorBound | EnergyErrorUnavailable
"""The energy-norm error of one solve."""


__all__ = ["EnergyError", "EnergyErrorBound", "EnergyErrorUnavailable", "ExactEnergyError"]
