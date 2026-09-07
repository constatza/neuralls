"""Descriptive labels for constructed preconditioner objects.

Builds short, human-readable structural descriptions of live ``torchalg``
preconditioner instances (grid levels; coarsening strategy: threshold +
prolongation-smoothing damping + realized coarse dimension, or fitted POD
rank as that same coarse dimension) for use in comparison plot legends and
axis labels. Cycle type is deliberately excluded — see
:func:`_describe_coarsening` — and Greek/math symbols (``L``, ``θ``, ``ω``,
``c``) are used throughout rather than spelled-out words, matching the
notation the study's own plots put on their axes. Every coarsening strategy
renders its coarse dimension as ``c=`` — POD-2G's fitted rank and
target-dimension coarsening's realized dimension are each that same
coarse dimension — so all methods read consistently side by side; grid
level count (``L=``) is shown for every strategy except POD-2G, which is
architecturally single-basis two-grid (a level count there would be
constant noise, not signal).

Detail is read from the constructed object's own attributes, never from the
TOML config that produced it: the object is the ground truth for what was
actually built (e.g. the POD basis's real fitted rank, which may differ from
a configured energy-threshold ``rank``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import IC0Preconditioner, ScheduledPreconditioner
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    AMGPreconditioner,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

__all__ = [
    "MAX_LABEL_LENGTH",
    "AggregationCoarseningDetail",
    "PODCoarseningDetail",
    "TargetDimensionCoarseningDetail",
    "build_preconditioner_labels",
    "coarsening_detail",
    "describe_preconditioner",
    "preconditioner_label",
]

MAX_LABEL_LENGTH = 45
"""Legend-text budget per preconditioner, name included.

Sized to the actual longest current entry (an AMG variant with ``L``, ``θ``,
``ω``, and ``c`` all shown: ``"amg-medium-theta (L=2, θ=0.25, ω=0.67,
c=15)"`` is 44 characters) plus a little headroom, not picked arbitrarily.

A comparison run typically legends several variants of the same type side by
side (e.g. three AMG theta values) — this keeps each entry to roughly one
line so the legend stays readable rather than pushing plots off-canvas.
"""


@dataclass(frozen=True)
class AggregationCoarseningDetail:
    """Structured facts about a built smoothed-aggregation coarsening level.

    Attributes:
        theta (float): Configured strength-of-connection threshold.
        omega (float): Configured Jacobi-smoothing damping for the
            prolongation smoother.
        coarse_dimension (int): Realized coarse dimension, read back by
            building the transfer operator — ``theta`` is a threshold, not a
            chosen dimension, so this is the only way to know it.
    """

    theta: float
    omega: float
    coarse_dimension: int


@dataclass(frozen=True)
class PODCoarseningDetail:
    """Structured facts about a fitted POD-2G coarsening basis.

    Attributes:
        rank (int): Actual fitted basis width (may differ from a configured
            energy-threshold ``rank``).
    """

    rank: int


@dataclass(frozen=True)
class TargetDimensionCoarseningDetail:
    """Structured facts about a coarsening strategy driven by a target dimension.

    Attributes:
        target_coarse_dim (int): The requested coarse dimension.
        realized_coarse_dim (int): The actual coarse dimension the closest-
            matching `theta` produced — may differ slightly from
            `target_coarse_dim`, since realized dimension vs. `theta` is a
            step function (see `TargetDimensionCoarsening`'s docstring).
    """

    target_coarse_dim: int
    realized_coarse_dim: int


type CoarseningDetail = (
    AggregationCoarseningDetail | PODCoarseningDetail | TargetDimensionCoarseningDetail | None
)


def coarsening_detail(coarsening: object, matrix: torch.Tensor) -> CoarseningDetail:
    """Extract structured facts from a coarsening strategy.

    Kept separate from string formatting so callers (tests included) can
    assert on the real values (a float, an int) instead of parsing them back
    out of rendered text.

    Args:
        coarsening (object): The AMG preconditioner's coarsening strategy
            object.
        matrix (torch.Tensor): The finest-level system matrix, used to read
            back the realized coarse dimension for aggregation coarsening.

    Returns:
        CoarseningDetail: The matching detail dataclass, or ``None`` for
            unrecognized strategies.
    """
    if isinstance(coarsening, PODCoarseningStrategy):
        return PODCoarseningDetail(rank=coarsening._basis.shape[1])
    if isinstance(coarsening, AggregationCoarsening):
        coarse_matrix, _ = coarsening.build_transfer(matrix)
        return AggregationCoarseningDetail(
            theta=coarsening._theta,
            omega=coarsening._omega,
            coarse_dimension=coarse_matrix.shape[0],
        )
    if isinstance(coarsening, TargetDimensionCoarsening):
        if coarsening._realized_coarse_dim is None:
            # Not yet built (no CG solve has triggered the hierarchy's
            # lazy build yet) — build once now rather than leave the
            # label with nothing to report; the search result is cached
            # afterward, so this never repeats the search.
            coarse_matrix, _ = coarsening.build_transfer(matrix)
            realized_coarse_dim = coarse_matrix.shape[0]
        else:
            realized_coarse_dim = coarsening._realized_coarse_dim
        return TargetDimensionCoarseningDetail(
            target_coarse_dim=coarsening._target_coarse_dim,
            realized_coarse_dim=realized_coarse_dim,
        )
    return None


def describe_preconditioner(precond: Preconditioner) -> str:
    """Build a short structural-detail string from a constructed preconditioner.

    Args:
        precond (Preconditioner): A constructed (already-instantiated)
            preconditioner object, e.g. as produced by
            ``composition.preconditioners.factory.create_preconditioner``.
            ``ScheduledPreconditioner`` wrappers are unwrapped to their
            primary preconditioner before inspection.

    Returns:
        str: A structural detail string, such as ``"L=2, θ=0.25, ω=0.67,
            c=17"`` for AMG (``θ``/``ω``/``c`` match the notation the
            study's own plots already put on their x-axis — see
            ``MAX_LABEL_LENGTH``, since a comparison run typically legends
            several AMG variants at once), or ``""`` for preconditioner
            types with no meaningful structural variants (e.g. ``Identity``,
            ``JacobiPreconditioner``, ``NeuralPreconditioner``).
    """
    if isinstance(precond, ScheduledPreconditioner):
        return describe_preconditioner(precond._primary)
    if isinstance(precond, AMGPreconditioner):
        return _describe_amg(precond)
    if isinstance(precond, IC0Preconditioner):
        return f"threshold={precond._threshold:.0e}"
    return ""


def preconditioner_label(name: str, precond: Preconditioner) -> str:
    """Combine a preconditioner's config name with its live structural detail.

    Args:
        name (str): Config-level preconditioner name (the dict key used
            throughout the comparison pipeline).
        precond (Preconditioner): The corresponding constructed
            preconditioner object.

    Returns:
        str: ``"{name} ({detail})"`` when structural detail is available
            (see :func:`describe_preconditioner`), else just ``name``.
    """
    detail = describe_preconditioner(precond)
    display_name = _display_name(name, precond)
    return f"{display_name} ({detail})" if detail else display_name


def _display_name(name: str, precond: Preconditioner) -> str:
    """Render machine-style preconditioner names as plot-friendly labels."""
    if isinstance(precond, ScheduledPreconditioner):
        return _display_name(name, precond._primary)
    return name.replace("_", " ").title()


def build_preconditioner_labels(
    preconditioners: dict[str, Preconditioner],
) -> dict[str, str]:
    """Build a name -> descriptive-label mapping for a set of preconditioners.

    Args:
        preconditioners (dict[str, Preconditioner]): Constructed
            preconditioner instances keyed by config name, e.g. as produced
            by ``PreconditionerService.create_preconditioner_set``.

    Returns:
        dict[str, str]: Mapping from each name to its
            :func:`preconditioner_label` string.
    """
    return {name: preconditioner_label(name, precond) for name, precond in preconditioners.items()}


def _describe_amg(precond: AMGPreconditioner) -> str:
    """Describe an AMG preconditioner's grid levels plus coarsening detail.

    Cycle type is left out — see :func:`_describe_coarsening` — and so is
    ``n_levels`` for POD-2G coarsening: ``PODCoarseningStrategy`` is
    architecturally a single-basis two-grid method, so its own docstring
    notes level count isn't a meaningful variant there. For aggregation
    coarsening, ``n_levels`` stays, written as ``L={n}`` (the standard
    multigrid symbol for level count, mirroring ``θ``/``ω``/``c`` rather
    than an English abbreviation): it is a real config knob that can differ
    between runs.

    Args:
        precond (AMGPreconditioner): Constructed AMG preconditioner instance.

    Returns:
        str: ``"L={n}, {coarsening detail}"`` for aggregation coarsening, or
            just the coarsening detail for POD-2G.
    """
    detail = _describe_coarsening(precond._coarsening, precond._matrix)
    if isinstance(precond._coarsening, PODCoarseningStrategy):
        return detail
    return f"L={precond._n_levels}, {detail}"


def _describe_coarsening(coarsening: object, matrix: torch.Tensor) -> str:
    """Render a coarsening strategy's structured detail as plot-legend text.

    Deliberately omits cycle type: cycle is fixed across every preconditioner
    in a typical comparison run (all variants share the same multigrid
    scaffolding — only the coarsening differs), so repeating it per legend
    entry would be redundant, not informative. ``θ``, ``ω``, and ``c`` are
    always the Greek/math symbols (never the spelled-out words) and match
    the notation the study's own plots already use on their axes (e.g.
    "Realized coarse dimension c"), so the legend stays consistent with the
    figure instead of introducing new shorthand a reader has to decode.
    ``c`` is used for both methods' coarse dimension — POD-2G's fitted rank
    *is* its coarse dimension, so there is no separate symbol for it.

    Args:
        coarsening (object): The AMG preconditioner's coarsening strategy
            object.
        matrix (torch.Tensor): The finest-level system matrix, forwarded to
            :func:`coarsening_detail`.

    Returns:
        str: ``"c={n}"`` for POD-2G or target-dimension coarsening,
            ``"θ={theta}, ω={omega}, c={n}"`` for aggregation coarsening
            (``c`` being the realized coarse dimension in every case), or
            the class name as a fallback for unrecognized
            ``CoarseningStrategy`` implementations.
    """
    match coarsening_detail(coarsening, matrix):
        case PODCoarseningDetail(rank=rank):
            return f"c={rank}"
        case TargetDimensionCoarseningDetail(realized_coarse_dim=c):
            return f"c={c}"
        case AggregationCoarseningDetail(theta=theta, omega=omega, coarse_dimension=c):
            return f"θ={theta:.2g}, ω={omega:.2g}, c={c}"
        case None:
            return type(coarsening).__name__
