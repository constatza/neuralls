"""Scientific descriptions of preconditioner configs and built preconditioners.

Configured facts come from the validated config and are known before execution:
every algorithmic parameter (defaults included), reproducibility seeds, and
scheduling controls only when non-default. Realized facts come from the built
object after setup. Each emits facts in the order it wants them shown. Checkpoint,
assignment, path, and config-name identity is never a fact.
"""

from __future__ import annotations

from functools import singledispatch

import torch
from torchalg.multigrid.hierarchy import MultigridHierarchy
from torchalg.preconditioners.base import Preconditioner, PreconditionerNotReadyError
from torchalg.preconditioners.implementations import (
    IC0Preconditioner,
    Identity,
    ILUPreconditioner,
    JacobiPreconditioner,
    NeuralPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    AggregationCoarsening,
    AMGPreconditioner,
    BootstrapAMGPreconditioner,
    NeuralCoarseningStrategy,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.amg.protocols import CoarseningStrategy
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.coarsening import (
    AggregationCoarsening as SparseAggregationCoarsening,
)
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner as SparseIC0Preconditioner
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner as SparseILUPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)

from neuralls.platform.config.models.preconditioner import (
    ConcretePreconditionerConfig,
    MissingArchitectureNameError,
)
from neuralls.platform.config.models.preconditioner_family import preconditioner_family
from neuralls.shared.fact_binding import describe_model
from neuralls.shared.scientific_description.catalog import (
    COARSE_DIMENSION,
    REALIZED_LEVELS,
    make_fact,
)
from neuralls.shared.scientific_description.models import ScientificDescription, ScientificFact

__all__ = [
    "MissingArchitectureNameError",
    "UnsupportedRealizedPreconditionerError",
    "describe_configured_preconditioner",
    "describe_realized_preconditioner",
]

type Facts = tuple[ScientificFact, ...]
type PODStrategy = PODCoarseningStrategy | SparsePODCoarseningStrategy


def describe_configured_preconditioner(cfg: ConcretePreconditionerConfig) -> ScientificDescription:
    """Describe ``cfg`` completely from configuration alone, with no I/O or setup."""
    return ScientificDescription(
        base_name=preconditioner_family(cfg).abbreviation(), facts=describe_model(cfg)
    )


# ==============================================================================
# Realized facts — read from the constructed object
# ==============================================================================


class UnsupportedRealizedPreconditionerError(TypeError):
    """A built preconditioner or coarsening class has no declared realized facts.

    Every class this repository builds declares its realized facts, possibly none. A class
    that does not is a new, undescribed method, so describing it fails instead of
    silently labelling it without its realized facts.
    """

    def __init__(self, kind: type) -> None:
        self.kind = kind
        super().__init__(
            f"No realized facts are declared for {kind.__module__}.{kind.__qualname__}; "
            "register its realized facts (an empty tuple when it has none)"
        )


def _first_coarse_dimension(precond: AMGPreconditioner) -> int:
    """Dimension of the first coarse level, read from the hierarchy ``setup`` already built.

    torchalg exposes no public accessor for the built hierarchy of a plain
    ``AMGPreconditioner``; reading it avoids building a transfer a second time, which would
    repeat the setup cost per matrix outside the timed and failure-handled setup.
    """
    match precond._hierarchy:
        case MultigridHierarchy(levels=(_, coarse, *_)):
            return int(coarse.matrix.shape[0])
        case _:
            raise PreconditionerNotReadyError(
                "A multigrid hierarchy with a coarse level is required to describe the "
                "realized coarse dimension; call setup() first"
            )


@singledispatch
def _coarsening_facts(coarsening: CoarseningStrategy, precond: AMGPreconditioner) -> Facts:
    raise UnsupportedRealizedPreconditionerError(type(coarsening))


@_coarsening_facts.register(PODCoarseningStrategy)
@_coarsening_facts.register(SparsePODCoarseningStrategy)
def _(coarsening: PODStrategy, precond: AMGPreconditioner) -> Facts:
    return (make_fact(COARSE_DIMENSION, coarsening.rank),)


@_coarsening_facts.register(AggregationCoarsening)
@_coarsening_facts.register(SparseAggregationCoarsening)
def _(
    coarsening: AggregationCoarsening | SparseAggregationCoarsening, precond: AMGPreconditioner
) -> Facts:
    return (make_fact(COARSE_DIMENSION, _first_coarse_dimension(precond)),)


@_coarsening_facts.register
def _(coarsening: TargetDimensionCoarsening, precond: AMGPreconditioner) -> Facts:
    return (make_fact(COARSE_DIMENSION, coarsening.realized_coarse_dim(precond.matrix)),)


@_coarsening_facts.register
def _(coarsening: NeuralCoarseningStrategy, precond: AMGPreconditioner) -> Facts:
    return ()


@singledispatch
def describe_realized_preconditioner(precond: Preconditioner) -> Facts:
    """Facts only the built object knows.

    Raises:
        UnsupportedRealizedPreconditionerError: If the class declares no realized facts.
    """
    raise UnsupportedRealizedPreconditionerError(type(precond))


@describe_realized_preconditioner.register(Identity)
@describe_realized_preconditioner.register(JacobiPreconditioner)
@describe_realized_preconditioner.register(SparseJacobiPreconditioner)
@describe_realized_preconditioner.register(ILUPreconditioner)
@describe_realized_preconditioner.register(SparseILUPreconditioner)
@describe_realized_preconditioner.register(IC0Preconditioner)
@describe_realized_preconditioner.register(SparseIC0Preconditioner)
@describe_realized_preconditioner.register(NeuralPreconditioner)
def _(precond: Preconditioner) -> Facts:
    return ()


@describe_realized_preconditioner.register
def _(precond: ScheduledPreconditioner) -> Facts:
    return describe_realized_preconditioner(precond.primary)


@describe_realized_preconditioner.register
def _(precond: AMGPreconditioner) -> Facts:
    return _coarsening_facts(precond.coarsening, precond)


def _hierarchy_facts(matrices: list[torch.Tensor]) -> Facts:
    return (
        make_fact(REALIZED_LEVELS, len(matrices)),
        make_fact(COARSE_DIMENSION, int(matrices[-1].shape[0])),
    )


@describe_realized_preconditioner.register(AdaptiveSAPreconditioner)
@describe_realized_preconditioner.register(SparseAdaptiveSAPreconditioner)
@describe_realized_preconditioner.register(BootstrapAMGPreconditioner)
@describe_realized_preconditioner.register(SparseBootstrapAMGPreconditioner)
def _(
    precond: AdaptiveSAPreconditioner
    | SparseAdaptiveSAPreconditioner
    | BootstrapAMGPreconditioner
    | SparseBootstrapAMGPreconditioner,
) -> Facts:
    return _hierarchy_facts(list(precond.result.matrices))
