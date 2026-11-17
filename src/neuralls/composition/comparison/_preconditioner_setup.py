"""Preconditioner creation, scheduling, input binding, and stopping criterion registry."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
from torchalg.preconditioners.base import BindableInputs, Preconditioner

from neuralls.composition.comparison.models import MatrixWorkload, PlannedSystem
from neuralls.composition.comparison.plan import SETUP_SEED_STREAM, derive_seed
from neuralls.composition.preconditioners.checkpoint_store import (
    NO_PRELOADED,
    CheckpointLoaders,
    CheckpointStore,
)
from neuralls.composition.preconditioners.descriptions import (
    Facts,
    describe_realized_preconditioner,
)
from neuralls.composition.preconditioners.factory import create_preconditioner
from neuralls.composition.preconditioners.ports import FitDataLoader
from neuralls.composition.preconditioners.preloading import preload_checkpoints
from neuralls.composition.preconditioners.schedule import (
    PreconditionerScheduleConfig,
    create_scheduled_preconditioner,
)
from neuralls.domain.presentation.methods import SystemRealization
from neuralls.platform.config.models.coarsening import NeuralPODCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.storage.training_artifacts import (
    load_array_source_sample,
    load_training_arrays,
)
from neuralls.shared.types import DatasetId, MatrixFormat, MatrixId, PreconditionerName
from neuralls.shared.workflow_failures import ComparisonMatrixFailureContext, ComparisonMatrixPhase

# Arrays always available from LinearSystem — never need loading from disk.
_SYSTEM_ARRAYS_ALWAYS_AVAILABLE: frozenset[str] = frozenset({"matrix"})


def matrix_key(system: PlannedSystem) -> MatrixId:
    """The matrix a planned system solves on, for per-matrix setup scoping."""
    return system.matrix.matrix_id


class PreconditionerService:
    """Service for creating preconditioners via the composition factory.

    Args:
        fit_data: Where a neural POD's parameter arrays are read from; `preload` reads them
            once per method so the timed build never touches the dataset.
        loaders: How checkpoints are loaded and on which device (`CheckpointLoaders.dlkit`).
        checkpoints: Where checkpoints are loaded ahead of a build (see `preload`).
            ``None`` means nothing is preloaded: every checkpoint is then read
            inside the build, and so inside its timed region.
        pod_basis: The basis a POD method fitted once before its timed setups
            (`_method_fit.fit_method`); every build of the method borrows it.
    """

    def __init__(
        self,
        fit_data: FitDataLoader,
        loaders: CheckpointLoaders | None = None,
        checkpoints: CheckpointStore | None = None,
        pod_basis: torch.Tensor | None = None,
    ) -> None:
        self._fit_data = fit_data
        self._loaders = loaders
        self._checkpoints = checkpoints
        self._pod_basis = pod_basis
        self._staged_fit_data: dict[str, tuple[np.ndarray, ...]] = {}

    def preload(self, matrix: torch.Tensor, config: PreconditionerConfig) -> None:
        """Load the checkpoints ``config`` depends on, outside any timed region.

        Loading a checkpoint is file I/O, not part of a preconditioner's
        algorithmic setup cost, and a checkpoint can serve many builds (every
        matrix of this method workload, every timing repeat); so it is loaded once, here,
        before the timed build, and the build borrows the loaded object. A
        no-op for configs without checkpoints and for a service built without a
        store.

        Args:
            matrix: System matrix; checkpoints are loaded for its dtype/device.
            config: Preconditioner configuration.
        """
        if self._checkpoints is not None:
            preload_checkpoints(
                config,
                matrix,
                self._checkpoints,
                self._loaders,
            )
        match config:
            case AMGPreconditionerConfig(coarsening=NeuralPODCoarseningConfig() as coarsening):
                if config.name not in self._staged_fit_data:
                    self._staged_fit_data[config.name] = tuple(
                        self._fit_data.parameters(coarsening.dataset_dir)
                    )
            case _:
                pass

    def create_preconditioner(
        self,
        matrix: torch.Tensor,
        config: PreconditionerConfig,
    ) -> Preconditioner:
        """Create a single preconditioner, borrowing whatever `preload` already loaded.

        Args:
            matrix: System matrix to precondition (shape: n x n).
            config: Preconditioner configuration.

        Returns:
            Preconditioner instance.
        """
        preloaded = self._checkpoints if self._checkpoints is not None else NO_PRELOADED
        return create_preconditioner(
            matrix,
            config,
            loaders=self._loaders,
            matrix_format=(
                MatrixFormat.CSR if matrix.layout is torch.sparse_csr else MatrixFormat.DENSE
            ),
            preloaded=preloaded,
            staged_fit_data=self._staged_fit_data.get(config.name),
            pod_basis=self._pod_basis,
        )


class RealizedFactCaptureError(RuntimeError):
    """Reading the realized facts of a built preconditioner failed.

    This is a labelling failure, not a scientific outcome of the method: it is never turned
    into a setup failure, and it stops the workload rather than letting solves continue
    under a legend that lacks their realized facts. ``context`` names the method and matrix.
    """

    def __init__(self, context: ComparisonMatrixFailureContext, cause: Exception) -> None:
        self.context = context
        super().__init__(f"Realized facts could not be read ({context!r}): {cause}")


def capture_system_realizations(
    preconditioner: Preconditioner, method: PreconditionerName, bundle: MatrixWorkload
) -> tuple[SystemRealization, ...]:
    """Read the facts only the built object knows, once for the matrix and all its systems.

    Setup already produced ``preconditioner``; reading it adds no setup. Every sibling RHS
    system shares the object, so they share its facts, and only these plain facts outlive
    it once the caller releases the preconditioner.

    Raises:
        RealizedFactCaptureError: If the facts cannot be read, with the method and matrix.
    """
    facts = _realized_facts(preconditioner, method, bundle)
    return tuple(
        SystemRealization(member.tag.system_id, bundle.matrix_id, facts)
        for member in bundle.members
    )


def _realized_facts(
    preconditioner: Preconditioner, method: PreconditionerName, bundle: MatrixWorkload
) -> Facts:
    """The realized facts of ``preconditioner``, or a failure naming the method and matrix."""
    try:
        return describe_realized_preconditioner(preconditioner)
    except Exception as exc:
        raise RealizedFactCaptureError(
            ComparisonMatrixFailureContext(
                phase=ComparisonMatrixPhase.REALIZED_FACT_CAPTURE,
                method=method,
                dataset=bundle.dataset,
                matrix_id=bundle.matrix_id,
            ),
            exc,
        ) from exc


def setup_config_for_matrix(
    config: PreconditionerConfig, dataset: DatasetId, matrix_id: MatrixId
) -> PreconditionerConfig:
    """The config one matrix's setup is built from: a randomized setup draws a seed per matrix.

    A seeded setup (aSA, BAMG) is built with a seed derived from ``config.seed``, the matrix set
    and the matrix (``derive_seed`` in the ``SETUP_SEED_STREAM``). The matrix set enters as the
    CRC-32 of its id, a key stable across processes, because matrix ids repeat across matrix
    sets and matrix ``k`` of two sets must not draw the same candidates. Every warm-up and timed repeat of one
    matrix builds from this same config, so its repeats stay identical and their fingerprints
    agree, while different matrices draw different candidates, which puts the seed's
    variability into the matrix-level interval at no extra cost. The method's identity, slug and facts keep the configured seed; only the
    build sees the derived one. A setup with no seed is returned unchanged.
    """
    match config:
        case AdaptiveSAPreconditionerConfig() | BootstrapAMGPreconditionerConfig():
            seed = derive_seed(
                config.seed,
                zlib.crc32(dataset.encode("utf-8")),
                matrix_id,
                stream=SETUP_SEED_STREAM,
            )
            return config.model_copy(update={"seed": seed})
        case _:
            return config


def setup_fingerprint(
    preconditioner: Preconditioner, method: PreconditionerName, bundle: MatrixWorkload
) -> int | None:
    """A cheap deterministic summary of a built preconditioner, to compare timed setup repeats.

    It is the class plus the realized facts every supported preconditioner already reports
    (``describe_realized_preconditioner``: level count and coarse dimension of a hierarchy),
    so a repeat that built a different hierarchy than the first makes the setup's repeats
    ``Diverged``. A preconditioner with no realized facts (Identity, Jacobi, ILU, IC(0),
    neural) has nothing a repeat could differ in, so it gets ``None`` and its repeats are
    stored ``Unchecked`` rather than claiming an agreement nothing checked. CRC-32 keeps the
    summary stable across processes and within the stored ``Int64`` column.

    Raises:
        RealizedFactCaptureError: If the facts cannot be read, with the method and matrix.
    """
    facts = _realized_facts(preconditioner, method, bundle)
    if not facts:
        return None
    return zlib.crc32(repr((type(preconditioner).__qualname__, facts)).encode("utf-8"))


def _schedule_preconditioner(
    cfg: PreconditionerConfig, matrix: torch.Tensor, primary: Preconditioner
) -> Preconditioner:
    """Wrap one built preconditioner with the scheduling its config asks for.

    Args:
        cfg: Config carrying the scheduling parameters.
        matrix: System matrix (needed for fallback creation).
        primary: The already-built preconditioner.

    Returns:
        The scheduled (or unscheduled) preconditioner.
    """
    schedule = PreconditionerScheduleConfig(
        start_iter=cfg.start_iter,
        limit_iters=cfg.limit_iters,
        fallback=cfg.fallback,
    )
    return create_scheduled_preconditioner(primary=primary, schedule=schedule, matrix=matrix)


def _collect_extra_name_positions(preconditioner: Preconditioner) -> dict[str, int]:
    """Map each extra-input name the preconditioner declares to its declaration index.

    Args:
        preconditioner: Scheduled preconditioner to inspect.

    Returns:
        Dict mapping each extra name to its position in ``extra_input_names``
        (first occurrence wins, preserving training-config declaration order); empty when
        the preconditioner binds no inputs.
    """
    if not isinstance(preconditioner, BindableInputs):
        return {}
    positions: dict[str, int] = {}
    for i, name in enumerate(preconditioner.extra_input_names):
        if name not in _SYSTEM_ARRAYS_ALWAYS_AVAILABLE:
            positions.setdefault(name, i)
    return positions


def _load_extra_inputs(matrix_path: Path, matrix_index: int) -> tuple[np.ndarray, ...]:
    """Read every parameter array for one matrix outside online timing.

    Args:
        matrix_path: Dataset directory containing ``parameters_N`` array files.
        matrix_index: Sample index to extract from each parameter array.

    Returns:
        Parameter arrays in manifest order. Torch tensor construction,
        device/dtype conversion and semantic binding deliberately happen
        later, inside setup timing.
    """
    if not matrix_path.is_dir():
        return ()
    try:
        arrays = load_training_arrays(matrix_path)
    except FileNotFoundError:
        return ()
    return tuple(
        np.array(load_array_source_sample(source, matrix_index), copy=True)
        for source in arrays.parameter_sources
    )


def _bind_extra_inputs(
    preconditioner: Preconditioner,
    matrix: torch.Tensor,
    staged_inputs: tuple[np.ndarray, ...],
) -> None:
    """Convert staged parameter arrays and bind them inside setup timing.

    Args:
        preconditioner: Preconditioner to inspect and bind.
        matrix: System matrix — always bound as ``"matrix"``.
        staged_inputs: Parameter arrays read before setup, in manifest order.
    """
    name_to_position = _collect_extra_name_positions(preconditioner)
    extra_data = {
        name: torch.as_tensor(staged_inputs[position], dtype=matrix.dtype, device=matrix.device)
        for name, position in name_to_position.items()
        if position < len(staged_inputs)
    }
    _bind_system_inputs(preconditioner, {"matrix": matrix, **extra_data})


def _bind_system_inputs(
    preconditioner: Preconditioner,
    system_data: dict[str, torch.Tensor],
) -> None:
    """Bind dataset-sourced extra inputs to a preconditioner that declares them.

    Reads ``extra_input_names`` from the preconditioner. For each name present in
    ``system_data``, calls ``bind_inputs()`` so the preconditioner can forward them to
    the model without changing the CG-facing apply(residual) interface.

    Args:
        preconditioner: The preconditioner to bind.
        system_data: Available named arrays (always includes ``"matrix"``).
    """
    if not isinstance(preconditioner, BindableInputs):
        return
    needed = {k: v for k, v in system_data.items() if k in preconditioner.extra_input_names}
    if needed:
        preconditioner.bind_inputs(**needed)
