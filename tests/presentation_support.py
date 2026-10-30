"""Shared type aliases for tests that build prepared method presentations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from neuralls.composition.assignments.presentation_preflight import DatasetPresentationLoader
from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig

type PreparedMethodFactory = Callable[[str], PreparedMethodPresentation]
"""The `prepared_method` fixture: the validated presentation of one method id."""


def pod_spec(dataset_dir: Path, rank: int, assignment: str = "fit-job") -> AMGPreconditionerConfig:
    """A POD-2G method fitted on ``dataset_dir`` by fit assignment ``assignment``."""
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=rank, assignment=assignment)
    )


NO_DATASET_READS: DatasetPresentationLoader = lambda directory: (_ for _ in ()).throw(
    AssertionError(f"unexpected manifest read of {directory}")
)
"""For methods and scopes that never describe a fit dataset: any read fails the test."""


def datasets_by_dir(by_dir: Mapping[Path, DatasetPresentation]) -> DatasetPresentationLoader:
    """An in-memory manifest reader: the presentation recorded for each directory.

    Raises ``KeyError`` for a directory no test registered, so a stray read fails loudly.
    """
    return lambda directory: by_dir[Path(directory)]
