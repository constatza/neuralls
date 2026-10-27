"""Shared type aliases for tests that build prepared method presentations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from neuralls.domain.presentation.methods import PreparedMethodPresentation
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    PODCoarseningConfig,
)

type PreparedMethodFactory = Callable[[str], PreparedMethodPresentation]
"""The `prepared_method` fixture: the validated presentation of one method id."""


def pod_spec(dataset_dir: Path, rank: int, assignment: str = "fit-job") -> AMGPreconditionerConfig:
    """A POD-2G method fitted on ``dataset_dir`` by fit assignment ``assignment``."""
    return AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=dataset_dir, rank=rank, assignment=assignment)
    )
