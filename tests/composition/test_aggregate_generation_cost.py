"""`aggregate_generation_cost` combines manifests already in memory."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.composition.comparison._offline_cost import aggregate_generation_cost
from neuralls.domain.solver.models.cost import Known, Unavailable
from neuralls.platform.storage.manifest import (
    DatasetArtifact,
    DatasetManifest,
    DatasetNormalization,
)
from neuralls.platform.storage.manifest_io import make_dataset_manifest
from neuralls.shared.types import CostProvenance

type ManifestFactory = Callable[[float | None, int | None], DatasetManifest]


@pytest.fixture
def manifest_with_cost() -> ManifestFactory:
    """Build a manifest whose only varying fields are the recorded generation costs."""
    matrix = DatasetArtifact(path="matrix.npy", format="npy", dtype="float64", shape=(2, 2))
    vector = DatasetArtifact(path="rhs.npy", format="npy", dtype="float64", shape=(2,))
    normalization = DatasetNormalization(
        type="none", matrix_norm=1.0, matrix_norm_type="spectral", scale={}
    )

    def build(seconds: float | None, memory: int | None = None) -> DatasetManifest:
        return make_dataset_manifest(
            matrix=matrix,
            rhs=vector,
            solutions=vector,
            normalization=normalization,
            generation_duration_seconds=seconds,
            generation_peak_memory_bytes=memory,
        )

    return build


def test_no_manifest_is_unavailable() -> None:
    assert isinstance(aggregate_generation_cost([]), Unavailable)


def test_manifests_without_a_duration_are_unavailable(manifest_with_cost: ManifestFactory) -> None:
    assert isinstance(aggregate_generation_cost([manifest_with_cost(None, 5)]), Unavailable)


def test_times_add_up_and_peak_memory_takes_the_maximum(
    manifest_with_cost: ManifestFactory,
) -> None:
    manifests = [
        manifest_with_cost(1.5, 100),
        manifest_with_cost(2.0, 300),
        manifest_with_cost(None, None),
    ]

    assert aggregate_generation_cost(manifests) == Known(3.5, 300, CostProvenance.HISTORICAL)


def test_unrecorded_memory_stays_unrecorded(manifest_with_cost: ManifestFactory) -> None:
    assert aggregate_generation_cost([manifest_with_cost(4.0, None)]) == Known(
        4.0, None, CostProvenance.HISTORICAL
    )
