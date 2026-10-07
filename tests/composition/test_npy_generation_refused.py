"""Generation refuses npy as an output format.

npy is readable as an existing dataset format, but it is not a generation output:
the buffered npy writer held every sample in memory, and streamed npy is future work.
The refusal is raised at the entry point of ``build_dataset`` before any directory
is created, so a refused build leaves nothing behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec


def test_npy_generation_is_refused_with_supported_formats_named(
    atomic_matrix_dir: Path,
    tmp_path: Path,
    atomic_spec: DatasetSpec,
) -> None:
    source = SourceSpec(matrix_path=str(atomic_matrix_dir / "A_000.txt"))
    dataset_dir = tmp_path / "ds"

    with pytest.raises(ValueError, match="zarr") as excinfo:
        build_dataset(source, atomic_spec, str(dataset_dir), dataset_format="npy")

    assert "hdf5" in str(excinfo.value)
    assert not dataset_dir.exists()
