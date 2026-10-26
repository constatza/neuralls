from __future__ import annotations

import pytest

from neuralls.platform.config.models.data_models import OutputConfig


def test_output_config_accepts_supported_dataset_formats() -> None:
    assert OutputConfig(dataset_format="zarr").dataset_format == "zarr"
    assert OutputConfig(dataset_format="npy").dataset_format == "npy"


def test_output_config_rejects_unknown_dataset_format() -> None:
    with pytest.raises(ValueError):
        OutputConfig(dataset_format="parquet")
