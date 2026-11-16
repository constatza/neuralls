"""The runtime patch that gives an energy-POD fit job its dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
from dlkit.infrastructure.config.job_config import FitJobConfig

from neuralls.composition.assignments.runtime_model_patcher import patch_energy_inner_product

DATASET_DIR = Path("processed/pod-dataset")


def _fit_job(**model_extra: object) -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit", "seed": 42},
            "model": {
                "name": "PODCoarseningFittable",
                "module_path": "neuralls.composition.preconditioners.pod_fittable",
                "rank": 8,
                **model_extra,
            },
            "data": {"name": "FlexibleDataset", "features": [], "targets": []},
        }
    )


def test_energy_inner_product_gains_the_job_dataset() -> None:
    patched = patch_energy_inner_product(_fit_job(inner_product="energy"), dataset_dir=DATASET_DIR)

    assert patched.model is not None
    assert patched.model.model_extra == {
        "rank": 8,
        "inner_product": {"kind": "energy", "dataset_dir": str(DATASET_DIR)},
    }


@pytest.mark.parametrize("model_extra", [{}, {"inner_product": "euclidean"}])
def test_other_models_are_returned_unchanged(model_extra: dict[str, object]) -> None:
    job = _fit_job(**model_extra)

    assert patch_energy_inner_product(job, dataset_dir=DATASET_DIR) is job
