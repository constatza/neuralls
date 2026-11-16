"""Point an energy-norm POD fit job at the dataset that defines its inner product."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import TrainableJobConfig
from neuralls.composition.preconditioners.pod_fittable import ENERGY_INNER_PRODUCT

_INNER_PRODUCT_FIELD = "inner_product"


def patch_energy_inner_product[T: TrainableJobConfig](settings: T, *, dataset_dir: Path) -> T:
    """Add the job's dataset to a `[model] inner_product = "energy"` entry.

    The energy weight is the mean of the dataset's train-fold matrices, and only the
    runtime knows which dataset a job fits from, so the TOML names the kind alone and
    this stage supplies the directory. Any other model is returned unchanged.
    """
    model = settings.model
    if model is None or (model.model_extra or {}).get(_INNER_PRODUCT_FIELD) != ENERGY_INNER_PRODUCT:
        return settings
    return settings.patch(
        {
            "model": {
                _INNER_PRODUCT_FIELD: {
                    "kind": ENERGY_INNER_PRODUCT,
                    "dataset_dir": str(dataset_dir),
                }
            }
        }
    )
