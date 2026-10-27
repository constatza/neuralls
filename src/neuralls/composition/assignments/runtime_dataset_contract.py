"""Composition-owned runtime dataset contract for supervised workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeDatasetContract:
    """Canonical runtime names for the supported supervised workflow."""

    primary_input_name: str
    matrix_input_name: str
    target_name: str
    prediction_name: str
    loss_target_key: str


def default_training_dataset_contract() -> RuntimeDatasetContract:
    """Return the default runtime dataset contract for training and inference."""
    return RuntimeDatasetContract(
        primary_input_name="x",
        matrix_input_name="matrix",
        target_name="y",
        prediction_name="y_pred",
        loss_target_key="targets.y",
    )
