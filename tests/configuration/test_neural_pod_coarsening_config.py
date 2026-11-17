"""Schema validation tests for NeuralPODCoarseningConfig.

Only schema/validity is tested here (per project convention: no tests
asserting on specific values inside a checked-in TOML) — parsing, discriminated
union resolution, the rank's mode-count rule, and the POD inner product's rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.coarsening import (
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import AMGPreconditionerConfig
from neuralls.shared.types import InnerProductKind

WORKSPACE = Path("workspace")


def test_neural_pod_coarsening_config_requires_input_names() -> None:
    """input_names is required — there is no default that could silently mismatch.

    Unlike the dataset's on-disk params artifact key (an enumeration
    artifact of how the dataset was written), the checkpoint's expected
    keyword-argument names are declared training-side and must be stated
    explicitly here; there is no way to infer them.
    """
    with pytest.raises(ValidationError):
        NeuralPODCoarseningConfig.model_validate(
            {
                "dataset_dir": str(WORKSPACE / "params-dataset"),
                "checkpoint_path": str(WORKSPACE / "model.ckpt"),
            }
        )


def test_amg_config_resolves_neural_pod_coarsening_via_discriminator() -> None:
    """AMGPreconditionerConfig.coarsening resolves to NeuralPODCoarseningConfig by method."""
    config = AMGPreconditionerConfig(
        coarsening={
            "method": "neural_pod",
            "dataset_dir": str(WORKSPACE / "params-dataset"),
            "checkpoint_path": str(WORKSPACE / "model.ckpt"),
            "input_names": ["young_modulus"],
            "rank": 12,
        },
    )
    assert isinstance(config.coarsening, NeuralPODCoarseningConfig)
    assert config.coarsening.rank == 12


@pytest.fixture
def neural_pod_fields() -> dict[str, object]:
    """Every field a NeuralPODCoarseningConfig needs besides its rank."""
    return {
        "dataset_dir": WORKSPACE / "params-dataset",
        "checkpoint_path": WORKSPACE / "model.ckpt",
        "input_names": ("young_modulus",),
    }


@pytest.mark.parametrize("rank", [0.99, 10.0, 0])
@pytest.mark.parametrize("neural", [False, True], ids=["pod", "neural-pod"])
def test_a_pod_rank_is_a_positive_mode_count(
    neural_pod_fields: dict[str, object], rank: float, neural: bool
) -> None:
    """A rank is the coarse dimension, so a fraction, a float or zero is rejected."""
    with pytest.raises(ValidationError) as error:
        if neural:
            NeuralPODCoarseningConfig.model_validate({**neural_pod_fields, "rank": rank})
        else:
            PODCoarseningConfig.model_validate({"dataset_dir": WORKSPACE, "rank": rank})
    assert [item["loc"] for item in error.value.errors()] == [("rank",)]


def test_an_energy_inner_product_on_an_inline_fit_is_rejected() -> None:
    """Only a fit job's checkpoint carries an energy basis; the inline fit is Euclidean."""
    with pytest.raises(ValidationError) as error:
        PODCoarseningConfig.model_validate({"dataset_dir": WORKSPACE, "inner_product": "energy"})
    assert [(item["type"], item["loc"]) for item in error.value.errors()] == [("value_error", ())]


def test_an_energy_fit_has_its_own_slug_and_a_euclidean_one_keeps_its_slug() -> None:
    """The inner product separates two otherwise equal fitted bases in every path."""
    euclidean = PODCoarseningConfig(dataset_dir=WORKSPACE, rank=10, assignment="fit_job_1")
    energy = euclidean.model_copy(update={"inner_product": InnerProductKind.ENERGY})

    assert (euclidean.slug(), energy.slug()) == (
        f"pod_rank10_n-1_{euclidean.checkpoint_slug()}",
        f"pod_rank10_n-1_energy_{euclidean.checkpoint_slug()}",
    )


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        pytest.param({"n_snapshots": 0}, ("n_snapshots",), id="zero"),
        pytest.param({"n_snapshots": -2}, ("n_snapshots",), id="negative"),
        pytest.param({"n_snapshots": 5, "assignment": "fit_job_1"}, (), id="checkpoint-backed"),
    ],
)
def test_n_snapshots_is_all_or_a_positive_inline_cut(
    fields: dict[str, object], loc: tuple[str, ...]
) -> None:
    """-1 keeps every snapshot; a checkpoint's basis is loaded, not refit, so it takes no cut."""
    with pytest.raises(ValidationError) as error:
        PODCoarseningConfig.model_validate({"dataset_dir": WORKSPACE, **fields})
    assert {item["loc"][:1] for item in error.value.errors()} == {loc}
