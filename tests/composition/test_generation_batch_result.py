"""A generated dataset is reported by its stamped scientific description, never its registry id."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

from neuralls.composition.assignments.assignment_subjects import case_notation
from neuralls.composition.generation import multi_generation
from neuralls.composition.generation.multi_generation import generate_batch
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.platform.config.models.experiments import CaseConfig, RegistryEntry


def test_each_result_carries_the_presentation_stamped_on_its_dataset(
    tmp_path: Path, dataset_presentation: DatasetPresentation
) -> None:
    cfg = CaseConfig(datasets=[RegistryEntry(id="residuals-registry-id", path=Path("d.toml"))])
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        with (
            patch.object(multi_generation, "process_data_from_config", return_value=tmp_path),
            patch.object(
                multi_generation, "load_dataset_presentation", return_value=dataset_presentation
            ),
        ):
            (result,) = generate_batch(cfg, tmp_path, MagicMock())
    finally:
        logger.remove(sink_id)

    assert result.dataset is dataset_presentation
    assert result.notation == case_notation(cfg)
    assert messages == [f"Test problem | Gaussian CG Residuals (k=20, N=100) ready: {tmp_path}"]
