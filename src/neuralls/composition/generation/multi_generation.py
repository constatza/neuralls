"""Batch dataset generation workflow: generate all datasets from a case config.

Key Functions:
    - ``generate_batch()``: Iterates ``[[datasets]]`` in the case config and
      calls ``process_data_from_config()`` for each entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neuralls.composition.assignments.assignment_subjects import case_notation
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.run_names import render_dataset_label
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result for a single generated dataset.

    Attributes:
        dataset_id: Dataset registry id (from ``[[datasets]]``).
        config_path: Resolved dataset config TOML path.
        output_dir: Directory written by ``process_data_from_config``.
        dataset: What the written dataset is, from the provenance stamped on it.
        notation: The case's symbol policy, which renders ``dataset``.
    """

    dataset_id: str
    config_path: Path
    output_dir: Path
    dataset: DatasetPresentation
    notation: NotationPolicy


def generate_batch(
    cfg: CaseConfig,
    configs_dir: Path,
    settings: NeurallsSettings,
    *,
    force: bool = False,
) -> list[GenerationResult]:
    """Generate all datasets listed in the case config.

    Iterates ``cfg.datasets``, resolves each entry's path relative to
    ``configs_dir``, and calls ``process_data_from_config()`` for each one.

    Args:
        cfg: Validated case configuration.
        configs_dir: Parent directory of the case TOML (for resolving
            relative dataset config paths).
        force: Regenerate every dataset even if a matching one already exists.

    Returns:
        List of ``GenerationResult`` in the same order as ``cfg.datasets``.
    """
    notation = case_notation(cfg)
    results: list[GenerationResult] = []
    for entry in cfg.datasets:
        config_path = (configs_dir / entry.path).resolve()
        output_dir = process_data_from_config(config_path, settings, force=force)
        result = GenerationResult(
            dataset_id=entry.id,
            config_path=config_path,
            output_dir=output_dir,
            dataset=load_dataset_presentation(output_dir),
            notation=notation,
        )
        results.append(result)
        logger.info("{} ready: {}", render_dataset_label(result.dataset, notation), output_dir)
    return results
