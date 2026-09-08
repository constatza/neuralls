"""Generation context and plan assembly for the generate workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neuralls.domain.generation.data_types import NormalizeType
from neuralls.domain.generation.plan import (
    GenerationPlan,
    StrategySpec,
    canonicalize_strategy_kind,
    plan_from_specs,
)
from neuralls.domain.generation.source_streams import EnumerateBy
from neuralls.domain.generation.specs import SourceSpec
from neuralls.platform.config.models.data_models import DataConfigFile, GenerationConfig
from neuralls.shared.types import DatasetFormat


@dataclass(frozen=True)
class DataGenerationContext:
    """Immutable context assembled from config inputs.

    Attributes:
        matrix_path: Path to the system matrix file or glob.
        rhs_path: Optional path to the RHS vector file or glob.
        solutions_path: Optional path to a solutions archive glob.
        solution_path: Optional path to a single solution file.
        sample_id_regex: Optional regex for extracting sample IDs.
        enumerate_by: Optional enumeration strategy for archive sources.
        include_indices: Restrict every glob-based stream to exactly these sample ids.
        exclude_indices: Drop these sample ids from every glob-based stream.
        dataset_dir: Target directory for the generated dataset.
        normalize: Normalization strategy applied to each sample.
        seed: Random seed for reproducibility.
        shuffle: Whether to shuffle samples after generation.
        replacement: Whether to sample with replacement.
        parameters_paths: Tuple of additional parameter file paths.
        dataset_format: Storage format family for persisted dataset artifacts.
    """

    matrix_path: str
    rhs_path: str | None
    solutions_path: str | None
    solution_path: str | None
    sample_id_regex: str | None
    enumerate_by: EnumerateBy | None
    include_indices: tuple[int, ...] | None
    exclude_indices: tuple[int, ...]
    dataset_dir: Path
    normalize: NormalizeType
    seed: int
    shuffle: bool
    replacement: bool
    parameters_paths: tuple[str, ...] = ()
    dataset_format: DatasetFormat = "hdf5"

    def source_spec(self, *, rhs_path: str | None = None) -> SourceSpec:
        """Project the source-side fields into the domain's SourceSpec.

        The domain layer cannot import this composition-layer context, so it
        receives this projection instead of either side re-declaring the same
        eight source fields at every call boundary.

        ``self.rhs_path`` and ``self.solutions_path`` are deliberately *not*
        projected: both are glob fallbacks consumed by archive resolution in
        this layer, not stream paths. Only the synthetic executor opens an RHS
        stream, and it passes the already-resolved path in as ``rhs_path``;
        archive-only executors leave it unset because the archive itself
        supplies the RHS.

        Args:
            rhs_path: Resolved RHS stream path, or None to open no RHS stream.
        """
        return SourceSpec(
            matrix_path=self.matrix_path,
            rhs_path=rhs_path,
            solution_path=self.solution_path,
            parameters_paths=self.parameters_paths,
            sample_id_regex=self.sample_id_regex,
            enumerate_by=self.enumerate_by,
            include_indices=self.include_indices,
            exclude_indices=self.exclude_indices,
        )


def _plan_from_generation_config(
    generation_cfg: GenerationConfig,
) -> GenerationPlan:
    """Translate a validated GenerationConfig into a GenerationPlan.

    Converts Pydantic StrategyConfig objects directly to StrategySpec objects,
    bypassing raw-dict re-parsing.

    Args:
        generation_cfg: Validated generation configuration.

    Returns:
        GenerationPlan with resolved strategy specs.

    Raises:
        ValueError: If no strategy entries are declared.
    """
    if not generation_cfg.strategy:
        raise ValueError(
            "Missing '[[generation.strategy]]' entries in config. "
            "Add at least one strategy with 'name' and 'samples' fields."
        )
    specs: list[StrategySpec] = []
    for sc in generation_cfg.strategy:
        options = sc.model_dump(mode="python", exclude={"name", "samples"}, exclude_none=True)
        specs.append(
            StrategySpec(
                raw_name=sc.name,
                kind=canonicalize_strategy_kind(sc.name),
                samples=sc.samples,
                options=options,
            )
        )
    return plan_from_specs(specs)


def _build_context(
    *,
    config: DataConfigFile,
) -> tuple[DataGenerationContext, GenerationPlan]:
    """Assemble generation context and plan from a fully resolved DataConfigFile.

    Args:
        config: Loaded and resolved data config (output.data_dir must be set).

    Returns:
        (DataGenerationContext, GenerationPlan) pair.

    Raises:
        ValueError: If output.data_dir is None, matrix_path is missing, or
            normalize is a bare bool.
    """
    if config.output.data_dir is None:
        raise ValueError("output.data_dir must be resolved before process_config().")
    dataset_dir = config.output.data_dir / config.id

    matrix_path = config.source.matrix_path
    if not matrix_path:
        raise ValueError("Missing 'source.matrix_path' in config")

    normalize_value = config.generation.normalize
    if isinstance(normalize_value, bool):
        raise TypeError(
            f"Invalid normalize value in config: {normalize_value} (bool). "
            "The 'normalize' parameter expects one of 'matrix', 'rhs', or 'none'."
        )
    normalize = cast(NormalizeType, str(normalize_value))

    plan = _plan_from_generation_config(config.generation)

    return DataGenerationContext(
        matrix_path=matrix_path,
        rhs_path=config.source.rhs_path,
        solutions_path=config.source.solutions_path,
        solution_path=config.source.solution_path,
        sample_id_regex=config.source.sample_id_regex,
        enumerate_by=config.source.enumerate_by,
        include_indices=config.source.include_indices,
        exclude_indices=config.source.exclude_indices,
        dataset_dir=dataset_dir,
        normalize=normalize,
        seed=config.generation.seed,
        shuffle=config.generation.shuffle,
        replacement=config.generation.replacement,
        parameters_paths=config.source.parameters_paths,
        dataset_format=config.output.dataset_format,
    ), plan
