"""build_dataset stamps structured scientific provenance, and stale datasets are refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuralls.composition.generation._context_builder import _build_context
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.platform.config.models.data_models import (
    DataConfigFile,
    GenerationConfig,
    OutputConfig,
    SourceConfig,
    StrategyConfig,
)
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError, manifest_path_for
from neuralls.platform.storage.manifest_io import (
    load_dataset_manifest,
    load_dataset_presentation,
)
from neuralls.shared.scientific_description import ScientificDescription
from neuralls.shared.scientific_description.presentation import DatasetSourceProvenance
from neuralls.shared.types import GenerationStrategyKind


@pytest.fixture
def problem() -> DatasetSourceProvenance:
    """A named source problem, as `[source] problem` would produce."""
    return DatasetSourceProvenance(ScientificDescription("Spectral 3D field"))


def test_build_records_source_and_every_strategy(
    tmp_path: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    problem: DatasetSourceProvenance,
) -> None:
    dataset_dir = build_dataset(
        source_spec, dataset_spec, str(tmp_path / "ds"), source_provenance=problem
    )

    presentation = load_dataset_presentation(dataset_dir)

    assert presentation.source == problem
    assert [s.kind for s in presentation.generation.strategies] == [
        GenerationStrategyKind.NEUTRAL_ONES
    ]
    assert presentation.generation.generated_rows == 1
    assert presentation.generation.normalization == "none"


def test_manifest_stores_keys_and_values_not_notation_or_a_version(
    tmp_path: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    problem: DatasetSourceProvenance,
) -> None:
    dataset_dir = build_dataset(
        source_spec, dataset_spec, str(tmp_path / "ds"), source_provenance=problem
    )

    raw = json.loads(manifest_path_for(dataset_dir).read_text(encoding="utf-8"))

    assert "schema" not in raw
    text = json.dumps(raw["provenance"])
    assert '"key": "sample_count"' in text
    assert "\\mathrm" not in text
    assert "Spectral 3D field" in text


def test_dataset_without_provenance_is_refused_with_a_regeneration_message(
    tmp_path: Path, source_spec: SourceSpec, dataset_spec: DatasetSpec
) -> None:
    dataset_dir = build_dataset(source_spec, dataset_spec, str(tmp_path / "ds"))

    with pytest.raises(MissingDatasetProvenanceError, match="Regenerate"):
        load_dataset_presentation(dataset_dir)


def test_tampered_provenance_is_rejected_on_load(
    tmp_path: Path,
    source_spec: SourceSpec,
    dataset_spec: DatasetSpec,
    problem: DatasetSourceProvenance,
) -> None:
    dataset_dir = build_dataset(
        source_spec, dataset_spec, str(tmp_path / "ds"), source_provenance=problem
    )
    path = manifest_path_for(dataset_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["provenance"]["generation"]["strategies"][0]["facts"][0]["key"] = "not_a_registered_fact"
    path.write_text(json.dumps(raw), encoding="utf-8")
    load_dataset_manifest.cache_clear()

    with pytest.raises(KeyError):
        load_dataset_presentation(dataset_dir)


def test_source_problem_name_must_be_a_plain_label() -> None:
    from pydantic import ValidationError

    from neuralls.platform.config.models.data_models import SourceConfig

    assert SourceConfig(matrix_path="m.mtx", problem="Spectral 3D field").problem
    with pytest.raises(ValidationError):
        SourceConfig(matrix_path="m.mtx", problem="spectral_3d_field")


def _config(problem: str | None, data_dir: Path) -> DataConfigFile:
    """A resolved config; the matrix path is only stored, never opened."""
    return DataConfigFile(
        id="ds",
        source=SourceConfig(matrix_path="unopened/*.mtx", problem=problem),
        generation=GenerationConfig(
            normalize="none", strategy=[StrategyConfig(name="neutral_ones", samples=1)]
        ),
        output=OutputConfig(data_dir=data_dir),
    )


def test_context_carries_the_configured_problem_as_provenance(tmp_path: Path) -> None:
    named, _ = _build_context(config=_config("Spectral 3D field", tmp_path))
    unnamed, _ = _build_context(config=_config(None, tmp_path))

    assert named.source_provenance == DatasetSourceProvenance(
        ScientificDescription("Spectral 3D field")
    )
    assert unnamed.source_provenance is None
