"""A presentation error aborts the whole comparison batch, through the real per-entry preparation.

Nothing here replaces `_prepare_comparison_entry`: only checkpoint resolution
(`_resolve_comparison_specs`), source validation and MLflow are stubbed, so the
tests fail if presentation validation ever moves under the per-entry handler that
turns a resolution `ValueError` into a failed-comparison outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w

from neuralls.composition.assignments import comparison_batch
from neuralls.composition.assignments.comparison_batch import (
    ResolvedComparisonSpecs,
    _ComparisonResolutionContext,
    _matrix_scope,
    run_comparison_batch,
)
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.composition.preconditioners.descriptions import MissingArchitectureNameError
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.presentation.methods import DuplicateMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.models.preconditioner import (
    NeuralPreconditionerConfig,
    PreconditionerConfig,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.shared.scientific_description.models import ScientificDescription
from tests.dataset_support import build_test_dataset, strip_provenance
from tests.presentation_support import pod_spec
from tests.workflows.comparison_support import (
    MLFLOW_MODULE,
    make_settings,
    write_experiments_config,
)

pytestmark = pytest.mark.deferred_io("generation-io")

type ResolvedSpecs = Callable[[list[PreconditionerConfig]], list[int]]

DATASET_ID = "ds"


@pytest.fixture
def twin_pod_dataset_dirs(
    tmp_path: Path, atomic_source: SourceSpec, atomic_spec: DatasetSpec
) -> tuple[Path, Path]:
    """Two separately generated datasets with identical provenance."""
    problem = DatasetSourceProvenance(ScientificDescription("Atomic test"))
    dirs = []
    for name in ("copy-a", "copy-b"):
        target = tmp_path / name
        build_test_dataset(atomic_source, atomic_spec, str(target), source_provenance=problem)
        dirs.append(target)
    return dirs[0], dirs[1]


@pytest.fixture
def resolved_specs(monkeypatch: pytest.MonkeyPatch) -> ResolvedSpecs:
    """Make checkpoint resolution return the given specs; the returned list counts calls."""

    def install(specs: list[PreconditionerConfig]) -> list[int]:
        calls: list[int] = []

        def resolve(*_args: object) -> ResolvedComparisonSpecs:
            calls.append(1)
            return ResolvedComparisonSpecs(specs=specs, warnings=(), checkpoint_digests={})

        monkeypatch.setattr(comparison_batch, "_resolve_comparison_specs", resolve)
        monkeypatch.setattr(comparison_batch, "_validate_comparison_sources", lambda cfg: None)
        return calls

    return install


def _run_expecting_abort(tmp_path: Path, error: type[Exception]) -> MagicMock:
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)
    with (
        patch(
            "neuralls.composition.assignments.comparison_batch.resolve_comparison_config",
            return_value=MagicMock(),
        ),
        patch(MLFLOW_MODULE) as mock_mlflow,
        pytest.raises(error),
    ):
        run_comparison_batch(experiments_config, ComparisonParams())
    return mock_mlflow


@pytest.mark.usefixtures("stub_input_gate", "stub_planning", "stub_matrix_scope")
class TestPresentationFailureAbortsTheBatch:
    def test_a_neural_method_without_an_architecture(
        self, tmp_path: Path, resolved_specs: ResolvedSpecs
    ) -> None:
        calls = resolved_specs(
            [
                NeuralPreconditionerConfig(
                    assignment="a1", model_ref=TrainedAssignmentRefConfig(assignment_id="a1")
                )
            ]
        )

        mock_mlflow = _run_expecting_abort(tmp_path, MissingArchitectureNameError)

        assert calls, "resolution ran, so the failure came from presentation"
        mock_mlflow.start_run.assert_not_called()

    def test_a_pod_method_over_a_dataset_without_provenance(
        self, tmp_path: Path, resolved_specs: ResolvedSpecs, stale_dataset_dir: Path
    ) -> None:
        calls = resolved_specs([pod_spec(stale_dataset_dir, 10)])

        mock_mlflow = _run_expecting_abort(tmp_path, MissingDatasetProvenanceError)

        assert calls
        mock_mlflow.start_run.assert_not_called()

    def test_two_pod_methods_with_distinct_assignments_over_identical_datasets(
        self,
        tmp_path: Path,
        resolved_specs: ResolvedSpecs,
        twin_pod_dataset_dirs: tuple[Path, Path],
    ) -> None:
        first, second = twin_pod_dataset_dirs
        calls = resolved_specs(
            [pod_spec(first, 10, assignment="fit-a"), pod_spec(second, 10, assignment="fit-b")]
        )

        mock_mlflow = _run_expecting_abort(tmp_path, DuplicateMethodPresentationError)

        assert calls
        mock_mlflow.start_run.assert_not_called()


@pytest.fixture
def case_cfg_and_path(tmp_path: Path, atomic_source: SourceSpec) -> tuple[CaseConfig, Path]:
    """A case registry whose one dataset config points at `tmp_path/processed/ds`."""
    config_dir = tmp_path / "case"
    (config_dir / "datasets").mkdir(parents=True)
    dataset_config = {
        "id": DATASET_ID,
        "source": {"matrix_path": str(tmp_path / "unused.txt"), "problem": "Atomic test"},
        "generation": {
            "normalize": "none",
            "seed": 1,
            "strategy": [{"name": "neutral_ones", "samples": 1}],
        },
        "output": {"data_dir": str(tmp_path / "processed")},
    }
    with (config_dir / "datasets" / f"{DATASET_ID}.toml").open("wb") as handle:
        tomli_w.dump(dataset_config, handle)
    cfg = CaseConfig.model_validate(
        {"datasets": [{"id": DATASET_ID, "path": f"datasets/{DATASET_ID}.toml"}]}
    )
    return cfg, config_dir / "case.toml"


def _planned() -> MagicMock:
    planned = MagicMock()
    planned.entry.matrix_dataset = DATASET_ID
    return planned


def _scope_of(tmp_path: Path, cfg: CaseConfig, case_path: Path) -> ComparisonScopePresentation:
    context = _ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=case_path,
        settings=make_settings(tmp_path),
        load_dataset=load_dataset_presentation,
        master_cfg=cfg,
    )
    return _matrix_scope(_planned(), context)


def test_the_matrix_scope_is_read_from_the_dataset_the_registry_resolves(
    tmp_path: Path,
    case_cfg_and_path: tuple[CaseConfig, Path],
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
) -> None:
    cfg, case_path = case_cfg_and_path
    dataset_dir = tmp_path / "processed" / DATASET_ID
    build_test_dataset(
        atomic_source,
        atomic_spec,
        str(dataset_dir),
        source_provenance=DatasetSourceProvenance(ScientificDescription("Atomic test")),
    )

    scope = _scope_of(tmp_path, cfg, case_path)

    assert scope.dataset == load_dataset_presentation(dataset_dir)


def test_a_matrix_dataset_without_provenance_aborts_scope_resolution(
    tmp_path: Path,
    case_cfg_and_path: tuple[CaseConfig, Path],
    atomic_source: SourceSpec,
    atomic_spec: DatasetSpec,
) -> None:
    cfg, case_path = case_cfg_and_path
    dataset_dir = tmp_path / "processed" / DATASET_ID
    build_test_dataset(atomic_source, atomic_spec, str(dataset_dir))
    strip_provenance(dataset_dir)

    with pytest.raises(MissingDatasetProvenanceError):
        _scope_of(tmp_path, cfg, case_path)
