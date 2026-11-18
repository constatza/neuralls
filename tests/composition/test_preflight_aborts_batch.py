"""A presentation error aborts the whole comparison batch, through the real per-entry preparation.

Nothing here replaces `prepare_comparison_entry`: only checkpoint resolution
(`resolve_comparison_specs`), source validation and MLflow are stubbed, so the
tests fail if presentation validation ever moves under the per-entry handler that
turns a resolution `ValueError` into a failed-comparison outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomli_w
from dlkit.infrastructure.io.index import save_split_indices
from dlkit.infrastructure.types.split import IndexSplit

from neuralls.composition.assignments import comparison_prepare
from neuralls.composition.assignments.comparison_batch import (
    run_comparison_batch,
)
from neuralls.composition.assignments.comparison_prepare import (
    matrix_scope,
)
from neuralls.composition.assignments.comparison_resolution import (
    ComparisonResolutionContext,
    ResolvedComparisonSpecs,
)
from neuralls.composition.comparison.models import ComparisonParams
from neuralls.composition.preconditioners.descriptions import MissingArchitectureNameError
from neuralls.domain.generation.specs import DatasetSpec, SourceSpec
from neuralls.domain.presentation.methods import DuplicateMethodPresentationError
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.platform.config.case_path import CaseConfigPath
from neuralls.platform.config.models.checkpoint_refs import TrainedAssignmentRefConfig
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest import MissingDatasetProvenanceError
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.shared.constants import SPLIT_FILENAME
from neuralls.shared.scientific_description.models import ScientificDescription
from tests.dataset_support import build_test_dataset, strip_provenance
from tests.fakes.comparison_runs import RecordingTrackingServices
from tests.presentation_support import pod_spec
from tests.workflows.comparison_support import (
    make_settings,
    write_experiments_config,
)

pytestmark = pytest.mark.deferred_io("dataset-fixture-io")

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
        # A POD method fits the train fold, so its dataset needs a pinned split.
        save_split_indices(
            IndexSplit(train=(0,), validation=(), test=(), predict=None), target / SPLIT_FILENAME
        )
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

        monkeypatch.setattr(comparison_prepare, "resolve_comparison_specs", resolve)
        monkeypatch.setattr(comparison_prepare, "validate_comparison_sources", lambda cfg: None)
        return calls

    return install


def _run_expecting_abort(
    tmp_path: Path,
    settings: NeurallsSettings,
    tracking: RecordingTrackingServices,
    error: type[Exception],
) -> None:
    experiments_config = tmp_path / "experiments.toml"
    write_experiments_config(experiments_config, with_comparisons=True)
    with (
        patch(
            "neuralls.composition.assignments.comparison_prepare.resolve_comparison_config",
            return_value=MagicMock(),
        ),
        pytest.raises(error),
    ):
        run_comparison_batch(
            CaseConfigPath.resolve(experiments_config),
            ComparisonParams(),
            settings,
            tracking=tracking,
        )


@pytest.mark.usefixtures(
    "stub_input_gate", "stub_planning", "stub_matrix_scope", "stub_setup_tracking"
)
class TestPresentationFailureAbortsTheBatch:
    def test_a_neural_method_without_an_architecture(
        self,
        tmp_path: Path,
        neuralls_settings: NeurallsSettings,
        resolved_specs: ResolvedSpecs,
        fake_tracking: RecordingTrackingServices,
    ) -> None:
        calls = resolved_specs(
            [
                NeuralPreconditionerConfig(
                    assignment="a1", model_ref=TrainedAssignmentRefConfig(assignment_id="a1")
                )
            ]
        )

        _run_expecting_abort(
            tmp_path, neuralls_settings, fake_tracking, MissingArchitectureNameError
        )

        assert calls, "resolution ran, so the failure came from presentation"
        assert fake_tracking.runs.sessions == []

    def test_inline_pod_over_a_dataset_without_provenance_aborts(
        self,
        tmp_path: Path,
        neuralls_settings: NeurallsSettings,
        resolved_specs: ResolvedSpecs,
        fake_tracking: RecordingTrackingServices,
        stale_dataset_dir: Path,
    ) -> None:
        spec = AMGPreconditionerConfig(
            coarsening=PODCoarseningConfig(dataset_dir=stale_dataset_dir, rank=10)
        )
        calls = resolved_specs([spec])

        _run_expecting_abort(
            tmp_path, neuralls_settings, fake_tracking, MissingDatasetProvenanceError
        )

        assert calls
        assert fake_tracking.runs.sessions == []

    def test_two_pod_methods_over_identical_datasets(
        self,
        tmp_path: Path,
        neuralls_settings: NeurallsSettings,
        resolved_specs: ResolvedSpecs,
        fake_tracking: RecordingTrackingServices,
        twin_pod_dataset_dirs: tuple[Path, Path],
    ) -> None:
        first, second = twin_pod_dataset_dirs
        calls = resolved_specs([pod_spec(first, 10), pod_spec(second, 10)])

        _run_expecting_abort(
            tmp_path, neuralls_settings, fake_tracking, DuplicateMethodPresentationError
        )

        assert calls
        assert fake_tracking.runs.sessions == []


@pytest.fixture
def case_cfg_and_path(tmp_path: Path, atomic_source: SourceSpec) -> tuple[CaseConfig, Path]:
    """A case registry whose one dataset config points at `tmp_path/processed/ds`."""
    config_dir = tmp_path / "case"
    (config_dir / "datasets").mkdir(parents=True)
    dataset_config = {
        "id": DATASET_ID,
        "source": {"matrix_path": str(tmp_path / "unused.txt"), "problem": "Atomic test"},
        "generation": {
            "scaling": {"kind": "none"},
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
    context = ComparisonResolutionContext(
        topology=MagicMock(),
        case_config_path=case_path,
        settings=make_settings(tmp_path),
        load_dataset=load_dataset_presentation,
        tracking=MagicMock(),
        master_cfg=cfg,
    )
    return matrix_scope(_planned(), context)


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
