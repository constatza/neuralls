"""Test fixtures for graph-cg tests."""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from types import ModuleType

import h5py
import matplotlib.figure
import numpy as np
import pytest
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    MethodPresentationCandidate,
    PreparedComparisonPresentation,
    PreparedMethodPresentation,
    StandaloneMethodPresentation,
    UnvalidatedComparisonPresentation,
)
from neuralls.domain.presentation.planner import prepare_presentations
from neuralls.domain.presentation.provenance import (
    DatasetGenerationProvenance,
    DatasetPresentation,
    DatasetSourceProvenance,
    DerivedSeed,
    StrategyProvenance,
)
from neuralls.domain.presentation.run_names import (
    ComparisonRunName,
    MethodRunName,
    NamedRun,
    SystemQualifier,
    TrackedRun,
    TrainingPresentation,
    Untimed,
)
from neuralls.domain.presentation.titles import ComparisonScopePresentation, EditorialPresentation
from neuralls.domain.solver.models.figure_scope import AllDatasetsScope
from neuralls.platform.config.context import ConfigContext
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.reporting.figure_text import Legend, method_legend
from neuralls.platform.reporting.outcome_plots import GroupStyle
from neuralls.platform.storage.dataset_readers import resolve_dataset_artifacts
from neuralls.shared.enum_codecs import encode_row_kind_array
from neuralls.shared.scientific_description.catalog import ITERATION_COUNT, make_fact
from neuralls.shared.scientific_description.models import (
    ScientificDescription,
)
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import DatasetId, GenerationStrategyKind, PreconditionerName, RowKind

REPO_ROOT = Path(__file__).resolve().parent.parent


def _patch_default_paths_for_loaded_modules(
    path_values: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch already-imported modules that cache default path constants."""
    str_values = {
        "DEFAULT_PROCESSED_DIR": str(path_values["DEFAULT_PROCESSED_DATA_DIR"]),
        "DEFAULT_RESULTS_DIR": str(path_values["DEFAULT_OUTPUT_DIR"]),
        "DEFAULT_FIGURES_DIR_STR": str(path_values["DEFAULT_FIGURES_DIR"]),
    }
    all_values: dict[str, object] = {**path_values, **str_values}
    all_values["DEFAULT_CHECKPOINTS_DIR"] = path_values["DEFAULT_OUTPUT_DIR"] / "checkpoints"
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        for name, value in all_values.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value, raising=False)


def _patch_dlkit_environment(
    runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch DLKit's imported environment singleton to the per-test root."""
    from dlkit.infrastructure.config.environment import env as dlkit_env

    monkeypatch.setattr(dlkit_env, "internal_dir", str(runtime_root / ".dlkit"), raising=False)


@pytest.fixture
def runtime_root(tmp_path: Path) -> Path:
    """Per-test runtime root for all generated artifacts."""
    return (tmp_path / "runtime").resolve()


@pytest.fixture
def output_root(runtime_root: Path) -> Path:
    """Per-test output root."""
    return runtime_root / "output"


@pytest.fixture
def processed_root(runtime_root: Path) -> Path:
    """Per-test processed data root."""
    return runtime_root / "processed"


@pytest.fixture
def figures_root(runtime_root: Path) -> Path:
    """Per-test figures root."""
    return runtime_root / "figures"


@pytest.fixture
def mlflow_tracking_dir(runtime_root: Path) -> Path:
    """Per-test MLflow tracking root."""
    return runtime_root / "mlruns"


@pytest.fixture
def mlflow_artifact_dir(runtime_root: Path) -> Path:
    """Per-test MLflow artifact root."""
    return runtime_root / "mlartifacts"


@pytest.fixture(autouse=True)
def isolate_default_paths_with_tmp_path(
    runtime_root: Path,
    output_root: Path,
    processed_root: Path,
    figures_root: Path,
    mlflow_tracking_dir: Path,
    mlflow_artifact_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirect ambient defaults without creating files for pure tests."""
    case_config_path = runtime_root / "case.toml"
    env_map = {
        "NEURALLS_PROCESSED_DIR": str(processed_root),
        "NEURALLS_OUTPUT_DIR": str(output_root),
        "NEURALLS_CASE_CONFIG": str(case_config_path),
        # Force DLKit test artifacts away from repository-level tests/artifacts/.
        "DLKIT_TEST_MODE": "1",
        "DLKIT_TEST_ARTIFACT_ROOT": str(runtime_root / "dlkit_artifacts"),
        # Keep DLKit internals local to the test temp root.
        "DLKIT_ROOT_DIR": str(runtime_root),
        "DLKIT_INTERNAL_DIR": str(runtime_root / ".dlkit"),
    }
    monkeypatch.chdir(runtime_root.parent)
    for key, value in env_map.items():
        monkeypatch.setenv(key, value)

    from neuralls.shared import constants as neuralls_constants

    path_values = {
        "DEFAULT_OUTPUT_DIR": Path(env_map["NEURALLS_OUTPUT_DIR"]).resolve(),
        "DEFAULT_PROCESSED_DATA_DIR": Path(env_map["NEURALLS_PROCESSED_DIR"]).resolve(),
        "DEFAULT_FIGURES_DIR": figures_root.resolve(),
        "DEFAULT_MLRUNS_DIR": mlflow_tracking_dir.resolve(),
        "DEFAULT_MLARTIFACTS_DIR": mlflow_artifact_dir.resolve(),
    }
    for name, value in path_values.items():
        monkeypatch.setattr(neuralls_constants, name, value, raising=False)

    _patch_default_paths_for_loaded_modules(path_values, monkeypatch)
    _patch_dlkit_environment(runtime_root, monkeypatch)


@pytest.fixture(autouse=True)
def redirect_tempfile_under_pytest_tmp(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send library `tempfile` use (batch work roots, staging dirs) under pytest's managed tmp."""
    monkeypatch.setattr(
        tempfile, "tempdir", str(tmp_path_factory.mktemp("tempfile")), raising=False
    )


@pytest.fixture(autouse=True)
def skip_figure_encoding(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Figure.savefig` create an empty file instead of encoding a PNG.

    Figure building and all artist state stay real, so tests that inspect a figure or
    assert that a file was produced are unaffected; only the costly raster encoding is
    skipped. Tests whose subject is the encoded image opt out with `@pytest.mark.real_render`.
    """
    if request.node.get_closest_marker("real_render") is not None:
        return
    encode = matplotlib.figure.Figure.savefig

    def touch(
        self: matplotlib.figure.Figure, fname: object, *args: object, **kwargs: object
    ) -> None:
        if isinstance(fname, str | Path):
            Path(fname).touch()
            return
        encode(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", touch)


# Real local MLflow store: migrate the schema once per session, copy it per test.
@pytest.fixture(scope="session")
def mlflow_db_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create the migrated MLflow schema once for isolated per-test copies."""
    template_dir = tmp_path_factory.mktemp("mlflow-template")
    template_path = template_dir / "mlflow.db"
    client = MlflowClient(tracking_uri=f"sqlite:///{template_path.as_posix()}")
    client.search_experiments()
    return template_path


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path, mlflow_db_template: Path) -> str:
    """Local isolated MLflow store copied from a pre-migrated template.

    Args:
        tmp_path: Pytest-provided isolated temp directory.
        mlflow_db_template: Session-scoped database containing only the schema.

    Returns:
        A ``sqlite:///...`` tracking URI string.
    """
    database_path = tmp_path / "mlflow.db"
    copy2(mlflow_db_template, database_path)
    return f"sqlite:///{database_path.as_posix()}"


@pytest.fixture
def neuralls_settings(
    runtime_root: Path, processed_root: Path, output_root: Path
) -> NeurallsSettings:
    """Resolved settings fixture backed by per-test temporary roots."""
    raw = runtime_root / "raw"
    for directory in (raw, processed_root, output_root):
        directory.mkdir(parents=True, exist_ok=True)
    (runtime_root / "case.toml").touch()
    return NeurallsSettings(
        _env_file=[],
        processed_dir=processed_root,
        output_dir=output_root,
    )


@pytest.fixture
def config_context(neuralls_settings: NeurallsSettings, tmp_path: Path) -> ConfigContext:
    """Config context fixture anchored to a temporary config path."""
    config_path = tmp_path / "config.toml"
    config_path.touch()
    return ConfigContext(config_path=config_path, settings=neuralls_settings)


@pytest.fixture
def minimal_data_config_toml(
    tmp_path: Path,
    runtime_root: Path,
) -> Path:
    """Minimal valid dataset config for loader and processing tests."""
    config_path = tmp_path / "test.toml"
    processed = runtime_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    matrix_path = processed / "matrix.mtx"
    matrix_path.touch()
    config_path.write_text(
        """
id = "test-dataset"

[source]
matrix_path = "${NEURALLS_PROCESSED_DIR}/matrix.mtx"

[generation]

[output]
""".strip()
    )
    return config_path


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure loguru to be safe against closed streams during cleanup."""

    def safe_sink(message):
        try:
            sys.stderr.write(message)
        except ValueError:
            pass  # Ignore "I/O operation on closed file"

    logger.remove()
    logger.add(safe_sink)


@pytest.fixture
def small_spd_matrix() -> np.ndarray:
    """Small 2x2 symmetric positive definite test matrix.

    Returns:
        2x2 SPD matrix with known properties
    """
    return np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)


@pytest.fixture
def small_rhs() -> np.ndarray:
    """Small 2D RHS vector for testing.

    Returns:
        2D RHS vector
    """
    return np.array([1.0, 0.0], dtype=np.float64)


@pytest.fixture
def archive_solutions() -> np.ndarray:
    """Pre-computed archive solutions for testing.

    Returns:
        Array of 3 solution vectors (2D each)
    """
    return np.array(
        [[0.5, 0.3], [0.2, 0.8], [0.1, 0.4]],
        dtype=np.float64,
    )


@pytest.fixture
def archive_rhs(small_spd_matrix: np.ndarray, archive_solutions: np.ndarray) -> np.ndarray:
    """Pre-computed RHS vectors from archive solutions.

    Args:
        small_spd_matrix: Test matrix A
        archive_solutions: Archive solution vectors

    Returns:
        Array of RHS vectors b = A @ x
    """
    return np.array(
        [small_spd_matrix @ x for x in archive_solutions],
        dtype=np.float64,
    )


@pytest.fixture
def test_seed() -> int:
    """Deterministic seed for reproducible tests.

    Returns:
        Random seed value
    """
    return 42


@pytest.fixture
def warning_messages() -> Iterator[list[str]]:
    """Collect loguru WARNING-level messages emitted during a test."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    yield messages
    logger.remove(sink_id)


ARRAY_STORE_SEED: int = 7
ARRAY_STORE_TOTAL_ROWS: int = 6
ARRAY_STORE_FIRST_BATCH_ROWS: int = 2
ARRAY_STORE_N: int = 3
ARRAY_STORE_M: int = 4


@dataclass(frozen=True)
class DenseRowsFixture:
    """Seeded dense rows split into two offset batches, shaped like a generated dataset."""

    total_rows: int
    first_batch_rows: int
    matrices: np.ndarray
    vectors: np.ndarray


@pytest.fixture
def dense_rows_fixture() -> DenseRowsFixture:
    rng = np.random.default_rng(ARRAY_STORE_SEED)
    return DenseRowsFixture(
        total_rows=ARRAY_STORE_TOTAL_ROWS,
        first_batch_rows=ARRAY_STORE_FIRST_BATCH_ROWS,
        matrices=rng.standard_normal((ARRAY_STORE_TOTAL_ROWS, ARRAY_STORE_N, ARRAY_STORE_M)),
        vectors=rng.standard_normal((ARRAY_STORE_TOTAL_ROWS, ARRAY_STORE_N)),
    )


@pytest.fixture
def zarr_store_root(tmp_path: Path) -> Path:
    return tmp_path / "store.zarr"


@pytest.fixture
def hdf5_store_path(tmp_path: Path) -> Path:
    return tmp_path / "store.h5"


@pytest.fixture
def overwrite_row_kind() -> Callable[[Path, list[RowKind]], None]:
    """Replace a hdf5 dataset's stored row-kind array in place, through the artifact's key."""

    def _overwrite(dataset_dir: Path, kinds: list[RowKind]) -> None:
        artifact = resolve_dataset_artifacts(dataset_dir).row_kind
        assert artifact is not None and artifact.key is not None
        with h5py.File(str(artifact.path), "r+") as f:
            f[artifact.key][...] = encode_row_kind_array(kinds)

    return _overwrite


@pytest.fixture
def stub_input_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the generated-and-pinned-inputs gate for tests that hand-build their datasets.

    The gate has its own tests; end-to-end comparison tests that write a tiny
    dataset directly (no generation stage) would otherwise be refused as stale.
    """
    monkeypatch.setattr(
        "neuralls.composition.assignments.comparison_batch.require_current_inputs",
        lambda *_args, **_kwargs: None,
    )


def _build_diag_matrix_dataset(
    root: Path, matrix_ids: tuple[int, ...], rows: int, dimension: int = 2
) -> Path:
    """Build an hdf5 dataset of `rows` rows over one diagonal SPD matrix per id in `matrix_ids`."""
    from scipy.io import mmwrite
    from scipy.sparse import csr_array

    from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
    from tests.dataset_support import build_test_dataset

    raw = root / "raw"
    for matrix_id in matrix_ids:
        case_dir = raw / str(matrix_id)
        case_dir.mkdir(parents=True)
        mmwrite(
            str(case_dir / "K_ff.mtx"),
            csr_array(np.diag([float(matrix_id), *([1.0] * (dimension - 1))])),
        )
    dataset_dir = root / "dataset"
    build_test_dataset(
        SourceSpec(matrix_path=str(raw / "*" / "K_ff.mtx")),
        DatasetSpec(
            mixture=MixtureSpec(counts={"neutral_ones": rows}, seed=42, shuffle=False),
            normalize="none",
        ),
        str(dataset_dir),
        dataset_format="hdf5",
    )
    return dataset_dir


@pytest.fixture
def sparse_id_dataset(tmp_path: Path) -> Path:
    """Two matrices whose sample ids (5, 9) differ from every physical row position.

    The multi-matrix layout stores one matrix copy per row, so rows 0-1 hold
    matrix id 5 and rows 2-3 hold matrix id 9; neither id is a valid row.
    """
    return _build_diag_matrix_dataset(tmp_path / "multi", (5, 9), rows=4)


@pytest.fixture
def single_matrix_dataset(tmp_path: Path) -> Path:
    """One matrix (id 7) with four right-hand-side rows."""
    return _build_diag_matrix_dataset(tmp_path / "single", (7,), rows=4)


@pytest.fixture
def pin_test_fold() -> Callable[[Path, tuple[int, ...]], None]:
    """Write a `split.json` holding out the given rows, as generation would."""
    from dlkit.infrastructure.io.index import save_split_indices
    from dlkit.infrastructure.types.split import IndexSplit

    def _pin(dataset_dir: Path, test_rows: tuple[int, ...]) -> None:
        save_split_indices(
            IndexSplit(train=(), validation=(), test=test_rows, predict=None),
            dataset_dir / "split.json",
        )

    return _pin


@pytest.fixture
def larger_single_matrix_dataset(tmp_path: Path) -> Path:
    """A single-matrix dataset of 3x3 systems, to pair with the 2x2 fixtures."""
    return _build_diag_matrix_dataset(tmp_path / "larger", (3,), rows=4, dimension=3)


@pytest.fixture
def stub_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan every `[[comparisons]]` entry onto a shared matrix without reading any dataset.

    For batch tests that drive orchestration with mocked configs and have no
    generated dataset for the planner to classify; planning has its own tests.
    """
    from neuralls.composition.comparison.models import MatrixRef, PlannedSystem
    from neuralls.domain.solver.models.provenance import SharedSingleMatrix
    from neuralls.platform.config.models.experiments import CaseConfig
    from neuralls.shared.types import MatrixId

    def _plan(case_cfg: CaseConfig, *_args: object) -> tuple[PlannedSystem, ...]:
        return tuple(
            PlannedSystem(
                entry=entry,
                matrix=MatrixRef(matrix_id=MatrixId(0), row=0),
                rhs_row=None,
                rhs_ordinal=0,
                provenance=SharedSingleMatrix(raw_matrix_index=0),
            )
            for entry in case_cfg.comparisons
        )

    monkeypatch.setattr("neuralls.composition.assignments.comparison_batch.plan_comparisons", _plan)


@pytest.fixture
def dataset_presentation() -> DatasetPresentation:
    """A small dataset presentation for tests that need one but not its content."""
    return DatasetPresentation(
        source=DatasetSourceProvenance(ScientificDescription("Test problem")),
        generation=DatasetGenerationProvenance(
            strategies=(
                StrategyProvenance(
                    GenerationStrategyKind.GAUSSIAN_RESIDUALS,
                    rows=100,
                    parameters=(make_fact(ITERATION_COUNT, 20),),
                    seed=DerivedSeed(),
                ),
            ),
            generated_rows=100,
            normalization="matrix",
            root_seed=42,
        ),
    )


@pytest.fixture
def prepared_presentation(
    dataset_presentation: DatasetPresentation,
) -> PreparedComparisonPresentation:
    """A validated one-method cohort for tests that build prepared executions directly."""
    method = ScientificDescription("Jacobi")
    return prepare_presentations(
        UnvalidatedComparisonPresentation(
            methods=(
                MethodPresentationCandidate(
                    PreconditionerName("jacobi"), StandaloneMethodPresentation(method)
                ),
            ),
            scope=ComparisonScopePresentation(dataset_presentation),
        )
    )


@pytest.fixture
def prepared_method(
    dataset_presentation: DatasetPresentation,
) -> Callable[[str], PreparedMethodPresentation]:
    """Build the validated presentation of a single standalone method by its id."""

    def build(method_id: str) -> PreparedMethodPresentation:
        cohort = prepare_presentations(
            UnvalidatedComparisonPresentation(
                methods=(
                    MethodPresentationCandidate(
                        PreconditionerName(method_id),
                        StandaloneMethodPresentation(ScientificDescription("Jacobi")),
                    ),
                ),
                scope=ComparisonScopePresentation(dataset_presentation),
            )
        )
        return cohort.methods[0]

    return build


@pytest.fixture
def stub_matrix_scope(
    monkeypatch: pytest.MonkeyPatch, dataset_presentation: DatasetPresentation
) -> ComparisonScopePresentation:
    """Resolve every comparison's matrix dataset to a fixed presentation.

    The batch reads a dataset manifest through the case registry; tests that mock
    the registry and the datasets use this to supply the presentation instead.
    """
    scope = ComparisonScopePresentation(dataset_presentation)
    monkeypatch.setattr(
        "neuralls.composition.assignments.comparison_batch._matrix_scope",
        lambda planned, context: scope,
    )
    return scope


@pytest.fixture
def notation() -> NotationPolicy:
    """The default MathText notation plots render with."""
    return NotationPolicy.default(RenderTarget.MATHTEXT)


@pytest.fixture
def legend_of(notation: NotationPolicy) -> Callable[..., Legend]:
    """Build a legend naming each given series ``Method <name>`` (distinct, standalone)."""

    def build(*names: str) -> Legend:
        return method_legend(
            {
                name: StandaloneMethodPresentation(
                    ScientificDescription(f"Method {name.replace('_', ' ')}")
                )
                for name in names
            },
            notation,
        )

    return build


@pytest.fixture
def prepared_cohort(
    dataset_presentation: DatasetPresentation,
) -> Callable[..., PreparedComparisonPresentation]:
    """Build a validated cohort of standalone methods, one per id, each named ``Method <id>``."""

    def build(*method_ids: str) -> PreparedComparisonPresentation:
        return prepare_presentations(
            UnvalidatedComparisonPresentation(
                methods=tuple(
                    MethodPresentationCandidate(
                        PreconditionerName(method_id),
                        StandaloneMethodPresentation(
                            ScientificDescription(f"Method {method_id.replace('_', ' ')}")
                        ),
                    )
                    for method_id in method_ids
                ),
                scope=ComparisonScopePresentation(dataset_presentation),
            )
        )

    return build


@pytest.fixture
def group_datasets(
    dataset_presentation: DatasetPresentation,
) -> dict[DatasetId, DatasetPresentation]:
    """The dataset presentations a report batch over dataset ``ds`` supplies."""
    return {DatasetId("ds"): dataset_presentation}


@pytest.fixture
def group_style_of(
    notation: NotationPolicy, legend_of: Callable[..., Legend]
) -> Callable[..., GroupStyle]:
    """Build the style of a group figure whose series are the given method names."""

    def build(*names: str, title: str = "group") -> GroupStyle:
        return GroupStyle(
            title=title,
            scope=AllDatasetsScope(),
            notation=notation,
            legend=legend_of(*names),
            families={},
            color_keys={},
            marker_keys={},
        )

    return build


@pytest.fixture
def training_presentation(dataset_presentation: DatasetPresentation) -> TrainingPresentation:
    """What a training run trains, for tests that prepare or finalize a training run."""
    return TrainingPresentation(
        DatasetQualifiedMethodPresentation(ScientificDescription("POD-2G"), dataset_presentation),
        NotationPolicy.default(),
    )


@pytest.fixture
def comparison_run() -> NamedRun:
    """The typed name of a comparison run, for outcomes that only need some run."""
    return NamedRun(
        ComparisonRunName(EditorialPresentation("Test comparison"), SystemQualifier(()), Untimed()),
        NotationPolicy.default(),
    )


@pytest.fixture
def training_presentation_of(
    dataset_presentation: DatasetPresentation,
) -> Callable[[str], TrainingPresentation]:
    """Build the training presentation of a model with the given method name."""

    def build(method_name: str) -> TrainingPresentation:
        return TrainingPresentation(
            DatasetQualifiedMethodPresentation(
                ScientificDescription(method_name), dataset_presentation
            ),
            NotationPolicy.default(),
        )

    return build


@pytest.fixture
def child_runs_of() -> Callable[..., dict[str, TrackedRun]]:
    """Build the nested per-method runs of a comparison result, one standalone method each."""

    def build(*result_keys: str) -> dict[str, TrackedRun]:
        return {
            key: TrackedRun(
                NamedRun(
                    MethodRunName(StandaloneMethodPresentation(ScientificDescription("Jacobi"))),
                    NotationPolicy.default(),
                ),
                {},
            )
            for key in result_keys
        }

    return build
