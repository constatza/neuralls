"""The complete spectral comparison states every scientific fact once, in every sink, with no id.

One cohort drives the whole check: Identity, Jacobi, IC(0), two AMGs, aSA with ``k=3`` and
``k=7``, BAMG, and POD-2G of ranks 10 and 100 fitted on Gaussian CG residuals and on
smoother-filtered probes, all over a ``Spectral 3D field`` dataset. Plot legends, MLflow run
names, tags and parameters, and group reports are produced by the real adapters and share
the cohort's structured presentations; the method ids stay machine-only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.presentation_preflight import prepare_comparison_presentation
from neuralls.composition.comparison.postprocessing.group_reporting import report_all_levels
from neuralls.composition.tracking.run_specs import build_benchmark_method_run_spec
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.presentation.methods import PreparedComparisonPresentation
from neuralls.domain.presentation.provenance import DatasetPresentation, DatasetSourceProvenance
from neuralls.domain.presentation.titles import ComparisonScopePresentation
from neuralls.domain.solver.models.group_outcomes import GroupAggregated
from neuralls.domain.solver.models.group_presentation import (
    GroupMethodPresentation,
    GroupPresentation,
)
from neuralls.platform.config.models.coarsening import AggregationCoarseningConfig
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.reporting.figure_text import method_legend
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.tracking.metrics_logging import log_configured_method_facts
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.shared.digest import canonical_digest
from neuralls.shared.scientific_description.catalog import REALIZED_LEVELS
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import NotationPolicy, RenderTarget
from neuralls.shared.types import DatasetId, PreconditionerName
from tests.dataset_support import build_test_dataset
from tests.presentation_support import pod_spec
from tests.realized_support import completed_workload

pytestmark = pytest.mark.deferred_io("mlflow-run-tracking")

PLAIN = NotationPolicy.default(RenderTarget.PLAIN)
MATHTEXT = NotationPolicy.default(RenderTarget.MATHTEXT)

SEED = 2024
MATRIX_SIZE = 4
MATRIX_COUNT = 3
SOURCE_PROBLEM = "Spectral 3D field"
SYSTEM_ID = "system-0"
FIT_SWEEPS = 3
"""Jacobi sweeps / CG iterations of the POD fit datasets."""
POD_RANKS = (10, 100)
FIT_DIR_NAMES = ("gaussian-cg-fit", "filtered-probe-fit")
"""Directory basenames of the POD fit datasets; they must never reach visible text."""
ASSIGNMENT_IDS = ("fit-gaussian", "fit-probes")
"""Training-assignment ids of the POD methods; they must never reach visible text."""
UNDERSCORE_BLOB = re.compile(r"[A-Za-z0-9]+_[A-Za-z0-9]+")
"""An identifier-shaped token: letters and digits joined by an underscore."""

EXPECTED_LEGEND = [
    "Identity",
    "Jacobi",
    "IC0 (τ=0)",
    "AMG (L=2, θ=0.25)",
    "AMG (L=2, θ=0.5)",
    "aSA (L=2, k=3, θ=0)",
    "aSA (L=2, k=7, θ=0)",
    "BAMG (L=10)",
]


@pytest.fixture
def matrix_glob(tmp_path: Path) -> str:
    """Seeded SPD matrices as individually globbable text files."""
    rng = np.random.default_rng(SEED)
    directory = tmp_path / "matrices"
    directory.mkdir()
    for index in range(MATRIX_COUNT):
        base = rng.standard_normal((MATRIX_SIZE, MATRIX_SIZE))
        np.savetxt(directory / f"A_{index:03d}.txt", base @ base.T + 5.0 * np.eye(MATRIX_SIZE))
    return str(directory / "A_*.txt")


def _generate(matrix_glob: str, target: Path, mixture: MixtureSpec) -> Path:
    """Generate one dataset of the spectral source with real structured provenance."""
    build_test_dataset(
        SourceSpec(matrix_path=matrix_glob),
        DatasetSpec(mixture=mixture, normalize="matrix", write_batch_size=4),
        str(target),
        source_provenance=DatasetSourceProvenance(ScientificDescription(SOURCE_PROBLEM)),
    )
    return target


def _trace_mixture(strategy: str) -> MixtureSpec:
    return MixtureSpec(
        counts={strategy: 8},
        seed=SEED,
        shuffle=False,
        strategy_overrides={strategy: {"stop": FIT_SWEEPS, "start": 1}},
    )


@pytest.fixture
def spectral_dataset(matrix_glob: str, tmp_path: Path) -> DatasetPresentation:
    """The matrix dataset the comparison runs on."""
    target = _generate(
        matrix_glob, tmp_path / "spectral", MixtureSpec(counts={"gaussian_forward": 8}, seed=SEED)
    )
    return load_dataset_presentation(target)


@pytest.fixture
def fit_dirs(matrix_glob: str, tmp_path: Path) -> tuple[Path, Path]:
    """POD fit datasets: Gaussian CG residuals, and smoother-filtered probes."""
    gaussian_cg, filtered_probes = (
        _generate(matrix_glob, tmp_path / name, _trace_mixture(strategy))
        for name, strategy in zip(
            FIT_DIR_NAMES, ("gaussian_residuals", "smoother_filtered_probes"), strict=True
        )
    )
    return gaussian_cg, filtered_probes


@pytest.fixture
def configs(fit_dirs: tuple[Path, Path]) -> tuple[PreconditionerConfig, ...]:
    return (
        StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        StandardPreconditionerConfig(type=PreconditionerType.JACOBI),
        IC0PreconditionerConfig(),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25)),
        AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.5)),
        AdaptiveSAPreconditionerConfig(num_candidates=3),
        AdaptiveSAPreconditionerConfig(num_candidates=7),
        BootstrapAMGPreconditionerConfig(),
        *(
            pod_spec(directory, rank, assignment=assignment)
            for directory, assignment in zip(fit_dirs, ASSIGNMENT_IDS, strict=True)
            for rank in POD_RANKS
        ),
    )


@pytest.fixture
def cohort(
    configs: tuple[PreconditionerConfig, ...], spectral_dataset: DatasetPresentation
) -> PreparedComparisonPresentation:
    return prepare_comparison_presentation(
        configs, ComparisonScopePresentation(spectral_dataset), load_dataset_presentation
    )


def _plain_label(cohort: PreparedComparisonPresentation) -> dict[str, str]:
    """The one-line plain legend text of every method, by method id."""
    legend = method_legend({m.method_id: m.compact for m in cohort.methods}, PLAIN)
    return {m.method_id: legend.label(m.method_id).replace("\n", " ") for m in cohort.methods}


def _forbidden(text: str, *, blobs: bool = True) -> list[str]:
    """Every identifier-like leak in visible ``text``: ids, slugs, paths, brackets, blobs.

    MathText legitimately contains ``nu_1``-style subscripts, so its check passes ``blobs=False``.
    """
    leaks = [
        fragment
        for fragment in (
            "assignment=",
            "config=",
            "method |",
            "[",
            "]",
            *FIT_DIR_NAMES,
            *ASSIGNMENT_IDS,
        )
        if fragment in text
    ]
    return [*leaks, *UNDERSCORE_BLOB.findall(text)] if blobs else leaks


def test_plain_legend_distinguishes_every_method_by_its_scientific_facts(
    cohort: PreparedComparisonPresentation,
) -> None:
    labels = list(_plain_label(cohort).values())

    assert labels[:8] == [text.replace("\n", " ") for text in EXPECTED_LEGEND]
    assert len(set(labels)) == len(labels) == 12
    assert "k=3" in labels[5]
    assert "k=7" in labels[6]
    assert labels[7] == "BAMG (L=10)"


def test_each_pod_label_has_its_rank_and_its_fit_dataset_described_once(
    cohort: PreparedComparisonPresentation,
) -> None:
    pod_labels = list(_plain_label(cohort).values())[8:]

    for label, (rank, fit) in zip(
        pod_labels,
        [(r, f) for f in ("Gaussian CG Residuals", "Jacobi-Filtered Probes") for r in POD_RANKS],
        strict=True,
    ):
        method, dataset = label.split(" | ")
        assert method == f"POD-2G (r={rank})"
        assert dataset.startswith(f"{fit} (")
        assert label.count(fit) == 1


def test_visible_text_has_no_ids_slugs_paths_brackets_or_underscore_blobs(
    cohort: PreparedComparisonPresentation,
) -> None:
    for method_id, label in _plain_label(cohort).items():
        assert _forbidden(label) == [], method_id
        assert method_id not in label


def test_mlflow_runs_share_the_legend_presentation_and_keep_ids_in_tags_and_facts_in_params(
    cohort: PreparedComparisonPresentation,
    mlflow_experiment: str,
    mlflow_tracking_uri: str,
) -> None:
    plain = _plain_label(cohort)
    for prepared in cohort.methods:
        run, tags = build_benchmark_method_run_spec(
            prepared=prepared,
            notation=PLAIN,
            config_digest=canonical_digest(prepared.method_id),
            system_ids=(SYSTEM_ID,),
        )
        with mlflow.start_run(run_name=run_name_text(run), tags=tags.as_mlflow_tags()):
            log_configured_method_facts(prepared)

    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(mlflow_experiment)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time ASC"])

    assert [run.info.run_name for run in runs] == list(plain.values())
    assert [run.data.tags["method_id"] for run in runs] == list(plain)
    for run in runs:
        visible = [run.info.run_name, run.data.tags["method_label"]]
        assert [leak for text in visible for leak in _forbidden(text)] == []
        assert run.data.params
        assert all(key.startswith(("method.", "training_dataset.")) for key in run.data.params)
        # Parameters are machine-readable: stable enum values such as `gaussian_residuals` are
        # intended; only ids, slugs, paths and brackets are leaks.
        assert [
            leak for value in run.data.params.values() for leak in _forbidden(value, blobs=False)
        ] == []


@dataclass(slots=True)
class _RecordingSink:
    presentations: list[GroupPresentation] = field(default_factory=list)

    def report(self, outcome: GroupAggregated, *, presentation: GroupPresentation) -> None:
        del outcome
        self.presentations.append(presentation)


def _group_methods(
    cohort: PreparedComparisonPresentation, depth_by_matrix: dict[int, int]
) -> Mapping[PreconditionerName, GroupMethodPresentation]:
    sink = _RecordingSink()
    report_all_levels(
        tuple(completed_workload(m, depth_by_matrix) for m in cohort.methods),
        sink,
        datasets={DatasetId("ds"): cohort.scope.dataset},
        notation=PLAIN,
    )
    return sink.presentations[0].methods


def test_group_states_a_realized_fact_only_when_every_matrix_agrees(
    cohort: PreparedComparisonPresentation,
) -> None:
    varying = _group_methods(cohort, {1: 2, 2: 3})
    agreeing = _group_methods(cohort, {1: 2, 2: 2})

    assert set(varying) == {PreconditionerName(m.method_id) for m in cohort.methods}
    for method in varying.values():
        assert method.invariant_facts == ()
        assert all(fact.key != REALIZED_LEVELS for fact in method.compact_facts)
    for method in agreeing.values():
        assert any(fact.key == REALIZED_LEVELS for fact in method.compact_facts)


def test_method_ids_stay_the_stable_machine_names_of_their_configs(
    cohort: PreparedComparisonPresentation, configs: tuple[PreconditionerConfig, ...]
) -> None:
    ids = [m.method_id for m in cohort.methods]

    assert ids == [spec.name for spec in configs]
    assert len(set(ids)) == 12
    assert all(method_id not in label for method_id, label in _plain_label(cohort).items())
