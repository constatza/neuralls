"""A dataset's provenance records the rows each strategy actually produced, and its seeds.

Every case is a real small build. The expected rows come from a second, independent run of
the batch stream (rows counted by the batch's strategy), never from the spec.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation.file_sources import MatrixReaders
from neuralls.domain.generation.orchestration import open_batch_stream
from neuralls.domain.generation.plan import canonicalize_strategy_kind
from neuralls.domain.generation.seeds import derive_seed, derive_strategy_seed
from neuralls.domain.generation.specs import DatasetSpec, MixtureSpec, SourceSpec
from neuralls.domain.presentation.provenance import DerivedSeed, ExplicitSeed
from neuralls.platform.storage.datasets import load_dense_training_arrays
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.platform.storage.matrix_readers import read_dense_npy, read_matrix
from neuralls.shared.types import GenerationStrategyKind
from tests.dataset_support import build_test_dataset

pytestmark = pytest.mark.deferred_io("generation-io")

_READERS = MatrixReaders(
    generic=partial(read_matrix, lazy=True), dense=partial(read_dense_npy, lazy=True)
)
_DIMENSION = 4
_ARCHIVE_FILES = 5
_WINDOW = {"stop": 3, "start": 1}
"""A trace window yielding several rows per base system (K > 1)."""
_BUDGET = 6


@dataclass(frozen=True, slots=True)
class Case:
    """One mixture to build: its counts and strategy options."""

    counts: Mapping[str, int]
    overrides: Mapping[str, Mapping[str, Any]]


@pytest.fixture
def archives(tmp_path: Path, spd_matrix_file: Path) -> Mapping[str, str]:
    """Solution and RHS archive globs whose RHS files satisfy ``A @ x = b``."""
    matrix = np.loadtxt(spd_matrix_file)
    solutions = tmp_path / "solutions"
    rhs = tmp_path / "rhs"
    solutions.mkdir()
    rhs.mkdir()
    generator = np.random.default_rng(3)
    for index in range(_ARCHIVE_FILES):
        solution = generator.standard_normal(_DIMENSION)
        np.savetxt(solutions / f"x_{index:02d}.txt", solution)
        np.savetxt(rhs / f"b_{index:02d}.txt", matrix @ solution)
    return {"solutions_glob": str(solutions / "x_*.txt"), "rhs_glob": str(rhs / "b_*.txt")}


def _generated(kind: GenerationStrategyKind, archives: Mapping[str, str]) -> Case:
    """The single-strategy case that exercises ``kind`` with its required options."""
    name = kind.value
    match kind:
        case GenerationStrategyKind.RESIDUALS:
            window = {**_WINDOW, "solutions_glob": archives["solutions_glob"]}
            return Case({name: _BUDGET}, {name: window})
        case (
            GenerationStrategyKind.GAUSSIAN_RESIDUALS
            | GenerationStrategyKind.SMOOTHER_FILTERED_PROBES
        ):
            return Case({name: _BUDGET}, {name: _WINDOW})
        case GenerationStrategyKind.SPARSE_RHS:
            return Case({name: 3}, {name: {"indices": [0], "values": [1.0]}})
        case GenerationStrategyKind.SOLUTION_ARCHIVE | GenerationStrategyKind.SCALED_SOLUTIONS:
            return Case({name: -1}, {name: {"solutions_glob": archives["solutions_glob"]}})
        case GenerationStrategyKind.RHS_ARCHIVE:
            return Case({name: -1}, {name: {"rhs_glob": archives["rhs_glob"]}})
        case GenerationStrategyKind.VALIDATED_ARCHIVE:
            return Case({name: 3}, {name: dict(archives)})
        case GenerationStrategyKind.KRYLOV:
            return Case({name: 3}, {name: {"krylov_iters": 2}})
        case _:
            return Case({name: 3}, {})


@pytest.fixture
def case_for(archives: Mapping[str, str]) -> Callable[[GenerationStrategyKind], Case]:
    """Factory of the single-strategy case for a kind."""
    return partial(_generated, archives=archives)


def _spec(case: Case, *, seed: int = 11) -> DatasetSpec:
    return DatasetSpec(
        mixture=MixtureSpec(
            counts=dict(case.counts), seed=seed, strategy_overrides=dict(case.overrides)
        ),
        normalize="none",
    )


def _produced_rows(source: SourceSpec, spec: DatasetSpec) -> Counter[str]:
    """Rows per strategy name from a separate run of the batch stream."""
    mixture = spec.mixture
    solver = make_solver()
    with_solvers = DatasetSpec(
        mixture=MixtureSpec(
            counts=mixture.counts,
            mix=mixture.mix,
            total=mixture.total,
            seed=mixture.seed,
            strategy_overrides=mixture.strategy_overrides,
            solver_overrides={"residuals": solver, "gaussian_residuals": solver},
        ),
        normalize=spec.normalize,
    )
    stream = open_batch_stream(source, with_solvers, batch_size=2, readers=_READERS)
    produced: Counter[str] = Counter()
    for batch in stream.batches:
        produced[batch.strategy_name] += len(batch)
    return produced


def _assert_recorded_rows_match_reality(
    source: SourceSpec, spec: DatasetSpec, dataset_dir: Path
) -> None:
    expected = {
        canonicalize_strategy_kind(name): rows
        for name, rows in _produced_rows(source, spec).items()
    }
    generation = load_dataset_presentation(dataset_dir).generation
    recorded = {strategy.kind: strategy.rows for strategy in generation.strategies}
    persisted = load_dense_training_arrays(str(dataset_dir))[0].shape[0]

    assert recorded == expected
    assert sum(recorded.values()) == generation.generated_rows == persisted


@pytest.mark.parametrize("kind", list(GenerationStrategyKind), ids=lambda kind: kind.value)
def test_every_strategy_records_the_rows_it_produced(
    kind: GenerationStrategyKind,
    case_for: Callable[[GenerationStrategyKind], Case],
    source_spec: SourceSpec,
    tmp_path: Path,
) -> None:
    """Kills: recomputing rows from the spec (x K again, dropped -1), dropping a strategy."""
    spec = _spec(case_for(kind))
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    _assert_recorded_rows_match_reality(source_spec, spec, Path(dataset_dir))


@pytest.mark.parametrize("name", ["residuals", "gaussian_residuals", "smoother_filtered_probes"])
def test_trace_budget_is_a_row_budget_not_a_base_system_count(
    name: str,
    case_for: Callable[[GenerationStrategyKind], Case],
    source_spec: SourceSpec,
    tmp_path: Path,
) -> None:
    """Kills: stamping ``budget * K``; the window yields K>1 rows per base system."""
    spec = _spec(case_for(GenerationStrategyKind(name)))
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    generation = load_dataset_presentation(dataset_dir).generation
    assert [strategy.rows for strategy in generation.strategies] == [_BUDGET]


def test_mixture_of_counts_records_each_strategy(
    case_for: Callable[[GenerationStrategyKind], Case], source_spec: SourceSpec, tmp_path: Path
) -> None:
    """Kills: dropping per-strategy rows for multi-strategy mixtures."""
    residuals = case_for(GenerationStrategyKind.RESIDUALS)
    forward = case_for(GenerationStrategyKind.GAUSSIAN_FORWARD)
    spec = _spec(Case({**residuals.counts, **forward.counts}, residuals.overrides))
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    _assert_recorded_rows_match_reality(source_spec, spec, Path(dataset_dir))


def test_mixture_of_mix_and_total_records_each_strategy(
    source_spec: SourceSpec, tmp_path: Path
) -> None:
    """Kills: resolving ``mix``/``total`` differently from generation."""
    spec = DatasetSpec(
        mixture=MixtureSpec(
            mix={"krylov": 1.0, "normal": 2.0},
            total=9,
            seed=5,
            strategy_overrides={"krylov": {"krylov_iters": 2}},
        ),
        normalize="none",
    )
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    _assert_recorded_rows_match_reality(source_spec, spec, Path(dataset_dir))
    generation = load_dataset_presentation(dataset_dir).generation
    assert generation.generated_rows == 9


@pytest.fixture
def received_seeds(monkeypatch: pytest.MonkeyPatch) -> dict[str, int | None]:
    """Seed each strategy's config carried when generation actually ran it."""
    from neuralls.domain.generation import strategy_rows

    seen: dict[str, int | None] = {}
    real = strategy_rows.run_generation

    def _spy(strategy_name: str, matrix: Any, *args: Any, **kwargs: Any) -> Any:
        seen[strategy_name] = kwargs["cfg"]["seed"]
        return real(strategy_name, matrix, *args, **kwargs)

    monkeypatch.setattr(strategy_rows, "run_generation", _spy)
    return seen


def test_explicit_strategy_seed_is_stamped_as_the_seed_the_strategy_received(
    source_spec: SourceSpec, tmp_path: Path, received_seeds: dict[str, int | None]
) -> None:
    """Kills: stamping a derived value in place of an explicit per-strategy seed."""
    spec = _spec(Case({"normal": 3}, {"normal": {"seed": 77}}))
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    strategy = load_dataset_presentation(dataset_dir).generation.strategies[0]
    assert strategy.seed == ExplicitSeed(77)
    assert received_seeds["normal"] == 77


def test_unseeded_strategy_stamps_the_root_the_received_seed_derives_from(
    source_spec: SourceSpec, tmp_path: Path, received_seeds: dict[str, int | None]
) -> None:
    """Kills: stamping ``derive_strategy_seed(root, name)``, a value no stream ever used."""
    root = 11
    spec = _spec(Case({"normal": 3}, {}), seed=root)
    dataset_dir = build_test_dataset(source_spec, spec, str(tmp_path / "ds"))

    generation = load_dataset_presentation(dataset_dir).generation
    assert generation.strategies[0].seed == DerivedSeed()
    assert generation.root_seed == root
    first_binding = derive_seed(generation.root_seed, "binding", 0)
    assert received_seeds["normal"] == derive_strategy_seed(first_binding, "normal")
    assert received_seeds["normal"] != derive_strategy_seed(root, "normal")
