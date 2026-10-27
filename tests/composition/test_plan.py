"""Planning turns case entries into concrete systems bound to matrices, rows and seeds."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from neuralls.composition.comparison.models import MatrixRef
from neuralls.composition.comparison.plan import _system_seed, derive_seed, plan_systems
from neuralls.domain.solver.models.provenance import HeldOutMatrix, SharedSingleMatrix
from neuralls.platform.config.models.comparison import (
    DatasetRhsSourceModel,
    RandomMask,
    RhsSourceModelBase,
)
from neuralls.platform.config.models.experiments import CaseConfig
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.types import MatrixId

type PinTestFold = Callable[[Path, tuple[int, ...]], None]

_GAUSSIAN = {"kind": "gaussian"}
_FIRST_ONE = {"kind": "first_n", "n": 1}


def _case(
    *,
    comparisons: list[dict[str, object]] | None = None,
    sweeps: list[dict[str, object]] | None = None,
    datasets: tuple[str, ...] = ("ds",),
) -> CaseConfig:
    return CaseConfig.model_validate(
        {
            "datasets": [{"id": name, "path": f"{name}.toml"} for name in datasets],
            "comparisons": comparisons or [],
            "comparison_sweeps": sweeps or [],
        }
    )


def _comparison(entry_id: str = "c", **fields: object) -> dict[str, object]:
    return {"id": entry_id, "matrix_dataset": "ds", "rhs_source": _GAUSSIAN, **fields}


def _sweep(prefix: str = "sw", **fields: object) -> dict[str, object]:
    return {
        "id_prefix": prefix,
        "matrix_dataset": "ds",
        "rhs_source": _GAUSSIAN,
        "selector": _FIRST_ONE,
        **fields,
    }


class TestDeriveSeed:
    def test_is_deterministic(self) -> None:
        assert derive_seed(42, MatrixId(5), 0) == derive_seed(42, MatrixId(5), 0)

    def test_differs_across_matrices_ordinals_and_bases(self) -> None:
        seeds = {
            derive_seed(42, MatrixId(5), 0),
            derive_seed(42, MatrixId(9), 0),
            derive_seed(42, MatrixId(5), 1),
            derive_seed(43, MatrixId(5), 0),
        }

        assert len(seeds) == 4

    def test_missing_base_seed_counts_as_zero(self) -> None:
        assert derive_seed(None, MatrixId(5), 2) == derive_seed(0, MatrixId(5), 2)


class TestSingleMatrixComparisons:
    def test_one_system_keeps_the_entry_id_and_shares_the_matrix(
        self, single_matrix_dataset: Path
    ) -> None:
        (system,) = plan_systems(_case(comparisons=[_comparison()]), {"ds": single_matrix_dataset})

        assert system.entry.id == "c"
        assert system.matrix == MatrixRef(matrix_id=MatrixId(7), row=0)
        assert system.rhs_row is None
        assert system.provenance == SharedSingleMatrix(raw_matrix_index=0)

    def test_rhs_count_expands_into_independently_seeded_systems(
        self, single_matrix_dataset: Path
    ) -> None:
        case = _case(comparisons=[_comparison(rhs_count=3, seed=42)])

        systems = plan_systems(case, {"ds": single_matrix_dataset})

        assert [s.entry.id for s in systems] == ["c-0", "c-1", "c-2"]
        assert [s.entry.seed for s in systems] == [
            derive_seed(42, MatrixId(7), ordinal) for ordinal in range(3)
        ]
        assert len({s.entry.seed for s in systems}) == 3

    def test_dataset_kind_takes_the_first_held_out_rows_on_the_one_matrix(
        self, single_matrix_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(single_matrix_dataset, (1, 3))
        case = _case(
            comparisons=[_comparison(rhs_source={"kind": "dataset"}, rhs_count=2)],
        )

        systems = plan_systems(case, {"ds": single_matrix_dataset})

        assert [s.rhs_row for s in systems] == [1, 3]
        assert {s.matrix for s in systems} == {MatrixRef(matrix_id=MatrixId(7), row=0)}

    def test_dataset_kind_rejects_more_rows_than_the_fold_holds(
        self, single_matrix_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(single_matrix_dataset, (1, 3))
        case = _case(comparisons=[_comparison(rhs_source={"kind": "dataset"}, rhs_count=3)])

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": single_matrix_dataset})

    def test_a_multi_matrix_dataset_is_rejected_naming_the_dataset_and_rule(
        self, sparse_id_dataset: Path
    ) -> None:
        with pytest.raises(ValueError):
            plan_systems(_case(comparisons=[_comparison()]), {"ds": sparse_id_dataset})


class TestTraitDrivenPlanning:
    def test_a_fixed_vector_is_one_unseeded_draw(self, single_matrix_dataset: Path) -> None:
        raw = {"kind": "raw_rhs", "path": "rhs.npy", "row_kind": "standard"}

        (system,) = plan_systems(
            _case(comparisons=[_comparison(rhs_source=raw, seed=11)]),
            {"ds": single_matrix_dataset},
        )
        assert system.entry.seed == 11

        with pytest.raises(ValueError):
            plan_systems(
                _case(comparisons=[_comparison(rhs_source=raw, rhs_count=2)]),
                {"ds": single_matrix_dataset},
            )

    def test_a_masked_unseeded_source_still_gets_a_seed_per_draw(self) -> None:
        masked = DatasetRhsSourceModel(mask=RandomMask())

        assert _system_seed(masked, 5, MatrixId(1), 0) != _system_seed(masked, 5, MatrixId(1), 1)
        assert _system_seed(DatasetRhsSourceModel(), 5, MatrixId(1), 1) == 5

    def test_seeded_sources_get_a_derived_seed_per_draw(self, single_matrix_dataset: Path) -> None:
        case = _case(comparisons=[_comparison(rhs_count=2, seed=11)])

        first, second = plan_systems(case, {"ds": single_matrix_dataset})

        assert first.entry.seed == derive_seed(11, MatrixId(7), 0)
        assert second.entry.seed == derive_seed(11, MatrixId(7), 1)

    def test_every_planned_system_carries_a_provenance(
        self, single_matrix_dataset: Path, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        single = plan_systems(_case(comparisons=[_comparison()]), {"ds": single_matrix_dataset})
        swept = plan_systems(_case(sweeps=[_sweep()]), {"ds": sparse_id_dataset})

        assert all(
            isinstance(system.provenance, SharedSingleMatrix | HeldOutMatrix)
            for system in (*single, *swept)
        )

    def test_a_new_source_declaring_traits_is_planned_without_a_kind_list_edit(self) -> None:
        class ToySeededSource(RhsSourceModelBase):
            generated: ClassVar[bool] = True
            seeded: ClassVar[bool] = True
            single_draw: ClassVar[bool] = False

        class ToyFixedSource(ToySeededSource):
            seeded: ClassVar[bool] = False

        assert _system_seed(ToySeededSource(), 5, MatrixId(1), 2) == derive_seed(5, MatrixId(1), 2)
        assert _system_seed(ToyFixedSource(), 5, MatrixId(1), 2) == 5


class TestFixedSources:
    def test_a_fixed_vector_cannot_be_drawn_repeatedly(self, single_matrix_dataset: Path) -> None:
        raw = {"kind": "raw_rhs", "path": "rhs.npy", "row_kind": "standard"}
        case = _case(comparisons=[_comparison(rhs_source=raw, rhs_count=2)])

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": single_matrix_dataset})

    def test_a_fixed_vector_keeps_the_entry_seed(self, single_matrix_dataset: Path) -> None:
        raw = {"kind": "raw_rhs", "path": "rhs.npy", "row_kind": "standard"}
        case = _case(comparisons=[_comparison(rhs_source=raw, seed=11)])

        (system,) = plan_systems(case, {"ds": single_matrix_dataset})

        assert system.entry.seed == 11


class TestSweeps:
    def test_first_n_selects_held_out_matrices_by_id_and_binds_their_first_row(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))

        (system,) = plan_systems(_case(sweeps=[_sweep()]), {"ds": sparse_id_dataset})

        assert system.entry.id == "sw-5"
        assert system.matrix == MatrixRef(matrix_id=MatrixId(5), row=0)
        assert system.provenance == HeldOutMatrix(matrix_id=MatrixId(5), raw_matrix_index=0)

    def test_only_held_out_matrices_are_candidates_and_ids_differ_from_rows(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (2, 3))

        (system,) = plan_systems(_case(sweeps=[_sweep()]), {"ds": sparse_id_dataset})

        assert system.matrix == MatrixRef(matrix_id=MatrixId(9), row=2)

    def test_full_fraction_selects_every_held_out_matrix(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        selector = {"kind": "fraction", "p": 1.0, "seed": 0}

        systems = plan_systems(_case(sweeps=[_sweep(selector=selector)]), {"ds": sparse_id_dataset})

        assert [s.entry.id for s in systems] == ["sw-5", "sw-9"]

    def test_partial_fraction_is_reproducible_for_a_seed(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        case = _case(sweeps=[_sweep(selector={"kind": "fraction", "p": 0.5, "seed": 3})])

        first = plan_systems(case, {"ds": sparse_id_dataset})
        second = plan_systems(case, {"ds": sparse_id_dataset})

        assert len(first) == 1
        assert [s.entry.id for s in first] == [s.entry.id for s in second]

    def test_a_fraction_that_selects_nothing_is_an_error(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        case = _case(sweeps=[_sweep(selector={"kind": "fraction", "p": 0.1, "seed": 0})])

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": sparse_id_dataset})

    def test_every_selected_matrix_draws_its_own_rhs_seeds(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        case = _case(
            sweeps=[_sweep(selector={"kind": "first_n", "n": 2}, rhs_count=2, seed=42)],
        )

        systems = plan_systems(case, {"ds": sparse_id_dataset})

        assert [s.entry.id for s in systems] == ["sw-5-0", "sw-5-1", "sw-9-0", "sw-9-1"]
        assert len({s.entry.seed for s in systems}) == 4

    def test_dataset_kind_takes_each_matrixs_own_held_out_rows(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 2, 3))
        case = _case(
            sweeps=[
                _sweep(
                    selector={"kind": "first_n", "n": 2},
                    rhs_source={"kind": "dataset"},
                    rhs_count=1,
                )
            ]
        )

        systems = plan_systems(case, {"ds": sparse_id_dataset})

        assert [(s.matrix.matrix_id, s.rhs_row, s.matrix.row) for s in systems] == [
            (5, 0, 0),
            (9, 2, 2),
        ]

    def test_dataset_kind_rejects_more_rows_than_a_matrix_has_held_out(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 2, 3))
        case = _case(sweeps=[_sweep(rhs_source={"kind": "dataset"}, rhs_count=2)])

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": sparse_id_dataset})

    def test_a_single_matrix_dataset_is_rejected_naming_the_dataset_and_rule(
        self, single_matrix_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(single_matrix_dataset, (0, 1))

        with pytest.raises(ValueError):
            plan_systems(_case(sweeps=[_sweep()]), {"ds": single_matrix_dataset})

    def test_a_dataset_without_a_pinned_split_asks_for_generation(
        self, sparse_id_dataset: Path
    ) -> None:
        with pytest.raises(StageInputsStale):
            plan_systems(_case(sweeps=[_sweep()]), {"ds": sparse_id_dataset})


class TestCaseWideChecks:
    def test_duplicate_planned_ids_are_rejected(
        self, sparse_id_dataset: Path, pin_test_fold: PinTestFold
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        case = _case(sweeps=[_sweep(), _sweep()])

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": sparse_id_dataset})

    def test_datasets_of_different_system_size_are_rejected(
        self, single_matrix_dataset: Path, larger_single_matrix_dataset: Path
    ) -> None:
        case = _case(
            comparisons=[
                _comparison("small"),
                _comparison("large", matrix_dataset="big"),
            ],
            datasets=("ds", "big"),
        )

        with pytest.raises(ValueError):
            plan_systems(case, {"ds": single_matrix_dataset, "big": larger_single_matrix_dataset})

    def test_comparisons_and_sweeps_plan_in_declared_order(
        self,
        single_matrix_dataset: Path,
        sparse_id_dataset: Path,
        pin_test_fold: PinTestFold,
    ) -> None:
        pin_test_fold(sparse_id_dataset, (0, 1, 2, 3))
        case = _case(
            comparisons=[_comparison("one")],
            sweeps=[_sweep("many", matrix_dataset="multi")],
            datasets=("ds", "multi"),
        )

        systems = plan_systems(case, {"ds": single_matrix_dataset, "multi": sparse_id_dataset})

        assert [s.entry.id for s in systems] == ["one", "many-5"]
