"""Tests for the dlkit `Fittable` adapter over `PODCoarseningStrategy`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.preconditioners.pod_fittable import (
    EnergyInnerProduct,
    EuclideanInnerProduct,
    PODCoarseningFittable,
    parse_inner_product,
)


@pytest.fixture
def snapshot_batches() -> list[dict[str, dict[str, torch.Tensor]]]:
    """Two dataloader-shaped batches whose `targets.y` entries are POD snapshots.

    Mirrors the `TensorDict`-nested-under-"targets" batch shape
    `FlexibleDataset`/`ArrayDataModule` actually yield (see
    `RuntimeDatasetContract.loss_target_key == "targets.y"`), split across two
    batches to exercise concatenation across the whole dataloader, not just
    one batch.
    """
    rng = np.random.default_rng(0)
    first = torch.as_tensor(rng.standard_normal((3, 5)), dtype=torch.float64)
    second = torch.as_tensor(rng.standard_normal((2, 5)), dtype=torch.float64)
    return [{"targets": {"y": first}}, {"targets": {"y": second}}]


def test_fit_concatenates_batches_and_delegates_to_tensor_fit(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """fit(dataloader) materializes all batches' targets.y and fits the basis."""
    model = PODCoarseningFittable(rank=2)

    model.fit(snapshot_batches)

    assert model.is_fitted()
    assert isinstance(model, PODCoarseningStrategy)
    assert model._basis.shape == (5, 2)


def test_fit_respects_custom_target_name(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """A non-default target_name reads that key instead of 'y'."""
    renamed_batches = [
        {"targets": {"solutions": batch["targets"]["y"]}} for batch in snapshot_batches
    ]
    model = PODCoarseningFittable(rank=2, target_name="solutions")

    model.fit(renamed_batches)

    assert model.is_fitted()


class _OneShotBatchIterable:
    """Iterable that raises if consumed more than once.

    Catches a regression back to `list(snapshots)`, which would force
    `fit()` to drain the whole iterable up front instead of streaming it —
    the actual defect behind a `cuMemHostAlloc` OOM against a real
    `pin_memory=True` dataloader (this fixture doesn't reproduce the OOM
    itself, no GPU needed for that; it only pins down the single-pass
    consumption contract that avoids it).
    """

    def __init__(self, batches: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        self._batches = batches
        self._consumed = False

    def __iter__(self) -> Iterator[dict[str, dict[str, torch.Tensor]]]:
        if self._consumed:
            raise RuntimeError("dataloader iterated more than once")
        self._consumed = True
        return iter(self._batches)


def test_fit_iterates_dataloader_without_eager_listing(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """fit() must consume the dataloader in a single pass, never `list(snapshots)`."""
    model = PODCoarseningFittable(rank=2)

    model.fit(_OneShotBatchIterable(snapshot_batches))

    assert model.is_fitted()


def test_fit_rejects_empty_dataloader() -> None:
    """An empty dataloader raises a clear ValueError, not an opaque torch error."""
    model = PODCoarseningFittable(rank=2)

    with pytest.raises(ValueError):
        model.fit([])


def test_hparams_exposes_configured_rank_before_fit() -> None:
    """`.hparams` is readable before fit() — dlkit's tracking logger reads it first."""
    model = PODCoarseningFittable(rank=0.99)

    assert dict(model.hparams) == {"rank": 0.99, "inner_product": "euclidean"}


def test_dataloader_fit_forwards_row_scales_like_the_parent_tensor_fit(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """Explicit row scales reach the parent fit on the dataloader path unchanged."""
    row_scales = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5], dtype=torch.float64)
    snapshots = torch.cat([batch["targets"]["y"] for batch in snapshot_batches])
    reference = PODCoarseningStrategy(rank=2)
    reference.fit(snapshots, row_scales=row_scales)
    model = PODCoarseningFittable(rank=2)

    model.fit(snapshot_batches, row_scales=row_scales)

    torch.testing.assert_close(model._basis.cpu(), reference._basis, rtol=0.0, atol=0.0)


def test_dataloader_fit_forwards_the_inner_product_like_the_parent_tensor_fit(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """An inner-product weight reaches the parent fit on the dataloader path unchanged."""
    snapshots = torch.cat([batch["targets"]["y"] for batch in snapshot_batches])
    n_dofs = snapshots.shape[1]
    weight = torch.diag(torch.linspace(1.0, 4.0, n_dofs, dtype=torch.float64))
    reference = PODCoarseningStrategy(rank=2)
    reference.fit(snapshots, inner_product=weight)
    model = PODCoarseningFittable(rank=2)

    model.fit(snapshot_batches, inner_product=weight)

    torch.testing.assert_close(model._basis.cpu(), reference._basis, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("euclidean", EuclideanInnerProduct()),
        (
            {"kind": "energy", "dataset_dir": "processed/pod"},
            EnergyInnerProduct(dataset_dir=Path("processed/pod")),
        ),
    ],
)
def test_inner_product_parses_a_kind_name_or_a_patched_table(
    value: str | dict[str, str], expected: EuclideanInnerProduct | EnergyInnerProduct
) -> None:
    assert parse_inner_product(value) == expected


def test_energy_without_the_patched_dataset_is_rejected() -> None:
    """A bare `"energy"` never reached the runtime patcher, so its weight is undefined."""
    with pytest.raises(ValidationError):
        PODCoarseningFittable(rank=2, inner_product="energy")


def test_explicit_weight_wins_and_construction_reads_nothing(
    snapshot_batches: list[dict[str, dict[str, torch.Tensor]]],
) -> None:
    """A configured energy dataset that does not exist is never read when a weight is given.

    A checkpoint rebuild calls the constructor with the job's patched table on any machine,
    so neither construction nor an explicitly weighted fit may touch the dataset.
    """
    snapshots = torch.cat([batch["targets"]["y"] for batch in snapshot_batches])
    weight = torch.diag(torch.linspace(1.0, 4.0, snapshots.shape[1], dtype=torch.float64))
    reference = PODCoarseningStrategy(rank=2)
    reference.fit(snapshots, inner_product=weight)
    model = PODCoarseningFittable(
        rank=2, inner_product={"kind": "energy", "dataset_dir": "missing/pod-dataset"}
    )

    model.fit(snapshots, inner_product=weight)

    torch.testing.assert_close(model._basis, reference._basis, rtol=0.0, atol=0.0)
