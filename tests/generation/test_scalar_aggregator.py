"""Cross-binding scalars: first matrix norm kept, one scale factor recorded per matrix."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.scalar_aggregate import (
    BindingScale,
    ConflictingMatrixScaleError,
    ScalarAggregator,
    ScaleField,
    ScaleSummary,
)
from neuralls.shared.types import MatrixSampleId


@pytest.fixture
def make_binding_scale() -> Callable[..., BindingScale]:
    """Build one binding's scalar observation, defaulting to matrix 0 with agreeing values."""

    def _build(
        matrix_sample_id: int = 0, matrix_norm_value: float = 1.0, alpha: float = 3.0
    ) -> BindingScale:
        return BindingScale(
            matrix_sample_id=MatrixSampleId(matrix_sample_id),
            matrix_norm_value=matrix_norm_value,
            alpha=alpha,
        )

    return _build


def test_scalar_aggregator_records_alpha_per_matrix_sample_id(
    make_binding_scale: Callable[..., BindingScale],
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_sample_id=4, alpha=3.0))
    aggregator.observe(make_binding_scale(matrix_sample_id=7, alpha=5.0))
    aggregator.observe(make_binding_scale(matrix_sample_id=4, alpha=3.0))

    assert aggregator.result() == ScaleSummary(
        matrix_norm=1.0, matrix_scales={MatrixSampleId(4): 3.0, MatrixSampleId(7): 5.0}
    )


def test_scalar_aggregator_rejects_conflicting_alpha_for_one_matrix(
    make_binding_scale: Callable[..., BindingScale],
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_sample_id=2, alpha=3.0))

    with pytest.raises(ConflictingMatrixScaleError) as error:
        aggregator.observe(make_binding_scale(matrix_sample_id=2, alpha=4.0))

    assert (error.value.matrix_sample_id, error.value.first, error.value.second) == (2, 3.0, 4.0)


def test_scalar_aggregator_keeps_first_norm_and_reports_disagreement(
    make_binding_scale: Callable[..., BindingScale],
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_sample_id=0, matrix_norm_value=1.0))
    aggregator.observe(make_binding_scale(matrix_sample_id=1, matrix_norm_value=2.0))
    summary = aggregator.result()

    assert summary.matrix_norm == 1.0
    assert summary.disagreements == frozenset({ScaleField.MATRIX_NORM})


@pytest.mark.parametrize(
    ("second_norm", "expect_disagreement"),
    [
        pytest.param(1.0 + 0.5e-10, False, id="inside-rtol"),
        pytest.param(1.0 + 1e-6, True, id="outside-rtol"),
    ],
)
def test_scalar_aggregator_applies_isclose_tolerance(
    make_binding_scale: Callable[..., BindingScale],
    second_norm: float,
    expect_disagreement: bool,
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_norm_value=1.0))
    aggregator.observe(make_binding_scale(matrix_norm_value=second_norm))
    disagreements = aggregator.result().disagreements

    assert (ScaleField.MATRIX_NORM in disagreements) is expect_disagreement


def test_scalar_aggregator_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        ScalarAggregator().result()
