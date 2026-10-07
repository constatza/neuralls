"""Cross-binding scalar agreement: first value kept, disagreement flagged, warnings unchanged."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from neuralls.domain.generation.scalar_aggregate import BindingScale, ScalarAggregator
from neuralls.shared.types import ScaleMetadata

_NORM_WARNING = (
    "Bindings produced different normalized matrix norms; "
    "the dataset manifest keeps a representative matrix_norm only."
)
_SCALE_WARNING = (
    "Bindings produced different matrix value scales; "
    "the dataset manifest omits a shared reversible matrix scale."
)
_METADATA_WARNING = (
    "Bindings produced different scale metadata payloads; "
    "the dataset manifest intentionally stores no shared scale metadata."
)
_SHARED_METADATA: ScaleMetadata = {"spectral_radius_bound": 1.5, "dimension_scale": 2.0}


@pytest.fixture
def make_binding_scale() -> Callable[..., BindingScale]:
    """Build one binding's scalar observation, defaulting to an agreeing binding."""

    def _build(
        matrix_norm_value: float = 1.0,
        matrix_value_scale: float = 3.0,
        scale_params: ScaleMetadata | None = _SHARED_METADATA,
    ) -> BindingScale:
        return BindingScale(
            matrix_norm_value=matrix_norm_value,
            matrix_value_scale=matrix_value_scale,
            scale_params=scale_params,
        )

    return _build


def test_scalar_aggregator_matches_current_warnings(
    make_binding_scale: Callable[..., BindingScale],
    warning_messages: list[str],
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_norm_value=1.0, matrix_value_scale=3.0))
    aggregator.observe(
        make_binding_scale(
            matrix_norm_value=2.0,
            matrix_value_scale=5.0,
            scale_params={"spectral_radius_bound": 2.5, "dimension_scale": 2.0},
        )
    )
    summary = aggregator.result()

    assert summary.matrix_norm == 1.0
    assert summary.matrix_value_scale == 1.0
    assert summary.scale_metadata is None
    assert warning_messages == [_NORM_WARNING, _SCALE_WARNING, _METADATA_WARNING]


def test_scalar_aggregator_agreement_emits_no_warning(
    make_binding_scale: Callable[..., BindingScale],
    warning_messages: list[str],
) -> None:
    aggregator = ScalarAggregator()
    for _ in range(3):
        aggregator.observe(make_binding_scale())
    summary = aggregator.result()

    assert summary.matrix_norm == 1.0
    assert summary.matrix_value_scale == 3.0
    assert summary.scale_metadata == _SHARED_METADATA
    assert warning_messages == []


@pytest.mark.parametrize(
    ("second_norm", "expect_disagreement"),
    [
        pytest.param(1.0 + 0.5e-10, False, id="inside-rtol"),
        pytest.param(1.0 + 1e-6, True, id="outside-rtol"),
    ],
)
def test_scalar_aggregator_applies_isclose_tolerance(
    make_binding_scale: Callable[..., BindingScale],
    warning_messages: list[str],
    second_norm: float,
    expect_disagreement: bool,
) -> None:
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_norm_value=1.0))
    aggregator.observe(make_binding_scale(matrix_norm_value=second_norm))
    aggregator.result()

    assert (_NORM_WARNING in warning_messages) is expect_disagreement


def test_scalar_aggregator_keeps_only_first_value(
    make_binding_scale: Callable[..., BindingScale],
) -> None:
    """Agreement is checked against the first binding; later values are never stored."""
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_norm_value=4.0))
    for _ in range(1000):
        aggregator.observe(make_binding_scale(matrix_norm_value=4.0))

    assert aggregator.result().matrix_norm == 4.0


def test_scalar_aggregator_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No norm or scale values"):
        ScalarAggregator().result()


def test_agreeing_bindings_resolve_to_their_shared_scalars(
    make_binding_scale: Callable[..., BindingScale],
) -> None:
    """Two agreeing bindings resolve to their shared norm, scale and metadata."""
    aggregator = ScalarAggregator()
    aggregator.observe(make_binding_scale(matrix_norm_value=2.0, matrix_value_scale=7.0))
    aggregator.observe(make_binding_scale(matrix_norm_value=2.0, matrix_value_scale=7.0))
    summary = aggregator.result()

    assert summary.matrix_norm == 2.0
    assert summary.matrix_value_scale == 7.0
    assert summary.scale_metadata == _SHARED_METADATA
