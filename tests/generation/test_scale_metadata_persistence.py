"""Tests for scale metadata resolution in data generation."""

from neuralls.domain.generation.scalar_aggregate import BindingScale, ScalarAggregator


def test_multi_matrix_resolution_omits_shared_scale_metadata() -> None:
    """Mixed binding scales drop manifest-level reversible scale metadata."""
    aggregator = ScalarAggregator()
    aggregator.observe(
        BindingScale(
            matrix_norm_value=1.0,
            matrix_value_scale=3.0,
            scale_params={"spectral_radius_bound": 1.5, "dimension_scale": 2.0},
        )
    )
    aggregator.observe(
        BindingScale(
            matrix_norm_value=2.0,
            matrix_value_scale=5.0,
            scale_params={"spectral_radius_bound": 2.5, "dimension_scale": 2.0},
        )
    )
    summary = aggregator.result()

    assert summary.matrix_norm == 1.0
    assert summary.matrix_value_scale == 1.0
    assert summary.scale_metadata is None
