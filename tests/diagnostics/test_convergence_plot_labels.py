from __future__ import annotations

from types import SimpleNamespace

from neuralls.platform.reporting.convergence_plots import _build_convergence_label


def test_build_convergence_label_adds_limit_for_neural_only() -> None:
    """Neural methods show limit metadata in legend labels."""
    label = _build_convergence_label(
        method_name="neural_method",
        meta={"preconditioner_type": "neural", "limit_iters": 10},
    )
    assert label == "neural_method (limit=10)"


def test_build_convergence_label_omits_limit_for_non_neural() -> None:
    """Non-neural methods do not include limit metadata in labels."""
    label = _build_convergence_label(
        method_name="jacobi",
        meta={"preconditioner_type": "jacobi", "limit_iters": 10},
    )
    assert label == "jacobi"


def test_build_convergence_label_preserves_trace_fields_and_limit() -> None:
    """Label preserves tr/ap fields and appends neural limit metadata."""
    label = _build_convergence_label(
        method_name="neural_trace",
        meta={
            "residual_iters": 5,
            "applied_iters": 7,
            "preconditioner_type": "neural",
            "limit_iters": 3,
        },
    )
    assert label == "neural_trace (tr=5, ap=7, limit=3)"


def test_build_convergence_label_supports_object_metadata() -> None:
    """Object-style metadata keeps compatibility with existing callers."""
    meta = SimpleNamespace(
        residual_iters=4,
        applied_iters=None,
        preconditioner_type="neural",
        limit_iters=2,
    )
    label = _build_convergence_label(method_name="neural_obj", meta=meta)
    assert label == "neural_obj (tr=4, ap=4, limit=2)"
