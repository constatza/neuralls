"""Tests for CgErrorsStrategy and the snapshot-scaling config of both snapshot strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.domain.generation import run_generation
from neuralls.domain.generation.interfaces import TracingSolverCallable
from neuralls.domain.generation.strategy_configs import (
    CgErrorsConfig,
    JacobiProbesConfig,
    NormSnapshotScaling,
    NoSnapshotScaling,
)
from neuralls.domain.normalization import ResidualTraceSamples
from neuralls.shared.fact_binding import describe_model
from neuralls.shared.scientific_description.catalog import (
    BETA,
    DAMPING,
    ITERATION_COUNT,
    METRIC,
    STEPS,
    make_fact,
)
from tests.fakes.archive_source import InMemoryArchiveSource
from tests.generation_support import residual_trace_rows

type RunCgErrors = Callable[[dict[str, Any], np.ndarray | None], ResidualTraceSamples]


@pytest.fixture
def archive_source() -> InMemoryArchiveSource:
    """An empty in-memory archive; cg_errors never reads it."""
    return InMemoryArchiveSource()


@pytest.fixture
def cg_errors(
    spd_matrix: np.ndarray,
    residual_solver: TracingSolverCallable,
    archive_source: InMemoryArchiveSource,
) -> RunCgErrors:
    """Run ``cg_errors`` on the SPD matrix with a given config and optional single RHS."""

    def _run(cfg: dict[str, Any], single_rhs: np.ndarray | None) -> ResidualTraceSamples:
        return residual_trace_rows(
            run_generation(
                "cg_errors",
                spd_matrix,
                cfg=cfg,
                solver=residual_solver,
                archive_source=archive_source,
                single_rhs=single_rhs,
            )
        ).residual_traces

    return _run


@pytest.mark.parametrize("samples", [1, 7, 8])
def test_cg_errors_emit_exactly_the_row_budget(
    cg_errors: RunCgErrors, spd_matrix: np.ndarray, samples: int
) -> None:
    traces = cg_errors({"samples": samples, "stop": 3, "start": 1, "seed": 0}, None)

    n = spd_matrix.shape[0]
    assert (traces.residuals.shape, traces.solutions.shape) == ((samples, n), (samples, n))


@pytest.mark.parametrize(
    "scaling",
    [
        pytest.param({"kind": "none"}, id="unscaled"),
        pytest.param({"kind": "norm", "norm": "energy"}, id="energy"),
    ],
)
def test_cg_error_rows_satisfy_a_e_equals_r(
    cg_errors: RunCgErrors, spd_matrix: np.ndarray, scaling: dict[str, str]
) -> None:
    traces = cg_errors(
        {"samples": 10, "stop": 4, "start": 1, "seed": 0, "snapshot_scaling": scaling}, None
    )

    np.testing.assert_allclose(
        traces.residuals, traces.solutions @ spd_matrix.T, rtol=1e-9, atol=1e-9
    )


def test_cg_errors_ignore_a_binding_rhs(cg_errors: RunCgErrors, spd_matrix: np.ndarray) -> None:
    """The probes are the strategy's own systems: a shared RHS changes nothing."""
    cfg = {"samples": 6, "stop": 3, "start": 1, "seed": 4}

    with_rhs = cg_errors(cfg, np.ones(spd_matrix.shape[0]))
    without = cg_errors(cfg, None)

    np.testing.assert_array_equal(with_rhs.solutions, without.solutions)
    np.testing.assert_array_equal(with_rhs.residuals, without.residuals)


@pytest.mark.parametrize("config_type", [CgErrorsConfig, JacobiProbesConfig])
def test_snapshot_scaling_defaults_to_none(config_type: type[CgErrorsConfig]) -> None:
    config = config_type.model_validate({"samples": 3, "stop": 5})

    assert config.snapshot_scaling == NoSnapshotScaling()


@pytest.mark.parametrize("config_type", [CgErrorsConfig, JacobiProbesConfig])
def test_norm_snapshot_scaling_parses_with_unit_default_power(
    config_type: type[CgErrorsConfig],
) -> None:
    config = config_type.model_validate(
        {"samples": 3, "stop": 5, "snapshot_scaling": {"kind": "norm", "norm": "energy"}}
    )

    assert config.snapshot_scaling == NormSnapshotScaling(norm="energy", power=1.0)


@pytest.mark.parametrize(
    "scaling",
    [
        pytest.param({"kind": "norm", "norm": "a"}, id="unknown-norm"),
        pytest.param({"kind": "norm", "norm": "l2", "power": -0.5}, id="negative-power"),
        pytest.param({"kind": "norm"}, id="missing-norm"),
        pytest.param({"kind": "unit"}, id="unknown-kind"),
        pytest.param({"kind": "none", "norm": "l2"}, id="norm-on-none"),
    ],
)
@pytest.mark.parametrize("config_type", [CgErrorsConfig, JacobiProbesConfig])
def test_invalid_snapshot_scaling_is_rejected(
    config_type: type[CgErrorsConfig], scaling: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        config_type.model_validate({"samples": 3, "stop": 5, "snapshot_scaling": scaling})


def test_cg_errors_facts_show_the_snapshot_scaling() -> None:
    config = CgErrorsConfig.model_validate(
        {
            "samples": 3,
            "stop": 5,
            "snapshot_scaling": {"kind": "norm", "norm": "energy", "power": 0.5},
        }
    )

    assert describe_model(config) == (
        make_fact(ITERATION_COUNT, 5),
        make_fact(METRIC, "energy"),
        make_fact(BETA, 0.5),
    )


def test_jacobi_probes_facts_without_snapshot_scaling() -> None:
    config = JacobiProbesConfig.model_validate({"samples": 3, "stop": 5})

    assert describe_model(config) == (
        make_fact(STEPS, 5),
        make_fact(DAMPING, "auto"),
    )
