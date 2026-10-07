"""Trajectory windows must never keep step 0, the base pair.

Iterate 0 of every CG trace is the base pair (r0 = b, e0 = x*), because
generation always starts CG at x0 = 0. An omitted start defaults to -1 (the
last step only), and any start that resolves to step 0 on a full-length
trajectory is rejected by the shared window validator, so the base pair can
never leak into the dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from neuralls.composition.generation.default_services import make_solver
from neuralls.domain.generation import run_generation
from neuralls.domain.generation.step_window import StepWindow
from neuralls.domain.generation.strategy_configs import (
    ResidualErrorConfig,
    SmootherFilteredProbesConfig,
)
from neuralls.domain.generation.trace_utils import (
    TrajectoryShortfallError,
    require_full_trajectory,
)

TRAJECTORY_CONFIG_TYPES = (
    ResidualErrorConfig,
    SmootherFilteredProbesConfig,
)
WINDOW_STOP = 3
PROBE_COUNT = 2
"""Only read by SmootherFilteredProbesConfig, which requires `samples`; harmless for the others."""


@pytest.fixture
def make_trajectory_config() -> Callable[..., Any]:
    """Build a trajectory config of the given type with a window ending at ``WINDOW_STOP``."""

    def _build(config_type: type, **window: Any) -> Any:
        fields: dict[str, Any] = {"stop": WINDOW_STOP, "samples": PROBE_COUNT, **window}
        return config_type(**fields)

    return _build


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_start_zero_is_rejected_naming_the_field(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    with pytest.raises(ValidationError, match="start"):
        make_trajectory_config(config_type, start=0)


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_start_omitted_defaults_to_last_step(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    config = make_trajectory_config(config_type)
    assert list(config.window.resolve_indices(WINDOW_STOP + 1)) == [WINDOW_STOP]


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_negative_one_keeps_last_step_only(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    config = make_trajectory_config(config_type, start=-1, stop=5, step=3)
    assert list(config.window.resolve_indices(5 + 1)) == [5]


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_negative_stop_is_accepted(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    config = make_trajectory_config(config_type, start=-7, stop=7)
    assert list(config.window.resolve_indices(7 + 1)) == list(range(1, 8))


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_start_below_negative_stop_rejected(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    """start = -(stop + 1) resolves to step 0 on a full trajectory, so it is rejected."""
    with pytest.raises(ValidationError, match="start"):
        make_trajectory_config(config_type, start=-8, stop=7)


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_step_never_yields_step_zero(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    """Every accepted default or negative start, over several stops and steps, keeps step >= 1."""
    for stop in range(1, 7):
        for step in range(1, 5):
            starts: list[int | None] = [None, *range(-1, -stop - 1, -1)]
            for start in starts:
                config = make_trajectory_config(config_type, start=start, stop=stop, step=step)
                indices = config.window.resolve_indices(stop + 1)
                assert len(indices) > 0
                assert indices[0] >= 1


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_step_with_positive_start_one(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    config = make_trajectory_config(config_type, start=1, step=2, stop=7)
    assert list(config.window.resolve_indices(7 + 1)) == [1, 3, 5, 7]


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_stop_one_default(config_type: type, make_trajectory_config: Callable[..., Any]) -> None:
    config = make_trajectory_config(config_type, stop=1)
    assert list(config.window.resolve_indices(1 + 1)) == [1]


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_start_equal_stop_rejected(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    """StepWindow requires start < stop, so start == stop is refused for every trajectory config."""
    with pytest.raises(ValueError, match="start"):
        make_trajectory_config(config_type, start=WINDOW_STOP)


def test_early_exit_negative_start_rejected() -> None:
    """A truncated trajectory is rejected by step index even when its row count matches.

    Full window stop=3, start=-2 expects steps [2, 3] (2 rows). A trajectory
    that stopped after step 2 has rows 0..2 and keeps steps [1, 2]: also 2 rows,
    but the wrong steps, so it must raise.
    """
    window = StepWindow(stop=WINDOW_STOP, start=-2)
    expected = window.resolve_indices(WINDOW_STOP + 1)
    truncated = window.resolve_indices(WINDOW_STOP)
    assert len(truncated) == len(expected)
    with pytest.raises(TrajectoryShortfallError):
        require_full_trajectory(truncated, expected, 0, WINDOW_STOP)


def test_full_length_passes_index_check() -> None:
    window = StepWindow(stop=WINDOW_STOP, start=-2)
    expected = window.resolve_indices(WINDOW_STOP + 1)
    require_full_trajectory(window.resolve_indices(WINDOW_STOP + 1), expected, 0, WINDOW_STOP)


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_start_one_is_accepted(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    config = make_trajectory_config(config_type, start=1)
    assert config.start == 1


@pytest.mark.parametrize("config_type", TRAJECTORY_CONFIG_TYPES)
def test_negative_start_reaching_step_zero_is_rejected(
    config_type: type, make_trajectory_config: Callable[..., Any]
) -> None:
    """start = -(stop + 1) resolves to index 0 on a full trajectory; it must not pass."""
    with pytest.raises(ValidationError, match="start"):
        make_trajectory_config(config_type, start=-(WINDOW_STOP + 1))


SPD_SIZE = 6
BASE_SYSTEMS = 2
STOP = 3
START = 1
ROWS_PER_BASE_SYSTEM = STOP - START + 1
"""K = stop - start + 1: rows emitted per base system for an explicit window."""
RELATIVE_TOLERANCE = 1e-12


@pytest.fixture
def seeded_spd_matrix() -> np.ndarray:
    """Small symmetric positive-definite matrix from a fixed seed."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((SPD_SIZE, SPD_SIZE))
    return a.T @ a + np.eye(SPD_SIZE)


@pytest.fixture
def seeded_rhs() -> np.ndarray:
    """Right-hand side from a fixed seed, shared by the base system."""
    return np.random.default_rng(5).standard_normal(SPD_SIZE)


def _rows_close_to(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Boolean mask of rows equal to ``target`` within the relative tolerance."""
    return np.array([np.allclose(row, target, rtol=RELATIVE_TOLERANCE, atol=0.0) for row in rows])


def test_residuals_start_one_emits_k_rows_per_base_system(
    seeded_spd_matrix: np.ndarray, seeded_rhs: np.ndarray
) -> None:
    """Each base system emits K rows, and none is the base pair (b, x*).

    Expected counts: samples = K * BASE_SYSTEMS = 6 gives exactly two base
    systems (ceil(6 / 3) = 2) with three rows each, so the total is 6.
    Iterate 0 (r0 = b, e0 = x*) is excluded by start = 1, so no residual row
    may equal b and no error row may equal x*.
    """
    cfg = {"samples": ROWS_PER_BASE_SYSTEM * BASE_SYSTEMS, "stop": STOP, "start": START, "seed": 0}
    result = run_generation(
        "residuals",
        seeded_spd_matrix,
        cfg=cfg,
        solver=make_solver(),
        single_rhs=seeded_rhs,
    )
    traces = result.error_traces
    assert traces is not None
    assert traces.residuals is not None
    assert traces.residuals.shape[0] == ROWS_PER_BASE_SYSTEM * BASE_SYSTEMS
    for system in range(BASE_SYSTEMS):
        assert int(np.sum(traces.sample_indices == system)) == ROWS_PER_BASE_SYSTEM

    x_star = np.linalg.solve(seeded_spd_matrix, seeded_rhs)
    assert not _rows_close_to(traces.residuals, seeded_rhs).any()
    assert not _rows_close_to(traces.errors, x_star).any()
