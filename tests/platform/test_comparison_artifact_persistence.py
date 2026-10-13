"""Tests for the comparison-artifact persistence added alongside scaling-law
support: per-system scalar costs extended into `comparison.toml`, and 1D
iteration histories moved out of `comparison.json`'s inline lists into
`histories.h5`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import h5py
import numpy as np

from neuralls.domain.solver.models.config import ComparisonData, ComparisonGeneral, SolverParams
from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonResult,
    StageCost,
)
from neuralls.platform.reporting.artifacts import write_comparison_artifacts
from neuralls.shared.types import CostProvenance


def _solver_params(tmp_path: Path) -> ComparisonGeneral:
    return ComparisonGeneral(
        params=SolverParams(
            rtol=1.0e-6,
            atol=1.0e-14,
            max_iterations=10,
            stopping_criterion="residual_norm",
            m_max=20,
        ),
        data=ComparisonData(
            matrix_path=tmp_path / "matrix.npy",
            rhs_path=tmp_path / "rhs.npy",
        ),
    )


def _result_with_histories_and_costs(name: str) -> CGComparisonResult:
    return CGComparisonResult(
        x=np.array([1.0, 2.0]),
        converged=True,
        iterations=3,
        residual=1.0e-8,
        residual_abs=1.0e-9,
        residual_history_rel=[1.0, 0.1, 1.0e-8],
        residual_history_abs=[1.0, 0.1, 1.0e-9],
        preconditioner=name,
        initial_guess=np.zeros(2),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
        error_history_a_rel=[1.0, 0.5, 0.01],
        error_bound_a_rel=[1.1, 0.6, 0.02],
        generation_cost=StageCost(
            wall_time_seconds=2.0, peak_memory_bytes=1024, provenance=CostProvenance.HISTORICAL
        ),
        setup_cost=0.25,
        setup_peak_memory_bytes=2048,
        solve_time_seconds=0.75,
        solve_peak_memory_bytes=4096,
    )


def _comparison_result(tmp_path: Path) -> ComparisonResult:
    return ComparisonResult(
        results={
            "jacobi": _result_with_histories_and_costs("jacobi"),
            "identity": CGComparisonResult(
                x=np.zeros(2),
                converged=True,
                iterations=1,
                residual=1.0,
                residual_abs=1.0,
                residual_history_rel=[],
                residual_history_abs=[],
                preconditioner="identity",
                initial_guess=np.zeros(2),
                exact_error=None,
                rhs_norm=1.0,
                breakdown=False,
            ),
        },
        summary="ok",
        solver_params=_solver_params(tmp_path),
        preconditioners=("jacobi", "identity"),
    )


def test_histories_written_to_hdf5_not_json(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    manifest = write_comparison_artifacts(result=_comparison_result(tmp_path), work_root=work_root)

    with h5py.File(manifest.histories_h5, "r") as h5file:
        assert set(h5file.keys()) == {"jacobi"}  # identity measured no histories, so no group
        np.testing.assert_allclose(h5file["jacobi"]["residual_history_rel"][:], [1.0, 0.1, 1.0e-8])
        np.testing.assert_allclose(h5file["jacobi"]["error_history_a_rel"][:], [1.0, 0.5, 0.01])

    comparison_json = manifest.comparison_json.read_text(encoding="utf-8")
    assert "residual_history_rel" not in comparison_json
    assert "error_history_a_rel" not in comparison_json


def test_comparison_toml_includes_every_measured_cost_field(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    manifest = write_comparison_artifacts(result=_comparison_result(tmp_path), work_root=work_root)

    with manifest.comparison_toml.open("rb") as fh:
        payload = tomllib.load(fh)

    assert payload["iterations"] == {"jacobi": 3, "identity": 1}
    # setup_cost is always a real measured float (never None) — identity's
    # 0.0 is a genuine zero-time build, so it belongs in the table, not
    # skipped as if unmeasured.
    assert payload["setup_time_seconds"] == {"jacobi": 0.25, "identity": 0.0}
    # solve/generation time and peak memory were never measured for
    # identity (None, not a fabricated zero) — correctly absent.
    assert payload["solve_time_seconds"] == {"jacobi": 0.75}
    assert payload["generation_time_seconds"] == {"jacobi": 2.0}
    assert payload["peak_memory_bytes"] == {"jacobi": 4096}
    assert "identity" not in payload["solve_time_seconds"]
