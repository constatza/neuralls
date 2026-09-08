"""Typed artifact writing for comparison workflows."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from numpy.typing import NDArray

from neuralls.domain.solver.models.result import (
    CGComparisonResult,
    ComparisonRecommendations,
    ComparisonResult,
    PlotPaths,
)
from neuralls.platform.reporting.serialization import to_json_primitive


@dataclass(frozen=True)
class ArrayArtifact:
    """Reference to a persisted numpy array artifact.

    Args:
        path: Relative path within the output directory.
        shape: Array shape tuple.
        dtype: Numpy dtype string.
    """

    path: Path
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ComparisonArtifactManifest:
    """Manifest of files emitted by comparison artifact writing.

    Args:
        comparison_toml: Path to comparison summary TOML.
        comparison_json: Path to comparison summary JSON.
        recommendations_json: Path to ranked recommendations JSON.
        summary_txt: Path to text summary.
        config_copy: Path to config file copy.
        arrays: Tuple of array artifact references.
    """

    comparison_toml: Path
    comparison_json: Path
    recommendations_json: Path
    summary_txt: Path
    config_copy: Path | None
    arrays: tuple[ArrayArtifact, ...] = ()


@dataclass(frozen=True)
class PendingArrayArtifact:
    """In-memory numpy artifact waiting to be saved."""

    reference: ArrayArtifact
    values: NDArray[Any]


@dataclass(frozen=True)
class SerializedSolverResult:
    """JSON-safe solver result payload."""

    iterations: int
    residual: float
    residual_abs: float | None = None
    converged: bool | None = None
    residual_history_rel: tuple[float, ...] = ()
    residual_history_abs: tuple[float, ...] = ()
    preconditioner: str | None = None
    exact_error: float | None = None
    rhs_norm: float | None = None
    breakdown: bool | None = None
    error: str | None = None
    x: ArrayArtifact | None = None
    initial_guess: ArrayArtifact | None = None


@dataclass(frozen=True)
class SerializedComparisonPayload:
    """JSON-safe comparison payload without raw numpy arrays."""

    summary: str
    preconditioners: tuple[str, ...]
    condition_numbers: dict[str, float]
    plot_paths: PlotPaths
    recommendations: ComparisonRecommendations
    results: dict[str, SerializedSolverResult]


def _sanitize_path_part(value: str) -> str:
    """Convert an arbitrary key into a stable path-safe fragment."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "value"


def _array_reference(
    path_parts: tuple[str, ...],
    values: NDArray[Any],
) -> PendingArrayArtifact:
    """Create a typed artifact reference for a numpy array."""
    path = (
        Path("arrays")
        .joinpath(*[_sanitize_path_part(part) for part in path_parts])
        .with_suffix(".npy")
    )
    reference = ArrayArtifact(
        path=path,
        shape=tuple(int(dimension) for dimension in values.shape),
        dtype=str(values.dtype),
    )
    return PendingArrayArtifact(reference=reference, values=values)


def _serialize_solver_result(
    label: str,
    result: CGComparisonResult,
) -> tuple[SerializedSolverResult, tuple[PendingArrayArtifact, ...]]:
    """Build a typed serialized solver payload and detached arrays."""
    solution_artifact = _array_reference(("results", label, "x"), result.x)
    guess_artifact = _array_reference(("results", label, "initial_guess"), result.initial_guess)
    payload = SerializedSolverResult(
        iterations=result.iterations,
        residual=result.residual,
        residual_abs=result.residual_abs,
        converged=result.converged,
        residual_history_rel=tuple(result.residual_history_rel),
        residual_history_abs=tuple(result.residual_history_abs),
        preconditioner=result.preconditioner,
        exact_error=result.exact_error,
        rhs_norm=result.rhs_norm,
        breakdown=result.breakdown,
        error=result.error,
        x=solution_artifact.reference,
        initial_guess=guess_artifact.reference,
    )
    return payload, (solution_artifact, guess_artifact)


def extract_array_artifacts(
    result: ComparisonResult,
) -> tuple[SerializedComparisonPayload, tuple[PendingArrayArtifact, ...]]:
    """Detach numpy arrays from a typed comparison result."""
    serialized_results: dict[str, SerializedSolverResult] = {}
    pending_arrays: list[PendingArrayArtifact] = []
    for label, entry in result.results.items():
        if not isinstance(entry, CGComparisonResult):
            continue
        serialized_entry, arrays = _serialize_solver_result(label, entry)
        serialized_results[label] = serialized_entry
        pending_arrays.extend(arrays)

    payload = SerializedComparisonPayload(
        summary=result.summary,
        preconditioners=tuple(result.preconditioners),
        condition_numbers=dict(result.condition_numbers),
        plot_paths=result.plot_paths,
        recommendations=result.recommendations,
        results=serialized_results,
    )
    return payload, tuple(pending_arrays)


def serialize_comparison_payload(payload: SerializedComparisonPayload) -> dict[str, Any]:
    """Convert a typed comparison payload to a JSON-ready dict."""
    serialized = to_json_primitive(payload)
    if not isinstance(serialized, dict):
        raise TypeError("Serialized comparison payload must be a mapping.")
    return serialized


def save_numpy_artifacts(
    work_root: Path,
    arrays: tuple[PendingArrayArtifact, ...],
) -> tuple[ArrayArtifact, ...]:
    """Persist detached numpy arrays as binary artifacts."""
    for array_artifact in arrays:
        output_path = work_root / array_artifact.reference.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, array_artifact.values, allow_pickle=False)
    return tuple(array.reference for array in arrays)


def _stage_plot_path(source: Path, work_root: Path) -> Path:
    """Copy a plot into the staged artifacts directory and return its relative path."""
    if not source.exists():
        raise FileNotFoundError(f"Comparison plot does not exist: {source}")
    destination = work_root / "figures" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination.relative_to(work_root)


def _stage_plot_paths(plot_paths: PlotPaths, work_root: Path) -> PlotPaths:
    """Stage comparison plots under work_root/figures and return relative paths."""
    staged = {
        key: _stage_plot_path(path, work_root) for key, path in plot_paths.to_mapping().items()
    }
    return PlotPaths.from_mapping(staged)


def _stage_result_plots(
    result: ComparisonResult,
    work_root: Path,
) -> ComparisonResult:
    """Rewrite result plot paths to staged artifact-relative paths."""
    staged_paths = _stage_plot_paths(result.plot_paths, work_root)
    return replace(result, plot_paths=staged_paths)


def _save_comparison_toml(
    result: ComparisonResult,
    output_path: Path,
) -> None:
    """Save scalar comparison diagnostics to a TOML file."""
    iterations = {name: entry.iterations for name, entry in result.results.items()}
    residuals = {name: entry.residual for name, entry in result.results.items()}
    payload = {
        "condition_number": dict(result.condition_numbers),
        "iterations": iterations,
        "final_residual": residuals,
    }
    with open(output_path, "wb") as fh:
        tomli_w.dump(payload, fh)


def write_comparison_artifacts(
    *,
    result: ComparisonResult,
    work_root: Path,
    comparison_config: Path | None = None,
) -> ComparisonArtifactManifest:
    """Write all structured comparison artifacts to disk."""
    staged_result = _stage_result_plots(result, work_root)
    payload, pending_arrays = extract_array_artifacts(staged_result)
    config_copy: Path | None = None
    if comparison_config is not None:
        config_copy = work_root / "config" / comparison_config.name
        config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(comparison_config, config_copy)
    manifest = ComparisonArtifactManifest(
        comparison_toml=work_root / "comparison.toml",
        comparison_json=work_root / "comparison.json",
        recommendations_json=work_root / "recommendations.json",
        summary_txt=work_root / "summary.txt",
        config_copy=config_copy,
        arrays=save_numpy_artifacts(work_root, pending_arrays),
    )
    _save_comparison_toml(staged_result, manifest.comparison_toml)
    manifest.comparison_json.write_text(
        json.dumps(serialize_comparison_payload(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest.summary_txt.write_text(payload.summary, encoding="utf-8")
    manifest.recommendations_json.write_text(
        json.dumps(to_json_primitive(payload.recommendations), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
