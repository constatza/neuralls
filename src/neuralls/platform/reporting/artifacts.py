"""Typed artifact writing for comparison workflows."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import h5py
import numpy as np
import tomli_w
from numpy.typing import NDArray

from neuralls.domain.solver.cost_metrics import iterations_to_tolerances
from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.cost_fields import COST_FIELD_REGISTRY
from neuralls.domain.solver.models.plot_paths import PlotPaths
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.recommendations import ComparisonRecommendations
from neuralls.domain.solver.models.result import ComparisonResult
from neuralls.domain.solver.models.statistics import (
    AlwaysCheaper,
    BreakEvenAt,
    CheaperOnlyBelow,
    MethodComparison,
    NeverCheaper,
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
        histories_h5: Path to the per-preconditioner iteration-history HDF5 file.
        config_copy: Path to config file copy.
        arrays: Tuple of array artifact references.
    """

    comparison_toml: Path
    comparison_json: Path
    recommendations_json: Path
    summary_txt: Path
    histories_h5: Path
    config_copy: Path | None
    arrays: tuple[ArrayArtifact, ...] = ()


@dataclass(frozen=True)
class PendingArrayArtifact:
    """In-memory numpy artifact waiting to be saved."""

    reference: ArrayArtifact
    values: NDArray[Any]


@dataclass(frozen=True)
class SerializedSolverResult:
    """JSON-safe solver result payload.

    1D iteration histories (`residual_history_rel/abs`,
    `error_history_a_rel`, `error_bound_a_rel`) live in `histories.h5`
    instead of here — a single source of truth for array data, not
    duplicated as inline JSON lists.
    """

    iterations: int
    residual: float
    residual_abs: float | None = None
    converged: bool | None = None
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


_HISTORY_FIELDS: tuple[str, ...] = (
    "residual_history_rel",
    "residual_history_abs",
    "error_history_a_rel",
    "error_bound_a_rel",
    "setup_time_samples",
    "solve_time_samples",
)
"""`CGComparisonResult` fields written to `histories.h5` — 1D arrays (per-iteration
histories and per-repeat timing samples), never inlined as JSON (see
`SerializedSolverResult`)."""


class ScalingLawTomlFields(TypedDict):
    """The measured constants a preconditioner's scaling-law curve is computed from."""

    n_systems: int
    n_converged: int
    setup_cost_total_seconds: float
    total_solve_time_seconds: float
    total_iterations: int
    share_setup: bool
    share_generation: bool
    generation_cost_total_seconds: NotRequired[float]
    solve_time_std_seconds: NotRequired[float]


class ConvergenceTomlFields(TypedDict, total=False):
    """ERT with its bootstrap interval and the converged runs' iteration statistics."""

    ert_seconds: float
    ert_ci_low: float
    ert_ci_high: float
    iterations_mean: float
    iterations_median: float
    iterations_std: float


class ComparisonTomlFields(TypedDict, total=False):
    """A method's comparison with the group's reference, each field present when defined."""

    reference: str
    break_even_systems: float
    speedup_median: float
    speedup_ci_low: float
    speedup_ci_high: float
    wilcoxon_p: float


class ProjectedTomlEntry(ScalingLawTomlFields, ConvergenceTomlFields, ComparisonTomlFields):
    """One preconditioner's scaling-law numbers as written to the projected TOML.

    `total_solve_time_seconds` and `total_iterations` cover every run (failed
    ones at their measured cost); the stds and `iterations_*` describe the
    `n_converged` of `n_systems` converged runs. `ert_seconds` is the pooled
    cost per solved system of COCO (Hansen et al. 2016, arXiv:1605.03560)
    with a bootstrap interval
    `ert_ci_low/high`; `iterations_mean/median/std` describe the converged runs.
    The comparison fields exist for a method with a reference: `reference` names
    it, `break_even_systems` is the solved-system count beyond which the method
    is cheaper (only when the break-even is a finite count), `speedup_*` is the
    median of reference time / method time over systems both solved with its
    bootstrap interval, and `wilcoxon_p` the signed-rank p-value of the log
    speedups (Wilcoxon 1945). Every optional field is omitted when it is
    undefined.
    """


class UnconvergedTomlEntry(TypedDict):
    """A preconditioner that converged on no system: nothing to project, only the count."""

    n_systems: int
    n_converged: int


def projected_toml_payload(
    projected: Mapping[str, ProjectedComparisonResult],
    convergence: Mapping[str, ConvergenceSummary],
    comparisons: Mapping[str, MethodComparison],
) -> dict[str, ProjectedTomlEntry | UnconvergedTomlEntry]:
    """Build one group's scaling-law numbers as a TOML-ready payload.

    Pure scalars per preconditioner — the plotted curve is a deterministic
    function of exactly these fields, so this is the source of truth the PNG
    is derived from, not a separate, possibly-drifting summary of it. Optional
    numbers are omitted rather than written as sentinels. Every preconditioner
    of ``convergence`` appears, so a method that never converged is visible
    rather than silently absent.
    """
    payload: dict[str, ProjectedTomlEntry | UnconvergedTomlEntry] = {}
    for name, summary in convergence.items():
        result = projected.get(name)
        if result is None:
            payload[name] = UnconvergedTomlEntry(
                n_systems=summary.n_systems, n_converged=summary.n_converged
            )
            continue
        comparison = comparisons.get(name)
        payload[name] = ProjectedTomlEntry(
            **_projected_entry(result),
            **_convergence_fields(summary),
            **(_comparison_fields(comparison) if comparison is not None else {}),
        )
    return payload


def _projected_entry(result: ProjectedComparisonResult) -> ScalingLawTomlFields:
    """The scaling-law constants of one projection, optional numbers only when defined."""
    entry = ScalingLawTomlFields(
        n_systems=result.n_systems,
        n_converged=result.n_converged,
        setup_cost_total_seconds=result.setup_cost_total_seconds,
        total_solve_time_seconds=result.total_solve_time_seconds,
        total_iterations=result.total_iterations,
        share_setup=result.setup_sharing.share_setup,
        share_generation=result.setup_sharing.share_generation,
    )
    if result.generation_cost_total_seconds is not None:
        entry["generation_cost_total_seconds"] = result.generation_cost_total_seconds
    if result.solve_time_std_seconds is not None:
        entry["solve_time_std_seconds"] = result.solve_time_std_seconds
    return entry


def _convergence_fields(summary: ConvergenceSummary) -> ConvergenceTomlFields:
    """ERT with its interval and the converged runs' iteration statistics, when defined."""
    fields = ConvergenceTomlFields()
    if summary.ert_seconds is not None:
        fields["ert_seconds"] = summary.ert_seconds
    if summary.ert_interval is not None:
        fields["ert_ci_low"] = summary.ert_interval.low
        fields["ert_ci_high"] = summary.ert_interval.high
    if summary.iterations is not None:
        fields["iterations_mean"] = summary.iterations.mean
        fields["iterations_median"] = summary.iterations.median
        if summary.iterations.std is not None:
            fields["iterations_std"] = summary.iterations.std
    return fields


def _comparison_fields(comparison: MethodComparison) -> ComparisonTomlFields:
    """The reference, break-even count, speedup and Wilcoxon p-value of one method."""
    fields = ComparisonTomlFields(reference=comparison.reference)
    match comparison.break_even:
        case BreakEvenAt(n_systems=n_systems):
            fields["break_even_systems"] = n_systems
        case AlwaysCheaper() | NeverCheaper() | CheaperOnlyBelow():
            pass
    paired = comparison.paired
    if paired is None:
        return fields
    fields["speedup_median"] = paired.median_speedup
    if paired.median_speedup_ci is not None:
        fields["speedup_ci_low"] = paired.median_speedup_ci.low
        fields["speedup_ci_high"] = paired.median_speedup_ci.high
    if paired.wilcoxon_p is not None:
        fields["wilcoxon_p"] = paired.wilcoxon_p
    return fields


def write_projected_toml(
    projected: Mapping[str, ProjectedComparisonResult],
    convergence: Mapping[str, ConvergenceSummary],
    comparisons: Mapping[str, MethodComparison],
    output_path: Path,
) -> None:
    """Write one group's scaling-law numbers to `output_path` as TOML."""
    with open(output_path, "wb") as fh:
        tomli_w.dump(projected_toml_payload(projected, convergence, comparisons), fh)


def write_histories_h5(result: ComparisonResult, output_path: Path) -> None:
    """Write every preconditioner's 1D iteration histories to one HDF5 file.

    One dataset per field, under `/<preconditioner_name>/<field>`, omitting
    any field that's `None` or empty for that preconditioner — mirrors the
    "omit if unmeasured" convention `_plot_cost_barplot` already uses.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5file:
        for name, entry in result.results.items():
            if not isinstance(entry, CGComparisonResult):
                continue
            group = None
            for field_name in _HISTORY_FIELDS:
                values = getattr(entry, field_name)
                if not values:
                    continue
                if group is None:
                    group = h5file.create_group(name.replace("/", "_"))  # "/" would nest
                group.create_dataset(field_name, data=np.asarray(values, dtype=np.float64))


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


def _iterations_to_tolerance_table(result: ComparisonResult) -> dict[str, dict[str, int]]:
    """Per-tolerance, per-preconditioner iterations; a rung a run never reached is omitted."""
    rtol = result.solver_params.params.rtol
    table: dict[str, dict[str, int]] = {}
    for name, entry in result.results.items():
        if not isinstance(entry, CGComparisonResult):
            continue
        for tolerance, iterations in iterations_to_tolerances(entry, rtol).items():
            if iterations is not None:
                table.setdefault(f"{tolerance:.0e}", {})[name] = iterations
    return table


def _save_comparison_toml(
    result: ComparisonResult,
    output_path: Path,
) -> None:
    """Save scalar comparison diagnostics to a TOML file.

    Every field in `COST_FIELD_REGISTRY` is written per preconditioner,
    skipped for one that didn't measure it — the same registry the cost
    bar-charts read from (`composition/comparison/_plots.py`), so a new
    measured cost is added here by extending that one list, not by also
    editing this function.
    """
    iterations = {name: entry.iterations for name, entry in result.results.items()}
    residuals = {name: entry.residual for name, entry in result.results.items()}
    payload: dict[str, Any] = {"iterations": iterations, "final_residual": residuals}
    for spec in COST_FIELD_REGISTRY:
        values = {
            name: value
            for name, entry in result.results.items()
            if (value := spec.extractor(entry)) is not None
        }
        if values:
            payload[spec.field_name] = values
    payload["iterations_to_tolerance"] = _iterations_to_tolerance_table(result)
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
        histories_h5=work_root / "histories.h5",
        config_copy=config_copy,
        arrays=save_numpy_artifacts(work_root, pending_arrays),
    )
    _save_comparison_toml(staged_result, manifest.comparison_toml)
    write_histories_h5(staged_result, manifest.histories_h5)
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
