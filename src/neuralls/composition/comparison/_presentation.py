"""Presentation labels for one resolved comparison run."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.comparison.models import ComparisonPaths, ResolvedComparisonInput


def _path_label(path: Path) -> str:
    """Return a concise label for a source file or directory."""
    return path.stem or path.name


def _rhs_label(paths: ComparisonPaths, resolved_input: ResolvedComparisonInput | None) -> str:
    """Describe the single RHS source selected for a comparison run."""
    if resolved_input is None:
        return _path_label(paths.rhs)

    if resolved_input.rhs_dataset_id is not None:
        return _path_label(Path(resolved_input.rhs_dataset_id))

    source_kind = resolved_input.rhs_source_kind
    source_path = (resolved_input.rhs_source_params or {}).get("path")
    if source_kind is not None and isinstance(source_path, str | Path):
        return f"{source_kind}:{_path_label(Path(source_path))}"
    if source_kind is not None:
        return str(source_kind)
    return _path_label(paths.rhs)


def build_comparison_source_context(
    paths: ComparisonPaths,
    resolved_input: ResolvedComparisonInput | None,
) -> str:
    """Build the canonical matrix/RHS label shared by logs and plots."""
    matrix_label = (
        resolved_input.matrix_dataset_id
        if resolved_input is not None
        else _path_label(paths.matrix)
    )
    return f"matrix={matrix_label} | rhs={_rhs_label(paths, resolved_input)}"


def build_comparison_plot_title(context: str, system_size: int | None) -> str:
    """Add the system size to the canonical comparison context for plot titles."""
    if system_size is None:
        return context
    return f"{context}\nN={system_size}"
