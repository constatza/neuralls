"""Convergence-history curves of solver results, one figure per set of methods."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from loguru import logger

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_text import (
    LEGEND_FONT_SIZE,
    FigureTitle,
    Legend,
    ScopeSubtitle,
    draw_legend,
    title_text,
)
from neuralls.platform.reporting.figure_writer import finish_group_figure, use_headless_backend
from neuralls.platform.reporting.series_style import resolve_styles

DEFAULT_LINE_MARKER_SIZE = 2.0
DEFAULT_SCATTER_MARKER_AREA = 4.0


def _meta_value(meta: Any, key: str) -> Any:
    """Extract a value from metadata by key, supporting both dicts and objects.

    Args:
        meta: Metadata as a ``Mapping`` or object with attributes.
        key: Key/attribute name to look up.

    Returns:
        The value for ``key``, or ``None`` if not found.
    """
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None)


def _build_convergence_label(method_name: str, meta: Any | None) -> str:
    """Build a legend label for a convergence plot entry.

    Adds trace training/applied iteration counts and neural limit metadata
    when present. Non-neural methods never get a ``limit`` annotation.

    Args:
        method_name: Base method name used as the label prefix.
        meta: Optional metadata mapping or object. Recognised keys:
            - ``residual_iters``: Training iteration count.
            - ``applied_iters``: Applied iteration count (falls back to
              ``residual_iters`` when ``None``).
            - ``preconditioner_type``: Preconditioner type string.
            - ``limit_iters``: Iteration limit (shown for neural only).

    Returns:
        Human-readable legend label string.
    """
    if meta is None:
        return method_name
    details: list[str] = []
    residual_iters = _meta_value(meta, "residual_iters")
    applied_iters = _meta_value(meta, "applied_iters") or residual_iters
    if residual_iters is not None:
        details.extend([f"tr={residual_iters}", f"ap={applied_iters}"])
    preconditioner_type = _meta_value(meta, "preconditioner_type")
    limit_iters = _meta_value(meta, "limit_iters")
    if (
        str(preconditioner_type).lower() == "neural"
        and isinstance(limit_iters, int)
        and limit_iters >= 0
    ):
        details.append(f"limit={limit_iters}")
    if not details:
        return method_name
    return f"{method_name} ({', '.join(details)})"


@dataclass(frozen=True, slots=True)
class ConvergenceCurve:
    """One method's per-iteration history on one plotted metric."""

    values: tuple[float, ...]
    converged: bool


_UNCONVERGED_ALPHA = 0.6


def _curve_from_result(
    result: CGComparisonResult | Mapping[str, Any], history_attr: str
) -> ConvergenceCurve | None:
    """The history of ``result`` as a curve, or ``None`` when it holds none."""
    if isinstance(result, CGComparisonResult):
        history = getattr(result, history_attr)
        converged = result.converged
    elif isinstance(result, Mapping):
        history = result.get(history_attr)
        if history_attr == "residual_history_rel":
            history = history or result.get("residuals")
        converged = bool(result.get("converged", True))
    else:
        return None
    if history is None or len(history) == 0:
        return None
    return ConvergenceCurve(tuple(float(value) for value in history), converged)


def plot_convergence_comparison(
    results: Mapping[str, CGComparisonResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    target: FigureTarget | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    subtitle: ScopeSubtitle | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
    threshold_rel: float | None = None,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    history_attr: str = "residual_history_rel",
    ylabel: str = "Relative Residual $\\|r\\| / \\|b\\|$",
    *,
    legend: Legend,
    show_titles: bool = False,
) -> None:
    """Plot the ``history_attr`` histories of solver results on one figure.

    Adapter over ``plot_convergence_curves``; a result without a history is
    skipped with a warning. ``metadata`` maps a result key to optional
    training annotations appended to its legend label.
    """
    curves: dict[str, ConvergenceCurve] = {}
    for method_name, result in results.items():
        curve = _curve_from_result(result, history_attr)
        if curve is None:
            logger.warning(f"Method '{method_name}' has no residual history to plot")
            continue
        curves[method_name] = curve
    plot_convergence_curves(
        curves,
        legend=legend,
        target=target,
        show=show,
        title=title,
        subtitle=subtitle,
        rtol=rtol,
        atol=atol,
        max_iterations=max_iterations,
        threshold_rel=threshold_rel,
        families=families,
        color_keys=color_keys,
        marker_keys=marker_keys,
        marker_size=marker_size,
        ylabel=ylabel,
        metadata=metadata,
        show_titles=show_titles,
    )


def plot_convergence_curves(
    curves: Mapping[str, ConvergenceCurve],
    *,
    legend: Legend,
    target: FigureTarget | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    subtitle: ScopeSubtitle | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
    threshold_rel: float | None = None,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    ylabel: str = "Relative Residual $\\|r\\| / \\|b\\|$",
    metadata: Mapping[str, Any] | None = None,
    show_titles: bool = False,
) -> None:
    """Plot one semilog curve per key of ``curves``.

    Args:
        curves: Per-iteration values by series key; a curve that did not converge is
            drawn translucent.
        legend: Text of every plotted series, keyed like ``curves``; a curve without an
            entry raises.
        target: Where to save the plot, one file per figure format.
        show: Whether to show plot.
        title: Optional title for the plot.
        subtitle: Optional scope shown below the core metric title, rendered here.
        rtol: Optional relative tolerance parameter to display.
        atol: Optional absolute tolerance parameter to display.
        max_iterations: Optional max iterations parameter to display.
        threshold_rel: Optional relative residual at which the solver stops, drawn as a
            dashed horizontal line.
        families: Optional plot-style family per key; same-family lines share a
            linestyle.
        color_keys: Optional color-axis key per series; series sharing a key share a
            color. Colors come from the Okabe-Ito palette, extended with generated hues.
        marker_keys: Optional marker-axis key per series; series sharing a key share a
            marker.
        marker_size: Marker diameter for convergence-history points.
        ylabel: Y-axis label.
        metadata: Optional per-key training annotations appended to the legend label.
        show_titles: Whether the figure carries its title and subtitle.
    """
    use_headless_backend()
    fig, ax = plt.subplots(figsize=(10, 6))
    metadata = dict(metadata or {})
    line_styles = resolve_styles(families, color_keys, marker_keys) if families else {}

    for key, curve in curves.items():
        label = _build_convergence_label(legend.label(key), metadata.get(key))
        style = dict(line_styles.get(key, {"marker": "o", "linestyle": "-"}))
        size_scale = style.pop("markersize_scale", 1.0)
        if not curve.converged:
            style["alpha"] = _UNCONVERGED_ALPHA
        ax.semilogy(
            range(len(curve.values)),
            curve.values,
            label=label,
            markersize=marker_size * size_scale,
            **style,
        )

    if threshold_rel is not None:
        ax.axhline(threshold_rel, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            f"stopping threshold {threshold_rel:.1e}",
            xy=(1.0, threshold_rel),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            fontsize=LEGEND_FONT_SIZE,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)

    # Build subtitle from non-None parameters
    subtitle_parts = [subtitle.text] if subtitle is not None else []
    if rtol is not None:
        subtitle_parts.append(f"rtol={rtol:.0e}")
    if atol is not None:
        subtitle_parts.append(f"atol={atol:.0e}")
    if max_iterations is not None:
        subtitle_parts.append(f"maxiter={max_iterations}")

    ax.grid(True, alpha=0.3)
    draw_legend(ax, legend, inside_loc="upper right")

    finish_group_figure(
        fig,
        ax,
        title=title_text(title) if title is not None else "Convergence Comparison",
        subtitle=", ".join(subtitle_parts),
        show_titles=show_titles,
        target=target,
        show=show,
        saved_message="Saved convergence comparison plot to",
    )


def plot_error_convergence_comparison(
    results: Mapping[str, CGComparisonResult | Mapping[str, Any]],
    **kwargs: Any,
) -> None:
    """Plot relative energy-norm error ``||e_k||_A / ||e_0||_A`` per preconditioner.

    Thin wrapper over ``plot_convergence_comparison`` reading
    ``error_history_a_rel``; accepts the same keyword arguments.

    Args:
        results (Mapping[str, CGComparisonResult | Mapping[str, Any]]): Results by method name.
        **kwargs (Any): Forwarded to ``plot_convergence_comparison``.
    """
    plot_convergence_comparison(
        results,
        history_attr="error_history_a_rel",
        ylabel="Relative energy error $\\|e_k\\|_A / \\|e_0\\|_A$",
        **kwargs,
    )
