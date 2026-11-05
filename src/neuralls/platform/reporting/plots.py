"""Plotting utilities for graph-cg project."""

from __future__ import annotations

import colorsys
import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.platform.reporting.figure_formats import FigureTarget
from neuralls.platform.reporting.figure_text import (
    FigureTitle,
    Legend,
    ScopeSubtitle,
    draw_legend,
    title_text,
)

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


_LINESTYLE_POOL: tuple[str | tuple[float, tuple[float, ...]], ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
)
"""Assigned per family, most-conventional-first (matches matplotlib's own default
linestyle cycle): solid/dashed/dashdot/dotted, then two dense variants for overflow."""

_MARKER_POOL: dict[str, float] = {
    "o": 1.0,
    ".": 1.6,
    "s": 1.0,
    "^": 1.1,
    "D": 1.1,
    "v": 1.1,
    "<": 1.1,
    ">": 1.1,
    "p": 1.0,
    "*": 1.3,
    "h": 1.0,
    "X": 0.9,
    "P": 0.9,
    "+": 1.4,
    "x": 1.4,
    "1": 1.5,
    "2": 1.5,
    "3": 1.5,
    "4": 1.5,
    "|": 1.3,
    "_": 1.3,
    "d": 1.2,
    "H": 1.0,
}
"""Marker shape -> `marker_size` multiplier, in assignment order. First 13 keys
are solid shapes (dot included — it reads fine on its own, unlike the
overlapping thin glyphs below); the remaining 10 line-drawn markers (+, x,
1-4, |, _) are deprioritized only because they're easy to confuse with each
other in a dense legend, used as an overflow tier once more than 13 distinct
values need a marker. The multiplier balances visual weight: matplotlib draws
every marker glyph in the same bounding box, but thin glyphs read much
smaller than solid ones at an identical `markersize`, while bulky filled ones
(X, P) read larger — 1.0 means no correction needed."""

_OKABE_ITO_HEX: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)
"""Okabe-Ito colorblind-safe categorical palette (Wong, Nature Methods 2011) —
the standard qualitative palette for scientific figures. Used for the first 8
distinct color keys in a plot; see `_generate_hues` for overflow beyond that."""


def _generate_hues(count: int) -> list[tuple[float, float, float]]:
    """Generate `count` evenly spaced, fixed-lightness/saturation hues.

    Used only once a plot needs more distinct colors than the standard
    Okabe-Ito palette provides (8). Separation between hues is `360/count`
    degrees regardless of how large `count` gets, so this pool never
    "depletes" the way a fixed named palette does.

    Args:
        count: Number of distinct hues to generate.

    Returns:
        list[tuple[float, float, float]]: RGB triples in [0, 1].
    """
    return [colorsys.hls_to_rgb(i / count, 0.5, 0.65) for i in range(count)]


def _color_pool(count: int) -> list[Any]:
    """Resolve `count` distinct colors: Okabe-Ito first, generated hues beyond 8.

    Args:
        count: Number of distinct colors needed.

    Returns:
        list[Any]: Matplotlib-compatible colors, length `count`.
    """
    if count <= len(_OKABE_ITO_HEX):
        return list(_OKABE_ITO_HEX[:count])
    return list(_OKABE_ITO_HEX) + _generate_hues(count - len(_OKABE_ITO_HEX))


def _assign_axis(keys: Mapping[str, Hashable], palette: Sequence[Any]) -> dict[str, Any]:
    """Assign each distinct key a palette slot, in first-seen order, cycling on overflow.

    Args:
        keys: Axis key per method name (plot label).
        palette: Ordered pool of visual codes (linestyles, markers, or colors)
            to draw from.

    Returns:
        dict[str, Any]: Method name -> palette value.
    """
    unique_keys = dict.fromkeys(keys.values())
    palette_by_key = dict(zip(unique_keys, itertools.cycle(palette)))
    return {name: palette_by_key[key] for name, key in keys.items()}


def _resolve_styles(
    families: Mapping[str, Hashable],
    color_keys: Mapping[str, Hashable] | None,
    marker_keys: Mapping[str, Hashable] | None,
) -> dict[str, dict[str, Any]]:
    """Build a per-method matplotlib style dict from three independent axes.

    Linestyle always follows `family` (lowest-cardinality attribute, fewest
    legible linestyles). Marker and color follow `marker_keys`/`color_keys`
    when given (falling back to `family` per method when a method has no
    entry), so callers can make e.g. same-dataset lines share a color and
    same-weighting-scheme lines share a marker within one family, instead of
    every same-family line collapsing onto one marker/linestyle pair.

    Any group of methods left sharing an identical (linestyle, marker, color)
    triple — the case when no override keys are given at all — gets its color
    spread across a lightness ramp so it still stays visually distinguishable.

    Args:
        families: Family key per method name (plot label).
        color_keys: Optional color-axis key per method name.
        marker_keys: Optional marker-axis key per method name.

    Returns:
        dict[str, dict[str, Any]]: Method name -> ``{"marker", "linestyle", "color"}``.
    """
    color_keys = color_keys or {}
    marker_keys = marker_keys or {}
    effective_color_keys = {name: color_keys.get(name, family) for name, family in families.items()}
    effective_marker_keys = {
        name: marker_keys.get(name, family) for name, family in families.items()
    }

    linestyles = _assign_axis(families, _LINESTYLE_POOL)
    markers = _assign_axis(effective_marker_keys, list(_MARKER_POOL))
    n_colors = len(dict.fromkeys(effective_color_keys.values()))
    colors = _assign_axis(effective_color_keys, _color_pool(max(n_colors, 1)))

    styles: dict[str, dict[str, Any]] = {
        name: {"marker": markers[name], "linestyle": linestyles[name], "color": colors[name]}
        for name in families
    }

    collision_groups: dict[tuple[Any, Any, Any], list[str]] = {}
    for name, style in styles.items():
        triple = (style["linestyle"], style["marker"], tuple(np.asarray(style["color"]).flat))
        collision_groups.setdefault(triple, []).append(name)
    for members in collision_groups.values():
        if len(members) <= 1:
            continue
        rgb: Any = matplotlib.colors.to_rgb(styles[members[0]]["color"])
        hue, _, saturation = colorsys.rgb_to_hls(*rgb)
        for member, lightness in zip(members, np.linspace(0.7, 0.35, len(members))):
            styles[member]["color"] = colorsys.hls_to_rgb(hue, lightness, saturation)

    for name, style in styles.items():
        style["markersize_scale"] = _MARKER_POOL.get(style["marker"], 1.0)
    return styles


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
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    metadata = dict(metadata or {})
    line_styles = _resolve_styles(families, color_keys, marker_keys) if families else {}

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
            fontsize=8,
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


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    *,
    metric_name: str,
    target: FigureTarget | None = None,
    show: bool = False,
    title: str | None = None,
    horizontal: bool = False,
) -> None:
    """Bar chart comparing a single metric across labelled experiments.

    Each bar is labelled with text the caller rendered from a presentation (see
    ``plot_method_bars``); the labels must be unique because they key the bar colours.

    Args:
        labels: Rendered bar labels. Must be unique.
        values: Metric value for each corresponding label.
        metric_name: Human-readable metric description used as y-axis label
            and figure title (e.g. ``"Mean Absolute Error (eval/mae)"``).
        target: Where to save the figure, one file per figure format. Parent directories are created
            automatically if they do not exist.
        show: Whether to call ``plt.show()`` (disabled by default for
            headless / batch use).
        title: Optional title for the plot. Defaults to "Metric Comparison: {metric_name}".
        horizontal: If True, use horizontal bars (barh) and swap axis labels.
    """
    label_lines = sum(label.count("\n") + 1 for label in labels)
    if horizontal:
        fig, ax = plt.subplots(figsize=(5, max(3, label_lines * 0.6)))
    else:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    order = [
        label for label, _ in sorted(zip(labels, values), key=lambda pair: pair[1], reverse=True)
    ]
    colormap = matplotlib.colormaps["Set1"]
    colors = colormap(np.linspace(0, 0.9, max(len(labels), 1)))
    palette = dict(zip(labels, colors))
    bar_kwargs = {
        "order": order,
        "hue": labels,
        "palette": palette,
        "legend": False,
        "alpha": 0.85,
        "edgecolor": "black",
        "linewidth": 0.7,
        "ax": ax,
    }

    if horizontal:
        sns.barplot(x=values, y=labels, **bar_kwargs)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel("Experiment", fontsize=12)
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        sns.barplot(x=labels, y=values, **bar_kwargs)
        ax.tick_params(axis="x", labelsize=11, rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.set_xlabel("Experiment", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    finish_group_figure(
        fig,
        ax,
        title=title or f"Metric Comparison: {metric_name}",
        subtitle=None,
        target=target,
        show=show,
        saved_message="Saved metric comparison plot to",
    )


def plot_method_bars(
    legend: Legend,
    names: Sequence[str],
    values: Sequence[float],
    *,
    metric_name: str,
    title: FigureTitle | None = None,
    horizontal: bool = False,
    target: FigureTarget | None = None,
) -> None:
    """Bar chart of one metric per method, each bar labelled by its legend entry.

    The bar labels are rendered from ``legend`` here; a name without an entry raises.
    """
    plot_metric_comparison(
        [legend.label(name) for name in names],
        list(values),
        metric_name=metric_name,
        title=title_text(title) if title is not None else None,
        horizontal=horizontal,
        target=target,
    )


def group_series_styles(
    names: Sequence[str],
    families: Mapping[str, Hashable] | None,
    color_keys: Mapping[str, Hashable] | None,
    marker_keys: Mapping[str, Hashable] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve per-preconditioner styles for the group-level (cross-system) plots.

    Uses the same resolution as ``plot_convergence_comparison`` so a
    preconditioner keeps one colour and marker across every comparison plot.
    Without ``families`` each name is its own family: still Okabe-Ito and
    distinct, just not shared with the convergence plots.

    Args:
        names: Preconditioner names to style, in legend order.
        families: Optional family per name (as the convergence plots receive).
        color_keys: Optional colour-axis key per name.
        marker_keys: Optional marker-axis key per name.

    Returns:
        dict[str, dict[str, Any]]: Name -> style with ``marker``, ``linestyle``,
        ``color`` and ``markersize_scale``.
    """
    effective_families = families or {name: name for name in names}
    return _resolve_styles(effective_families, color_keys, marker_keys)


def finish_group_figure(
    fig: Any,
    ax: Any,
    *,
    title: str,
    subtitle: str | None,
    target: FigureTarget | None,
    show: bool,
    saved_message: str,
) -> None:
    """Title, save and close/show a figure; the only place a report figure is written."""
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if subtitle:
        ax.set_title(subtitle, fontsize=9)
    fig.tight_layout()

    if target is not None:
        target.directory.mkdir(parents=True, exist_ok=True)
        for path in target.paths:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            logger.info(f"{saved_message}: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
