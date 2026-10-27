"""Plotting utilities for graph-cg project."""

from __future__ import annotations

import colorsys
import itertools
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from neuralls.domain.solver.models.cg_result import CGComparisonResult
from neuralls.domain.solver.models.cost import StageCost
from neuralls.platform.reporting.figure_text import (
    FigureTitle,
    Legend,
    ScopeSubtitle,
    draw_legend,
    title_text,
)
from neuralls.shared.types import CostProvenance

DEFAULT_LINE_MARKER_SIZE = 2.0
DEFAULT_SCATTER_MARKER_AREA = 4.0


def plot_residual_history(
    results: dict[str, dict[str, Any]],
    save_path: str | Path | None = None,
    show: bool = False,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
) -> None:
    """Plot residual history for different methods.

    Args:
        results: Dictionary of method results with 'residuals' key
        save_path: Path to save plot
        show: Whether to show plot
        marker_size: Marker diameter for residual-history points
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, result in results.items():
        # Handle both dict and dataclass results
        residuals = None
        if hasattr(result, "residual_history_rel"):
            # getattr, not direct access: result's declared type doesn't
            # include this attribute (only some dataclass variants have it,
            # guarded dynamically by hasattr) — direct access defeats ty's
            # hasattr-narrowing and produces a spurious type error.
            residuals = getattr(result, "residual_history_rel")  # noqa: B009
        elif isinstance(result, dict):
            residuals = result.get("residual_history_rel") or result.get("residuals")
        else:
            residuals = None

        if residuals:
            iterations = range(len(residuals))
            ax.semilogy(iterations, residuals, "o-", label=method_name, markersize=marker_size)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    ax.set_title("Convergence History (Relative)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved residual history plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


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


def plot_convergence_comparison(
    results: Mapping[str, CGComparisonResult | Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    subtitle: ScopeSubtitle | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    history_attr: str = "residual_history_rel",
    ylabel: str = "Relative Residual $\\|r\\| / \\|b\\|$",
    *,
    legend: Legend,
) -> None:
    """Plot convergence comparison between preconditioners.

    Args:
        results: Dictionary of method results
        metadata: Optional metadata dict mapping method name -> NeuralPreconditionerMetadata
        save_path: Path to save plot
        show: Whether to show plot
        title: Optional title for the plot
        subtitle: Optional scope shown below the core metric title, rendered here.
        rtol: Optional relative tolerance parameter to display
        atol: Optional absolute tolerance parameter to display
        max_iterations: Optional max iterations parameter to display
        families: Optional plot-style family per method name (see
            ``preconditioner_family.preconditioner_family``). Drives linestyle:
            same-family lines always share a linestyle.
        color_keys: Optional color-axis key per method name (e.g. a POD-2G
            fit dataset). Methods sharing a key share a color; methods
            omitted here fall back to their family. Colors come from the
            Okabe-Ito palette, extended with generated hues beyond 8 keys.
        marker_keys: Optional marker-axis key per method name (e.g. a POD-2G
            weighting scheme). Methods sharing a key share a marker; methods
            omitted here fall back to their family.
        marker_size: Marker diameter for convergence-history points.
        history_attr: Per-iteration history field to plot, read from each result.
        ylabel: Y-axis label matching ``history_attr``.
        legend: Text of every plotted series, keyed by result key. Result identity and
            style lookup always remain keyed by ``results``; a result without an entry
            raises.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    metadata = dict(metadata or {})
    line_styles = _resolve_styles(families, color_keys, marker_keys) if families else {}

    for method_name, result in results.items():
        # Handle both dict and dataclass results
        if isinstance(result, CGComparisonResult):
            residuals = getattr(result, history_attr)
        elif isinstance(result, Mapping):
            residuals = result.get(history_attr)
            if history_attr == "residual_history_rel":
                residuals = residuals or result.get("residuals")
        else:
            residuals = None

        if residuals and len(residuals) > 0:
            iterations = range(len(residuals))

            label = _build_convergence_label(legend.label(method_name), metadata.get(method_name))
            style = dict(line_styles.get(method_name, {"marker": "o", "linestyle": "-"}))
            size_scale = style.pop("markersize_scale", 1.0)

            ax.semilogy(
                iterations, residuals, label=label, markersize=marker_size * size_scale, **style
            )
        else:
            # Log warning for methods with no history
            logger.warning(f"Method '{method_name}' has no residual history to plot")

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

    if subtitle_parts:
        ax.set_title(", ".join(subtitle_parts), fontsize=9)

    fig.suptitle(
        title_text(title) if title is not None else "Convergence Comparison",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    draw_legend(ax, legend, inside_loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved convergence comparison plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


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


def plot_error_bound_comparison(
    results: Mapping[str, CGComparisonResult | Mapping[str, Any]],
    **kwargs: Any,
) -> None:
    """Plot the Golub-Meurant lower bound on ``||e_k||_A / ||e_0||_A`` per preconditioner.

    Thin wrapper over ``plot_convergence_comparison`` reading
    ``error_bound_a_rel`` — the fallback energy-error signal used when the exact
    value (``error_history_a_rel``) is unavailable (no reference solution);
    accepts the same keyword arguments.

    Args:
        results (Mapping[str, CGComparisonResult | Mapping[str, Any]]): Results by method name.
        **kwargs (Any): Forwarded to ``plot_convergence_comparison``.
    """
    plot_convergence_comparison(
        results,
        history_attr="error_bound_a_rel",
        ylabel="Energy-error lower bound $\\|e_k\\|_A / \\|e_0\\|_A$ (Golub–Meurant)",
        **kwargs,
    )


def plot_work_precision(
    results: Mapping[str, CGComparisonResult],
    *,
    legend: Legend,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    marker_size: float = DEFAULT_SCATTER_MARKER_AREA,
) -> None:
    """Work-precision diagram: final achieved precision vs. solve time.

    The standard cost/accuracy tradeoff plot from the iterative-solver
    benchmarking literature (log-log, one point per method) — distinct from
    ``plot_convergence_comparison``, which plots precision against iteration
    *index* rather than measured solve time. Offline generation and setup costs
    are intentionally excluded from this solve-only diagnostic.

    Args:
        results: Dictionary of method results.
        legend: Text of every plotted series, keyed by result key.
        families: Optional plot-style family per method name — see
            ``plot_convergence_comparison``.
        color_keys: Optional color-axis key per method name.
        marker_keys: Optional marker-axis key per method name.
        save_path: Path to save plot.
        show: Whether to call ``plt.show()``.
        title: Optional title for the plot.
        marker_size: Scatter marker area.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    line_styles = _resolve_styles(families, color_keys, marker_keys) if families else {}

    for method_name, result in results.items():
        solve_time = result.solve_time_seconds
        if solve_time is None or solve_time <= 0 or result.residual <= 0:
            logger.warning(f"Method '{method_name}' has no cost data to plot on work-precision.")
            continue

        style = dict(
            line_styles.get(method_name, {"marker": "o", "linestyle": "None", "color": None})
        )
        style["linestyle"] = "None"
        size_scale = style.pop("markersize_scale", 1.0)

        ax.scatter(
            [solve_time],
            [result.residual],
            label=legend.label(method_name),
            s=marker_size * 20 * size_scale,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Solve time (s)")
    ax.set_ylabel("Final Relative Residual $\\|r\\| / \\|b\\|$")
    plot_title = (
        f"{title_text(title)}\nSolve-Time Work-Precision"
        if title is not None
        else "Solve-Time Work-Precision Diagram"
    )
    fig.suptitle(plot_title, fontsize=13, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    draw_legend(ax, legend, inside_loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved work-precision plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _method_wall_times(
    result: CGComparisonResult, n_points: int, *, solve_only: bool = False
) -> list[float]:
    """Time axis, in seconds on the method's *own* clock, for an `n_points` history.

    Every method is timed from its own start: generation + setup is the offset
    at which its first history point sits, and the solve is spread uniformly
    over the remaining intervals. Methods are never chained one after another,
    so a curve's position depends only on that method's cost, not on the order
    methods happened to run in.

    A history holds the initial value plus one entry per iteration
    (`n_points = iterations + 1`), so the solve covers `n_points - 1` intervals
    and the last point lands exactly on generation + setup + solve — the same
    total that the work-precision plot places this method at.

    Uniform spacing is an assumption: per-iteration wall time is not recorded,
    so an iteration-varying preconditioner is drawn smoothed.

    With ``solve_only`` the offset is dropped, so every method starts at t=0
    and curves compare convergence speed alone (setup/generation are shown by
    the time-breakdown and work-precision plots).
    """
    offset = 0.0
    if not solve_only:
        offset = result.setup_cost
        if (
            result.generation_cost is not None
            and result.generation_cost.provenance is not CostProvenance.UNAVAILABLE
        ):
            offset += result.generation_cost.wall_time_seconds
    solve = result.solve_time_seconds or 0.0
    n_intervals = max(n_points - 1, 1)
    return [offset + solve * i / n_intervals for i in range(n_points)]


def plot_residual_vs_time(
    results: Mapping[str, CGComparisonResult],
    legend: Legend,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    solve_only: bool = False,
) -> None:
    """Convergence curves in wall time (not iteration): residual vs cumulative cost.

    Complements iteration-based convergence curves by showing how residual
    decreases as actual wall time accumulates (generation + setup + solve).
    Accounts for setup cost overhead and per-iteration speed variation.

    Each method is drawn on its own clock (see ``_method_wall_times``): its
    curve starts at (generation + setup) and increases linearly through the
    solve, ending at the method's total time. Methods are not chained.

    Args:
        results: Dictionary of method results.
        legend: Text of every plotted series, keyed by result key.
        families: Optional plot-style family per method name.
        color_keys: Optional color-axis key per method name.
        marker_keys: Optional marker-axis key per method name.
        save_path: Path to save plot.
        show: Whether to call ``plt.show()``.
        title: Optional title for the plot.
        marker_size: Marker diameter for convergence points.
        solve_only: Start every curve at t=0 on a linear axis, excluding
            generation and setup time (see ``_method_wall_times``).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = _resolve_styles(families, color_keys, marker_keys) if families else {}

    for method_name, result in results.items():
        if not result.residual_history_rel or result.solve_time_seconds is None:
            logger.warning(f"Method '{method_name}' has no residual history or solve time to plot.")
            continue

        residuals = result.residual_history_rel
        if len(residuals) == 0:
            continue

        wall_times = _method_wall_times(result, len(residuals), solve_only=solve_only)

        style = dict(line_styles.get(method_name, {"marker": "o", "linestyle": "-", "color": None}))
        size_scale = style.pop("markersize_scale", 1.0)

        ax.semilogy(
            wall_times,
            residuals,
            label=legend.label(method_name),
            markersize=marker_size * size_scale,
            **style,
        )

    if solve_only:
        ax.set_xlim(left=0.0)
        ax.set_xlabel("Solve time [s]")
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Wall time: generation + setup + solve [s]")
    ax.set_ylabel("Relative Residual $\\|r\\| / \\|b\\|$")
    fig.suptitle(
        title_text(title) if title is not None else "Convergence vs Wall Time",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    draw_legend(ax, legend, inside_loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved residual vs time plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_error_vs_time(
    results: Mapping[str, CGComparisonResult],
    legend: Legend,
    families: Mapping[str, Hashable] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    solve_only: bool = False,
) -> None:
    """Energy-norm error convergence vs wall time (not iteration).

    Complements iteration-based error convergence by showing how actual error
    (relative energy norm ||e||_A / ||e_0||_A) decreases as wall time accumulates.
    Only plots methods with available error_history_a_rel.

    Each method is drawn on its own clock (see ``_method_wall_times``): its
    curve starts at (generation + setup) and increases linearly through the
    solve, ending at the method's total time. Methods are not chained.

    Args:
        results: Dictionary of method results.
        legend: Text of every plotted series, keyed by result key.
        families: Optional plot-style family per method name.
        color_keys: Optional color-axis key per method name.
        marker_keys: Optional marker-axis key per method name.
        save_path: Path to save plot.
        show: Whether to call ``plt.show()``.
        title: Optional title for the plot.
        marker_size: Marker diameter for convergence points.
        solve_only: Start every curve at t=0 on a linear axis, excluding
            generation and setup time (see ``_method_wall_times``).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = _resolve_styles(families, color_keys, marker_keys) if families else {}

    for method_name, result in results.items():
        if not result.error_history_a_rel or result.solve_time_seconds is None:
            logger.debug(
                f"Method '{method_name}' has no energy-norm error history to plot vs time."
            )
            continue

        errors = result.error_history_a_rel
        if len(errors) == 0:
            continue

        wall_times = _method_wall_times(result, len(errors), solve_only=solve_only)

        style = dict(line_styles.get(method_name, {"marker": "o", "linestyle": "-", "color": None}))
        size_scale = style.pop("markersize_scale", 1.0)

        ax.semilogy(
            wall_times,
            errors,
            label=legend.label(method_name),
            markersize=marker_size * size_scale,
            **style,
        )

    if solve_only:
        ax.set_xlim(left=0.0)
        ax.set_xlabel("Solve time [s]")
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Wall time: generation + setup + solve [s]")
    ax.set_ylabel("Relative Energy-Norm Error $\\|e_k\\|_A / \\|e_0\\|_A$")
    fig.suptitle(
        title_text(title) if title is not None else "Error Convergence vs Wall Time",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    draw_legend(ax, legend, inside_loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved error vs time plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _extract_iterations(result: Mapping[str, int]) -> int:
    """Read `iterations` off one noise-robustness result entry.

    Args:
        result: One method's result at one noise level.

    Returns:
        The result's iteration count, or 0 if absent.
    """
    return result.get("iterations", 0)


def plot_noise_robustness(
    noise_results: Mapping[str, Mapping[str, Mapping[str, int]]],
    save_path: str | Path | None = None,
    show: bool = False,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
) -> None:
    """Plot noise robustness analysis results.

    Args:
        noise_results: Nested mapping: noise_level -> method -> result, each
            result at minimum carrying an ``"iterations"`` count.
        save_path: Optional path to save the plot
        show: Whether to show plot
        marker_size: Marker diameter for noise-series points
    """
    sorted_levels = sorted(noise_results.items(), key=lambda item: float(item[0]))
    methods: set[str] = set()
    for level_results in noise_results.values():
        methods.update(level_results.keys())
    method_names = sorted(methods)

    # Use matplotlib Set1 colormap for consistent colors
    colormap = matplotlib.colormaps["Set1"]
    colors = colormap(np.linspace(0, 1, max(len(method_names), 3)))
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(method_names)}

    fig = plt.figure(figsize=(12, 8))

    for method in method_names:
        iterations = []
        levels_numeric = []

        for level, level_results in sorted_levels:
            if method not in level_results:
                continue
            iterations.append(_extract_iterations(level_results[method]))
            levels_numeric.append(float(level))

        if iterations:
            plt.plot(
                levels_numeric,
                iterations,
                "o-",
                label=method,
                color=method_colors[method],
                linewidth=2,
                markersize=marker_size,
            )

    plt.xlabel("Noise Level (%)")
    plt.ylabel("CG Iterations to Convergence")
    plt.title("Noise Robustness Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    *,
    metric_name: str,
    legend: Mapping[str, str] | None = None,
    save_path: Path | None = None,
    show: bool = False,
    title: str | None = None,
    horizontal: bool = False,
) -> None:
    """Bar chart comparing a single metric across labelled experiments.

    Experiments are identified by short labels (e.g. "1", "2", "3") on the
    x-axis or y-axis (when horizontal). When ``legend`` is provided it is embedded
    below the plot as a text block so the full experiment identity is preserved
    without cluttering the axis.

    Args:
        labels: Short axis labels for each bar (e.g. ``["1", "2", "3"]``). Must
            be unique — ``legend`` (when given) is keyed by these same values.
        values: Metric value for each corresponding label.
        metric_name: Human-readable metric description used as y-axis label
            and figure title (e.g. ``"Mean Absolute Error (eval/mae)"``).
        legend: Optional mapping from label to full experiment description
            (e.g. ``{"1": "ffnn_test_solutions (run: abc123…)"}``) embedded
            below the chart.
        save_path: Path to save the figure. Parent directories are created
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

    ax.set_title(title or f"Metric Comparison: {metric_name}", fontsize=13, fontweight="bold")

    if legend:
        legend_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(legend.items()))
        legend_text = f"Legend:\n{legend_lines}"
        fig.text(
            0.01,
            -0.05,
            legend_text,
            fontsize=8,
            fontfamily="monospace",
            va="top",
            wrap=True,
        )
        fig.subplots_adjust(bottom=0.05 + 0.03 * len(legend))

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved metric comparison plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_method_bars(
    legend: Legend,
    names: Sequence[str],
    values: Sequence[float],
    *,
    metric_name: str,
    title: FigureTitle | None = None,
    horizontal: bool = False,
    save_path: Path | None = None,
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
        save_path=save_path,
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
    effective_families = families if families else {name: name for name in names}
    return _resolve_styles(effective_families, color_keys, marker_keys)


def finish_group_figure(
    fig: Any,
    ax: Any,
    *,
    title: str,
    subtitle: str | None,
    save_path: Path | None,
    show: bool,
    saved_message: str,
) -> None:
    """Title, save and close/show a group-level figure (shared by both plots)."""
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if subtitle:
        ax.set_title(subtitle, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"{saved_message}: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


_TIME_BREAKDOWN_UNAVAILABLE_HATCH = "///"
"""Hatch pattern marking a stage segment whose cost is UNAVAILABLE-provenance —
the number is a load/assembly artifact, not a real cost, so it must never read
identically to a genuine MEASURED/HISTORICAL segment."""


def _lighten(hex_color: str, amount: float = 0.5) -> tuple[float, float, float]:
    """Blend `hex_color` toward white by `amount` (0=no change, 1=white).

    Used to give a HISTORICAL segment a visually distinct-but-solid shade of
    its stage's base color (still a real, trustworthy number — just not
    measured this run — so it stays solid, never hatched).
    """
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _stage_segment_style(base_color: str, provenance: CostProvenance | None) -> dict[str, Any]:
    """Resolve one stage segment's matplotlib fill style from its provenance.

    Exhaustive `match` over every `CostProvenance` member plus an explicit
    `None` case (stage not applicable), mirroring the
    `PreconditionerFamily.abbreviation()` convention (`shared/types.py`) so a
    future fourth provenance value fails loudly at review time rather than
    silently rendering like one of the existing three.

    Args:
        base_color: This stage's Okabe-Ito base color.
        provenance: The segment's `StageCost.provenance`, or `None` if the
            stage doesn't apply to this method (contributes a 0-width,
            unstyled segment).

    Returns:
        dict[str, Any]: Keyword args for `Axes.barh` (`color`, optionally
        `hatch`).
    """
    match provenance:
        case None:
            return {"color": base_color}
        case CostProvenance.MEASURED:
            return {"color": base_color}
        case CostProvenance.HISTORICAL:
            return {"color": _lighten(base_color)}
        case CostProvenance.UNAVAILABLE:
            return {"color": base_color, "hatch": _TIME_BREAKDOWN_UNAVAILABLE_HATCH}


def plot_time_breakdown_barplot(
    results: Mapping[str, CGComparisonResult],
    legend: Legend,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    title: FigureTitle | None = None,
) -> None:
    """Stacked horizontal bar chart of each method's generation/setup/solve time.

    Complements the separate setup-time/solve-time bar charts with a diagnostic
    view of separately named generation, setup, and solve stages. This is not
    an online workload total; canonical event summaries own that quantity.
    Each segment is styled by its own ``StageCost.provenance``
    (solid for a real MEASURED/HISTORICAL cost, hatched for UNAVAILABLE — a
    real, unrecorded cost, not a true zero) so a silently-failed historical
    lookup can never render identically to a genuinely fast build. A method
    missing all three components is skipped (with a warning); a method
    missing only some has those components treated as zero-width.

    Args:
        results: Dictionary of method results.
        legend: Text of every plotted series, keyed by result key.
        save_path: Path to save plot.
        show: Whether to show plot.
        title: Optional title for the plot.
    """

    def _stage_parts(cost: StageCost | None) -> tuple[float, CostProvenance | None]:
        if cost is None:
            return 0.0, None
        return cost.wall_time_seconds, cost.provenance

    entries: list[
        tuple[str, float, CostProvenance | None, float, CostProvenance | None, float]
    ] = []
    for name, result in results.items():
        if (
            result.generation_cost is None
            and result.setup_cost == 0.0
            and result.solve_time_seconds is None
        ):
            logger.warning(
                f"Method '{name}' has no generation/setup/solve time to plot on time breakdown."
            )
            continue
        if result.has_unavailable_cost:
            logger.warning(
                f"Method '{name}' has an unavailable generation cost — "
                "its time-breakdown segment will be hatched, not a real zero."
            )
        generation_seconds, generation_provenance = _stage_parts(result.generation_cost)
        entries.append(
            (
                legend.label(name),
                generation_seconds,
                generation_provenance,
                result.setup_cost,
                CostProvenance.MEASURED,
                result.solve_time_seconds or 0.0,
            )
        )
    entries.sort(key=lambda entry: entry[1] + entry[3] + entry[5], reverse=True)

    plot_labels = [entry[0] for entry in entries]
    generation_values = [entry[1] for entry in entries]
    setup_values = [entry[3] for entry in entries]
    solve_values = [entry[5] for entry in entries]

    generation_color, setup_color, solve_color = (
        _OKABE_ITO_HEX[2],
        _OKABE_ITO_HEX[0],
        _OKABE_ITO_HEX[1],
    )

    fig, ax = plt.subplots(figsize=(8, max(3, len(plot_labels) * 0.6)))
    y_pos = np.arange(len(plot_labels))

    for i, (_, _, generation_provenance, _, setup_provenance, _) in enumerate(entries):
        ax.barh(
            y_pos[i],
            generation_values[i],
            edgecolor="black",
            linewidth=0.5,
            **_stage_segment_style(generation_color, generation_provenance),
        )
        ax.barh(
            y_pos[i],
            setup_values[i],
            left=generation_values[i],
            edgecolor="black",
            linewidth=0.5,
            **_stage_segment_style(setup_color, setup_provenance),
        )
        ax.barh(
            y_pos[i],
            solve_values[i],
            left=generation_values[i] + setup_values[i],
            color=solve_color,
            edgecolor="black",
            linewidth=0.5,
        )

    legend_handles = [
        matplotlib.patches.Patch(color=generation_color, label="Generation"),
        matplotlib.patches.Patch(color=setup_color, label="Setup"),
        matplotlib.patches.Patch(color=solve_color, label="Solve"),
    ]
    if any(
        generation_provenance is CostProvenance.UNAVAILABLE
        or setup_provenance is CostProvenance.UNAVAILABLE
        for _, _, generation_provenance, _, setup_provenance, _ in entries
    ):
        legend_handles.append(
            matplotlib.patches.Patch(
                facecolor="white",
                edgecolor="black",
                hatch=_TIME_BREAKDOWN_UNAVAILABLE_HATCH,
                label="Cost unavailable",
            )
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(handles=legend_handles, loc="lower right")
    fig.suptitle(
        title_text(title) if title is not None else "Time Breakdown (Generation + Setup + Solve)",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved time breakdown plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_data_norms(
    data_dir: str | Path, save_path: str | Path | None = None, show: bool = False
) -> None:
    """Plot RHS and solution norm distributions for a dataset.

    Args:
        data_dir: Path to dataset directory containing dataset artifacts.
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        None
    """
    data_dir = Path(data_dir)

    # Load data from dataset storage
    from neuralls.platform.storage.dataset_views import load_dataset

    data = load_dataset(data_dir, variant="dataset")
    rhs_samples = data["rhs"].astype(np.float64, copy=False)
    sol_samples = data["solutions"].astype(np.float64, copy=False)

    # Calculate norms
    rhs_norms = np.linalg.norm(rhs_samples, axis=1)
    sol_norms = np.linalg.norm(sol_samples, axis=1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RHS norms histogram
    ax1.hist(rhs_norms, bins=50, alpha=0.7, edgecolor="black")
    ax1.axvline(
        np.mean(rhs_norms),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(rhs_norms):.3e}",
    )
    ax1.axvline(
        np.median(rhs_norms),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(rhs_norms):.3e}",
    )
    ax1.set_xlabel("||b|| (RHS Norm)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("RHS Norm Distribution", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot solution norms histogram
    ax2.hist(sol_norms, bins=50, alpha=0.7, edgecolor="black", color="orange")
    ax2.axvline(
        np.mean(sol_norms),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(sol_norms):.3e}",
    )
    ax2.axvline(
        np.median(sol_norms),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(sol_norms):.3e}",
    )
    ax2.set_xlabel("||x|| (Solution Norm)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Solution Norm Distribution", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Add dataset info as suptitle
    dataset_name = data_dir.name
    fig.suptitle(f"Data Norms: {dataset_name}", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved data norms plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
