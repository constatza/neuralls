"""Plotting utilities for graph-cg project."""

from __future__ import annotations

import colorsys
import itertools
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from neuralls.domain.solver.models.result import CGComparisonResult
from neuralls.platform.config.models.preconditioner_family import PreconditionerFamilyKey

DEFAULT_LINE_MARKER_SIZE = 2.0
DEFAULT_SCATTER_MARKER_AREA = 4.0
DEFAULT_DIAGNOSTIC_SCATTER_MARKER_AREA = 2.0


def plot_parity_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample: int = 0,
    save_path: str | Path | None = None,
    show: bool = False,
    marker_area: float = DEFAULT_SCATTER_MARKER_AREA,
) -> None:
    """Create parity and residuals plots.

    Args:
        y_true: True values
        y_pred: Predicted values
        sample: Sample number for title
        save_path: Path to save plot
        show: Whether to show plot
        marker_area: Marker area for scatter points
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calculate metrics
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size > 0 else 0.0
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax = axes[0]
    y_min = float(np.min([y_true.min(), y_pred.min()]))
    y_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0

    set1 = matplotlib.colormaps["Set1"]
    ax.scatter(y_true, y_pred, s=marker_area, alpha=0.7, color=set1(0.0))
    ax.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        linestyle="dashed",
        color=set1(0.1),
        label="y = x",
    )
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity — sample {sample}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(
        0.02,
        0.98,
        f"R²={r2:.3f}\nRMSE={rmse:.3e}\nMAE={mae:.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    # Residuals plot
    ax = axes[1]
    ax.scatter(y_pred, residuals, s=marker_area, alpha=0.7, color=set1(0.0))
    ax.axhline(0.0, color=set1(0.1), linestyle="dashed", label="residual = 0")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved parity+residuals plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


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
    families: Mapping[str, PreconditionerFamilyKey],
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
    title: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iterations: int | None = None,
    families: Mapping[str, PreconditionerFamilyKey] | None = None,
    color_keys: Mapping[str, Hashable] | None = None,
    marker_keys: Mapping[str, Hashable] | None = None,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
    history_attr: str = "residual_history_rel",
    ylabel: str = "Relative Residual $\\|r\\| / \\|b\\|$",
) -> None:
    """Plot convergence comparison between preconditioners.

    Args:
        results: Dictionary of method results
        metadata: Optional metadata dict mapping method name -> NeuralPreconditionerMetadata
        save_path: Path to save plot
        show: Whether to show plot
        title: Optional title for the plot
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

            label = _build_convergence_label(method_name, metadata.get(method_name))
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
    subtitle_parts = []
    if rtol is not None:
        subtitle_parts.append(f"rtol={rtol:.0e}")
    if atol is not None:
        subtitle_parts.append(f"atol={atol:.0e}")
    if max_iterations is not None:
        subtitle_parts.append(f"maxiter={max_iterations}")

    if subtitle_parts:
        ax.set_title(", ".join(subtitle_parts), fontsize=9)

    fig.suptitle(title or "Convergence Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
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


def plot_noise_robustness(
    noise_results: dict[str, dict[str, dict]],
    save_path: str | Path | None = None,
    show: bool = False,
    marker_size: float = DEFAULT_LINE_MARKER_SIZE,
) -> None:
    """Plot noise robustness analysis results.

    Args:
        noise_results: Nested dict: noise_level -> method -> results
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
            if method in level_results:
                result = level_results[method]
                # Handle both dict and dataclass results
                iters = 0
                if hasattr(result, "iterations"):
                    # getattr, not direct access: same hasattr-narrowing
                    # issue as plot_residual_history above.
                    iters = getattr(result, "iterations")  # noqa: B009
                elif isinstance(result, dict):
                    iters = result.get("iterations", 0)
                else:
                    iters = 0
                iterations.append(iters)
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


# Private helpers for plot_prediction_diagnostics
def _compute_prediction_stats(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """Compute comprehensive statistics for predictions and targets.

    Args:
        y_pred: Predicted values
        y_true: True target values

    Returns:
        Tuple of (pred_stats, true_stats, error_stats, scale_ratio)
    """
    pred_stats = {
        "min": float(np.min(y_pred)),
        "max": float(np.max(y_pred)),
        "mean": float(np.mean(y_pred)),
        "median": float(np.median(y_pred)),
        "std": float(np.std(y_pred)),
        "norm": float(np.linalg.norm(y_pred)),
    }

    true_stats = {
        "min": float(np.min(y_true)),
        "max": float(np.max(y_true)),
        "mean": float(np.mean(y_true)),
        "median": float(np.median(y_true)),
        "std": float(np.std(y_true)),
        "norm": float(np.linalg.norm(y_true)),
    }

    error = y_pred - y_true
    error_stats = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }

    scale_ratio = pred_stats["norm"] / max(true_stats["norm"], 1e-15)

    return pred_stats, true_stats, error_stats, scale_ratio


def _plot_distribution_row(
    fig: Any,
    gs: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_stats: dict[str, float],
    true_stats: dict[str, float],
    error_stats: dict[str, float],
) -> None:
    """Plot histograms for predictions, targets, and errors.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        y_pred: Predicted values
        y_true: True target values
        pred_stats: Prediction statistics
        true_stats: Target statistics
        error_stats: Error statistics
    """
    error = y_pred - y_true

    # Prediction distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(y_pred, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        pred_stats["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pred_stats['mean']:.3e}",
    )
    ax1.axvline(
        pred_stats["median"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pred_stats['median']:.3e}",
    )
    ax1.set_xlabel("Value", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Prediction Distribution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Target distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_true, bins=50, alpha=0.7, color="orange", edgecolor="black")
    ax2.axvline(
        true_stats["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {true_stats['mean']:.3e}",
    )
    ax2.axvline(
        true_stats["median"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {true_stats['median']:.3e}",
    )
    ax2.set_xlabel("Value", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Target Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Error distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(error, bins=50, alpha=0.7, color="red", edgecolor="black")
    ax3.axvline(0, color="black", linestyle="-", linewidth=2)
    ax3.axvline(
        error_stats["mae"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"MAE: {error_stats['mae']:.3e}",
    )
    ax3.set_xlabel("Error (Pred - True)", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3)


def _plot_timeseries_row(
    fig: Any,
    gs: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Plot time series views of predictions and targets.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        y_pred: Predicted values
        y_true: True target values
    """
    n_show = min(200, len(y_pred))
    indices = np.arange(n_show)

    # Predictions time series
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(indices, y_pred[:n_show], "b-", linewidth=1, alpha=0.7, label="Predictions")
    ax4.set_xlabel("Index", fontsize=11)
    ax4.set_ylabel("Value", fontsize=11)
    ax4.set_title(f"Predictions (first {n_show} values)", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc="upper left")

    # Targets time series
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(indices, y_true[:n_show], "orange", linewidth=1, alpha=0.7, label="Targets")
    ax5.set_xlabel("Index", fontsize=11)
    ax5.set_ylabel("Value", fontsize=11)
    ax5.set_title(f"Targets (first {n_show} values)", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9, loc="upper left")

    # Overlay
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(indices, y_pred[:n_show], "b-", linewidth=1, alpha=0.6, label="Predictions")
    ax6.plot(indices, y_true[:n_show], "orange", linewidth=1, alpha=0.6, label="Targets")
    ax6.set_xlabel("Index", fontsize=11)
    ax6.set_ylabel("Value", fontsize=11)
    ax6.set_title(f"Overlay (first {n_show} values)", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9, loc="upper left")


def _plot_stats_bar_chart(
    fig: Any,
    gs: Any,
    pred_stats: dict[str, float],
    true_stats: dict[str, float],
) -> None:
    """Plot statistics comparison bar chart.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        pred_stats: Prediction statistics
        true_stats: Target statistics
    """
    ax = fig.add_subplot(gs[2, 0])
    stats_names = ["min", "max", "mean", "median", "std", "norm"]
    pred_vals = [pred_stats[k] for k in stats_names]
    true_vals = [true_stats[k] for k in stats_names]
    x_pos = np.arange(len(stats_names))
    width = 0.35

    ax.bar(x_pos - width / 2, pred_vals, width, label="Predictions", alpha=0.8, color="blue")
    ax.bar(x_pos + width / 2, true_vals, width, label="Targets", alpha=0.8, color="orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Statistics Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("symlog")


def _plot_parity_subplot(
    fig: Any,
    gs: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Plot parity subplot.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        y_pred: Predicted values
        y_true: True target values
    """
    ax = fig.add_subplot(gs[2, 1])
    y_min = float(np.min([y_true.min(), y_pred.min()]))
    y_max = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    ax.scatter(
        y_true,
        y_pred,
        s=DEFAULT_DIAGNOSTIC_SCATTER_MARKER_AREA,
        alpha=0.5,
        color="purple",
    )
    ax.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        "k--",
        linewidth=2,
        label="Perfect prediction",
    )
    ax.set_xlabel("True Values", fontsize=11)
    ax.set_ylabel("Predicted Values", fontsize=11)
    ax.set_title("Parity Plot", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.axis("equal")


def _plot_summary_text(
    fig: Any,
    gs: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_stats: dict[str, float],
    true_stats: dict[str, float],
    scale_ratio: float,
    sample: int,
) -> None:
    """Plot summary text panel.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        y_pred: Predicted values
        y_true: True target values
        pred_stats: Prediction statistics
        true_stats: Target statistics
        scale_ratio: Ratio of prediction norm to target norm
        sample: Sample number for title
    """
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    error = y_pred - y_true
    error_stats = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }

    summary_text = f"""
DIAGNOSTIC SUMMARY (Sample {sample})

PREDICTIONS:
  Min:    {pred_stats["min"]:.6e}
  Max:    {pred_stats["max"]:.6e}
  Mean:   {pred_stats["mean"]:.6e}
  Median: {pred_stats["median"]:.6e}
  Std:    {pred_stats["std"]:.6e}
  ||pred||: {pred_stats["norm"]:.6e}

TARGETS:
  Min:    {true_stats["min"]:.6e}
  Max:    {true_stats["max"]:.6e}
  Mean:   {true_stats["mean"]:.6e}
  Median: {true_stats["median"]:.6e}
  Std:    {true_stats["std"]:.6e}
  ||true||: {true_stats["norm"]:.6e}

ERRORS:
  MAE:     {error_stats["mae"]:.6e}
  RMSE:    {error_stats["rmse"]:.6e}
  Max Abs: {error_stats["max_abs"]:.6e}

SCALE ANALYSIS:
  ||pred|| / ||true||: {scale_ratio:.6e}

{"WARNING: Scale ratio > 10x!" if scale_ratio > 10 or scale_ratio < 0.1 else "Scale ratio looks reasonable"}
    """
    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
    )


def _plot_comparison_row(
    fig: Any,
    gs: Any,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_stats: dict[str, float],
    true_stats: dict[str, float],
    scale_ratio: float,
    sample: int,
) -> None:
    """Plot statistical comparisons, parity plot, and summary.

    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        y_pred: Predicted values
        y_true: True target values
        pred_stats: Prediction statistics
        true_stats: Target statistics
        scale_ratio: Ratio of prediction norm to target norm
        sample: Sample number for title
    """
    _plot_stats_bar_chart(fig, gs, pred_stats, true_stats)
    _plot_parity_subplot(fig, gs, y_pred, y_true)
    _plot_summary_text(fig, gs, y_pred, y_true, pred_stats, true_stats, scale_ratio, sample)


def plot_prediction_diagnostics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sample: int = 0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Create comprehensive diagnostic plots for predictions vs targets.

    This function creates separate visualizations for predictions and targets
    to help identify scaling issues, normalization problems, and data quality issues.

    Args:
        y_pred: Predicted values (1D array)
        y_true: True target values (1D array)
        sample: Sample number for title
        save_path: Path to save plot
        show: Whether to show plot
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    # Compute statistics
    pred_stats, true_stats, error_stats, scale_ratio = _compute_prediction_stats(y_pred, y_true)

    # Create comprehensive diagnostic figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Row 1: Distributions
    _plot_distribution_row(fig, gs, y_pred, y_true, pred_stats, true_stats, error_stats)

    # Row 2: Time series
    _plot_timeseries_row(fig, gs, y_pred, y_true)

    # Row 3: Comparisons and summary
    _plot_comparison_row(fig, gs, y_pred, y_true, pred_stats, true_stats, scale_ratio, sample)

    fig.suptitle("Prediction Diagnostics - Scaling Analysis", fontsize=16, fontweight="bold")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved diagnostic plot to: {save_path}")

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
        labels: Short axis labels for each bar (e.g. ``["1", "2", "3"]``).
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
    if horizontal:
        fig, ax = plt.subplots(figsize=(5, max(3, len(labels) * 0.6)))
    else:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))

    x_pos = np.arange(len(labels))
    colormap = matplotlib.colormaps["Set1"]
    colors = colormap(np.linspace(0, 0.9, max(len(labels), 1)))

    if horizontal:
        ax.barh(x_pos, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_ylabel("Experiment", fontsize=12)
        ax.set_xlabel(metric_name, fontsize=12)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.bar(x_pos, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11)
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
