"""Colour, marker and linestyle of every plotted series, assigned once per figure.

A method keeps one colour and marker across every comparison plot because all of them resolve
styles here: the history curves from their families, the group plots through ``method_palette``.
"""

from __future__ import annotations

import colorsys
import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.colors
import numpy as np

from neuralls.platform.reporting.group_style import GroupStyle

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


def resolve_styles(
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
class MethodPalette:
    """The drawn label, order and visual style of each method of one figure.

    ``labels`` maps a method name to its legend text in the order the names were given, which is
    the row order of a bar figure; ``styles`` holds the ``marker``, ``linestyle``, ``color`` and
    ``markersize_scale`` of every name.
    """

    labels: Mapping[str, str]
    styles: Mapping[str, Mapping[str, Any]]

    @property
    def order(self) -> list[str]:
        """The labels from the top row down."""
        return list(self.labels.values())

    @property
    def colors(self) -> dict[str, Any]:
        """The colour of each label."""
        return {self.labels[name]: style["color"] for name, style in self.styles.items()}

    def color(self, name: str) -> Any:
        return self.styles[name]["color"]

    def marker(self, name: str) -> Any:
        return self.styles[name]["marker"]

    def linestyle(self, name: str) -> Any:
        return self.styles[name]["linestyle"]


def method_palette(names: Sequence[str], style: GroupStyle) -> MethodPalette:
    """Labels and series styles of ``names``, in the order given, as every group figure draws them.

    Colours and markers are assigned in first-seen order, so a caller that wants the same method
    to look the same whatever subset a figure shows passes the names sorted. The styles use the
    same resolution as the convergence plots; without ``families`` each name is its own family:
    still Okabe-Ito and distinct, just not shared with the convergence plots.
    """
    families = style.families or {name: name for name in names}
    resolved = resolve_styles(families, style.color_keys, style.marker_keys)
    return MethodPalette(
        {name: style.legend.label(name) for name in names},
        {name: resolved[name] for name in names},
    )
