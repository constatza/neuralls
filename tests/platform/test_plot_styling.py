"""Tests for the 3-axis (linestyle/marker/color) comparison-plot styling engine.

Family alone used to drive every visual axis, so a dense same-family sweep
(e.g. 10+ POD-2G variants) collapsed onto one marker/linestyle pair and told
lines apart only by shading a single colormap. These pin the generalized
engine: linestyle still follows family, but marker/color follow explicit
per-line keys when given, with a fallback to family when they are not.
"""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.plots import (
    _OKABE_ITO_HEX,
    _assign_axis,
    _color_pool,
    _resolve_styles,
)
from neuralls.shared.types import PreconditionerFamily


@pytest.fixture
def pod2g_family_map() -> dict[str, PreconditionerFamily]:
    """Four POD-2G entries plus one AMG entry — all one plot's method names.

    Returns:
        Family key per method (plot label) name.
    """
    return {
        "pod2g-ds1-raw": PreconditionerFamily.POD2G,
        "pod2g-ds1-power": PreconditionerFamily.POD2G,
        "pod2g-ds2-raw": PreconditionerFamily.POD2G,
        "pod2g-ds2-power": PreconditionerFamily.POD2G,
        "amg-default": PreconditionerFamily.AMG,
    }


@pytest.fixture
def pod2g_color_keys() -> dict[str, str]:
    """Dataset-derived color key for the POD-2G entries only.

    Returns:
        Color-axis key per method name; AMG entry intentionally omitted so it
        falls back to family.
    """
    return {
        "pod2g-ds1-raw": "dataset-1",
        "pod2g-ds1-power": "dataset-1",
        "pod2g-ds2-raw": "dataset-2",
        "pod2g-ds2-power": "dataset-2",
    }


@pytest.fixture
def pod2g_marker_keys() -> dict[str, str]:
    """Weighting-scheme-derived marker key for the POD-2G entries only.

    Returns:
        Marker-axis key per method name; AMG entry intentionally omitted so
        it falls back to family.
    """
    return {
        "pod2g-ds1-raw": "raw",
        "pod2g-ds1-power": "power_norm",
        "pod2g-ds2-raw": "raw",
        "pod2g-ds2-power": "power_norm",
    }


def test_assign_axis_gives_each_distinct_key_its_own_slot() -> None:
    """Distinct keys map to distinct palette entries, in first-seen order."""
    assigned = _assign_axis(
        {"a": "x", "b": "y", "c": "x"}, palette=["circle", "square", "triangle"]
    )
    assert assigned == {"a": "circle", "b": "square", "c": "circle"}


def test_assign_axis_cycles_on_overflow() -> None:
    """More distinct keys than palette slots wraps back to the start."""
    keys = {"a": 1, "b": 2, "c": 3}
    assigned = _assign_axis(keys, palette=["x", "y"])
    assert assigned == {"a": "x", "b": "y", "c": "x"}


def test_color_pool_uses_okabe_ito_within_standard_size() -> None:
    """Up to 8 colors come straight from the Okabe-Ito palette, in order."""
    assert _color_pool(3) == list(_OKABE_ITO_HEX[:3])


def test_color_pool_extends_beyond_okabe_ito_with_distinct_generated_hues() -> None:
    """Beyond 8 keys, the first 8 stay Okabe-Ito and the rest are new, distinct colors."""
    pool = _color_pool(10)
    assert pool[:8] == list(_OKABE_ITO_HEX)
    assert len(pool[8:]) == 2
    assert len(set(pool)) == len(pool)


def test_same_dataset_different_weighting_shares_color_not_marker(
    pod2g_family_map: dict[str, PreconditionerFamily],
    pod2g_color_keys: dict[str, str],
    pod2g_marker_keys: dict[str, str],
) -> None:
    """Two POD-2G entries fit on the same dataset get the same color but different markers."""
    styles = _resolve_styles(pod2g_family_map, pod2g_color_keys, pod2g_marker_keys)

    assert styles["pod2g-ds1-raw"]["color"] == styles["pod2g-ds1-power"]["color"]
    assert styles["pod2g-ds1-raw"]["marker"] != styles["pod2g-ds1-power"]["marker"]


def test_same_weighting_different_dataset_shares_marker_not_color(
    pod2g_family_map: dict[str, PreconditionerFamily],
    pod2g_color_keys: dict[str, str],
    pod2g_marker_keys: dict[str, str],
) -> None:
    """Two POD-2G entries with the same weighting scheme get the same marker but different colors."""
    styles = _resolve_styles(pod2g_family_map, pod2g_color_keys, pod2g_marker_keys)

    assert styles["pod2g-ds1-raw"]["marker"] == styles["pod2g-ds2-raw"]["marker"]
    assert styles["pod2g-ds1-raw"]["color"] != styles["pod2g-ds2-raw"]["color"]


def test_all_pod2g_entries_share_family_linestyle(
    pod2g_family_map: dict[str, PreconditionerFamily],
    pod2g_color_keys: dict[str, str],
    pod2g_marker_keys: dict[str, str],
) -> None:
    """Linestyle is unaffected by color/marker keys — it always follows family."""
    styles = _resolve_styles(pod2g_family_map, pod2g_color_keys, pod2g_marker_keys)
    pod2g_linestyles = {
        styles[name]["linestyle"]
        for name, family in pod2g_family_map.items()
        if family == PreconditionerFamily.POD2G
    }
    assert len(pod2g_linestyles) == 1
    assert styles["amg-default"]["linestyle"] != next(iter(pod2g_linestyles))


def test_entries_without_override_keys_fall_back_to_family_grouping(
    pod2g_family_map: dict[str, PreconditionerFamily],
) -> None:
    """With no color_keys/marker_keys at all, same-family entries still get distinguishable colors."""
    styles = _resolve_styles(pod2g_family_map, color_keys=None, marker_keys=None)

    pod2g_names = [
        name for name, family in pod2g_family_map.items() if family == PreconditionerFamily.POD2G
    ]
    pod2g_colors = {styles[name]["color"] for name in pod2g_names}
    assert len(pod2g_colors) == len(pod2g_names)


def test_markersize_scale_present_for_every_style(
    pod2g_family_map: dict[str, PreconditionerFamily],
    pod2g_color_keys: dict[str, str],
    pod2g_marker_keys: dict[str, str],
) -> None:
    """Every resolved style carries a markersize_scale multiplier."""
    styles = _resolve_styles(pod2g_family_map, pod2g_color_keys, pod2g_marker_keys)
    assert all(isinstance(style["markersize_scale"], float) for style in styles.values())
