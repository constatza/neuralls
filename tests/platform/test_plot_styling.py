"""Tests for the comparison-plot styling engine.

Linestyle and marker follow family; color follows explicit per-line keys when
given, with a fallback to family when they are not.
"""

from __future__ import annotations

import pytest

from neuralls.platform.reporting.series_style import (
    _OKABE_ITO_HEX,
    _assign_axis,
    color_pool,
    resolve_styles,
)
from neuralls.shared.types import PreconditionerFamily


@pytest.fixture
def pod_family_map() -> dict[str, PreconditionerFamily]:
    """Four POD entries plus one AMG entry — all one plot's method names.

    Returns:
        Family key per method (plot label) name.
    """
    return {
        "pod-ds1-rank10": PreconditionerFamily.POD,
        "pod-ds1-rank20": PreconditionerFamily.POD,
        "pod-ds2-rank10": PreconditionerFamily.POD,
        "pod-ds2-rank20": PreconditionerFamily.POD,
        "amg-default": PreconditionerFamily.AMG,
    }


@pytest.fixture
def pod_color_keys() -> dict[str, str]:
    """Dataset-derived color key for the POD entries only.

    Returns:
        Color-axis key per method name; AMG entry intentionally omitted so it
        falls back to family.
    """
    return {
        "pod-ds1-rank10": "dataset-1",
        "pod-ds1-rank20": "dataset-1",
        "pod-ds2-rank10": "dataset-2",
        "pod-ds2-rank20": "dataset-2",
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
    assert color_pool(3) == list(_OKABE_ITO_HEX[:3])


def test_color_pool_extends_beyond_okabe_ito_with_distinct_generated_hues() -> None:
    """Beyond 8 keys, the first 8 stay Okabe-Ito and the rest are new, distinct colors."""
    pool = color_pool(10)
    assert pool[:8] == list(_OKABE_ITO_HEX)
    assert len(pool[8:]) == 2
    assert len(set(pool)) == len(pool)


def test_same_family_different_dataset_shares_marker_not_color(
    pod_family_map: dict[str, PreconditionerFamily],
    pod_color_keys: dict[str, str],
) -> None:
    """Two POD entries fit on different datasets get the same marker but different colors."""
    styles = resolve_styles(pod_family_map, pod_color_keys)

    assert styles["pod-ds1-rank10"]["marker"] == styles["pod-ds2-rank10"]["marker"]
    assert styles["pod-ds1-rank10"]["color"] != styles["pod-ds2-rank10"]["color"]


def test_all_pod_entries_share_family_linestyle(
    pod_family_map: dict[str, PreconditionerFamily],
    pod_color_keys: dict[str, str],
) -> None:
    """Linestyle is unaffected by color keys — it always follows family."""
    styles = resolve_styles(pod_family_map, pod_color_keys)
    pod_linestyles = {
        styles[name]["linestyle"]
        for name, family in pod_family_map.items()
        if family == PreconditionerFamily.POD
    }
    assert len(pod_linestyles) == 1
    assert styles["amg-default"]["linestyle"] != next(iter(pod_linestyles))


def test_entries_without_override_keys_fall_back_to_family_grouping(
    pod_family_map: dict[str, PreconditionerFamily],
) -> None:
    """With no color_keys at all, same-family entries still get distinguishable colors."""
    styles = resolve_styles(pod_family_map, color_keys=None)

    pod_names = [
        name for name, family in pod_family_map.items() if family == PreconditionerFamily.POD
    ]
    pod_colors = {styles[name]["color"] for name in pod_names}
    assert len(pod_colors) == len(pod_names)


def test_markersize_scale_present_for_every_style(
    pod_family_map: dict[str, PreconditionerFamily],
    pod_color_keys: dict[str, str],
) -> None:
    """Every resolved style carries a markersize_scale multiplier."""
    styles = resolve_styles(pod_family_map, pod_color_keys)
    assert all(isinstance(style["markersize_scale"], float) for style in styles.values())
