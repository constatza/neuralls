"""``SetupSharingPolicy``: the one place that splits a cost into one-time and repeated parts."""

from __future__ import annotations

import pytest

from neuralls.domain.solver.models.sharing import SetupSharingPolicy


@pytest.mark.parametrize(
    ("share_setup", "share_generation", "one_time", "repeated"),
    [
        (True, True, 2.5, 0.0),
        (True, False, 2.0, 0.5),
        (False, True, 0.5, 2.0),
        (False, False, 0.0, 2.5),
    ],
)
def test_each_component_is_one_time_or_repeated_never_both(
    share_setup, share_generation, one_time, repeated
) -> None:
    policy = SetupSharingPolicy(share_setup=share_setup, share_generation=share_generation)

    assert policy.one_time_seconds(2.0, 0.5) == pytest.approx(one_time)
    assert policy.repeated_seconds(2.0, 0.5) == pytest.approx(repeated)


def test_unavailable_generation_counts_as_zero_on_both_sides() -> None:
    policy = SetupSharingPolicy(share_setup=False, share_generation=True)

    assert policy.one_time_seconds(2.0, None) == 0.0
    assert policy.repeated_seconds(2.0, None) == pytest.approx(2.0)
