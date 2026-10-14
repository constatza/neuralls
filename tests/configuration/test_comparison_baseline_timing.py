"""``[comparison_baseline.timing]``: defaults, parsing, and the repeat cap."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralls.platform.config.models.experiments import ComparisonBaseline
from neuralls.shared.constants import MAX_TIMING_REPEATS
from neuralls.shared.timing import TimingRepeatPolicy, TimingStatistic


def test_timing_defaults_to_one_plain_timed_run() -> None:
    assert ComparisonBaseline().timing == TimingRepeatPolicy(repeats=1, warmup=0)


def test_timing_table_is_parsed_into_the_policy() -> None:
    baseline = ComparisonBaseline.model_validate(
        {"timing": {"repeats": 3, "warmup": 1, "statistic": "min"}}
    )

    assert baseline.timing == TimingRepeatPolicy(repeats=3, warmup=1, statistic=TimingStatistic.MIN)


def test_repeats_beyond_the_cap_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ComparisonBaseline.model_validate({"timing": {"repeats": MAX_TIMING_REPEATS + 1}})
