"""``repeat_timed_outcome``: warm-up, repeats, reduction, and first-execution memory."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from pydantic import ValidationError

from neuralls.shared import device as device_module
from neuralls.shared.device import (
    Agreed,
    Diverged,
    FailedTimedRuns,
    SingleRun,
    TimedRuns,
    Unchecked,
    repeat_timed_outcome,
)
from neuralls.shared.timing import TimingRepeatPolicy, TimingStatistic

CPU = torch.device("cpu")


def timed_runs(*args: object, **kwargs: object) -> TimedRuns:
    """`repeat_timed_outcome` for regions that do not raise: the successful outcome."""
    outcome = repeat_timed_outcome(*args, **kwargs)  # type: ignore[arg-type]
    assert isinstance(outcome, TimedRuns)
    return outcome


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[float]], None]:
    """Install a fake ``perf_counter`` where execution ``i`` lasts ``durations[i]`` seconds.

    Replaces the module's ``time`` reference only, so pytest's own timing is untouched.
    """

    def install(durations: list[float]) -> None:
        ticks: Iterator[float] = itertools.chain.from_iterable(
            (index * 1000.0, index * 1000.0 + duration) for index, duration in enumerate(durations)
        )
        monkeypatch.setattr(
            device_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
        )

    return install


def test_default_policy_is_a_single_plain_measurement(clock: Callable[[list[float]], None]) -> None:
    clock([2.0])
    calls: list[int] = []

    timed = timed_runs(lambda: calls.append(1) or "value", TimingRepeatPolicy(), device=CPU)

    assert len(calls) == 1
    assert timed.value == "value"
    assert timed.seconds == pytest.approx(2.0)
    assert timed.samples_seconds == (pytest.approx(2.0),)


def test_warmup_runs_are_executed_but_not_timed(clock: Callable[[list[float]], None]) -> None:
    clock([50.0, 1.0, 3.0, 2.0])
    calls: list[int] = []
    policy = TimingRepeatPolicy(repeats=3, warmup=1, statistic=TimingStatistic.MEDIAN)

    timed = timed_runs(lambda: calls.append(1), policy, device=CPU)

    assert len(calls) == 4
    assert timed.samples_seconds == pytest.approx((1.0, 3.0, 2.0))
    assert timed.seconds == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("statistic", "expected"),
    [
        (TimingStatistic.MEDIAN, 2.0),
        (TimingStatistic.MEAN, 2.0),
        (TimingStatistic.MIN, 1.0),
    ],
)
def test_statistic_reduces_the_samples(
    clock: Callable[[list[float]], None], statistic: TimingStatistic, expected: float
) -> None:
    clock([1.0, 3.0, 2.0])

    timed = timed_runs(lambda: None, TimingRepeatPolicy(repeats=3, statistic=statistic), device=CPU)

    assert timed.seconds == pytest.approx(expected)


def test_value_is_the_first_timed_executions(clock: Callable[[list[float]], None]) -> None:
    clock([1.0, 1.0, 1.0])
    counter = itertools.count()

    timed = timed_runs(lambda: next(counter), TimingRepeatPolicy(repeats=2, warmup=1), device=CPU)

    assert timed.value == 1  # execution 0 was warm-up


def test_a_repeat_whose_fingerprint_differs_logs_a_warning(
    clock: Callable[[list[float]], None],
) -> None:
    clock([1.0, 1.0])
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    counter = itertools.count()
    try:
        timed = timed_runs(
            lambda: next(counter),
            TimingRepeatPolicy(repeats=2),
            device=CPU,
            fingerprint=lambda value: value,
        )
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert timed.agreement == Diverged(first=0, other=1)


def test_only_the_first_divergence_is_reported(clock: Callable[[list[float]], None]) -> None:
    clock([1.0, 1.0, 1.0])
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    counter = itertools.count()
    try:
        timed = timed_runs(
            lambda: next(counter),
            TimingRepeatPolicy(repeats=3),
            device=CPU,
            fingerprint=lambda value: value,
        )
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert timed.agreement == Diverged(first=0, other=1)


def test_repeats_with_the_same_fingerprint_agree(clock: Callable[[list[float]], None]) -> None:
    clock([1.0, 1.0])

    timed = timed_runs(
        lambda: 7, TimingRepeatPolicy(repeats=2), device=CPU, fingerprint=lambda value: value
    )

    assert timed.agreement == Agreed()


def test_a_single_timed_run_has_nothing_to_compare(clock: Callable[[list[float]], None]) -> None:
    clock([1.0])

    timed = timed_runs(lambda: 7, TimingRepeatPolicy(), device=CPU, fingerprint=lambda value: value)

    assert timed.agreement == SingleRun()


def test_repeats_without_a_fingerprint_are_unchecked_not_agreed(
    clock: Callable[[list[float]], None],
) -> None:
    clock([1.0, 1.0])

    timed = timed_runs(lambda: 7, TimingRepeatPolicy(repeats=2), device=CPU)

    assert timed.agreement == Unchecked()


def test_a_raising_region_is_a_failed_outcome(clock: Callable[[list[float]], None]) -> None:
    clock([1.0])

    def boom() -> int:
        raise RuntimeError("boom")

    outcome = repeat_timed_outcome(boom, TimingRepeatPolicy(), device=CPU)

    assert isinstance(outcome, FailedTimedRuns)


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"repeats": 11}, {"warmup": -1}, {"bogus": 1}])
def test_policy_rejects_out_of_range_or_unknown_fields(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        TimingRepeatPolicy(**kwargs)
