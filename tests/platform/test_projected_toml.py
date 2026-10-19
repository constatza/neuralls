"""``projected_toml_payload``: converged-only numbers, ERT, comparisons and never-converged entries."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping

import pytest

from neuralls.domain.solver.models.convergence import ConvergenceSummary
from neuralls.domain.solver.models.projection import ProjectedComparisonResult
from neuralls.domain.solver.models.sharing import SetupSharingPolicy
from neuralls.domain.solver.models.statistics import (
    AlwaysCheaper,
    BreakEvenAt,
    Interval,
    IterationStats,
    MethodComparison,
    PairedComparison,
)
from neuralls.platform.reporting.artifacts import projected_toml_payload, write_projected_toml


def _entry(payload: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, object]:
    """One preconditioner's entry, read by key whichever TOML entry variant it is."""
    return payload[name]


@pytest.fixture
def projected() -> dict[str, ProjectedComparisonResult]:
    return {
        "amg": ProjectedComparisonResult(
            preconditioner_name="amg",
            n_systems=5,
            n_converged=3,
            setup_sharing=SetupSharingPolicy(share_setup=False, share_generation=False),
            setup_cost_total_seconds=5.0,
            generation_cost_total_seconds=None,
            total_solve_time_seconds=10.0,
            total_iterations=30,
        )
    }


@pytest.fixture
def convergence() -> dict[str, ConvergenceSummary]:
    return {
        "amg": ConvergenceSummary(n_systems=5, n_converged=3, ert_seconds=3.5),
        "none": ConvergenceSummary(n_systems=5, n_converged=0, ert_seconds=None),
    }


def test_projected_entry_carries_success_count_and_ert(projected, convergence) -> None:
    entry = _entry(projected_toml_payload(projected, convergence, {}), "amg")

    assert (entry["n_systems"], entry["n_converged"], entry["ert_seconds"]) == (5, 3, 3.5)


def test_never_converged_preconditioner_is_listed_with_counts_only(projected, convergence) -> None:
    entry = _entry(projected_toml_payload(projected, convergence, {}), "none")

    assert entry == {"n_systems": 5, "n_converged": 0}


@pytest.fixture
def detailed_convergence() -> dict[str, ConvergenceSummary]:
    return {
        "amg": ConvergenceSummary(
            n_systems=5,
            n_converged=3,
            ert_seconds=3.5,
            iterations=IterationStats(mean=10.0, median=9.0, std=2.0),
            ert_interval=Interval(low=3.0, high=4.5, confidence=0.95),
        )
    }


@pytest.fixture
def comparison() -> MethodComparison:
    return MethodComparison(
        reference="jacobi",
        break_even=BreakEvenAt(n_systems=12.5),
        paired=PairedComparison(
            n_both_solved=3,
            speedups=(1.0, 2.0, 3.0),
            median_speedup=2.0,
            median_speedup_ci=Interval(low=1.0, high=3.0, confidence=0.95),
            wilcoxon_p=0.25,
        ),
    )


def test_convergence_statistics_are_written_when_defined(projected, detailed_convergence) -> None:
    entry = _entry(projected_toml_payload(projected, detailed_convergence, {}), "amg")

    assert (entry["iterations_mean"], entry["iterations_median"], entry["iterations_std"]) == (
        10.0, 9.0, 2.0
    )  # fmt: skip
    assert (entry["ert_ci_low"], entry["ert_ci_high"]) == (3.0, 4.5)


def test_convergence_statistics_are_omitted_when_undefined(projected, convergence) -> None:
    entry = _entry(projected_toml_payload(projected, convergence, {}), "amg")

    assert not {"iterations_mean", "iterations_std", "ert_ci_low", "ert_ci_high"} & entry.keys()


def test_comparison_fields_are_written_when_defined(projected, convergence, comparison) -> None:
    entry = _entry(projected_toml_payload(projected, convergence, {"amg": comparison}), "amg")

    assert entry["reference"] == "jacobi"
    assert entry["break_even_systems"] == 12.5
    assert (entry["speedup_median"], entry["speedup_ci_low"], entry["speedup_ci_high"]) == (
        2.0, 1.0, 3.0
    )  # fmt: skip
    assert entry["wilcoxon_p"] == 0.25


def test_break_even_count_is_omitted_unless_it_is_a_finite_count(
    projected, convergence, comparison
) -> None:
    always = MethodComparison(reference="jacobi", break_even=AlwaysCheaper(), paired=None)

    entry = _entry(projected_toml_payload(projected, convergence, {"amg": always}), "amg")

    assert entry["reference"] == "jacobi"
    assert not {"break_even_systems", "speedup_median", "wilcoxon_p"} & entry.keys()


def test_undefined_interval_and_p_value_are_omitted(projected, convergence, comparison) -> None:
    bare = PairedComparison(
        n_both_solved=1,
        speedups=(2.0,),
        median_speedup=2.0,
        median_speedup_ci=None,
        wilcoxon_p=None,
    )
    method = MethodComparison(reference="jacobi", break_even=BreakEvenAt(1.0), paired=bare)

    entry = _entry(projected_toml_payload(projected, convergence, {"amg": method}), "amg")

    assert entry["speedup_median"] == 2.0
    assert not {"speedup_ci_low", "speedup_ci_high", "wilcoxon_p"} & entry.keys()


def test_never_converged_entry_stays_counts_only_despite_a_comparison(
    projected, convergence, comparison
) -> None:
    entry = _entry(projected_toml_payload(projected, convergence, {"none": comparison}), "none")

    assert entry == {"n_systems": 5, "n_converged": 0}


def test_written_toml_round_trips_the_payload(
    tmp_path, projected, detailed_convergence, comparison
) -> None:
    path = tmp_path / "projected.toml"

    write_projected_toml(projected, detailed_convergence, {"amg": comparison}, path)

    assert tomllib.loads(path.read_text())["amg"]["break_even_systems"] == 12.5
