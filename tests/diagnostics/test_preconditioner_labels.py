"""Tests for descriptive preconditioner plot labels.

Follows project conventions: fixtures over inline literals, type hints
throughout, no ``tempfile`` (no file I/O needed here — pure object
introspection).
"""

from __future__ import annotations

import pytest
import torch
from torchalg.preconditioners.base import Preconditioner
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    ScheduledPreconditioner,
)
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    AggregationCoarsening,
    AMGPreconditioner,
    BootstrapAMGPreconditioner,
    JacobiSmoother,
    TargetDimensionCoarsening,
    VCycle,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.platform.config.models.preconditioner import PreconditionerType
from neuralls.platform.reporting.preconditioner_labels import (
    MAX_LABEL_LENGTH,
    AggregationCoarseningDetail,
    PODCoarseningDetail,
    TargetDimensionCoarseningDetail,
    build_preconditioner_labels,
    coarsening_detail,
    describe_preconditioner,
    preconditioner_label,
)
from neuralls.shared.types import PreconditionerFamily

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tridiag_spd_matrix() -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix, suitable for AMG aggregation coarsening."""
    n = 5
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def rank2_snapshot_ensemble() -> torch.Tensor:
    """Snapshot ensemble whose exact numerical rank is 2 (6 samples, 5 dofs).

    Built as ``coeffs @ base`` from two independent random directions, so a
    POD fit with a high energy-capture threshold (e.g. 0.999) resolves to
    exactly 2 retained modes — deterministic given the fixed seed, and
    distinct from the configured threshold value itself, so tests can assert
    the description reports the *actual* fitted width rather than echoing
    the configured float.
    """
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    coeffs = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    return coeffs @ base


@pytest.fixture
def aggregation_amg_preconditioner(tridiag_spd_matrix: torch.Tensor) -> AMGPreconditioner:
    """Constructed AMG preconditioner using smoothed-aggregation coarsening."""
    coarsening = AggregationCoarsening(omega=0.67)
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


@pytest.fixture
def auto_omega_aggregation_amg_preconditioner(
    tridiag_spd_matrix: torch.Tensor,
) -> AMGPreconditioner:
    """Constructed AMG preconditioner with omega left unset (torchalg's auto per-matrix rule)."""
    coarsening = AggregationCoarsening()
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


@pytest.fixture
def pod_amg_preconditioner(
    tridiag_spd_matrix: torch.Tensor, rank2_snapshot_ensemble: torch.Tensor
) -> AMGPreconditioner:
    """Constructed AMG preconditioner using POD-2G coarsening with an energy-threshold rank."""
    coarsening = PODCoarseningStrategy(rank=0.999)
    coarsening.fit(rank2_snapshot_ensemble)
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


@pytest.fixture
def target_dim_amg_preconditioner(tridiag_spd_matrix: torch.Tensor) -> AMGPreconditioner:
    """Constructed AMG preconditioner using target-coarse-dimension coarsening."""
    coarsening = TargetDimensionCoarsening(
        target_coarse_dim=3, theta_min=0.05, theta_max=0.5, step=0.05, omega=0.67
    )
    smoother = JacobiSmoother(omega=0.67)
    cycle = VCycle(smoother=smoother, n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


@pytest.fixture
def adaptive_sa_preconditioner(tridiag_spd_matrix: torch.Tensor) -> AdaptiveSAPreconditioner:
    """Constructed adaptive SA-AMG (alpha-SA) preconditioner, forced to 2 levels."""
    return AdaptiveSAPreconditioner(tridiag_spd_matrix, max_levels=2, max_coarse=1, theta=0.0)


@pytest.fixture
def tridiag_spd_matrix_20() -> torch.Tensor:
    """Dense 20x20 SPD tridiagonal matrix, large enough for Bootstrap AMG to coarsen."""
    n = 20
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def bootstrap_amg_preconditioner(
    tridiag_spd_matrix_20: torch.Tensor,
) -> BootstrapAMGPreconditioner:
    """Constructed Bootstrap AMG (BAMG) preconditioner, forced to coarsen into 3 levels.

    ``tridiag_spd_matrix`` (5x5) is too small to coarsen at all under BAMG's
    compatible-relaxation criterion (a real end-to-end check showed it always
    settles on a single level, unlike aggregation/alpha-SA on the same
    matrix) — hence the larger dedicated fixture.
    """
    return BootstrapAMGPreconditioner(tridiag_spd_matrix_20, max_coarse=5, seed=0)


# ==============================================================================
# coarsening_detail — structured facts, no string parsing required to verify
# ==============================================================================


def test_coarsening_detail_reports_realized_coarse_dimension_for_aggregation(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Aggregation coarsening's realized coarse dimension is read back, not guessed.

    ``theta`` is a strength-of-connection threshold, not a chosen dimension —
    the only way to know how many aggregates it produced is to build the
    transfer operator and check its shape.

    Expected value is 2 (one boundary pair {0,1} + one interior triple
    {2,3,4}) under torchalg's field-standard three-pass
    ``standard_aggregation`` (VMB96/PyAMG's ``amg_core::standard_aggregation``),
    which replaced the single-pass ``greedy_aggregation`` this test
    previously pinned to 3 aggregates against.
    """
    detail = coarsening_detail(
        aggregation_amg_preconditioner.coarsening, aggregation_amg_preconditioner.matrix
    )

    assert detail == AggregationCoarseningDetail(theta=0.25, omega=0.67, coarse_dimension=2)


def test_coarsening_detail_reports_none_omega_when_left_to_auto_rule(
    auto_omega_aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """An unset omega is read back as None, not a guessed or defaulted float.

    torchalg resolves the actual damping per-matrix at solve time via its own
    spectral-radius rule; the label layer must not fabricate a value it never
    configured.
    """
    detail = coarsening_detail(
        auto_omega_aggregation_amg_preconditioner.coarsening,
        auto_omega_aggregation_amg_preconditioner.matrix,
    )

    assert isinstance(detail, AggregationCoarseningDetail)
    assert detail.omega is None


def test_coarsening_detail_reports_fitted_rank_for_pod(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G's detail is its actual fitted basis width, not the configured threshold."""
    detail = coarsening_detail(pod_amg_preconditioner.coarsening, pod_amg_preconditioner.matrix)

    assert detail == PODCoarseningDetail(rank=2)


def test_coarsening_detail_reports_realized_dimension_for_target_dim(
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Target-dimension coarsening's detail carries both the target and what was realized."""
    detail = coarsening_detail(
        target_dim_amg_preconditioner.coarsening, target_dim_amg_preconditioner.matrix
    )

    coarsening = target_dim_amg_preconditioner.coarsening
    assert isinstance(coarsening, TargetDimensionCoarsening)
    assert isinstance(detail, TargetDimensionCoarseningDetail)
    assert detail.target_coarse_dim == 3
    assert detail.realized_coarse_dim == coarsening.realized_coarse_dim(
        target_dim_amg_preconditioner.matrix
    )


# ==============================================================================
# describe_preconditioner — AMG with aggregation coarsening
# ==============================================================================


def test_describe_preconditioner_amg_aggregation_coarsening_has_detail(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """AMG with smoothed-aggregation coarsening reports non-empty structural detail.

    The actual facts (theta, omega, realized coarse dimension) are verified
    structurally by ``test_coarsening_detail_reports_realized_coarse_dimension_for_aggregation``.
    """
    assert describe_preconditioner(aggregation_amg_preconditioner) != ""


def test_describe_preconditioner_amg_aggregation_renders_auto_omega(
    auto_omega_aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """An unset omega renders as 'auto' in the label instead of crashing on None."""
    detail = describe_preconditioner(auto_omega_aggregation_amg_preconditioner)

    assert "ω=auto" in detail


# ==============================================================================
# describe_preconditioner — AMG with POD-2G coarsening
# ==============================================================================


def test_describe_preconditioner_amg_pod_coarsening_has_detail(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """AMG with POD-2G coarsening reports non-empty structural detail.

    That the reported rank is the actual fitted basis width (2) rather than
    the configured energy threshold (0.999) is verified structurally by
    ``test_coarsening_detail_reports_fitted_rank_for_pod``.
    """
    assert describe_preconditioner(pod_amg_preconditioner) != ""


# ==============================================================================
# describe_preconditioner — AMG with target-coarse-dimension coarsening
# ==============================================================================


def test_describe_preconditioner_target_dim_coarsening_has_detail(
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    """AMG with target-dimension coarsening reports non-empty structural detail."""
    assert describe_preconditioner(target_dim_amg_preconditioner) != ""


def test_describe_preconditioner_target_dim_keeps_level_count(
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Unlike POD-2G, target-dimension coarsening keeps L= — n_levels is a real knob for it too."""
    detail = describe_preconditioner(target_dim_amg_preconditioner)

    assert "L=" in detail
    assert "c=" in detail


# ==============================================================================
# describe_preconditioner — adaptive SA-AMG (alpha-SA)
# ==============================================================================


def test_describe_preconditioner_adaptive_sa_has_detail(
    adaptive_sa_preconditioner: AdaptiveSAPreconditioner,
) -> None:
    """Adaptive SA-AMG reports non-empty structural detail instead of the prebuilt placeholder.

    ``AdaptiveSAPreconditioner`` stores ``_PrebuiltCoarsening()`` as its
    ``_coarsening`` (torchalg's placeholder, never a real strategy) — this
    guards against the generic ``AMGPreconditioner`` branch matching first
    and rendering that placeholder's class name instead of real detail.
    """
    detail = describe_preconditioner(adaptive_sa_preconditioner)

    assert detail != ""
    assert "_PrebuiltCoarsening" not in detail
    assert "L=" in detail
    assert "k=" in detail
    assert "c=" in detail


def test_describe_preconditioner_adaptive_sa_reports_realized_coarse_dimension(
    adaptive_sa_preconditioner: AdaptiveSAPreconditioner,
) -> None:
    """The rendered `c=` matches the hierarchy's actual coarsest-level dimension."""
    detail = describe_preconditioner(adaptive_sa_preconditioner)

    realized = adaptive_sa_preconditioner.result.matrices[-1].shape[0]
    assert f"c={realized}" in detail


# ==============================================================================
# describe_preconditioner — Bootstrap AMG (BAMG)
# ==============================================================================


def test_describe_preconditioner_bootstrap_amg_has_detail(
    bootstrap_amg_preconditioner: BootstrapAMGPreconditioner,
) -> None:
    """Bootstrap AMG reports non-empty structural detail instead of the prebuilt placeholder.

    Same most-derived-first reasoning as the alpha-SA case above:
    ``BootstrapAMGPreconditioner`` also subclasses ``AMGPreconditioner`` and
    stores a prebuilt placeholder as its ``_coarsening``.
    """
    detail = describe_preconditioner(bootstrap_amg_preconditioner)

    assert detail != ""
    assert "PrebuiltCoarsening" not in detail
    assert "L=" in detail
    assert "k_r=" in detail
    assert "c=" in detail


def test_describe_preconditioner_bootstrap_amg_reports_realized_coarse_dimension(
    bootstrap_amg_preconditioner: BootstrapAMGPreconditioner,
) -> None:
    """The rendered `c=` matches the hierarchy's actual coarsest-level dimension."""
    detail = describe_preconditioner(bootstrap_amg_preconditioner)

    realized = bootstrap_amg_preconditioner.result.matrices[-1].shape[0]
    assert f"c={realized}" in detail


# ==============================================================================
# describe_preconditioner — plain preconditioners (no structural variants)
# ==============================================================================


@pytest.mark.parametrize(
    "precond",
    [
        Identity(),
    ],
)
def test_describe_preconditioner_identity_returns_empty_string(precond: Preconditioner) -> None:
    """Identity has no structural variants, so no detail is fabricated."""
    assert describe_preconditioner(precond) == ""


def test_describe_preconditioner_jacobi_returns_empty_string(
    tridiag_spd_matrix: torch.Tensor,
) -> None:
    """JacobiPreconditioner has no structural variants, so no detail is fabricated."""
    precond = JacobiPreconditioner(tridiag_spd_matrix)

    assert describe_preconditioner(precond) == ""


# ==============================================================================
# describe_preconditioner — ScheduledPreconditioner unwraps to its primary
# ==============================================================================


def test_describe_preconditioner_unwraps_scheduled_preconditioner(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """A scheduled AMG preconditioner still reports the wrapped AMG's structural detail."""
    scheduled = ScheduledPreconditioner(
        primary=aggregation_amg_preconditioner,
        fallback=Identity(),
        limit_iters=10,
        start_iter=0,
    )

    assert describe_preconditioner(scheduled) == describe_preconditioner(
        aggregation_amg_preconditioner
    )


# ==============================================================================
# preconditioner_label / build_preconditioner_labels
# ==============================================================================


def test_preconditioner_label_includes_amg_detail(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """A label combines the family abbreviation with whatever describe_preconditioner reports."""
    detail = describe_preconditioner(aggregation_amg_preconditioner)

    assert (
        preconditioner_label(PreconditionerFamily.AMG, aggregation_amg_preconditioner)
        == f"AMG ({detail})"
    )


def test_preconditioner_label_falls_back_to_bare_abbreviation_without_detail() -> None:
    """Labels fall back to the bare abbreviation when there is no structural detail."""
    label = preconditioner_label(PreconditionerType.IDENTITY, Identity())

    assert label == "Identity"


@pytest.mark.parametrize(
    "family",
    [PreconditionerFamily.AMG],
)
def test_preconditioner_label_stays_within_length_budget_for_amg(
    family: PreconditionerFamily, aggregation_amg_preconditioner: AMGPreconditioner
) -> None:
    """AMG legend entries stay short even with multiple theta variants compared side by side.

    Long per-entry legend text is what actually motivated dropping grid
    levels/cycle/smoother detail from ``describe_preconditioner`` — this is
    the regression check for that: a length property, not a wording check.
    """
    label = preconditioner_label(family, aggregation_amg_preconditioner)

    assert len(label) <= MAX_LABEL_LENGTH


def test_preconditioner_label_stays_within_length_budget_for_pod(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G legend entries stay within the same length budget as AMG's."""
    label = preconditioner_label(PreconditionerFamily.POD2G, pod_amg_preconditioner)

    assert len(label) <= MAX_LABEL_LENGTH


def test_preconditioner_label_uses_pod2g_abbreviation(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G always renders as the fixed abbreviation "POD-2G", from the family alone.

    Unlike the previous config-name-based approach, the base name no longer
    depends on any string a caller happened to put in ``cfg.name`` (e.g. an
    assignment display name carrying the fitted dataset's id) — the dataset
    is already conveyed by the legend's color/marker key, and the rank by
    the ``c=`` detail this label is combined with.
    """
    label = preconditioner_label(PreconditionerFamily.POD2G, pod_amg_preconditioner)

    assert label.startswith("POD-2G ")


def test_preconditioner_label_stays_within_length_budget_for_target_dim(
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Target-dimension-coarsening legend entries stay within the same length budget."""
    label = preconditioner_label(PreconditionerFamily.AMG, target_dim_amg_preconditioner)

    assert len(label) <= MAX_LABEL_LENGTH


def test_preconditioner_label_for_adaptive_sa_uses_asa_abbreviation(
    adaptive_sa_preconditioner: AdaptiveSAPreconditioner,
) -> None:
    """Adaptive SA-AMG renders with the "aSA" abbreviation from its type."""
    label = preconditioner_label(PreconditionerType.ADAPTIVE_SA_AMG, adaptive_sa_preconditioner)

    assert label.startswith("aSA ")


def test_preconditioner_label_for_bootstrap_amg_uses_bamg_abbreviation(
    bootstrap_amg_preconditioner: BootstrapAMGPreconditioner,
) -> None:
    """Bootstrap AMG renders with the "BAMG" abbreviation from its type."""
    label = preconditioner_label(PreconditionerType.BOOTSTRAP_AMG, bootstrap_amg_preconditioner)

    assert label.startswith("BAMG ")


@pytest.mark.parametrize("member", list(PreconditionerFamily))
def test_preconditioner_family_abbreviation_is_exhaustive(member: PreconditionerFamily) -> None:
    """Every `PreconditionerFamily` member has a non-empty abbreviation.

    A future member without a matching `abbreviation()` arm fails loudly here
    (and via `ty`'s exhaustiveness check on the `match`) rather than raising
    at label-rendering time.
    """
    assert member.abbreviation()


@pytest.mark.parametrize("member", list(PreconditionerType))
def test_preconditioner_type_abbreviation_is_exhaustive(member: PreconditionerType) -> None:
    """Every `PreconditionerType` member has a non-empty abbreviation.

    Same exhaustiveness guarantee as
    ``test_preconditioner_family_abbreviation_is_exhaustive``, for the finer
    ``PreconditionerType`` enum.
    """
    assert member.abbreviation()


def test_describe_preconditioner_pod_uses_c_not_rank_or_level_count(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G's coarse dimension is rendered as ``c=``, matching AMG's terminology.

    ``L=`` is also omitted: ``PODCoarseningStrategy`` is architecturally a
    single-basis two-grid method, so a constant level count would be noise,
    not signal, in a legend comparing POD-2G variants against each other or
    against AMG.
    """
    detail = describe_preconditioner(pod_amg_preconditioner)

    assert "c=" in detail
    assert "rank=" not in detail
    assert "L=" not in detail


def test_describe_preconditioner_amg_aggregation_still_shows_level_count(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """AMG-aggregation keeps ``L=`` — it's a real, independently-tunable knob there."""
    detail = describe_preconditioner(aggregation_amg_preconditioner)

    assert "L=" in detail


def test_build_preconditioner_labels_maps_each_name(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Each entry in the mapping equals what preconditioner_label returns for it."""
    preconditioners: dict[str, Preconditioner] = {
        "amg": aggregation_amg_preconditioner,
        "identity": Identity(),
    }
    families: dict[str, PreconditionerFamily | PreconditionerType] = {
        "amg": PreconditionerFamily.AMG,
        "identity": PreconditionerType.IDENTITY,
    }

    labels = build_preconditioner_labels(preconditioners, families)

    assert labels == {
        name: preconditioner_label(families[name], precond)
        for name, precond in preconditioners.items()
    }
