"""Complete scientific descriptions of every preconditioner config and built object."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
import torch
from pydantic import BaseModel
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
    VCycle,
)
from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy

from neuralls.composition.preconditioners.descriptions import (
    MissingArchitectureNameError,
    describe_configured_preconditioner,
    describe_realized_preconditioner,
)
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    CoarseningConfig,
    ConcretePreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPODCoarseningConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PODCoarseningConfig,
    PowerNormWeightingConfig,
    PreconditionerType,
    SmootherPersistenceWeightingConfig,
    SnapshotWeightingConfig,
    StandardPreconditionerConfig,
    TargetDimCoarseningConfig,
)
from neuralls.shared.fact_binding import Fact, unbound_fields
from neuralls.shared.scientific_description.catalog import (
    COARSE_DIMENSION,
    REALIZED_LEVELS,
)
from neuralls.shared.scientific_description.models import ScientificDescription
from neuralls.shared.scientific_description.policy import (
    NotationPolicy,
    RenderProfile,
    RenderTarget,
)
from neuralls.shared.scientific_description.render import render

pytestmark = pytest.mark.deferred_io("generation-io")

PLAIN = NotationPolicy.default(RenderTarget.PLAIN)
PROFILE = RenderProfile(name="description", include_context=True)


def _plain(description: ScientificDescription) -> str:
    return render(description, PROFILE, PLAIN).text


@pytest.fixture
def pod_dataset_dir(tmp_path: Path) -> Path:
    """A stand-in POD fit-dataset directory (the config validator does not require it to exist)."""
    return tmp_path / "pod-fit-dataset"


# --- exact labels pinned by the display contract ----------------------------------


def test_amg_label_lists_every_algorithmic_parameter() -> None:
    cfg = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.05))
    assert (
        _plain(describe_configured_preconditioner(cfg)) == "AMG (L=2, θ=0.05, ν₁=2, ν₂=2, ωₛ=auto)"
    )


def test_asa_label_includes_candidates_levels_and_coarse_limit() -> None:
    cfg = AdaptiveSAPreconditionerConfig(theta=0.05)
    assert (
        _plain(describe_configured_preconditioner(cfg))
        == "aSA (L=2, k=3, nᵢₜ=5, cₘₐₓ=10, θ=0.05, ω=1.33, s=0)"
    )


def test_bamg_label_includes_every_numerical_and_categorical_control() -> None:
    cfg = BootstrapAMGPreconditionerConfig(n_levels=10, theta_ad=0.5)
    assert _plain(describe_configured_preconditioner(cfg)) == (
        "BAMG (L=10, kᵣ=8, η=4, nᵦ=2, ν=5, δ=0.7, θₐd=0.5, q=4, γ=1.5, LSR=on, cₘₐₓ=10, s=0)"
    )


def test_pod_label_shows_rank_without_dataset_or_identity(pod_dataset_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(dataset_dir=pod_dataset_dir, rank=10, assignment="fit-a")
    )
    label = _plain(describe_configured_preconditioner(cfg))
    assert label == "POD-2G (r=10, ν₁=2, ν₂=2, ωₛ=auto)"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (PreconditionerType.IDENTITY, "Identity"),
        (PreconditionerType.JACOBI, "Jacobi"),
    ],
)
def test_parameterless_methods_invent_no_parameters(
    kind: PreconditionerType, expected: str
) -> None:
    assert (
        _plain(describe_configured_preconditioner(StandardPreconditionerConfig(type=kind)))
        == expected
    )


def test_ic0_label_shows_drop_tolerance_even_at_its_default() -> None:
    label = _plain(describe_configured_preconditioner(IC0PreconditionerConfig()))
    assert label == "IC0 (τ=0)"


# --- explicit-default equals omitted-default; any difference is visible ------------


def test_explicit_default_equals_omitted_default() -> None:
    omitted = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    explicit = AMGPreconditionerConfig(
        n_levels=2,
        pre_smoothing_steps=2,
        post_smoothing_steps=2,
        coarsening=AggregationCoarseningConfig(theta=0.25),
    )
    assert describe_configured_preconditioner(omitted) == describe_configured_preconditioner(
        explicit
    )


@pytest.mark.parametrize(
    ("variant", "base"),
    [
        (AdaptiveSAPreconditionerConfig(num_candidates=7), AdaptiveSAPreconditionerConfig()),
        (AdaptiveSAPreconditionerConfig(n_levels=3), AdaptiveSAPreconditionerConfig()),
        (AdaptiveSAPreconditionerConfig(candidate_iters=9), AdaptiveSAPreconditionerConfig()),
        (AdaptiveSAPreconditionerConfig(max_coarse=20), AdaptiveSAPreconditionerConfig()),
        (AdaptiveSAPreconditionerConfig(seed=4), AdaptiveSAPreconditionerConfig()),
        (BootstrapAMGPreconditionerConfig(k_r=9), BootstrapAMGPreconditionerConfig()),
        (BootstrapAMGPreconditionerConfig(use_lsr=False), BootstrapAMGPreconditionerConfig()),
        (BootstrapAMGPreconditionerConfig(caliber=5), BootstrapAMGPreconditionerConfig()),
        (BootstrapAMGPreconditionerConfig(seed=2), BootstrapAMGPreconditionerConfig()),
    ],
)
def test_any_algorithmic_change_changes_the_description(
    variant: ConcretePreconditionerConfig, base: ConcretePreconditionerConfig
) -> None:
    assert describe_configured_preconditioner(variant) != describe_configured_preconditioner(base)


# --- scheduling --------------------------------------------------------------------


def test_scheduling_controls_appear_only_when_non_default() -> None:
    default = _plain(
        describe_configured_preconditioner(
            AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
        )
    )
    scheduled = _plain(
        describe_configured_preconditioner(
            AMGPreconditionerConfig(
                coarsening=AggregationCoarseningConfig(),
                start_iter=5,
                limit_iters=20,
                fallback=PreconditionerType.JACOBI,
            )
        )
    )
    assert default == "AMG (L=2, θ=0.25, ν₁=2, ν₂=2, ωₛ=auto)"
    assert scheduled == "AMG (L=2, θ=0.25, ν₁=2, ν₂=2, ωₛ=auto, k₀=5, kₘₐₓ=20, fb=Jacobi)"


# --- POD, weighting, neural --------------------------------------------------------


def test_pod_weighting_and_snapshot_controls_are_visible(pod_dataset_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=pod_dataset_dir,
            rank=0.999,
            n_snapshots=200,
            weighting=PowerNormWeightingConfig(metric="a", beta=1.0),
        )
    )
    label = _plain(describe_configured_preconditioner(cfg))
    assert label == "POD-2G (r=0.999, Nₛ=200, w=PN, p=A, β=1, ν₁=2, ν₂=2, ωₛ=auto)"


def test_smoother_persistence_weighting_shows_its_sweeps(pod_dataset_dir: Path) -> None:
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir=pod_dataset_dir,
            weighting=SmootherPersistenceWeightingConfig(steps=7),
        )
    )
    assert (
        _plain(describe_configured_preconditioner(cfg))
        == "POD-2G (r=8, w=SP, ν=7, ν₁=2, ν₂=2, ωₛ=auto)"
    )


def test_target_dimension_coarsening_shows_its_target() -> None:
    cfg = AMGPreconditionerConfig(coarsening=TargetDimCoarseningConfig(target_coarse_dim=12))
    assert (
        _plain(describe_configured_preconditioner(cfg)) == "AMG (L=2, cₜ=12, ν₁=2, ν₂=2, ωₛ=auto)"
    )


def test_neural_architecture_comes_from_the_architecture_name_never_a_class_name() -> None:
    cfg = NeuralPreconditionerConfig(architecture="Graph Net", assignment="job-1")
    label = _plain(describe_configured_preconditioner(cfg))
    assert label == "Neural (arch=Graph Net)"


def test_neural_without_an_architecture_name_fails_instead_of_guessing() -> None:
    with pytest.raises(MissingArchitectureNameError):
        describe_configured_preconditioner(NeuralPreconditionerConfig())


def test_neural_architecture_must_be_a_plain_name() -> None:
    with pytest.raises(ValueError):
        describe_configured_preconditioner(NeuralPreconditionerConfig(architecture="graph_net"))


def test_neural_pod_and_neural_amg_describe_their_own_parameters(pod_dataset_dir: Path) -> None:
    pod = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir=pod_dataset_dir, rank=6, input_names=("x",)
        )
    )
    amg = NeuralAMGPreconditionerConfig(
        aggregation_omega=0.7, prolongation=NeuralTransferConfig(assignment="p")
    )
    assert _plain(describe_configured_preconditioner(pod)) == "POD-2G (r=6, ν₁=2, ν₂=2, ωₛ=auto)"
    assert (
        _plain(describe_configured_preconditioner(amg))
        == "Neural (L=2, ω=0.7, ν₁=2, ν₂=2, ωₛ=auto)"
    )


# --- every config field is bound ----------------------------------------------------


def _config_models() -> list[type[BaseModel]]:
    """Every config model a preconditioner description walks, found from the unions."""
    unions = (
        ConcretePreconditionerConfig,
        get_args(CoarseningConfig)[0],
        get_args(SnapshotWeightingConfig)[0],
    )
    return [model for union in unions for model in get_args(union)]


@pytest.mark.parametrize("model", _config_models(), ids=lambda model: model.__name__)
def test_every_config_field_is_bound(model: type[BaseModel]) -> None:
    """A new field fails here until it states whether it is a fact."""
    assert unbound_fields(model) == ()


@pytest.mark.parametrize("model", _config_models(), ids=lambda model: model.__name__)
def test_identity_and_provenance_fields_are_never_facts(model: type[BaseModel]) -> None:
    for name in ("assignment", "checkpoint_path", "dataset_dir", "model_ref"):
        if name in model.model_fields:
            assert not any(isinstance(m, Fact) for m in model.model_fields[name].metadata)


# --- realized facts ---------------------------------------------------------------


@pytest.fixture
def tridiag_spd_matrix() -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix."""
    n = 5
    return (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )


@pytest.fixture
def aggregation_amg(tridiag_spd_matrix: torch.Tensor) -> AMGPreconditioner:
    cycle = VCycle(smoother=JacobiSmoother(omega=0.67), n_pre=2, n_post=2)
    return AMGPreconditioner(
        tridiag_spd_matrix, coarsening=AggregationCoarsening(omega=0.67), cycle=cycle, n_levels=2
    ).setup(tridiag_spd_matrix)


@pytest.fixture
def pod_amg(tridiag_spd_matrix: torch.Tensor) -> AMGPreconditioner:
    generator = torch.Generator().manual_seed(0)
    ensemble = torch.randn(2, 5, generator=generator, dtype=torch.float64)
    coeffs = torch.randn(6, 2, generator=generator, dtype=torch.float64)
    coarsening = PODCoarseningStrategy(rank=0.999)
    coarsening.fit(coeffs @ ensemble)
    cycle = VCycle(smoother=JacobiSmoother(omega=0.67), n_pre=2, n_post=2)
    return AMGPreconditioner(tridiag_spd_matrix, coarsening=coarsening, cycle=cycle, n_levels=2)


def test_realized_aggregation_reports_the_coarse_dimension(
    aggregation_amg: AMGPreconditioner,
) -> None:
    facts = describe_realized_preconditioner(aggregation_amg)
    assert [(fact.key, fact.value) for fact in facts] == [(COARSE_DIMENSION, 2)]


def test_realized_pod_reports_the_fitted_width_not_the_energy_threshold(
    pod_amg: AMGPreconditioner,
) -> None:
    facts = describe_realized_preconditioner(pod_amg)
    assert [(fact.key, fact.value) for fact in facts] == [(COARSE_DIMENSION, 2)]


def test_realized_hierarchy_of_adaptive_sa_reports_levels_and_coarsest_size(
    tridiag_spd_matrix: torch.Tensor,
) -> None:
    precond = AdaptiveSAPreconditioner(max_levels=2, max_coarse=1, theta=0.0).setup(
        tridiag_spd_matrix
    )
    facts = {fact.key: fact.value for fact in describe_realized_preconditioner(precond)}
    assert facts[REALIZED_LEVELS] == len(precond.result.matrices)
    assert facts[COARSE_DIMENSION] == int(precond.result.matrices[-1].shape[0])


def test_parameterless_and_scheduled_preconditioners_unwrap_correctly(
    tridiag_spd_matrix: torch.Tensor, aggregation_amg: AMGPreconditioner
) -> None:
    assert describe_realized_preconditioner(Identity()) == ()
    assert describe_realized_preconditioner(JacobiPreconditioner().setup(tridiag_spd_matrix)) == ()
    scheduled: Preconditioner = ScheduledPreconditioner(aggregation_amg, start_iter=1)
    assert describe_realized_preconditioner(scheduled) == describe_realized_preconditioner(
        aggregation_amg
    )


def test_realized_bootstrap_matches_its_hierarchy() -> None:
    n = 20
    matrix = (
        2 * torch.eye(n, dtype=torch.float64)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), 1)
        + torch.diag(-torch.ones(n - 1, dtype=torch.float64), -1)
    )
    precond = BootstrapAMGPreconditioner(max_coarse=5, seed=0).setup(matrix)
    facts = {fact.key: fact.value for fact in describe_realized_preconditioner(precond)}
    assert facts[REALIZED_LEVELS] == len(precond.result.matrices)
