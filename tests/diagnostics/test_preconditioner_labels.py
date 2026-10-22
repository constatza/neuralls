"""Tests for descriptive preconditioner plot labels.

Follows project conventions: fixtures over inline literals, type hints
throughout, no ``tempfile`` (no file I/O needed here — pure object
introspection).
"""

from __future__ import annotations

from pathlib import Path

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

from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AggregationCoarseningConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.dlkit.predictor_adapter import DLKitNeuralPreconditioner, DLKitPredictor
from neuralls.platform.reporting.preconditioner_labels import (
    build_preconditioner_labels,
    coarsening_facts,
    configured_facts,
    contextualize_preconditioner_label,
    describe_preconditioner,
    preconditioner_label,
)
from neuralls.shared.scientific_description import (
    COARSE_DIMENSION,
    DAMPING,
    DROP_TOLERANCE,
    LEGEND_PROFILE,
    PRE_SMOOTHING_STEPS,
    THRESHOLD,
    DecimalFact,
    FieldImportance,
    IntegerFact,
    NotationPolicy,
    RenderTarget,
    TextFact,
)
from neuralls.shared.types import PreconditionerFamily

AGGREGATION = AggregationCoarseningConfig()

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
    return AdaptiveSAPreconditioner(max_levels=2, max_coarse=1, theta=0.0).setup(tridiag_spd_matrix)


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
    return BootstrapAMGPreconditioner(max_coarse=5, seed=0).setup(tridiag_spd_matrix_20)


# ==============================================================================
# coarsening_facts — realized values, no string parsing required to verify
# ==============================================================================


def test_coarsening_facts_report_realized_coarse_dimension_for_aggregation(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Aggregation coarsening's realized coarse dimension is read back, not guessed.

    ``theta`` is a strength-of-connection threshold, not a chosen dimension —
    the only way to know how many aggregates it produced is to build the
    transfer operator and check its shape (2 here: one boundary pair plus one
    interior triple under torchalg's three-pass standard aggregation).
    """
    facts = coarsening_facts(
        aggregation_amg_preconditioner.coarsening, aggregation_amg_preconditioner.matrix
    )

    essential = FieldImportance.ESSENTIAL
    assert facts == (
        DecimalFact(THRESHOLD, 0.25, essential),
        DecimalFact(DAMPING, 0.67, essential),
        IntegerFact(COARSE_DIMENSION, 2, essential),
    )


def test_coarsening_facts_report_auto_omega_as_text_not_a_guessed_float(
    auto_omega_aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """An unset omega is "auto" — torchalg resolves it per matrix; the label must not invent a value."""
    facts = coarsening_facts(
        auto_omega_aggregation_amg_preconditioner.coarsening,
        auto_omega_aggregation_amg_preconditioner.matrix,
    )

    assert TextFact(DAMPING, "auto", FieldImportance.ESSENTIAL) in facts


def test_coarsening_facts_report_fitted_rank_for_pod(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    """POD-2G's fact is its actual fitted basis width (2), not the configured threshold (0.999)."""
    facts = coarsening_facts(pod_amg_preconditioner.coarsening, pod_amg_preconditioner.matrix)

    assert facts == (IntegerFact(COARSE_DIMENSION, 2, FieldImportance.ESSENTIAL),)


def test_coarsening_facts_report_realized_dimension_for_target_dim(
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    coarsening = target_dim_amg_preconditioner.coarsening
    assert isinstance(coarsening, TargetDimensionCoarsening)

    facts = coarsening_facts(coarsening, target_dim_amg_preconditioner.matrix)

    realized = coarsening.realized_coarse_dim(target_dim_amg_preconditioner.matrix)
    assert facts == (IntegerFact(COARSE_DIMENSION, realized, FieldImportance.ESSENTIAL),)


# ==============================================================================
# describe_preconditioner — realized facts of the built object
# ==============================================================================


def test_describe_amg_aggregation_renders_theta_omega_and_coarse_dimension(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    assert describe_preconditioner(aggregation_amg_preconditioner) == "θ=0.25, ω=0.67, c=2"


def test_describe_amg_aggregation_renders_auto_omega(
    auto_omega_aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    assert "ω=auto" in describe_preconditioner(auto_omega_aggregation_amg_preconditioner)


def test_describe_pod_uses_c_only(pod_amg_preconditioner: AMGPreconditioner) -> None:
    """POD-2G's coarse dimension is ``c=``; no ``rank=`` word, no level count."""
    assert describe_preconditioner(pod_amg_preconditioner) == "c=2"


def test_describe_target_dim_reports_c(target_dim_amg_preconditioner: AMGPreconditioner) -> None:
    assert "c=" in describe_preconditioner(target_dim_amg_preconditioner)


def test_describe_adaptive_sa_reports_realized_hierarchy_not_the_placeholder_coarsening(
    adaptive_sa_preconditioner: AdaptiveSAPreconditioner,
) -> None:
    """aSA stores torchalg's prebuilt placeholder as its coarsening; its own registration wins by MRO."""
    detail = describe_preconditioner(adaptive_sa_preconditioner)

    realized = adaptive_sa_preconditioner.result.matrices[-1].shape[0]
    assert "Prebuilt" not in detail
    assert "L=" in detail
    assert "k=" in detail
    assert f"c={realized}" in detail


def test_describe_bootstrap_amg_reports_realized_hierarchy_not_the_placeholder_coarsening(
    bootstrap_amg_preconditioner: BootstrapAMGPreconditioner,
) -> None:
    detail = describe_preconditioner(bootstrap_amg_preconditioner)

    realized = bootstrap_amg_preconditioner.result.matrices[-1].shape[0]
    assert "Prebuilt" not in detail
    assert "L=" in detail
    assert "k_r=" in detail
    assert f"c={realized}" in detail


def test_describe_plain_preconditioners_fabricate_nothing(
    tridiag_spd_matrix: torch.Tensor,
) -> None:
    assert describe_preconditioner(Identity()) == ""
    assert describe_preconditioner(JacobiPreconditioner().setup(tridiag_spd_matrix)) == ""


def test_describe_unwraps_scheduled_preconditioner(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
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
# configured_facts — config-driven, defaults omitted, seeds never shown
# ==============================================================================


def test_ic0_default_threshold_shows_nothing() -> None:
    assert configured_facts(IC0PreconditionerConfig()) == ()


def test_ic0_non_default_threshold_shows_tau() -> None:
    facts = configured_facts(IC0PreconditionerConfig(threshold=1e-4))

    assert facts == (DecimalFact(DROP_TOLERANCE, 1e-4, FieldImportance.ESSENTIAL),)


def test_amg_default_shows_only_the_level_count() -> None:
    facts = configured_facts(AMGPreconditionerConfig(coarsening=AGGREGATION))

    assert [fact.key.value for fact in facts] == ["multigrid_levels"]


def test_pod_amg_has_no_level_count() -> None:
    """POD-2G is single-basis two-grid, so a level count would be constant noise."""
    cfg = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=Path("snapshots")))

    assert configured_facts(cfg) == ()


def test_amg_smoothing_steps_are_variant_facts_only_when_non_default() -> None:
    cfg = AMGPreconditionerConfig(coarsening=AGGREGATION, pre_smoothing_steps=3)

    facts = configured_facts(cfg)

    assert IntegerFact(PRE_SMOOTHING_STEPS, 3, FieldImportance.VARIANT) in facts
    assert all(fact.key.value != "post_smoothing_steps" for fact in facts)


@pytest.mark.parametrize(
    ("default", "tuned"),
    [
        (AdaptiveSAPreconditionerConfig(), AdaptiveSAPreconditionerConfig(theta=0.1)),
        (AdaptiveSAPreconditionerConfig(), AdaptiveSAPreconditionerConfig(candidate_iters=9)),
        (BootstrapAMGPreconditionerConfig(), BootstrapAMGPreconditionerConfig(eta=8)),
        (
            BootstrapAMGPreconditionerConfig(),
            BootstrapAMGPreconditionerConfig(n_bootstrap_cycles=5),
        ),
    ],
)
def test_adaptive_and_bootstrap_settings_show_only_when_tuned(
    default: PreconditionerConfig, tuned: PreconditionerConfig
) -> None:
    assert configured_facts(default) == ()  # type: ignore[arg-type]
    tuned_facts = configured_facts(tuned)  # type: ignore[arg-type]
    assert len(tuned_facts) == 1
    assert tuned_facts[0].importance is FieldImportance.VARIANT


@pytest.mark.parametrize(
    "cfg",
    [AdaptiveSAPreconditionerConfig(seed=7), BootstrapAMGPreconditionerConfig(seed=7)],
)
def test_seeds_never_appear(cfg: PreconditionerConfig) -> None:
    assert all("seed" not in fact.key.value for fact in configured_facts(cfg))  # type: ignore[arg-type]


# ==============================================================================
# preconditioner_label / build_preconditioner_labels
# ==============================================================================

_AMG_CFG = AMGPreconditionerConfig(coarsening=AGGREGATION)


def test_label_combines_family_configured_and_realized_facts(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    label = preconditioner_label(PreconditionerFamily.AMG, aggregation_amg_preconditioner, _AMG_CFG)

    assert label == "AMG (L=2, θ=0.25, ω=0.67, c=2)"


def test_amg_variants_that_differ_only_in_smoothing_get_distinct_labels(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    """They used to collide and fall back to a label with the raw ASCII key appended."""
    tuned = AMGPreconditionerConfig(coarsening=AGGREGATION, pre_smoothing_steps=3)

    default_label = preconditioner_label(
        PreconditionerFamily.AMG, aggregation_amg_preconditioner, _AMG_CFG
    )
    tuned_label = preconditioner_label(
        PreconditionerFamily.AMG, aggregation_amg_preconditioner, tuned
    )

    assert default_label != tuned_label
    assert tuned_label.endswith("ν1=3)")


def test_label_with_no_facts_is_the_bare_abbreviation() -> None:
    cfg = StandardPreconditionerConfig(type=PreconditionerType.IDENTITY)

    assert preconditioner_label(PreconditionerType.IDENTITY, Identity(), cfg) == "Identity"


def test_ic0_label_hides_the_default_threshold_and_shows_a_tuned_one() -> None:
    from torchalg.preconditioners.implementations import IC0Preconditioner

    default_label = preconditioner_label(
        PreconditionerType.IC0, IC0Preconditioner(), IC0PreconditionerConfig()
    )
    tuned_label = preconditioner_label(
        PreconditionerType.IC0,
        IC0Preconditioner(threshold=1e-4),
        IC0PreconditionerConfig(threshold=1e-4),
    )

    assert default_label == "IC0"
    assert tuned_label == "IC0 (τ=0.0001)"


def test_label_honours_the_configured_notation_policy(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    mathtext = NotationPolicy.default(RenderTarget.MATHTEXT)

    label = preconditioner_label(
        PreconditionerFamily.AMG, aggregation_amg_preconditioner, _AMG_CFG, policy=mathtext
    )

    assert "$\\theta=0.25$" in label
    assert "$\\omega=0.67$" in label


def test_label_honours_a_symbol_override(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    from neuralls.shared.scientific_description import SymbolSpec

    policy = NotationPolicy.default().with_overrides({THRESHOLD: SymbolSpec(plain="th", math="th")})

    label = preconditioner_label(
        PreconditionerFamily.AMG, aggregation_amg_preconditioner, _AMG_CFG, policy=policy
    )

    assert "th=0.25" in label


def test_every_label_stays_within_the_shared_legend_budget(
    aggregation_amg_preconditioner: AMGPreconditioner,
    pod_amg_preconditioner: AMGPreconditioner,
    target_dim_amg_preconditioner: AMGPreconditioner,
) -> None:
    """Even with every smoothing setting overridden, essential facts survive and the budget holds."""
    crowded = AMGPreconditionerConfig(
        coarsening=AGGREGATION, pre_smoothing_steps=7, post_smoothing_steps=9, smoother_omega=0.123
    )
    labels = [
        preconditioner_label(PreconditionerFamily.AMG, aggregation_amg_preconditioner, crowded),
        preconditioner_label(
            PreconditionerFamily.POD2G,
            pod_amg_preconditioner,
            AMGPreconditionerConfig(
                coarsening=PODCoarseningConfig(dataset_dir=Path("snapshots")), pre_smoothing_steps=7
            ),
        ),
        preconditioner_label(PreconditionerFamily.AMG, target_dim_amg_preconditioner, crowded),
    ]

    assert all(len(label) <= (LEGEND_PROFILE.max_length or 0) for label in labels)
    assert all("c=" in label for label in labels)


def test_pod_label_uses_the_pod2g_abbreviation_and_no_level_count(
    pod_amg_preconditioner: AMGPreconditioner,
) -> None:
    cfg = AMGPreconditionerConfig(coarsening=PODCoarseningConfig(dataset_dir=Path("snapshots")))

    label = preconditioner_label(PreconditionerFamily.POD2G, pod_amg_preconditioner, cfg)

    assert label == "POD-2G (c=2)"


def test_adaptive_sa_and_bamg_labels_use_their_abbreviations(
    adaptive_sa_preconditioner: AdaptiveSAPreconditioner,
    bootstrap_amg_preconditioner: BootstrapAMGPreconditioner,
) -> None:
    asa = preconditioner_label(
        PreconditionerType.ADAPTIVE_SA_AMG,
        adaptive_sa_preconditioner,
        AdaptiveSAPreconditionerConfig(),
    )
    bamg = preconditioner_label(
        PreconditionerType.BOOTSTRAP_AMG,
        bootstrap_amg_preconditioner,
        BootstrapAMGPreconditionerConfig(),
    )

    assert asa.startswith("aSA (")
    assert bamg.startswith("BAMG (")


def test_contextualize_appends_fit_dataset() -> None:
    label = contextualize_preconditioner_label(
        "POD-2G (c=10)", "smoother-filtered-probes-spheres-1000x"
    )

    assert label == "POD-2G (c=10) | smoother-filtered-probes-spheres-1000x"


def test_contextualize_preserves_unqualified_label() -> None:
    assert contextualize_preconditioner_label("Jacobi", None) == "Jacobi"


# ==============================================================================
# Neural preconditioners — architecture class name
# ==============================================================================


class _GraphNet(torch.nn.Module):
    """Stand-in model whose class name is the architecture shown in the label."""


class _OtherNet(torch.nn.Module):
    """A second architecture, to prove two neural variants no longer collide."""


class _FakeCheckpointPredictor:
    feature_names: tuple[str, ...] = ()

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def unload(self) -> None:
        """Nothing to release; present so `DLKitPredictor.cleanup()` runs quietly."""


class _FakeAdapter:
    """Hands out a DLKitPredictor wrapping a given model, with no checkpoint or DLKit involved."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def create_predictor(self, checkpoint_path, config_path=None, data_config_path=None):
        return DLKitPredictor(
            _FakeCheckpointPredictor(self._model),  # type: ignore[arg-type]
            "cpu",
            model_class_name=type(self._model).__name__,
        )


def _neural(model: torch.nn.Module) -> DLKitNeuralPreconditioner:
    return DLKitNeuralPreconditioner(_FakeAdapter(model).create_predictor(Path("unused.ckpt")))


def test_neural_label_names_the_model_architecture() -> None:
    cfg = NeuralPreconditionerConfig(assignment="pod-run")

    label = preconditioner_label(PreconditionerFamily.NEURAL, _neural(_GraphNet()), cfg)

    assert label == "Neural (assignment=pod-run, arch=_GraphNet)"


def test_two_neural_variants_with_different_architectures_get_distinct_labels() -> None:
    cfg = NeuralPreconditionerConfig(assignment="pod-run")

    first = preconditioner_label(PreconditionerFamily.NEURAL, _neural(_GraphNet()), cfg)
    second = preconditioner_label(PreconditionerFamily.NEURAL, _neural(_OtherNet()), cfg)

    assert first != second


def test_neural_architecture_with_an_underscore_survives_mathtext() -> None:
    """An underscore would be a subscript in mathtext and break the fragment."""
    mathtext = NotationPolicy.default(RenderTarget.MATHTEXT)
    cfg = NeuralPreconditionerConfig(assignment="pod-run")

    label = preconditioner_label(
        PreconditionerFamily.NEURAL, _neural(_GraphNet()), cfg, policy=mathtext
    )

    assert "assignment=pod-run" in label
    assert "\\_GraphNet" in label


def test_build_preconditioner_labels_keeps_neural_assignments_distinct() -> None:
    configs = {
        "network_a": NeuralPreconditionerConfig(assignment="network_a"),
        "network_b": NeuralPreconditionerConfig(assignment="network_b"),
    }
    preconditioners = {name: _neural(_GraphNet()) for name in configs}
    families = dict.fromkeys(configs, PreconditionerFamily.NEURAL)

    labels = build_preconditioner_labels(preconditioners, families, configs)

    assert labels == {
        "network_a": "Neural (assignment=network_a, arch=_GraphNet)",
        "network_b": "Neural (assignment=network_b, arch=_GraphNet)",
    }


# ==============================================================================
# Exhaustiveness and mapping
# ==============================================================================


@pytest.mark.parametrize("member", list(PreconditionerFamily))
def test_preconditioner_family_abbreviation_is_exhaustive(member: PreconditionerFamily) -> None:
    """A future member without an `abbreviation()` arm fails loudly here, not at render time."""
    assert member.abbreviation()


@pytest.mark.parametrize("member", list(PreconditionerType))
def test_preconditioner_type_abbreviation_is_exhaustive(member: PreconditionerType) -> None:
    assert member.abbreviation()


def test_build_preconditioner_labels_maps_each_name(
    aggregation_amg_preconditioner: AMGPreconditioner,
) -> None:
    preconditioners: dict[str, Preconditioner] = {
        "amg": aggregation_amg_preconditioner,
        "identity": Identity(),
    }
    families: dict[str, PreconditionerFamily | PreconditionerType] = {
        "amg": PreconditionerFamily.AMG,
        "identity": PreconditionerType.IDENTITY,
    }
    configs: dict[str, PreconditionerConfig] = {
        "amg": _AMG_CFG,
        "identity": StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
    }

    labels = build_preconditioner_labels(preconditioners, families, configs)

    assert labels == {
        name: preconditioner_label(families[name], precond, configs[name])
        for name, precond in preconditioners.items()
    }
