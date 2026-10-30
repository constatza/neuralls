"""Tests for the derived, never-user-settable preconditioner `name` (identity slug).

`name` used to be an arbitrary user-typed string; it's now a
``@computed_field`` derived from each config's own type and distinguishing
fields (each config's ``slug()``).
"""

from __future__ import annotations

import pytest

from neuralls.platform.config.models.checkpoint_refs import (
    ByAlias,
    ByRunId,
    LoggedModelRefConfig,
    RegisteredModelRefConfig,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.config.models.coarsening import (
    AggregationCoarseningConfig,
    NeuralPODCoarseningConfig,
    PODCoarseningConfig,
    TargetDimCoarseningConfig,
)
from neuralls.platform.config.models.preconditioner import (
    AdaptiveSAPreconditionerConfig,
    AMGPreconditionerConfig,
    BootstrapAMGPreconditionerConfig,
    IC0PreconditionerConfig,
    NeuralAMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    NeuralTransferConfig,
    PreconditionerType,
    StandardPreconditionerConfig,
)
from neuralls.platform.config.models.weighting import PowerNormWeightingConfig


def test_standard_preconditioner_slug_is_bare_type() -> None:
    """No distinguishing fields -> the slug is just the type."""
    assert StandardPreconditionerConfig(type=PreconditionerType.IDENTITY).name == "identity"
    assert StandardPreconditionerConfig(type=PreconditionerType.JACOBI).name == "jacobi"


def test_ic0_slug_includes_threshold() -> None:
    """IC0's only distinguishing field, the drop threshold, appears in its slug."""
    assert IC0PreconditionerConfig(threshold=0.001).name == "ic0_threshold0p001"
    assert (
        IC0PreconditionerConfig(threshold=0.002).name
        != IC0PreconditionerConfig(threshold=0.001).name
    )


def test_ic0_slug_is_ascii_for_large_and_small_exponents() -> None:
    """A `%g`-formatted exponent's `+`/`-` never leaks into the slug (MLflow-safe)."""
    assert IC0PreconditionerConfig(threshold=1e8).name == "ic0_threshold1e08"
    assert IC0PreconditionerConfig(threshold=1e-8).name == "ic0_threshold1eneg08"
    assert IC0PreconditionerConfig(threshold=1e8).name.isascii()


def test_amg_aggregation_slug_includes_theta_and_omega() -> None:
    """AMG aggregation coarsening's slug reflects theta/omega, including the auto-omega case."""
    cfg = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25, omega=0.67))
    assert "aggregation_theta0p25_omega0p67" in cfg.name

    auto_omega = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(theta=0.25))
    assert "aggregation_theta0p25_omegaauto" in auto_omega.name


def test_amg_slug_includes_levels_and_smoothing_steps() -> None:
    """AMG's own structural knobs (not just coarsening) distinguish otherwise-identical configs.

    Regression test: these fields were originally omitted from the slug,
    so two AMG configs differing only in e.g. `n_levels` collided on
    `name` and falsely tripped `validate_unique_preconditioner_keys`.
    """
    base = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig())
    more_levels = AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig(), n_levels=3)
    more_pre = AMGPreconditionerConfig(
        coarsening=AggregationCoarseningConfig(), pre_smoothing_steps=5
    )
    different_smoother_omega = AMGPreconditionerConfig(
        coarsening=AggregationCoarseningConfig(), smoother_omega=0.5
    )
    names = {base.name, more_levels.name, more_pre.name, different_smoother_omega.name}
    assert len(names) == 4


def test_amg_target_dim_slug_includes_target() -> None:
    """Target-dimension coarsening's slug reflects the requested target, not a realized value."""
    cfg = AMGPreconditionerConfig(coarsening=TargetDimCoarseningConfig(target_coarse_dim=10))
    assert "targetdim10" in cfg.name


def test_amg_pod_slug_includes_rank_and_weighting() -> None:
    """POD-2G coarsening's slug reflects rank and snapshot weighting."""
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir="/data/solutions",
            rank=8,
            weighting=PowerNormWeightingConfig(metric="l2", beta=1.0),
        )
    )
    assert "pod_rank8_n-1_powernorm_l2_beta1" in cfg.name


def test_amg_pod_slug_includes_checkpoint_identity_when_fitted() -> None:
    """A checkpoint-backed POD basis's slug carries its checkpoint identity, not just rank."""
    cfg = AMGPreconditionerConfig(
        coarsening=PODCoarseningConfig(
            dataset_dir="/data/solutions",
            rank=8,
            assignment="pod2g_run1",
        )
    )
    assert "assignment_pod2g_run1" in cfg.name


def test_neural_pod_slug_always_includes_checkpoint_identity() -> None:
    """Neural POD is always checkpoint-backed, so its slug always carries a checkpoint ref."""
    cfg = AMGPreconditionerConfig(
        coarsening=NeuralPODCoarseningConfig(
            dataset_dir="/data/params",
            input_names=("param",),
            rank=4,
            model_ref=RegisteredModelRefConfig(name="PodNet", selector=ByAlias(alias="best")),
        )
    )
    assert "neuralpod_rank4_n-1_registry_podnet_best" in cfg.name


def test_adaptive_sa_slug_includes_num_candidates_theta_max_coarse() -> None:
    """alpha-SA's slug reflects its real coarse-space-size knobs."""
    cfg = AdaptiveSAPreconditionerConfig(num_candidates=3, theta=0.05, max_coarse=10)
    assert "k3" in cfg.name
    assert "theta0p05" in cfg.name
    assert "maxcoarse10" in cfg.name


def test_adaptive_sa_slug_includes_previously_omitted_fields() -> None:
    """Regression test: candidate_iters/omega/seed/n_levels must also distinguish configs."""
    base = AdaptiveSAPreconditionerConfig()
    different_iters = AdaptiveSAPreconditionerConfig(candidate_iters=99)
    different_omega = AdaptiveSAPreconditionerConfig(omega=1.0)
    different_seed = AdaptiveSAPreconditionerConfig(seed=42)
    different_levels = AdaptiveSAPreconditionerConfig(n_levels=4)
    names = {
        base.name,
        different_iters.name,
        different_omega.name,
        different_seed.name,
        different_levels.name,
    }
    assert len(names) == 5


def test_bootstrap_amg_slug_includes_kr_cycles_max_coarse() -> None:
    """Bootstrap AMG's slug reflects its own structural knobs."""
    cfg = BootstrapAMGPreconditionerConfig(k_r=8, n_bootstrap_cycles=2, max_coarse=10)
    assert "kr8" in cfg.name
    assert "cycles2" in cfg.name
    assert "maxcoarse10" in cfg.name


def test_bootstrap_amg_slug_includes_previously_omitted_fields() -> None:
    """Regression test: eta/nu/delta/theta_ad/caliber/gamma/use_lsr/n_levels/seed must distinguish."""
    base = BootstrapAMGPreconditionerConfig()
    variants = [
        BootstrapAMGPreconditionerConfig(eta=99),
        BootstrapAMGPreconditionerConfig(nu=99),
        BootstrapAMGPreconditionerConfig(delta=0.1),
        BootstrapAMGPreconditionerConfig(theta_ad=0.1),
        BootstrapAMGPreconditionerConfig(caliber=99),
        BootstrapAMGPreconditionerConfig(gamma=0.1),
        BootstrapAMGPreconditionerConfig(use_lsr=False),
        BootstrapAMGPreconditionerConfig(n_levels=3),
        BootstrapAMGPreconditionerConfig(seed=42),
    ]
    names = {base.name, *(variant.name for variant in variants)}
    assert len(names) == 1 + len(variants)


def test_neural_slug_distinguishes_different_checkpoints() -> None:
    """Two neural preconditioners with different checkpoints get different slugs.

    This was the concrete gap found during review: the live-object-based
    render (`describe_preconditioner`) can't distinguish two neural
    preconditioners at all (always renders empty detail) — but the
    config-only slug can, because it reads `model_ref`/`assignment`
    directly from the config instead of a constructed object.
    """
    first = NeuralPreconditionerConfig(
        model_ref=RegisteredModelRefConfig(name="FFNN", selector=ByAlias(alias="best"))
    )
    second = NeuralPreconditionerConfig(
        model_ref=RegisteredModelRefConfig(name="FFNN", selector=ByAlias(alias="other"))
    )
    assert first.name != second.name


def test_neural_slug_prefers_logical_ref_over_resolved_local_path() -> None:
    """A resolved local checkpoint_path never overrides the stable model_ref-based slug.

    Checkpoint resolution (`model_resolution.py`) fills in `checkpoint_path`
    with a resolved *local* path once a `model_ref` is resolved — a path
    that can differ run to run for the same logical reference. The slug
    must stay keyed on the logical reference so the same comparison re-run
    later gets the same MLflow metric-key segment.
    """
    before = NeuralPreconditionerConfig(
        model_ref=LoggedModelRefConfig(selector=ByRunId(run_id="abc123")),
    )
    after = before.model_copy(update={"checkpoint_path": "/tmp/some/resolved/path.ckpt"})
    assert before.name == after.name


def test_neural_amg_slug_includes_both_transfer_operators() -> None:
    """Neural AMG's slug reflects both prolongation and restriction when both are set."""
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            model_ref=RegisteredModelRefConfig(name="ProlongNet", selector=ByAlias(alias="best"))
        ),
        restriction=NeuralTransferConfig(
            model_ref=RegisteredModelRefConfig(name="RestrictNet", selector=ByAlias(alias="best"))
        ),
    )
    assert "registry_prolongnet_best" in cfg.name
    assert "registry_restrictnet_best" in cfg.name


def test_neural_amg_slug_omits_restriction_when_unset() -> None:
    """Neural AMG with no explicit restriction (shared transfer) has no restriction segment."""
    cfg = NeuralAMGPreconditionerConfig(
        prolongation=NeuralTransferConfig(
            model_ref=RegisteredModelRefConfig(name="ProlongNet", selector=ByAlias(alias="best"))
        ),
    )
    assert cfg.name.endswith("registry_prolongnet_best")
    assert "restrictnet" not in cfg.name


def test_neural_amg_slug_includes_previously_omitted_fields() -> None:
    """Regression test: cycle/smoothing fields must also distinguish Neural AMG configs."""
    prolongation = NeuralTransferConfig(
        model_ref=RegisteredModelRefConfig(name="ProlongNet", selector=ByAlias(alias="best"))
    )
    base = NeuralAMGPreconditionerConfig(prolongation=prolongation)
    different_levels = NeuralAMGPreconditionerConfig(prolongation=prolongation, n_levels=3)
    different_pre = NeuralAMGPreconditionerConfig(prolongation=prolongation, pre_smoothing_steps=5)
    different_agg_omega = NeuralAMGPreconditionerConfig(
        prolongation=prolongation, aggregation_omega=0.5
    )
    names = {base.name, different_levels.name, different_pre.name, different_agg_omega.name}
    assert len(names) == 4


def test_trained_assignment_ref_slug_includes_assignment_id() -> None:
    """A TrainedAssignmentRefConfig model_ref's slug carries its assignment id."""
    cfg = NeuralPreconditionerConfig(
        model_ref=TrainedAssignmentRefConfig(assignment_id="ffnn_solutions")
    )
    assert cfg.name == "neural_assignment_ffnn_solutions"


def test_identical_configs_collide_on_name() -> None:
    """Two configs with identical type and fields are the same preconditioner by construction."""
    first = IC0PreconditionerConfig(threshold=0.001)
    second = IC0PreconditionerConfig(threshold=0.001)
    assert first.name == second.name


@pytest.mark.parametrize(
    "build",
    [
        lambda: StandardPreconditionerConfig(type=PreconditionerType.IDENTITY),
        lambda: IC0PreconditionerConfig(threshold=0.001),
        lambda: NeuralPreconditionerConfig(
            model_ref=RegisteredModelRefConfig(name="FFNN", selector=ByAlias(alias="best"))
        ),
        lambda: AMGPreconditionerConfig(coarsening=AggregationCoarseningConfig()),
        lambda: AdaptiveSAPreconditionerConfig(),
        lambda: BootstrapAMGPreconditionerConfig(),
    ],
)
def test_name_ignores_a_legacy_name_field(build) -> None:
    """A stray `name=` key (old TOML files) never changes the derived name."""
    cfg = build()
    with_legacy_name = type(cfg).model_validate({**cfg.model_dump(), "name": "some-legacy-name"})
    assert with_legacy_name.name == cfg.name
