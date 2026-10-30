"""Spec and checkpoint resolution for comparison entries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from dlkit.infrastructure.config.job_config import (
    FitJobConfig,
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from loguru import logger
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.assignments.assignment_subjects import (
    neural_spec_from_assignment,
    pod_fit_spec_from_assignment,
    resolve_assignment_inputs,
)
from neuralls.composition.assignments.comparison_topology import ComparisonTopology
from neuralls.composition.assignments.model_resolution import (
    AssignmentModelContext,
    TrainingLookup,
    resolve_preconditioner_models_with_warnings,
)
from neuralls.composition.assignments.presentation_preflight import (
    DatasetPresentationLoader,
)
from neuralls.composition.identity.training import training_identity
from neuralls.domain.identity import StageIdentity
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import (
    AssignmentEntry,
    CaseConfig,
)
from neuralls.platform.config.models.preconditioner import (
    CheckpointRefBearing,
    NeuralCheckpointRef,
    NeuralPreconditionerConfig,
    PreconditionerConfig,
)
from neuralls.platform.config.registry import get_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.artifact_access import (
    ArtifactLeaseManager,
    MlflowArtifactLeaseManager,
)
from neuralls.platform.tracking.extra_features import fetch_extra_feature_names
from neuralls.platform.tracking.mlflow_store import (
    MlflowIdentityStore,
)
from neuralls.platform.tracking.model_registry import build_registered_model_name
from neuralls.shared.digest import Digest, content_digest
from neuralls.shared.errors import StageInputsStale
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.types import (
    ComparisonRhsSourceKind,
)


def validate_neural_preconditioner(spec: Any) -> None:
    """Validate strict neural preconditioner requirements for comparisons."""
    if not isinstance(spec, NeuralPreconditionerConfig):
        return
    if spec.model_ref is None:
        raise ValueError(f"Neural preconditioner '{spec.name}' must define model_ref.")
    if spec.checkpoint_path is not None:
        raise ValueError(
            f"Neural preconditioner '{spec.name}' cannot use checkpoint_path/assignment "
            "legacy resolution in comparison configs."
        )


def resolve_neural_preconditioners(
    solver_specs: list[Any],
) -> list[PreconditionerConfig]:
    """Validate neural preconditioners and return unchanged specs."""
    for spec in solver_specs:
        validate_neural_preconditioner(spec)
    return solver_specs


def _iter_checkpoint_refs(
    specs: Sequence[PreconditionerConfig],
) -> Iterator[tuple[PreconditionerConfig, str, NeuralCheckpointRef]]:
    """Yield every ``(spec, label, ref)`` triple reachable from these specs.

    The single traversal of the `CheckpointRefBearing` protocol used by every
    consumer that asks a question about a comparison's checkpoint dependencies
    — "does anything need resolving", "which assignments are referenced",
    "what identity do the resolved checkpoints have". Specs that bear no
    checkpoint ref (static preconditioners, an unfitted POD coarsening) simply
    contribute nothing.
    """
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            continue
        for label, ref in spec.checkpoint_refs():
            yield spec, label, ref


def needs_model_resolution(specs: tuple[PreconditionerConfig, ...]) -> bool:
    """Return True when any checkpoint-bearing preconditioner needs model_ref lookup.

    Generic over `CheckpointRefBearing` (not `NeuralPreconditionerConfig`-only)
    so a checkpoint-backed `AMGPreconditionerConfig` (POD-2G fitted ahead of
    time via a `FitJobConfig` assignment) triggers resolution the same way a
    neural preconditioner does — one dispatch point, no per-kind branching.
    """
    return any(ref.model_ref is not None for _, _, ref in _iter_checkpoint_refs(specs))


def referenced_assignment_ids(
    specs: tuple[PreconditionerConfig, ...],
) -> tuple[str, ...]:
    """Return assignment ids explicitly referenced by checkpoint-bearing specs.

    Order is first-reference-wins, deduplicated — callers resolve one model
    context per id, and a stable order keeps their logs reproducible.
    """
    return tuple(
        dict.fromkeys(
            ref.assignment
            for _, _, ref in _iter_checkpoint_refs(specs)
            if ref.assignment is not None
        )
    )


def _assignment_training_identity(
    *,
    job_config_path: Path,
    data_config_path: Path,
    settings: NeurallsSettings,
) -> StageIdentity | None:
    """Derive an assignment's training identity, or ``None`` if its dataset is not ready.

    ``None`` makes strict model resolution fail for that assignment with a
    clear message (never a silent match against an unverifiable dataset).
    """
    try:
        return training_identity(
            job_config_path=job_config_path,
            data_config_path=data_config_path,
            settings=settings,
        )
    except FileNotFoundError, StageInputsStale:
        return None


def _build_master_assignment_contexts(
    assignment_ids: tuple[str, ...],
    settings: NeurallsSettings,
    *,
    master_cfg: CaseConfig,
    config_dir: Path,
) -> dict[str, AssignmentModelContext]:
    """Resolve dataset/model identity for all case-config assignments.

    Takes the case config and its resolution root directly, rather than a
    path to reload — callers that already have both in scope (the common
    case: this runs once per comparison entry needing model resolution)
    avoid re-reading and re-validating the same TOML file on every call.
    """
    contexts: dict[str, AssignmentModelContext] = {}
    for assignment_id in assignment_ids:
        binding = get_assignment_binding(master_cfg, config_dir, assignment_id)
        data_cfg = load_data_config(binding.data_config_path, settings)
        dataset_id = resolve_dataset_identity(
            data_cfg=data_cfg,
            config_path=binding.data_config_path,
        ).name
        contexts[assignment_id] = AssignmentModelContext(
            dataset_alias=dataset_id,
            model_name=build_registered_model_name(assignment_id),
            training=TrainingLookup(
                experiment=master_cfg.names.training,
                identity=_assignment_training_identity(
                    job_config_path=binding.job_config_path,
                    data_config_path=binding.data_config_path,
                    settings=settings,
                ),
            ),
        )
    return contexts


def existing_assignment_ids(specs: tuple[PreconditionerConfig, ...]) -> set[str]:
    """Return assignment ids already claimed by explicit checkpoint-bearing preconditioners."""
    return set(referenced_assignment_ids(specs))


@dataclass(frozen=True, slots=True)
class _AssignmentJobContext:
    """One assignment's resolved job settings plus its dataset directory."""

    job: AnyJobConfig
    dataset_dir: Path
    training: StageIdentity | None
    architecture: ValidatedLabel | None
    """The job registry entry's explicit ``architecture``; None when it declares none."""


def _resolve_assignment_job_context(
    entry: AssignmentEntry,
    *,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> _AssignmentJobContext:
    """Load one assignment's declared job and resolve its dataset directory.

    Both are needed to decide *what kind* of auto-generated preconditioner
    spec an assignment produces (`neural_specs_from_assignments`'s kind
    dispatch) and, for a `fit`-kind job, to populate the resulting
    `PODCoarseningConfig`'s required `dataset_dir`/`rank` fields.
    """
    inputs = resolve_assignment_inputs(entry, cfg=cfg, config_dir=config_dir, settings=settings)
    return _AssignmentJobContext(
        job=inputs.job,
        architecture=inputs.binding.architecture,
        dataset_dir=inputs.dataset_dir,
        training=_assignment_training_identity(
            job_config_path=inputs.binding.job_config_path,
            data_config_path=inputs.binding.data_config_path,
            settings=settings,
        ),
    )


def _extra_input_names_of_training_run(
    context: _AssignmentJobContext, *, experiment: str, client: MlflowClient
) -> tuple[str, ...]:
    """Read extra input names from the very run strict resolution will select.

    Uses the same identity lookup as model resolution, so the names can never
    come from a different run than the checkpoint that gets loaded.
    """
    if context.training is None:
        return ()
    reused = MlflowIdentityStore(
        tracking_uri=client.tracking_uri, experiment=experiment, require_checkpoint=True
    ).find(context.training)
    if reused is None or reused.run_id is None:
        return ()
    return fetch_extra_feature_names(reused.run_id, client=client)


def neural_specs_from_assignments(
    entries: Sequence[AssignmentEntry],
    claimed_ids: set[str],
    *,
    client: MlflowClient,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> list[PreconditionerConfig]:
    """Generate auto preconditioner stubs from assignment entries.

    Dispatches on each assignment's declared job kind: a `fit`-kind job
    (`FitJobConfig` — e.g. POD-2G basis fitting) yields an unresolved
    `AMGPreconditionerConfig`/`PODCoarseningConfig` stub; any other kind
    (today: `train`/`search`) yields the existing unresolved
    `NeuralPreconditionerConfig` stub. Both stubs reference the most recent
    run tagged with the assignment's id and get resolved to a concrete
    checkpoint later, symmetrically, by
    `resolve_preconditioner_models_with_warnings` (generic over
    `CheckpointRefBearing`). Closed for modification, open for extension: a
    future `Fittable` artifact kind adds one more dispatch branch here, not a
    new isinstance chain elsewhere.

    Args:
        entries: Assignment entries to convert.
        claimed_ids: Assignment ids already covered by explicit preconditioners.
        client: Configured MLflow client used to look up training run tags.
        cfg: The assignments' owning case config, for job/dataset resolution.
        config_dir: Directory `cfg`'s registry paths resolve relative to.
        settings: Runtime settings used to load each assignment's job/dataset config.

    Returns:
        List of auto-generated preconditioner configs, one per unclaimed entry.
    """
    specs: list[PreconditionerConfig] = []
    for entry in entries:
        if entry.id in claimed_ids:
            continue
        context = _resolve_assignment_job_context(
            entry, cfg=cfg, config_dir=config_dir, settings=settings
        )
        match context.job:
            case FitJobConfig() as job:
                specs.append(
                    pod_fit_spec_from_assignment(entry, job=job, dataset_dir=context.dataset_dir)
                )
            case TrainingJobConfig() | SearchJobConfig() | InferenceJobConfig():
                specs.append(
                    neural_spec_from_assignment(
                        entry,
                        extra_input_names=_extra_input_names_of_training_run(
                            context, experiment=cfg.names.training, client=client
                        ),
                        architecture=context.architecture,
                    )
                )
            case _ as unreachable:
                assert_never(unreachable)
    return specs


@dataclass(frozen=True, slots=True)
class ComparisonResolutionContext:
    """Everything needed to resolve one comparison entry's preconditioners.

    One instance is built once per `prepare_comparison_batch` call and threaded unchanged through
    `prepare_comparison_entry` -> `_resolve_comparison_specs` ->
    `_resolve_specs`, instead of each of those functions separately
    re-declaring `topology`/`case_config_path`/`settings`/`master_cfg` as
    their own loose parameters.

    Attributes:
        load_dataset: Reads a dataset's presentation from its directory, so scope and
            POD-2G method descriptions never touch a manifest directly.
        master_cfg: The already-loaded case config, when the caller has one
            in scope (the batch path) — avoids reloading it from
            `case_config_path` for every comparison entry. Standalone
            callers with no preloaded config leave this unset and pay one
            lazy reload, only when an assignment-bound model ref actually
            needs it.
        artifact_leases: Lease manager shared by every entry of a sweep, owned
            (entered and closed) by the caller. It downloads each MLflow
            artifact once however many entries reference it, and keeps the
            files alive until the sweep ends. Standalone callers leave this
            unset and each entry then owns a private manager.
    """

    topology: ComparisonTopology
    case_config_path: Path
    settings: NeurallsSettings
    load_dataset: DatasetPresentationLoader
    master_cfg: CaseConfig | None = None
    artifact_leases: MlflowArtifactLeaseManager | None = None
    reuse_comparisons: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedComparisonSpecs:
    """One comparison's preconditioners after checkpoint resolution.

    The `checkpoint_digests` are always derived from `specs` at construction
    (see `from_resolution`), so the specs and the checkpoint content the
    comparison's identity covers can never drift apart.

    Attributes:
        specs: Preconditioner configs with concrete checkpoint paths baked in.
        warnings: Human-readable notes about specs that failed to resolve and
            were skipped.
        checkpoint_digests: Content digest of every resolved checkpoint the
            specs depend on, keyed by spec position and ref label.
    """

    specs: list[PreconditionerConfig]
    warnings: tuple[str, ...]
    checkpoint_digests: Mapping[str, Digest]

    @classmethod
    def from_resolution(
        cls, specs: list[PreconditionerConfig], warnings: tuple[str, ...]
    ) -> ResolvedComparisonSpecs:
        """Bundle a resolution result with the checkpoint digests it implies."""
        return cls(specs=specs, warnings=warnings, checkpoint_digests=checkpoint_digests(specs))


def _resolve_specs(
    cfg: ComparisonConfig,
    context: ComparisonResolutionContext,
    artifact_leases: ArtifactLeaseManager,
) -> ResolvedComparisonSpecs:
    """Resolve model_ref preconditioners into concrete checkpoint paths."""
    specs = resolve_neural_preconditioners(list(cfg.preconditioners))
    if not needs_model_resolution(cfg.preconditioners):
        return ResolvedComparisonSpecs.from_resolution(specs, ())

    assignment_ids = referenced_assignment_ids(cfg.preconditioners)
    assignment_contexts = None
    if assignment_ids:
        resolved_master_cfg = (
            context.master_cfg
            if context.master_cfg is not None
            else load_validated_case_config(context.case_config_path, context.settings)[0]
        )
        assignment_contexts = _build_master_assignment_contexts(
            assignment_ids,
            context.settings,
            master_cfg=resolved_master_cfg,
            config_dir=context.case_config_path.resolve().parent,
        )
    resolution = resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=context.topology.model_store_tracking_uri,
        artifact_leases=artifact_leases,
        dataset_alias=cfg.general.data.dataset_alias,
        assignment_contexts=assignment_contexts,
        skip_unresolved=True,
    )
    return ResolvedComparisonSpecs.from_resolution(resolution.specs, resolution.warnings)


def checkpoint_digests(specs: Sequence[PreconditionerConfig]) -> dict[str, Digest]:
    """Content digest of every resolved checkpoint the specs depend on.

    The comparison depends on the checkpoint *bytes* it actually loads, not on
    which run they came from, so a retrained model with a different result
    changes the identity and an identical one does not. Refs without a
    resolved checkpoint (an unfitted inline POD coarsening) contribute nothing;
    their fit data is covered by the spec's own identity fields.
    """
    digests: dict[str, Digest] = {}
    for index, spec in enumerate(specs):
        for _, label, ref in _iter_checkpoint_refs([spec]):
            checkpoint_path = ref.resolved_checkpoint_path
            if checkpoint_path is None:
                continue
            digest_key = f"{index}:{label}"
            logger.info(
                "Inspecting checkpoint '{}' for preconditioner '{}' before hashing.",
                checkpoint_path,
                spec.name,
            )
            checkpoint_size = checkpoint_path.stat().st_size
            logger.info(
                "Hashing checkpoint '{}' for preconditioner '{}' ({} bytes).",
                checkpoint_path,
                spec.name,
                checkpoint_size,
            )
            digests[digest_key] = content_digest(checkpoint_path)
            logger.info(
                "Finished hashing checkpoint for preconditioner '{}'.",
                spec.name,
            )
    return digests


def require_rhs_source_kind(cfg: ComparisonConfig) -> ComparisonRhsSourceKind:
    """Return the configured RHS source kind or fail with a focused error."""
    kind = cfg.general.data.rhs_source_kind
    if kind is None:
        raise ValueError("Comparison config must define rhs_source.")
    return kind


def resolve_comparison_specs(
    cfg: ComparisonConfig,
    context: ComparisonResolutionContext,
    artifact_leases: ArtifactLeaseManager,
) -> ResolvedComparisonSpecs:
    """Resolve preconditioner checkpoints, rejecting a comparison with nothing left to run.

    Must run inside the same lease-manager scope as execution — leased
    checkpoint files are released when that scope closes.

    Raises:
        ValueError: When all preconditioners fail to resolve.
    """
    resolved = _resolve_specs(cfg, context, artifact_leases)
    if not resolved.specs:
        raise ValueError("No runnable preconditioners remain after model resolution.")
    return resolved
