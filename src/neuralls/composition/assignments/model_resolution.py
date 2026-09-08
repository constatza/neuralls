"""Resolve strict model references into concrete checkpoint paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

from neuralls.platform.config.models.dataset_identity import normalize_registry_id
from neuralls.platform.config.models.preconditioner import (
    CheckpointRefBearing,
    LoggedModelRefConfig,
    ModelRefConfig,
    NeuralCheckpointRef,
    PreconditionerConfig,
    RegisteredModelRefConfig,
)
from neuralls.platform.tracking.artifact_access import ArtifactLeaseManager
from neuralls.platform.tracking.artifact_selection import (
    CHECKPOINT_ARTIFACT_DIR,
    CHECKPOINT_FILE_EXTENSION,
)
from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint
from neuralls.platform.tracking.mlflow import quote_filter_value
from neuralls.platform.tracking.model_registry import CHECKPOINT_ARTIFACT_PATH_TAG

_DATASET_ALIAS_PLACEHOLDER = "@dataset"


@dataclass(frozen=True)
class AssignmentModelContext:
    """Per-assignment lookup context for comparison model resolution."""

    dataset_alias: str | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class ModelResolution:
    """Immutable model resolution result."""

    model_uri: str
    run_id: str
    checkpoint_path: Path


@dataclass(frozen=True)
class PreconditionerResolutionResult:
    """Resolved specs plus any skipped-resolution warnings."""

    specs: list[PreconditionerConfig]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoggedModelSearchResult:
    """Minimal logged-model lookup record for latest-run resolution."""

    run_id: str
    model_uri: str


def build_logged_model_uri(*, run_id: str, artifact_path: str) -> str:
    """Build a canonical MLflow logged-model URI."""
    normalized_artifact_path = artifact_path.strip("/")
    return f"runs:/{run_id}/{normalized_artifact_path}"


def build_registered_model_uri(
    model_name: str,
    *,
    version: int | None = None,
    alias: str | None = None,
) -> str:
    """Build a canonical MLflow registered-model URI."""
    if alias is not None:
        return f"models:/{model_name}@{alias}"
    if version is not None:
        return f"models:/{model_name}/{version}"
    raise ValueError("Registered model URI requires either alias or version.")


def search_registered_models(*, model_name: str, tracking_uri: str) -> list[Any]:
    """Return matching registered models by exact name.

    Only a genuine "no such registered model" response yields an empty list;
    any other MLflow error (auth, connectivity, ...) propagates so callers don't
    mistake a transient failure for a missing model.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        return [client.get_registered_model(model_name)]
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return []
        raise


def list_model_versions(model_name: str, *, tracking_uri: str) -> list[int]:
    """List registered model versions in ascending integer order."""
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    return sorted(int(str(version.version)) for version in versions)


def get_model_version(*, model_name: str, version: int, tracking_uri: str) -> Any:
    """Fetch one registered model version from MLflow."""
    client = MlflowClient(tracking_uri=tracking_uri)
    return client.get_model_version(model_name, str(version))


def _resolve_experiment_ids(
    *,
    client: MlflowClient,
    experiment_name: str | None,
    experiment_id: str | None,
) -> list[str]:
    """Resolve experiment scoping for logged-model lookup."""
    if experiment_id is not None:
        return [experiment_id]
    if experiment_name is not None:
        experiment = client.get_experiment_by_name(experiment_name)
        return [experiment.experiment_id] if experiment is not None else []
    experiments = client.search_experiments()
    return [experiment.experiment_id for experiment in experiments]


def _build_run_filter(*, run_name: str | None, tags: dict[str, str] | None) -> str:
    """Build an MLflow run filter string for logged-model lookup."""
    clauses: list[str] = []
    if run_name is not None:
        clauses.append(f"attributes.run_name = '{quote_filter_value(run_name)}'")
    for key, value in sorted((tags or {}).items()):
        clauses.append(f"tags.`{key}` = '{quote_filter_value(value)}'")
    return " and ".join(clauses)


def _run_matches_model_name(run: Run, model_name: str | None) -> bool:
    """Best-effort model-name filter for latest logged-model lookup."""
    if model_name is None:
        return True
    tags = run.data.tags or {}
    if tags.get("model_name") == model_name:
        return True
    return tags.get("mlflow.model.name") == model_name


def search_logged_models(
    *,
    model_name: str | None,
    experiment_name: str | None,
    experiment_id: str | None,
    run_name: str | None,
    tracking_uri: str,
    artifact_path: str,
    tags: dict[str, str] | None,
    max_results: int,
) -> list[LoggedModelSearchResult]:
    """Resolve candidate logged models from MLflow runs ordered newest first."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = _resolve_experiment_ids(
        client=client,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
    )
    if not experiment_ids:
        return []

    filter_string = _build_run_filter(run_name=run_name, tags=tags)
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=1,
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )
    return [
        LoggedModelSearchResult(
            run_id=run.info.run_id,
            model_uri=build_logged_model_uri(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
            ),
        )
        for run in runs
        if _run_matches_model_name(run, model_name)
    ]


def _resolve_primary_checkpoint(
    *,
    artifact_leases: ArtifactLeaseManager,
    run_id: str,
) -> Path | None:
    """Resolve the checkpoint under a run's canonical checkpoint artifact dir.

    Args:
        artifact_leases: Lease manager owning any temporary materialization.
        run_id: MLflow run whose artifacts to consult.

    Returns:
        The single checkpoint found there, or None when the directory can't be
        resolved or holds no checkpoint at all — both of which mean "try the
        fallback location". Ambiguity between distinct checkpoints is *not* a
        miss and propagates as `ValueError`.
    """
    try:
        checkpoint_root = artifact_leases.resolve_dir(run_id, CHECKPOINT_ARTIFACT_DIR).path
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Could not resolve '{}' artifacts for run {}: {}",
            CHECKPOINT_ARTIFACT_DIR,
            run_id,
            exc,
        )
        return None
    try:
        return find_single_checkpoint(checkpoint_root)
    except FileNotFoundError:
        return None


def _resolve_fallback_checkpoint(
    *,
    artifact_leases: ArtifactLeaseManager,
    run_id: str,
    fallback_artifact_path: str,
) -> Path:
    """Resolve a checkpoint from a ref's own artifact path — a file, or a dir to scan."""
    if PurePosixPath(fallback_artifact_path).suffix == CHECKPOINT_FILE_EXTENSION:
        return artifact_leases.resolve_file(run_id, fallback_artifact_path).path
    fallback_root = artifact_leases.resolve_dir(run_id, fallback_artifact_path).path
    return find_single_checkpoint(fallback_root)


def _resolve_checkpoint_for_run(
    *,
    run_id: str,
    artifact_leases: ArtifactLeaseManager,
    fallback_artifact_path: str,
) -> Path:
    """Resolve run artifacts and return a concrete checkpoint path.

    Consults the canonical checkpoint dir first, then the ref's own artifact
    path. Only a genuine miss advances to the fallback: an ambiguous set of
    distinct checkpoints raises `ValueError` from whichever location produced
    it, rather than being reported as "not found".

    Raises:
        FileNotFoundError: If neither location yields a checkpoint.
        ValueError: If either location holds several distinct checkpoints.
    """
    primary = _resolve_primary_checkpoint(artifact_leases=artifact_leases, run_id=run_id)
    if primary is not None:
        return primary

    logger.debug(
        "No checkpoint under '{}' for run {}. Falling back to '{}'.",
        CHECKPOINT_ARTIFACT_DIR,
        run_id,
        fallback_artifact_path,
    )
    try:
        return _resolve_fallback_checkpoint(
            artifact_leases=artifact_leases,
            run_id=run_id,
            fallback_artifact_path=fallback_artifact_path,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve checkpoint artifacts for run '{run_id}' "
            f"from '{CHECKPOINT_ARTIFACT_DIR}' or '{fallback_artifact_path}'."
        ) from exc


@dataclass(frozen=True)
class _ModelRefRequest:
    """One `model_ref` plus everything needed to resolve it against MLflow.

    A single request shape shared by every resolver in `_MODEL_REF_RESOLVERS`,
    so the caller dispatches by ref type without also having to know which
    arguments that particular ref kind happens to need.

    Attributes:
        ref: The `model_ref` config being resolved.
        tracking_uri: MLflow tracking URI to read metadata from.
        client: MLflow client bound to `tracking_uri`.
        artifact_leases: Lease manager owning artifact materialization.
        dataset_alias: Alias substituted for the `@dataset` placeholder.
        model_name: Registered model name supplied by an assignment binding.
    """

    ref: ModelRefConfig
    tracking_uri: str
    client: MlflowClient
    artifact_leases: ArtifactLeaseManager
    dataset_alias: str | None
    model_name: str | None


def _resolve_registered_version_number(
    *,
    ref: RegisteredModelRefConfig,
    model_name: str,
    tracking_uri: str,
) -> int:
    """Resolve which registered version to use: the pinned one, else the highest."""
    if ref.version is not None:
        return ref.version
    versions = list_model_versions(model_name, tracking_uri=tracking_uri)
    if not versions:
        raise ValueError(f"Registered model '{model_name}' has no versions")
    return max(versions)


def _resolve_registered_ref(request: _ModelRefRequest) -> ModelResolution:
    """Resolve a registered model reference.

    Registered model versions must have been created by
    ``register_logged_model``, which pins one unambiguous checkpoint file at
    registration time and records it under the ``checkpoint_artifact_path``
    version tag. Resolution here is a direct, O(1) download of that pinned
    artifact — there is no scanning, deduping, or best-checkpoint fallback;
    that scan-and-select contract only applies to raw run references
    (``LoggedModelRefConfig``, see ``_resolve_logged_ref``). A version created
    before pinning existed (no tag present) cannot be resolved and must be
    re-registered.

    ``ref.name`` is only reached when there is no assignment context: when a
    ``NeuralPreconditionerConfig.assignment`` is set, ``model_name`` (derived
    from the assignment binding) is the single source of truth and
    ``ref.name`` must be unset (enforced by a model validator on
    ``NeuralPreconditionerConfig``).
    """
    ref = request.ref
    if not isinstance(ref, RegisteredModelRefConfig):
        raise TypeError(f"Expected a registered model_ref, got {type(ref)}")

    model_name = ref.name or request.model_name
    if model_name is None:
        raise ValueError(
            "Registered model_ref.name is required unless supplied by an assignment binding."
        )
    if not search_registered_models(model_name=model_name, tracking_uri=request.tracking_uri):
        raise ValueError(f"Registered model '{model_name}' not found")

    if ref.alias is not None:
        alias = _resolve_registered_alias(ref.alias, dataset_alias=request.dataset_alias)
        model_uri = build_registered_model_uri(model_name, alias=alias)
        version = request.client.get_model_version_by_alias(model_name, alias)
    else:
        version_number = _resolve_registered_version_number(
            ref=ref,
            model_name=model_name,
            tracking_uri=request.tracking_uri,
        )
        version = get_model_version(
            model_name=model_name,
            version=version_number,
            tracking_uri=request.tracking_uri,
        )
        model_uri = build_registered_model_uri(model_name, version=version_number)
        if version.run_id is None:
            raise ValueError(
                f"Registered model '{model_name}' version {version_number} has no run_id."
            )

    run_id = version.run_id
    if run_id is None:
        raise ValueError(f"Registered model '{model_name}' could not be resolved to an MLflow run.")

    pinned_path = version.tags.get(CHECKPOINT_ARTIFACT_PATH_TAG)
    if pinned_path is None:
        raise ValueError(
            f"Registered model '{model_name}' version {version.version} predates "
            "checkpoint pinning and cannot be resolved — re-register it (register_logged_model) "
            "to pin an exact checkpoint before it can be used."
        )
    return ModelResolution(
        model_uri=model_uri,
        run_id=run_id,
        checkpoint_path=request.artifact_leases.resolve_file(run_id, pinned_path).path,
    )


def _resolve_registered_alias(alias: str, dataset_alias: str | None) -> str:
    """Resolve explicit alias and @dataset placeholder to canonical alias."""
    stripped = alias.strip()
    if stripped == _DATASET_ALIAS_PLACEHOLDER:
        if dataset_alias is None:
            raise ValueError(
                "model_ref alias '@dataset' requires general.data.dataset_alias "
                "or a neural assignment binding."
            )
        return normalize_registry_id(dataset_alias)
    return normalize_registry_id(stripped)


def _resolve_logged_ref(request: _ModelRefRequest) -> ModelResolution:
    """Resolve a logged-model reference."""
    ref = request.ref
    if not isinstance(ref, LoggedModelRefConfig):
        raise TypeError(f"Expected a logged model_ref, got {type(ref)}")

    if ref.run_id is not None:
        run_id = ref.run_id
        model_uri = build_logged_model_uri(run_id=run_id, artifact_path=ref.artifact_path)
    else:
        records = search_logged_models(
            model_name=ref.model_name,
            experiment_name=ref.experiment_name,
            experiment_id=ref.experiment_id,
            run_name=ref.run_name,
            tracking_uri=request.tracking_uri,
            artifact_path=ref.artifact_path,
            tags=ref.tags,
            max_results=100,
        )
        if not records:
            raise ValueError("No logged model matches the provided model_ref filters")
        latest = records[0]
        run_id = latest.run_id
        model_uri = latest.model_uri or build_logged_model_uri(
            run_id=run_id,
            artifact_path=ref.artifact_path,
        )

    return ModelResolution(
        model_uri=model_uri,
        run_id=run_id,
        checkpoint_path=_resolve_checkpoint_for_run(
            run_id=run_id,
            artifact_leases=request.artifact_leases,
            fallback_artifact_path=ref.artifact_path,
        ),
    )


# One resolver per `model_ref` kind, keyed by its config class. Adding a new
# ref kind means adding its config class and one resolver here — no caller
# anywhere re-tests the ref's type.
_MODEL_REF_RESOLVERS: Mapping[type, Callable[[_ModelRefRequest], ModelResolution]] = {
    RegisteredModelRefConfig: _resolve_registered_ref,
    LoggedModelRefConfig: _resolve_logged_ref,
}


def resolve_model_ref(
    *,
    spec: NeuralCheckpointRef,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
    dataset_alias: str | None = None,
    model_name: str | None = None,
) -> ModelResolution:
    """Resolve one checkpoint ref's `model_ref` to a concrete checkpoint."""
    ref = spec.model_ref
    _validate_artifact_lease_tracking_uri(
        tracking_uri=tracking_uri,
        artifact_leases=artifact_leases,
    )
    resolve = None if ref is None else _MODEL_REF_RESOLVERS.get(type(ref))
    if ref is None or resolve is None:
        raise TypeError(f"Unsupported model_ref type: {type(ref)}")

    return resolve(
        _ModelRefRequest(
            ref=ref,
            tracking_uri=tracking_uri,
            client=MlflowClient(tracking_uri=tracking_uri),
            artifact_leases=artifact_leases,
            dataset_alias=dataset_alias,
            model_name=model_name,
        )
    )


def _validate_artifact_lease_tracking_uri(
    *,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
) -> None:
    lease_tracking_uri = artifact_leases.tracking_uri
    if lease_tracking_uri is None or lease_tracking_uri == tracking_uri:
        return
    raise ValueError(
        "Model metadata tracking URI and artifact lease tracking URI must match: "
        f"{tracking_uri!r} != {lease_tracking_uri!r}."
    )


def resolve_preconditioner_models(
    *,
    specs: list[PreconditionerConfig],
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
    dataset_alias: str | None = None,
    assignment_contexts: dict[str, AssignmentModelContext] | None = None,
) -> list[PreconditionerConfig]:
    """Resolve all neural preconditioners to concrete checkpoint paths."""
    return resolve_preconditioner_models_with_warnings(
        specs=specs,
        tracking_uri=tracking_uri,
        artifact_leases=artifact_leases,
        dataset_alias=dataset_alias,
        assignment_contexts=assignment_contexts,
    ).specs


def _resolve_checkpoint_ref(
    ref: NeuralCheckpointRef,
    *,
    name: str,
    label: str,
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
    dataset_alias: str | None,
    assignment_contexts: dict[str, AssignmentModelContext] | None,
    skip_unresolved: bool,
) -> tuple[NeuralCheckpointRef, str | None]:
    """Resolve one checkpoint ref to a concrete checkpoint path.

    Already-set `checkpoint_path` is copied through unchanged; otherwise
    `model_ref` is resolved against MLflow and downloaded.

    Returns:
        Tuple of (resolved-or-original ref, warning message or `None`).
        A non-`None` warning means resolution was skipped (only possible
        when `skip_unresolved=True`) and the original ref is returned as-is.
    """
    display_name = f"{name} ({label})" if label else name
    if ref.checkpoint_path is not None:
        return ref.model_copy(update={"resolved_checkpoint_path": ref.checkpoint_path}), None
    if ref.model_ref is None:
        raise ValueError(f"'{display_name}' requires either checkpoint_path or model_ref.")

    context = (
        assignment_contexts.get(ref.assignment)
        if assignment_contexts is not None and ref.assignment is not None
        else None
    )
    try:
        resolution = resolve_model_ref(
            spec=ref,
            tracking_uri=tracking_uri,
            artifact_leases=artifact_leases,
            dataset_alias=context.dataset_alias if context is not None else dataset_alias,
            model_name=context.model_name if context is not None else None,
        )
    except (ValueError, FileNotFoundError, RuntimeError, OSError, KeyError) as exc:
        if not skip_unresolved:
            raise
        warning = f"Skipping {display_name}: {exc}"
        logger.warning(warning)
        return ref, warning

    checkpoint_path = resolution.checkpoint_path
    resolved_ref = ref.model_copy(
        update={
            "checkpoint_path": checkpoint_path,
            "resolved_checkpoint_path": checkpoint_path,
            "resolved_run_id": resolution.run_id,
        }
    )
    return resolved_ref, None


def resolve_preconditioner_models_with_warnings(
    *,
    specs: list[PreconditionerConfig],
    tracking_uri: str,
    artifact_leases: ArtifactLeaseManager,
    dataset_alias: str | None = None,
    assignment_contexts: dict[str, AssignmentModelContext] | None = None,
    skip_unresolved: bool = False,
) -> PreconditionerResolutionResult:
    """Resolve every checkpoint-bearing preconditioner spec, symmetrically.

    Dispatches purely on the `CheckpointRefBearing` protocol — a spec either
    exposes checkpoint refs (however many, wherever nested) or it doesn't;
    there is no branching on `PreconditionerType` here.
    """
    resolved: list[PreconditionerConfig] = []
    warnings: list[str] = []
    for spec in specs:
        if not isinstance(spec, CheckpointRefBearing):
            resolved.append(spec)
            continue
        refs = spec.checkpoint_refs()
        if not refs:
            resolved.append(spec)
            continue

        resolved_refs: list[tuple[str, NeuralCheckpointRef]] = []
        skipped = False
        for label, ref in refs:
            resolved_ref, warning = _resolve_checkpoint_ref(
                ref,
                name=spec.name,
                label=label,
                tracking_uri=tracking_uri,
                artifact_leases=artifact_leases,
                dataset_alias=dataset_alias,
                assignment_contexts=assignment_contexts,
                skip_unresolved=skip_unresolved,
            )
            if warning is not None:
                warnings.append(warning)
                skipped = True
                break
            resolved_refs.append((label, resolved_ref))

        if skipped:
            continue
        resolved.append(spec.with_resolved_refs(tuple(resolved_refs)))

    return PreconditionerResolutionResult(
        specs=resolved,
        warnings=tuple(warnings),
    )
