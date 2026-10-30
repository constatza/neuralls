"""Model catalog helpers for MLflow registration and naming."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from dlkit.mlflow import has_checkpoint_artifact
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

from neuralls.platform.config.job_metadata import read_job_metadata
from neuralls.platform.config.models.dataset_identity import normalize_registry_id
from neuralls.platform.tracking.checkpoint_selection import find_single_checkpoint

RESERVED_ALIASES: set[str] = {"latest"}
CHECKPOINT_ARTIFACT_PATH_TAG = "checkpoint_artifact_path"


@dataclass(frozen=True, slots=True)
class RegisteredModelRecord:
    """Immutable record for a registration event."""

    name: str
    version: int
    model_uri: str
    run_id: str


def build_registered_model_name(job_id: str) -> str:
    """Build canonical registered model name."""
    return job_id


def _normalize_aliases(aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate aliases while rejecting reserved values."""
    normalized_aliases: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        normalized = normalize_registry_id(alias)
        if normalized in RESERVED_ALIASES:
            raise ValueError(f"Alias '{normalized}' is reserved and cannot be assigned.")
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_aliases.append(normalized)
    return tuple(normalized_aliases)


def _warn_existing_registered_model_name(
    *,
    client: MlflowClient,
    registered_model_name: str,
) -> None:
    """Warn when registration will append a version to an existing model."""
    try:
        client.get_registered_model(registered_model_name)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return
        raise
    logger.warning(
        "Registered model '{}' already exists. Registering a new version.",
        registered_model_name,
    )


def _ensure_registered_model_exists(client: MlflowClient, name: str) -> None:
    """Create the registered model if it does not already exist.

    Args:
        client: MLflow client bound to the target tracking URI.
        name: Registered model name to ensure exists.

    Raises:
        MlflowException: Any error other than "already exists" propagates.
    """
    try:
        client.create_registered_model(name)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
            return
        raise


def _select_checkpoint_relative_path(
    *,
    client: MlflowClient,
    run_id: str,
) -> str:
    """Download a run's ``checkpoints/`` artifacts and select one unambiguous file.

    Args:
        client: MLflow client bound to the target tracking URI.
        run_id: MLflow run whose ``checkpoints/`` artifacts are inspected.

    Returns:
        The artifact-relative path of the selected checkpoint (e.g.
        ``"checkpoints/best.ckpt"``), suitable for building a ``runs:/`` URI.

    Raises:
        FileNotFoundError: If the run has no ``checkpoints/`` artifacts at all.
        ValueError: If multiple distinct, unresolvable checkpoints are found
            (propagated unchanged from ``find_single_checkpoint``).
    """
    no_checkpoints_message = (
        f"Run '{run_id}' has no 'checkpoints' artifacts to register. "
        "Registration requires a checkpoint artifact logged under "
        "'checkpoints/' before a model version can be pinned."
    )
    with TemporaryDirectory() as scratch_dir:
        try:
            root = Path(
                client.download_artifacts(
                    run_id=run_id,
                    path="checkpoints",
                    dst_path=scratch_dir,
                )
            )
        except Exception as exc:
            raise FileNotFoundError(no_checkpoints_message) from exc
        try:
            selected = find_single_checkpoint(root)
        except FileNotFoundError as exc:
            raise FileNotFoundError(no_checkpoints_message) from exc
        return f"checkpoints/{selected.relative_to(root).as_posix()}"


def register_logged_model(
    *,
    run_id: str,
    registered_model_name: str,
    tracking_uri: str,
    aliases: tuple[str, ...] = (),
    tags: Mapping[str, str] | None = None,
) -> RegisteredModelRecord:
    """Register a pinned checkpoint artifact and attach aliases.

    Selects exactly one unambiguous checkpoint file from the run's
    ``checkpoints/`` artifacts and pins it as the model version's ``source`` —
    once, deliberately, at registration time — so resolution later becomes a
    direct download with no scanning. Uses ``MlflowClient.create_model_version``
    directly (not the fluent ``mlflow.register_model``), since that fluent
    wrapper's validation rejects a bare single-file source and only accepts an
    MLmodel-flavored directory or a registered "Logged Model" entity.

    Args:
        run_id: MLflow run whose ``checkpoints/`` artifacts supply the pinned file.
        registered_model_name: Registered model name to create or append a version to.
        tracking_uri: MLflow tracking URI.
        aliases: Aliases to assign to the new version.
        tags: Additional version tags to set alongside the pinning metadata.

    Returns:
        Immutable record describing the new registered model version.

    Raises:
        FileNotFoundError: If the run has no ``checkpoints/`` artifacts.
        ValueError: If the run's ``checkpoints/`` artifacts are ambiguous
            (multiple distinct, unresolvable checkpoint candidates).
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    _warn_existing_registered_model_name(
        client=client,
        registered_model_name=registered_model_name,
    )
    _ensure_registered_model_exists(client, registered_model_name)

    relative_path = _select_checkpoint_relative_path(client=client, run_id=run_id)
    source = f"runs:/{run_id}/{relative_path}"
    model_version = client.create_model_version(
        name=registered_model_name,
        source=source,
        run_id=run_id,
    )
    version = int(model_version.version)
    for alias in _normalize_aliases(aliases):
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=alias,
            version=str(version),
        )
        resolved_version = client.get_model_version_by_alias(
            name=registered_model_name,
            alias=alias,
        )
        if int(resolved_version.version) != version:
            raise RuntimeError(
                "Alias assignment verification failed for "
                f"{registered_model_name}@{alias}: "
                f"expected version {version}, got {resolved_version.version}."
            )
    registration_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    effective_tags: dict[str, str] = dict(tags or {})
    effective_tags["registered_at"] = registration_ts
    effective_tags[CHECKPOINT_ARTIFACT_PATH_TAG] = relative_path
    for key, value in effective_tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=str(version),
            key=str(key),
            value=str(value),
        )

    return RegisteredModelRecord(
        name=registered_model_name,
        version=version,
        model_uri=source,
        run_id=run_id,
    )


def upload_checkpoint_artifacts(
    client: MlflowClient,
    run_id: str,
    checkpoint_path: Path,
) -> None:
    """Upload a checkpoint file to both canonical artifact paths in an MLflow run.

    Args:
        client: MLflow client bound to the correct tracking URI.
        run_id: Target run ID.
        checkpoint_path: Local path to the ``.ckpt`` file.
    """
    client.log_artifact(run_id, str(checkpoint_path), artifact_path="checkpoints")
    client.log_artifact(run_id, str(checkpoint_path), artifact_path="model")


def ensure_checkpoint_artifact(
    *,
    tracking_uri: str,
    run_id: str,
    checkpoint_path: Path,
) -> None:
    """Guarantee a checkpoint artifact exists under ``checkpoints/`` in an MLflow run.

    If the ``checkpoints`` artifact directory is empty or absent, uploads
    *checkpoint_path* to both ``checkpoints`` and ``model`` artifact paths.
    No-ops when the artifact is already present.

    Args:
        tracking_uri: MLflow tracking URI for the run.
        run_id: Target run ID.
        checkpoint_path: Local path to the permanent ``.ckpt`` file.
    """
    if not has_checkpoint_artifact(run_id, tracking_uri=tracking_uri):
        client = MlflowClient(tracking_uri=tracking_uri)
        upload_checkpoint_artifacts(client, run_id, checkpoint_path)
        logger.info("Uploaded missing checkpoint artifact for run {}", run_id)


def assign_dataset_alias_to_registered_model(
    *,
    tracking_uri: str,
    registered_model_name: str,
    run_id: str,
    dataset_alias: str,
) -> int | None:
    """Assign dataset alias to an already-registered model version for a run.

    Returns the resolved model version when found, otherwise ``None``.
    """
    alias = _normalize_aliases((dataset_alias,))[0]
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching = [mv for mv in versions if getattr(mv, "run_id", None) == run_id]
    if not matching:
        logger.warning(
            "No registered model version found for name='{}' and run_id='{}'. "
            "Skipping alias assignment for '{}'.",
            registered_model_name,
            run_id,
            alias,
        )
        return None

    selected = max(matching, key=lambda mv: int(str(mv.version)))
    if len(matching) > 1:
        logger.warning(
            "Multiple model versions found for name='{}' and run_id='{}'. "
            "Using highest version '{}'.",
            registered_model_name,
            run_id,
            selected.version,
        )

    version_str = str(selected.version)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version_str,
    )
    resolved = client.get_model_version_by_alias(
        name=registered_model_name,
        alias=alias,
    )
    resolved_version = int(str(resolved.version))
    if resolved_version != int(version_str):
        raise RuntimeError(
            "Alias assignment verification failed for "
            f"{registered_model_name}@{alias}: "
            f"expected version {version_str}, got {resolved.version}."
        )
    return resolved_version


def read_model_class_name(job_config_path: Path) -> str | None:
    """Read the resolved lower-case DLKit model class name from one job."""
    return read_job_metadata(job_config_path).model_name
