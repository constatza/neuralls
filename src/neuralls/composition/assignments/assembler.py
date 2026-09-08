"""Case-config loading and assignment assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from dlkit.infrastructure.config.job_config import FitJobConfig, SearchJobConfig, TrainingJobConfig
from loguru import logger

from neuralls.composition.assignments._job_types import AnyJobConfig, TrainableJobConfig
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.job_materializer import materialize_inference_job
from neuralls.composition.assignments.runtime_tracking_patcher import patch_training_tracking
from neuralls.composition.assignments.runtime_workspace_patcher import (
    patch_runtime_workspace_for_job,
)
from neuralls.platform.config.loaders import load_case_config, load_data_config
from neuralls.platform.config.models.experiments import CaseConfig, resolve_display_name
from neuralls.platform.config.models.workspace import (
    AssignmentBatch,
    AssignmentSpec,
    RunnableAssignment,
)
from neuralls.platform.config.registry import list_assignment_bindings
from neuralls.platform.config.resolution import (
    derive_output_root_from_tracking_uri,
    resolve_case_config_path,
    resolve_path_context,
)
from neuralls.platform.config.settings import NeurallsSettings, require_settings
from neuralls.platform.storage.workspaces import WorkspaceFactory
from neuralls.platform.tracking.environment import scoped_mlflow_environment
from neuralls.platform.tracking.mlflow import build_workflow_environment


def _base_name_from_settings(settings: AnyJobConfig, job_config_path: Path) -> str:
    """Resolve a stable run/base name from one lower-case DLKit job."""
    job_experiment = settings.experiment
    job_experiment_name = (
        getattr(job_experiment, "name", None) if job_experiment is not None else None
    )
    if isinstance(job_experiment_name, str) and job_experiment_name.strip():
        return job_experiment_name.strip()

    model = settings.model
    model_name = getattr(model, "name", None) if model is not None else None
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()

    return job_config_path.stem


def _require_trainable_job(settings: AnyJobConfig) -> TrainableJobConfig:
    """Narrow one loaded job to a kind runnable via the assignment/training pipeline."""
    if isinstance(settings, (TrainingJobConfig, SearchJobConfig, FitJobConfig)):
        return settings
    raise TypeError(
        f"Training mode requires a DLKit training, search, or fit job, got "
        f"{type(settings).__name__}."
    )


@dataclass(frozen=True)
class MlflowTopology:
    """Runtime MLflow topology for a training or inference workflow."""

    env: dict[str, str]
    experiment_name: str | None = None


@dataclass(frozen=True)
class AssignmentIdentity:
    """Who an assignment *is*, as declared by the case config that bound it.

    The unresolved counterpart to `AssignmentSpec`: the same identity fields,
    but every one of them still optional because nothing has been defaulted
    yet. `load_assignment` turns one of these into the resolved
    `AssignmentSpec` (filling `assignment_id` from the job's base name and
    `assignment_display_name` from `resolve_display_name`); every layer above
    it passes this object through instead of re-declaring the six fields.

    Attributes:
        assignment_id: Registry assignment key, or None for ad-hoc runs.
        assignment_display_name: Human-facing assignment label, or None to derive one.
        dataset_registry_id: `[[datasets]]` lookup key this run was bound to.
        dataset_display_name: Human-facing dataset label.
        job_registry_id: `[[jobs]]` lookup key this run was bound to.
        job_display_name: Human-facing job label.
    """

    assignment_id: str | None = None
    assignment_display_name: str | None = None
    dataset_registry_id: str | None = None
    dataset_display_name: str | None = None
    job_registry_id: str | None = None
    job_display_name: str | None = None

    @classmethod
    def from_spec(cls, spec: AssignmentSpec) -> AssignmentIdentity:
        """Project an already-resolved `AssignmentSpec` back onto its identity fields."""
        return cls(
            assignment_id=spec.assignment_id,
            assignment_display_name=spec.assignment_display_name,
            dataset_registry_id=spec.dataset_id,
            dataset_display_name=spec.dataset_display_name,
            job_registry_id=spec.job_id,
            job_display_name=spec.job_display_name,
        )


def load_validated_case_config(
    config_path: Path,
    neuralls_settings: NeurallsSettings | None = None,
) -> tuple[CaseConfig, Path]:
    """Load one case config and validate all registry-backed references."""
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=config_path,
    )
    cfg = load_case_config(config_path, neuralls_settings)
    config_dir = config_path.resolve().parent
    return cfg, config_dir


def _load_case_config(
    case_config_path: Path | None,
    neuralls_settings: NeurallsSettings | None = None,
) -> CaseConfig | None:
    """Load case topology when provided explicitly or via environment."""
    resolved_case_config = resolve_case_config_path(case_config_path)
    if resolved_case_config is None:
        return None
    cfg, _ = load_validated_case_config(resolved_case_config, neuralls_settings)
    return cfg


def _resolve_output_override(
    *,
    output_root: Path | None,
    case_cfg: CaseConfig | None,
    case_config_path: Path | None,
    neuralls_settings: NeurallsSettings,
) -> Path | None:
    """Resolve output root from explicit override or case config."""
    if output_root is not None:
        return output_root.resolve()
    if case_cfg is None or case_config_path is None:
        return neuralls_settings.output_dir

    tracking_uri = case_cfg.mlflow.tracking_uri
    if tracking_uri is None:
        return neuralls_settings.output_dir
    return derive_output_root_from_tracking_uri(tracking_uri, config_path=case_config_path)


def _build_default_mlflow_topology(path_ctx_output_root: Path) -> MlflowTopology:
    """Build default MLflow env rooted under the output directory."""
    runtime = build_workflow_environment(
        tracking_uri=None,
        artifact_location=None,
        default_output_root=path_ctx_output_root,
    )
    return MlflowTopology(env=runtime.env)


def _build_case_mlflow_topology(
    case_cfg: CaseConfig,
    case_config_path: Path,
    neuralls_settings: NeurallsSettings,
) -> MlflowTopology:
    """Build MLflow env from case config or derive it from output_dir."""
    runtime = build_workflow_environment(
        tracking_uri=case_cfg.mlflow.tracking_uri,
        artifact_location=case_cfg.mlflow.artifacts_destination,
        default_output_root=neuralls_settings.output_dir,
        config_path=case_config_path,
    )
    return MlflowTopology(
        env=runtime.env,
        experiment_name=case_cfg.names.training,
    )


def load_assignment(
    job_config_path: Path | None = None,
    data_config_path: Path | None = None,
    neuralls_settings: NeurallsSettings | None = None,
    output_root: Path | None = None,
    mode: str = "training",
    case_config_path: Path | None = None,
    identity: AssignmentIdentity | None = None,
) -> RunnableAssignment:
    """Load a single assignment configuration.

    Args:
        job_config_path: Path to the job configuration TOML.
        data_config_path: Path to the dataset configuration TOML.
        neuralls_settings: Resolved project settings, or None to resolve them here.
        output_root: Output root override for this assignment's workspace.
        mode: Either "training" or "inference".
        case_config_path: Case config that bound this assignment, if any.
        identity: Registry identity declared for this assignment by its case
            config. `dataset_registry_id` is required; everything else is
            defaulted here when unset.

    Returns:
        The fully resolved `RunnableAssignment`.
    """
    identity = identity or AssignmentIdentity()
    if job_config_path is None:
        raise ValueError("job_config_path is required.")
    if data_config_path is None:
        raise ValueError("data_config_path is required.")
    resolved_case_config_path = resolve_case_config_path(case_config_path)
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=resolved_case_config_path,
    )
    if mode not in ("training", "inference"):
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'training' or 'inference'.")
    case_cfg = _load_case_config(resolved_case_config_path, neuralls_settings)

    data_cfg = load_data_config(data_config_path, neuralls_settings)
    resolved_output_root = _resolve_output_override(
        output_root=output_root,
        case_cfg=case_cfg,
        case_config_path=resolved_case_config_path,
        neuralls_settings=neuralls_settings,
    )
    path_ctx = resolve_path_context(
        processed_root=neuralls_settings.processed_dir,
        output_root=neuralls_settings.output_dir,
        data_dir_override=data_cfg.output.data_dir,
        output_override=resolved_output_root,
    )
    mlflow_topology = (
        _build_case_mlflow_topology(
            case_cfg,
            resolved_case_config_path,
            neuralls_settings,
        )
        if case_cfg is not None and resolved_case_config_path is not None
        else _build_default_mlflow_topology(path_ctx.output_root)
    )

    with scoped_mlflow_environment(mlflow_topology.env):
        job_cfg = load_experiment_job(job_config_path, neuralls_settings)

    if identity.dataset_registry_id is None:
        raise ValueError(
            "dataset_registry_id is required. Pass it from the case config via load_assignment_batch()."
        )
    # On-disk paths key off the dataset config's own id (what generation used to
    # name the processed directory), not the case-registry alias, so the two can
    # never drift apart. dataset_registry_id stays on AssignmentSpec below purely
    # for tracking which case-config entry produced this run.
    dataset_id = data_cfg.id
    base_name = _base_name_from_settings(job_cfg, job_config_path)
    resolved_assignment_id = identity.assignment_id or base_name
    spec = AssignmentSpec(
        assignment_id=resolved_assignment_id,
        assignment_display_name=resolve_display_name(
            resolved_assignment_id,
            identity.assignment_display_name,
        ),
        dataset_id=identity.dataset_registry_id,
        dataset_display_name=identity.dataset_display_name,
        job_id=identity.job_registry_id,
        job_display_name=identity.job_display_name,
        job_config_path=job_config_path,
        data_config_path=data_config_path,
    )

    workspace = WorkspaceFactory(path_ctx.output_root, path_ctx.processed_root).create(
        dataset_id,
        base_name,
    )

    if mode == "inference":
        settings = materialize_inference_job(job_cfg)
        logger.debug("Loaded inference settings")
    else:
        trainable_job = _require_trainable_job(job_cfg)
        settings = patch_runtime_workspace_for_job(trainable_job, output_dir=workspace.root_dir)
        settings = patch_training_tracking(
            settings,
            uri=mlflow_topology.env.get("MLFLOW_TRACKING_URI"),
        )
        logger.debug("Loaded training settings")

    return RunnableAssignment(
        spec=spec,
        workspace=workspace,
        settings=settings,
        mlflow_env=mlflow_topology.env,
    )


def load_assignment_batch(
    case_config_path: Path,
    neuralls_settings: NeurallsSettings | None = None,
) -> AssignmentBatch:
    """Load all assignments from one case config file."""
    neuralls_settings = require_settings(
        neuralls_settings,
        case_config_path=case_config_path,
    )
    if not case_config_path.exists():
        raise FileNotFoundError(f"Case config not found: {case_config_path}")

    cfg, config_dir = load_validated_case_config(case_config_path, neuralls_settings)
    output_root = neuralls_settings.output_dir
    bindings = list_assignment_bindings(cfg, config_dir)
    if not bindings:
        raise ValueError(
            "No assignments defined. Expected [[assignments]] entries with id, dataset, job fields."
        )

    resolved_assignments: list[RunnableAssignment] = []
    for binding in bindings:
        if not binding.data_config_path.exists():
            raise FileNotFoundError(
                f"Assignment '{binding.assignment_id}': Dataset config not found: {binding.data_config_path}"
            )
        if not binding.job_config_path.exists():
            raise FileNotFoundError(
                f"Assignment '{binding.assignment_id}': Job config not found: {binding.job_config_path}"
            )

        assignment = load_assignment(
            job_config_path=binding.job_config_path,
            data_config_path=binding.data_config_path,
            neuralls_settings=neuralls_settings,
            output_root=output_root,
            case_config_path=case_config_path,
            identity=AssignmentIdentity(
                assignment_id=binding.assignment_id,
                assignment_display_name=binding.assignment_display_name,
                dataset_registry_id=binding.dataset_id,
                dataset_display_name=binding.dataset_display_name,
                job_registry_id=binding.job_id,
                job_display_name=binding.job_display_name,
            ),
        )

        if binding.checkpoint_path is not None:
            if not binding.checkpoint_path.exists():
                logger.warning(
                    "Assignment '{}': Checkpoint not found: {}",
                    binding.assignment_id,
                    binding.checkpoint_path,
                )
            assignment = replace(
                assignment,
                spec=assignment.spec.model_copy(
                    update={"checkpoint_path": binding.checkpoint_path}
                ),
            )

        resolved_assignments.append(assignment)

    return AssignmentBatch(
        output_root=output_root,
        assignments=resolved_assignments,
    )
