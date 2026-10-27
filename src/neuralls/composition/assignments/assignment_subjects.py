"""What each case assignment trains, and the typed subject that names its runs.

An assignment's job kind decides the method it produces: a ``fit`` job is a POD-2G basis, any
other trainable job a neural preconditioner. The same stubs feed the comparison's
auto-generated preconditioners and the training and evaluation run names, so a model is
described one way everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.config.job_config import FitJobConfig

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.presentation_preflight import describe_method
from neuralls.domain.presentation.methods import (
    DatasetQualifiedMethodPresentation,
    StandaloneMethodPresentation,
)
from neuralls.domain.presentation.run_names import (
    EditorialTrainingSubject,
    TrainingPresentation,
    TrainingSubject,
)
from neuralls.domain.presentation.titles import EditorialPresentation
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PODCoarseningConfig,
    PreconditionerType,
    TrainedAssignmentRefConfig,
)
from neuralls.platform.config.models.presentation import ScientificLabelsConfig
from neuralls.platform.config.registry import ResolvedAssignmentBinding, resolve_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.storage.manifest_io import load_dataset_presentation
from neuralls.shared.scientific_description.models import ValidatedLabel
from neuralls.shared.scientific_description.policy import NotationPolicy


@dataclass(frozen=True, slots=True)
class AssignmentInputs:
    """One assignment's registry binding, loaded job, and generated dataset directory."""

    binding: ResolvedAssignmentBinding
    job: AnyJobConfig
    dataset_dir: Path


def resolve_assignment_inputs(
    entry: AssignmentEntry,
    *,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> AssignmentInputs:
    """Load one assignment's declared job and resolve its dataset directory."""
    binding = resolve_assignment_binding(cfg, config_dir, entry)
    job = load_experiment_job(binding.job_config_path, settings)
    data_cfg = load_data_config(binding.data_config_path, settings)
    dataset_id = resolve_dataset_identity(
        data_cfg=data_cfg,
        config_path=binding.data_config_path,
    ).name
    return AssignmentInputs(
        binding=binding, job=job, dataset_dir=settings.processed_dir / dataset_id
    )


def neural_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    extra_input_names: tuple[str, ...],
    architecture: ValidatedLabel | None,
) -> NeuralPreconditionerConfig:
    """Build an unresolved neural preconditioner stub for a train/search-kind assignment.

    The result key (`spec.name`) is derived from the config's own fields
    (`preconditioner.py::preconditioner_slug`), not set here — it always
    embeds `entry.id` via the `assignment` field below, so it stays
    traceable to the assignment without needing a separately-chosen name.
    References the assignment's training run through a strict
    `TrainedAssignmentRefConfig` (same experiment, FINISHED, checkpoint present,
    current dataset), rather than a registry entry or a tag search — the
    registry is reserved for deliberate/manual promotion. Also fetches the
    ``neuralls.extra_feature_names`` tag from the training run so that FiLM
    models receive their condition tensor during comparison.
    """
    return NeuralPreconditionerConfig(
        type=PreconditionerType.NEURAL,
        assignment=entry.id,
        model_ref=TrainedAssignmentRefConfig(assignment_id=entry.id),
        extra_input_names=extra_input_names,
        architecture=None if architecture is None else architecture.text,
    )


def pod_fit_spec_from_assignment(
    entry: AssignmentEntry,
    *,
    job: FitJobConfig,
    dataset_dir: Path,
) -> AMGPreconditionerConfig:
    """Build an unresolved AMG/POD stub for a `fit`-kind (POD-2G) assignment.

    Mirrors `_neural_spec_from_assignment`'s "derived result key, unresolved
    stub referencing the assignment's training run" shape, but for
    `AMGPreconditionerConfig(coarsening=PODCoarseningConfig(...))` instead of
    `NeuralPreconditionerConfig` — the checkpoint-shaped side of the kind
    dispatch in `neural_specs_from_assignments`. `dataset_dir`/`rank` are
    populated from the assignment's own job/dataset so the config stays
    valid (required fields) even before resolution runs; once
    `resolved_checkpoint_path` is set by
    `resolve_preconditioner_models_with_warnings`, the factory reconstructs
    the fitted basis from the checkpoint and never reads these.
    """
    rank = getattr(job.model, "rank", None)
    if rank is None:
        raise ValueError(
            f"Assignment '{entry.id}' job declares run.type='fit' but its model has no "
            "'rank' hyperparameter — expected a PODCoarseningStrategy-shaped model."
        )
    return AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=dataset_dir,
            rank=rank,
            assignment=entry.id,
            model_ref=TrainedAssignmentRefConfig(assignment_id=entry.id),
        ),
    )


def training_subject(
    entry: AssignmentEntry,
    *,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
) -> TrainingSubject:
    """What the assignment's training and evaluation runs are about, from typed facts.

    An explicit assignment ``display_name`` is an editorial title shown beside the dataset;
    otherwise the method the job trains is described exactly as a comparison describes it,
    qualified by the dataset it trains on. No registry id is ever used as text.

    Raises:
        MissingDatasetProvenanceError: When the dataset predates scientific provenance.
        MissingArchitectureNameError: When a neural job declares no architecture name.
    """
    inputs = resolve_assignment_inputs(entry, cfg=cfg, config_dir=config_dir, settings=settings)
    dataset = load_dataset_presentation(inputs.dataset_dir)
    editorial = (entry.display_name or "").strip()
    if editorial:
        return EditorialTrainingSubject(EditorialPresentation(editorial), dataset)
    match inputs.job:
        case FitJobConfig():
            stub = pod_fit_spec_from_assignment(
                entry, job=inputs.job, dataset_dir=inputs.dataset_dir
            )
        case _:
            stub = neural_spec_from_assignment(
                entry, extra_input_names=(), architecture=inputs.binding.architecture
            )
    described = describe_method(stub)
    match described:
        case DatasetQualifiedMethodPresentation():
            return described
        case StandaloneMethodPresentation(method=method):
            return DatasetQualifiedMethodPresentation(method=method, dataset=dataset)


def case_notation(cfg: CaseConfig) -> NotationPolicy:
    """The symbol policy the case's names and labels are rendered with."""
    baseline = cfg.comparison_baseline
    labels = ScientificLabelsConfig() if baseline is None else baseline.presentation
    return labels.notation_policy()


def training_presentations(
    cfg: CaseConfig,
    *,
    config_dir: Path,
    settings: NeurallsSettings,
    assignment_ids: Sequence[str],
) -> dict[str, TrainingPresentation]:
    """Describe every named assignment before any run opens, so a bad one aborts the batch.

    Raises:
        KeyError: When an id is not one of the case's assignments.
    """
    entries = {entry.id: entry for entry in cfg.assignments}
    notation = case_notation(cfg)
    return {
        assignment_id: TrainingPresentation(
            training_subject(
                entries[assignment_id], cfg=cfg, config_dir=config_dir, settings=settings
            ),
            notation,
        )
        for assignment_id in assignment_ids
    }
