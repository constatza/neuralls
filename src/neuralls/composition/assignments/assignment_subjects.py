"""What each case assignment trains, and the typed subject that names its runs.

An assignment's job kind decides the method it produces: a ``pod`` recipe is a POD basis the
comparison fits, a train or search job a neural preconditioner. The neural stub feeds both
the comparison's auto-generated preconditioners and the training and evaluation run names,
so a model is described one way everywhere; a POD recipe has no training run to name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from neuralls.composition.assignments._job_types import AnyJobConfig
from neuralls.composition.assignments.job_loader import load_experiment_job
from neuralls.composition.assignments.presentation_preflight import (
    DatasetPresentationLoader,
)
from neuralls.composition.comparison.method_data_origin import fold_rows
from neuralls.composition.generation.ports import SplitStore
from neuralls.composition.generation.split_pinning import require_pinned_split
from neuralls.composition.preconditioners.descriptions import describe_configured_preconditioner
from neuralls.domain.presentation.methods import (
    qualify_method,
)
from neuralls.domain.presentation.run_names import (
    EditorialTrainingSubject,
    TrainingPresentation,
    TrainingSubject,
)
from neuralls.domain.presentation.titles import EditorialPresentation
from neuralls.domain.solver.models.cost import FitFolds
from neuralls.platform.config.loaders import load_data_config
from neuralls.platform.config.models.checkpoint_refs import TrainedAssignmentRefConfig
from neuralls.platform.config.models.coarsening import PODCoarseningConfig
from neuralls.platform.config.models.dataset_identity import resolve_dataset_identity
from neuralls.platform.config.models.experiments import AssignmentEntry, CaseConfig
from neuralls.platform.config.models.figure_style import FigureStyleConfig
from neuralls.platform.config.models.pod_job import PODJobConfig
from neuralls.platform.config.models.preconditioner import (
    AMGPreconditionerConfig,
    NeuralPreconditionerConfig,
    PreconditionerType,
)
from neuralls.platform.config.models.presentation import ScientificLabelsConfig
from neuralls.platform.config.registry import ResolvedAssignmentBinding, resolve_assignment_binding
from neuralls.platform.config.settings import NeurallsSettings
from neuralls.platform.tracking.run_names import render_training_label
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
    (each config's `slug()`), not set here — it always
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


def pod_spec_from_assignment(job: PODJobConfig, *, dataset_dir: Path) -> AMGPreconditionerConfig:
    """Build the AMG/POD method a ``pod`` recipe assignment declares.

    The recipe's rank, inner product and snapshot cut, paired with the assignment's
    dataset: the comparison fits the basis from that dataset's train fold, once per
    method, so the spec references no checkpoint and needs no resolution.
    """
    return AMGPreconditionerConfig(
        type=PreconditionerType.AMG,
        coarsening=PODCoarseningConfig(
            dataset_dir=dataset_dir,
            rank=job.pod.rank,
            inner_product=job.pod.inner_product,
            n_snapshots=job.pod.n_snapshots,
        ),
    )


def training_subject(
    entry: AssignmentEntry,
    *,
    cfg: CaseConfig,
    config_dir: Path,
    settings: NeurallsSettings,
    load_dataset: DatasetPresentationLoader,
    splits: SplitStore,
) -> TrainingSubject:
    """What the assignment's training and evaluation runs are about, from typed facts.

    An explicit assignment ``display_name`` is an editorial title shown beside the dataset;
    otherwise the neural method the job trains is qualified by its assignment dataset and
    the rows of its pinned split the job reads, the train and validation folds. No registry
    id is ever used as text. Only trained assignments have a subject: a POD recipe is fitted
    inside the comparison, so callers leave it out first.

    Raises:
        MissingDatasetProvenanceError: When the dataset predates scientific provenance.
        MissingArchitectureNameError: When a neural job declares no architecture name.
        StageInputsStale: When the dataset has no pinned split.
        TypeError: When the assignment is a POD recipe.
    """
    inputs = resolve_assignment_inputs(entry, cfg=cfg, config_dir=config_dir, settings=settings)
    if isinstance(inputs.job, PODJobConfig):
        raise TypeError(f"Assignment '{entry.id}' is a POD recipe; it has no training run.")
    dataset = load_dataset(inputs.dataset_dir)
    if entry.display_name is not None and entry.display_name.strip():
        return EditorialTrainingSubject(EditorialPresentation(entry.display_name.strip()), dataset)
    stub = neural_spec_from_assignment(
        entry, extra_input_names=(), architecture=inputs.binding.architecture
    )
    split = require_pinned_split(inputs.dataset_dir, splits)
    return qualify_method(
        describe_configured_preconditioner(stub),
        dataset,
        fold_rows(split, FitFolds.TRAIN_AND_VALIDATION),
    )


def case_notation(cfg: CaseConfig) -> NotationPolicy:
    """The symbol policy the case's names and labels are rendered with."""
    baseline = cfg.comparison_baseline
    labels = ScientificLabelsConfig() if baseline is None else baseline.presentation
    return labels.notation_policy()


def case_figure_style(cfg: CaseConfig) -> FigureStyleConfig:
    """The layout the case's report figures are drawn with."""
    baseline = cfg.comparison_baseline
    return FigureStyleConfig() if baseline is None else baseline.figure_style


def training_presentations(
    cfg: CaseConfig,
    *,
    config_dir: Path,
    settings: NeurallsSettings,
    assignment_ids: Sequence[str],
    load_dataset: DatasetPresentationLoader,
    splits: SplitStore,
) -> dict[str, TrainingPresentation]:
    """Describe every named assignment before any run opens, so a bad one aborts the batch.

    Raises:
        KeyError: When an id is not one of the case's assignments.
    """
    entries = {entry.id: entry for entry in cfg.assignments}
    notation = case_notation(cfg)
    presentations = {
        assignment_id: TrainingPresentation(
            training_subject(
                entries[assignment_id],
                cfg=cfg,
                config_dir=config_dir,
                settings=settings,
                load_dataset=load_dataset,
                splits=splits,
            ),
            notation,
        )
        for assignment_id in assignment_ids
    }
    require_distinct_training_labels(presentations)
    return presentations


def require_distinct_training_labels(presentations: Mapping[str, TrainingPresentation]) -> None:
    """Reject assignments a reader could not tell apart, before any run opens.

    Two assignments that render identically would share a bar label and a run name, so a
    result could not be traced to the assignment that produced it.

    Raises:
        IndistinguishableTrainingError: When two assignments render to the same text.
    """
    by_text: dict[str, list[str]] = {}
    for assignment_id, presentation in presentations.items():
        by_text.setdefault(render_training_label(presentation), []).append(assignment_id)
    for text, assignment_ids in by_text.items():
        if len(assignment_ids) > 1:
            raise IndistinguishableTrainingError(text, tuple(assignment_ids))


class IndistinguishableTrainingError(ValueError):
    """Two assignments of one batch describe the same trained model on the same dataset."""

    def __init__(self, text: str, assignment_ids: tuple[str, ...]) -> None:
        self.text = text
        self.assignment_ids = assignment_ids
        super().__init__(
            f"assignments {', '.join(assignment_ids)} are both described as {text!r}; give "
            "each a distinct architecture name or an explicit display_name"
        )
