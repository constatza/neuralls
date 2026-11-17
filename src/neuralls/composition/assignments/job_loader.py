"""Job loading boundary for DLKit-backed assignment composition."""

from __future__ import annotations

from pathlib import Path

from neuralls.composition.assignments._job_types import AnyJobConfig, DlkitJobConfig
from neuralls.platform.config.dlkit_bridge import load_job_config
from neuralls.platform.config.models.pod_job import PODJobConfig
from neuralls.platform.config.settings import NeurallsSettings


def load_experiment_job(
    job_config_path: Path,
    settings: NeurallsSettings,
) -> AnyJobConfig:
    """Load one typed job, a lower-case DLKit job or a POD recipe, for assignment composition."""
    return load_job_config(job_config_path, settings)


def is_pod_recipe(job_config_path: Path, settings: NeurallsSettings) -> bool:
    """Whether the job is a POD recipe, which the comparison fits and dlkit never runs."""
    return isinstance(load_experiment_job(job_config_path, settings), PODJobConfig)


def load_dlkit_job(job_config_path: Path, settings: NeurallsSettings) -> DlkitJobConfig:
    """Load one job that dlkit runs (train, search or predict).

    Raises:
        TypeError: When the job is a POD recipe; callers leave those assignments out first.
    """
    match load_experiment_job(job_config_path, settings):
        case PODJobConfig():
            raise TypeError(
                f"{job_config_path} is a POD recipe: the comparison fits it, dlkit never runs it."
            )
        case job:
            return job
