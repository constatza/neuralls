"""Thin adapter from neuralls job paths to DLKit lower-case job configs and POD recipes."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dlkit.common.errors import ConfigValidationError
from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    FitJobConfig,
    InferenceJobConfig,
    MultiRunJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from neuralls.platform.config.models.pod_job import POD_RUN_TYPE, PODJobConfig
from neuralls.platform.config.settings import NeurallsSettings

type DlkitJobSettings = TrainingJobConfig | InferenceJobConfig | SearchJobConfig
type AnyJobSettings = DlkitJobSettings | PODJobConfig


def load_job_config(path: Path, settings: NeurallsSettings) -> AnyJobSettings:
    """Load one job: a POD recipe, or a lower-case DLKit job through DLKit's native loader.

    A ``run.type = "pod"`` file is a neuralls recipe that dlkit never runs (the
    comparison fits the basis), so it is validated here into ``PODJobConfig``.
    Every other type goes to ``load_job()``. Legacy uppercase manifests are not
    special-cased: they have no ``[run]`` section, so DLKit's own validation
    already fails fast with ``ConfigValidationError: No run.type found...``.

    ``load_job()``'s return type also covers ``ConvergenceJobConfig``/
    ``MultiRunJobConfig`` (sweep-level configs, not one job) and dlkit's one-shot
    ``FitJobConfig``, which no neuralls workflow runs; all three are rejected,
    since this function's contract is one train, search, predict or POD job.

    Raises:
        ConfigValidationError: *path* is a sweep-level or dlkit fit config.
        pydantic.ValidationError: *path* is a POD recipe with a missing or unknown key.
    """
    del settings
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    match raw:
        case {"run": {"type": str() as run_type}} if run_type == POD_RUN_TYPE:
            return PODJobConfig.model_validate(raw)
        case _:
            return _load_dlkit_job(path)


def _load_dlkit_job(path: Path) -> DlkitJobSettings:
    """Load one single-job DLKit config, rejecting the kinds no neuralls workflow runs."""
    loaded = load_job(path)
    if isinstance(loaded, (ConvergenceJobConfig, MultiRunJobConfig, FitJobConfig)):
        raise ConfigValidationError(
            f"{path} is a {type(loaded).__name__}, not a single job config — "
            "load_job_config() only loads train/predict/search/pod jobs."
        )
    return loaded
