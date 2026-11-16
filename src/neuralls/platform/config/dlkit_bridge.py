"""Thin adapter from neuralls job paths to DLKit lower-case job configs."""

from __future__ import annotations

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

from neuralls.platform.config.settings import NeurallsSettings

type AnyJobSettings = TrainingJobConfig | InferenceJobConfig | SearchJobConfig | FitJobConfig


def load_job_config(path: Path, settings: NeurallsSettings) -> AnyJobSettings:
    """Load one lower-case DLKit job config through DLKit's native loader.

    Legacy uppercase manifests are not special-cased: they have no ``[run]``
    section, so DLKit's own validation in ``load_job()`` already fails fast
    with ``ConfigValidationError: No run.type found...``.

    ``load_job()``'s return type also covers ``ConvergenceJobConfig``/
    ``MultiRunJobConfig`` (sweep-level configs, not one job) — rejected here
    since this function's contract is "one job", matching its callers
    (per-assignment training/search/fit/inference). ``FitJobConfig`` (one-shot,
    non-gradient fit jobs, e.g. POD basis fitting) is a single job like any
    other and is returned like any other — consumed via the same
    assignment/training pipeline as ``TrainingJobConfig``/``SearchJobConfig``
    (see ``composition/assignments/_job_types.py::TrainableJobConfig``).

    Raises:
        ConfigValidationError: *path* resolves to a sweep-level config
            (``run.type`` of ``"converge"`` or ``"multirun"``).
    """
    del settings
    loaded = load_job(path)
    if isinstance(loaded, (ConvergenceJobConfig, MultiRunJobConfig)):
        raise ConfigValidationError(
            f"{path} is a {type(loaded).__name__}, not a single job config — "
            "load_job_config() only loads train/predict/search/fit jobs."
        )
    return loaded
