"""The root MLflow run of one report: opened with its source tags and all-matrix-sets params."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager

import mlflow

from neuralls.domain.solver.models.report_scope import AllMatrixSets
from neuralls.platform.tracking.scope_params import scope_params


@contextmanager
def root_report_run(run_name: str, tags: Mapping[str, str]) -> Generator[str]:
    """Open the report's root run, log its scope params, and yield its run id."""
    with mlflow.start_run(run_name=run_name, tags=dict(tags)) as run:
        mlflow.log_params(scope_params(AllMatrixSets()))
        yield run.info.run_id
