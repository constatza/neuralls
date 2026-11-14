"""The root MLflow run of one report: opened with its source tags and all-matrix-sets params."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager

from neuralls.domain.solver.models.report_scope import AllMatrixSets
from neuralls.platform.tracking.run_writer import RunWriter, open_run
from neuralls.platform.tracking.scope_params import scope_params


@contextmanager
def root_report_run(
    writer: RunWriter, experiment_id: str, run_name: str, tags: Mapping[str, str]
) -> Generator[str]:
    """Open the report's root run, log its scope params, and yield its run id."""
    with open_run(writer, experiment_id, run_name, tags) as run_id:
        writer.log_params(run_id, scope_params(AllMatrixSets()))
        yield run_id
