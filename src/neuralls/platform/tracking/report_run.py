"""The root MLflow run of one report: a child of its session, with all-matrix-sets params."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager

from neuralls.domain.presentation.run_names import NamedRun
from neuralls.domain.solver.models.report_scope import AllMatrixSets
from neuralls.platform.tracking.run_names import run_name_text
from neuralls.platform.tracking.run_writer import RunWriter, open_run
from neuralls.platform.tracking.scope_params import scope_params
from neuralls.shared.types import SessionRunId


@contextmanager
def root_report_run(
    writer: RunWriter,
    experiment_id: str,
    run: NamedRun,
    tags: Mapping[str, str],
    *,
    session_run_id: SessionRunId,
) -> Generator[str]:
    """Open the report's root under its session, log its scope params, and yield its run id."""
    with open_run(
        writer, experiment_id, run_name_text(run), tags, parent_run_id=session_run_id
    ) as run_id:
        writer.log_params(run_id, scope_params(AllMatrixSets()))
        yield run_id
