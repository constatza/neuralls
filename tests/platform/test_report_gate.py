from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from neuralls.composition.assignments.comparison_report import _finished_report
from neuralls.domain.solver.reporting.report_levels import ReportLevelName
from neuralls.platform.reporting.figure_index import (
    FIGURE_INDEX_FILE,
    FigureIndexEntry,
    write_figure_index,
)
from neuralls.platform.reporting.figure_metadata import FigureKind

type LogReport = Callable[..., str]

EXPERIMENT = "Reports-Case"
SOURCE_TAGS = {
    "source_session_run_id": "session-1",
    "notation_digest": "n",
    "report_schema_version": "2",
}


@pytest.fixture
def client(mlflow_tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=mlflow_tracking_uri)


@pytest.fixture
def experiment_id(client: MlflowClient) -> str:
    return client.create_experiment(EXPERIMENT)


@pytest.fixture
def log_report(client: MlflowClient, experiment_id: str, tmp_path: Path) -> LogReport:
    """Create a report root run; ``index`` is the text of its figure index, ``None`` for none."""

    def log(*, index: str | None, status: str = "FINISHED") -> str:
        run = client.create_run(experiment_id, tags={"phase": "report", **SOURCE_TAGS})
        if index is not None:
            path = tmp_path / run.info.run_id / FIGURE_INDEX_FILE
            path.parent.mkdir()
            path.write_text(index, encoding="utf-8")
            client.log_artifact(run.info.run_id, str(path))
        client.set_terminated(run.info.run_id, status=status)
        return run.info.run_id

    return log


@pytest.fixture
def valid_index(tmp_path: Path) -> str:
    path = tmp_path / "index.toml"
    entry = FigureIndexEntry(
        "r", "a/f.png", "a/f.toml", next(iter(FigureKind)), next(iter(ReportLevelName)), "scope"
    )
    write_figure_index(path, [entry])
    return path.read_text(encoding="utf-8")


def test_a_finished_report_with_a_valid_index_is_reused(
    client: MlflowClient, log_report: LogReport, valid_index: str
) -> None:
    run_id = log_report(index=valid_index)

    assert _finished_report(client, EXPERIMENT, SOURCE_TAGS) == run_id


@pytest.mark.parametrize("index", [None, "not = [valid", "other = 1"])
def test_a_finished_report_without_a_valid_index_is_not_reused(
    client: MlflowClient, log_report: LogReport, index: str | None
) -> None:
    log_report(index=index)

    assert _finished_report(client, EXPERIMENT, SOURCE_TAGS) is None


def test_a_failed_report_is_not_reused_and_an_older_valid_one_is(
    client: MlflowClient, log_report: LogReport, valid_index: str
) -> None:
    good = log_report(index=valid_index)
    log_report(index=valid_index, status="FAILED")

    assert _finished_report(client, EXPERIMENT, SOURCE_TAGS) == good
