"""MLflow-backed `IdentityStore`: the single reader of reusable runs."""

from __future__ import annotations

import json
import math
import tempfile
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from dlkit.mlflow import has_checkpoint_artifact
from loguru import logger
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from neuralls.domain.identity import IdentityTag, Reused, StageIdentity
from neuralls.platform.reporting.benchmark_events import (
    BENCHMARK_EVENT_SCHEMA_VERSION,
    read_setup_events,
    read_solve_attempts,
)
from neuralls.platform.tracking.mlflow import quote_filter_value
from neuralls.shared.digest import Digest, Json, canonical_digest

_PAGE_SIZE = 50
_NEWEST_FIRST = ["attributes.start_time DESC", "attributes.run_id ASC"]
_METHOD_PHASE = "comparison_method"
_METHOD_SCHEMA_VERSION = str(BENCHMARK_EVENT_SCHEMA_VERSION)
_METHOD_ARTIFACTS = frozenset(
    {
        "setup_events.parquet",
        "solve_attempts.parquet",
        "summary.toml",
        "method_config.toml",
        "method_diagnostics.json",
    }
)


def _tag_clause(tag: IdentityTag, value: str) -> str:
    return f"tags.`{tag}` = '{quote_filter_value(value)}'"


@dataclass(frozen=True)
class MlflowIdentityStore:
    """Finds the newest FINISHED run in one experiment carrying an identity's key.

    Attributes:
        tracking_uri (str): MLflow tracking URI.
        experiment (str): Experiment name the stage's runs live in; runs from
            any other experiment are never considered.
        require_checkpoint (bool): Also require a real checkpoint artifact
            (training runs); a FINISHED run without one is skipped in favour
            of an older run with the same key.
    """

    tracking_uri: str
    experiment: str
    require_checkpoint: bool = False

    def find(self, identity: StageIdentity) -> Reused | None:
        """Return the newest matching run, or ``None``.

        Ordering is deterministic (start time, then run id) and pagination
        walks every match, so a pile of newer broken runs cannot hide an older
        good one.
        """
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment)
        if experiment is None:
            return None
        for run in self._matching_runs(client, experiment.experiment_id, identity):
            run_id = run.info.run_id
            if not self.require_checkpoint or has_checkpoint_artifact(
                run_id, tracking_uri=self.tracking_uri
            ):
                return Reused(run_id=run_id)
            logger.warning(
                "Run {} matches {} identity {} but has no checkpoint artifact; trying an older run.",
                run_id,
                identity.stage,
                identity.key[:19],
            )
        return None

    def _matching_runs(
        self, client: MlflowClient, experiment_id: str, identity: StageIdentity
    ) -> Iterator[Run]:
        filter_string = " and ".join(
            [
                "attributes.status = 'FINISHED'",
                _tag_clause(IdentityTag.STAGE, identity.stage),
                _tag_clause(IdentityTag.KEY, identity.key),
            ]
        )
        token: str | None = None
        while True:
            page = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                order_by=_NEWEST_FIRST,
                max_results=_PAGE_SIZE,
                page_token=token,
            )
            yield from page
            token = page.token
            if not token:
                return


def find_reusable_benchmark_method(
    *,
    tracking_uri: str,
    experiment: str,
    method: str,
    config_digest: Digest,
    system_ids: tuple[str, ...],
    method_config: Mapping[str, Json],
) -> Reused | None:
    """Find a complete schema-v1 method child, ignoring legacy system runs.

    A matching tag set is insufficient: all canonical raw artifacts must be
    present before a method workload can be reused.  The query deliberately
    requires ``phase=comparison_method`` and the schema tag, so old
    comparison runs without a schema marker execute under the new topology.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        return None
    membership = canonical_digest(tuple(system_ids))
    escaped_method = method.replace("'", "''")
    clauses = [
        "attributes.status = 'FINISHED'",
        "tags.`phase` = 'comparison_method'",
        f"tags.`benchmark_event_schema_version` = '{_METHOD_SCHEMA_VERSION}'",
        f"tags.`method` = '{escaped_method}'",
        f"tags.`method_config_digest` = '{config_digest}'",
        f"tags.`workload_membership_digest` = '{membership}'",
    ]
    runs = client.search_runs(
        experiment_ids=[experiment_obj.experiment_id],
        filter_string=" and ".join(clauses),
        order_by=_NEWEST_FIRST,
    )
    for run in runs:
        artifacts = {
            info.path.rsplit("/", 1)[-1]: info.path
            for info in client.list_artifacts(run.info.run_id, path="method_workload")
            if not info.is_dir
        }
        if not _METHOD_ARTIFACTS.issubset(artifacts):
            continue
        with tempfile.TemporaryDirectory(prefix="neuralls-method-reuse-") as temp_dir:
            try:
                paths = {
                    name: client.download_artifacts(
                        run.info.run_id, path=artifact, dst_path=temp_dir
                    )
                    for name, artifact in artifacts.items()
                    if name in _METHOD_ARTIFACTS
                }
                _validate_method_artifacts(
                    paths,
                    method=method,
                    config_digest=config_digest,
                    system_ids=system_ids,
                    method_config=method_config,
                )
            except OSError, ValueError, TypeError, KeyError, tomllib.TOMLDecodeError:
                continue
        return Reused(run_id=run.info.run_id)
    return None


def _validate_method_artifacts(
    paths: dict[str, str],
    *,
    method: str,
    config_digest: Digest,
    system_ids: tuple[str, ...],
    method_config: Mapping[str, Json],
) -> None:
    """Validate the canonical Task 5 artifacts before accepting a cache hit."""
    setup_path = Path(paths["setup_events.parquet"])
    solve_path = Path(paths["solve_attempts.parquet"])
    summary_path = Path(paths["summary.toml"])
    config_path = Path(paths["method_config.toml"])
    setup = read_setup_events(setup_path)
    solve = read_solve_attempts(solve_path, setup_events_path=setup_path)
    summary = tomllib.loads(summary_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(paths["method_diagnostics.json"]).read_text(encoding="utf-8"))
    if config != dict(method_config):
        raise ValueError("method config artifact does not match requested workload")
    _validate_diagnostics(diagnostics, method=method, system_ids=system_ids)
    if summary.get("schema_version") != BENCHMARK_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported method summary schema")
    method_summary = summary.get("method")
    workload_summary = summary.get("workload")
    expected_ids = list(system_ids)
    if not isinstance(method_summary, dict) or not isinstance(workload_summary, dict):
        raise TypeError("method summary is missing identity fields")
    if method_summary.get("name") != method or method_summary.get("config_digest") != config_digest:
        raise ValueError("method summary identity does not match requested workload")
    if workload_summary.get("membership_digest") != canonical_digest(tuple(system_ids)):
        raise ValueError("method summary membership does not match requested workload")
    if workload_summary.get("system_ids") != expected_ids:
        raise ValueError("method summary system membership does not match requested workload")
    _validate_event_methods(setup, solve, method)
    solved_ids = tuple(solve.get_column("system_id").to_list())
    if len(set(solved_ids)) != len(solved_ids) or not set(solved_ids).issubset(system_ids):
        raise ValueError("solve artifact membership does not match requested workload")
    _validate_summary(summary, setup, solve, expected_ids=system_ids, solved_ids=solved_ids)


def _validate_diagnostics(diagnostics: object, *, method: str, system_ids: tuple[str, ...]) -> None:
    """Validate diagnostic JSON membership and method identity."""
    if not isinstance(diagnostics, list):
        raise TypeError("method diagnostics must be a JSON list")
    records: list[dict[str, object]] = []
    for record in diagnostics:
        if not isinstance(record, dict):
            raise TypeError("method diagnostic record must be an object")
        if record.get("method") != method:
            raise ValueError("diagnostic method does not match requested workload")
        if not isinstance(record.get("system_id"), str):
            raise TypeError("diagnostic record is missing system ID")
        records.append(record)
    diagnostic_ids = tuple(str(record["system_id"]) for record in records)
    if len(set(diagnostic_ids)) != len(diagnostic_ids) or set(diagnostic_ids) != set(system_ids):
        raise ValueError("diagnostic membership does not match requested workload")


def _validate_event_methods(setup: pl.DataFrame, solve: pl.DataFrame, method: str) -> None:
    """Require every canonical event frame to belong to the requested method."""
    for frame in (setup, solve):
        methods = {str(value) for value in frame.get_column("method").to_list()}
        if methods != ({method} if methods else set()):
            raise ValueError("event frame method does not match requested workload")


def _validate_summary(
    summary: Mapping[str, object],
    setup: pl.DataFrame,
    solve: pl.DataFrame,
    *,
    expected_ids: tuple[str, ...],
    solved_ids: tuple[str, ...],
) -> None:
    """Reconcile summary counts, membership, and online totals with event frames."""
    counts = summary.get("counts")
    online = summary.get("online")
    if not isinstance(counts, dict) or not isinstance(online, dict):
        raise TypeError("method summary is missing counts or online totals")
    expected_missing = [system_id for system_id in expected_ids if system_id not in solved_ids]
    if counts.get("planned_rhs") != len(expected_ids):
        raise ValueError("summary planned count does not match workload membership")
    if counts.get("attempted_rhs") != len(solved_ids):
        raise ValueError("summary attempted count does not match solve events")
    converged = sum(value == "converged" for value in solve.get_column("solve_status").to_list())
    if counts.get("converged") != converged:
        raise ValueError("summary converged count does not match solve events")
    if expected_missing:
        if counts.get("missing_system_ids") != expected_missing:
            raise ValueError("summary missing membership does not match solve events")
    elif "missing_system_ids" in counts:
        raise ValueError("complete summary must not declare missing systems")

    setup_total = math.fsum(float(value) for value in setup.get_column("setup_s").to_list())
    solve_total = math.fsum(float(value) for value in solve.get_column("solve_s").to_list())
    workload_total = setup_total + solve_total
    _require_close(online.get("setup_s"), setup_total, "setup total")
    _require_close(online.get("solve_s"), solve_total, "solve total")
    _require_close(online.get("workload_s"), workload_total, "workload total")
    if solved_ids:
        _require_close(
            online.get("per_attempted_rhs_s"),
            workload_total / len(solved_ids),
            "attempted-RHS average",
        )
    elif "per_attempted_rhs_s" in online:
        raise ValueError("empty solve membership must not declare an average")


def _require_close(value: object, expected: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"summary {label} is missing")
    if not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"summary {label} does not match event frames")
