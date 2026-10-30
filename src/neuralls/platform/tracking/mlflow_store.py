"""MLflow-backed `IdentityStore`: the single reader of reusable runs."""

from __future__ import annotations

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
    METHOD_ARTIFACT_DIR,
    MethodBenchmarkArtifacts,
    read_convergence,
    read_planned_systems,
    read_presentation,
    read_setup_events,
    read_solve_attempts,
)
from neuralls.platform.tracking.mlflow import quote_filter_value
from neuralls.platform.tracking.mlflow_client import MlflowClientFactory
from neuralls.shared.digest import Digest, Json, ordered_workload_digest

_PAGE_SIZE = 50
_NEWEST_FIRST = ["attributes.start_time DESC", "attributes.run_id ASC"]
_METHOD_PHASE = "comparison_method"
_METHOD_ARTIFACTS = frozenset(MethodBenchmarkArtifacts.file_names())


def _tag_clause(tag: IdentityTag, value: str) -> str:
    return f"tags.`{tag}` = '{quote_filter_value(value)}'"


@dataclass(frozen=True, slots=True)
class MlflowIdentityStore:
    """Finds the newest FINISHED run in one experiment carrying an identity's key.

    Attributes:
        tracking_uri (str): MLflow tracking URI.
        experiment (str): Experiment name the stage's runs live in; runs from
            any other experiment are never considered.
        require_checkpoint (bool): Also require a real checkpoint artifact
            (training runs); a FINISHED run without one is skipped in favour
            of an older run with the same key.
        client_factory (MlflowClientFactory): Builds the client used for run
            lookup; defaults to the real ``MlflowClient``.
    """

    tracking_uri: str
    experiment: str
    require_checkpoint: bool = False
    client_factory: MlflowClientFactory = MlflowClient

    def find(self, identity: StageIdentity) -> Reused | None:
        """Return the newest matching run, or ``None``.

        Ordering is deterministic (start time, then run id) and pagination
        walks every match, so a pile of newer broken runs cannot hide an older
        good one.
        """
        client = self.client_factory(tracking_uri=self.tracking_uri)
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
    system_identities: tuple[Digest, ...] | None = None,
) -> Reused | None:
    """Find a complete method child, ignoring legacy system runs.

    A matching tag set is insufficient: all canonical raw artifacts must be
    present before a method workload can be reused.  The query deliberately
    requires ``phase=comparison_method``, so legacy per-system comparison
    runs execute under the method-child topology.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_obj = client.get_experiment_by_name(experiment)
    if experiment_obj is None:
        return None
    membership = ordered_workload_digest(system_ids, system_identities)
    escaped_method = method.replace("'", "''")
    clauses = [
        "attributes.status = 'FINISHED'",
        "tags.`phase` = 'comparison_method'",
        f"tags.`method_id` = '{escaped_method}'",
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
            for info in client.list_artifacts(run.info.run_id, path=METHOD_ARTIFACT_DIR)
            if not info.is_dir
        }
        missing = sorted(_METHOD_ARTIFACTS - artifacts.keys())
        if missing:
            logger.info(
                "Run {} is not reusable for method {}: missing artifacts {}.",
                run.info.run_id,
                method,
                missing,
            )
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
                    system_identities=system_identities,
                    method_config=method_config,
                )
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                tomllib.TOMLDecodeError,
            ) as exc:
                logger.info(
                    "Run {} is not reusable for method {}: {}", run.info.run_id, method, exc
                )
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
    system_identities: tuple[Digest, ...] | None = None,
) -> None:
    """Validate the canonical Task 5 artifacts before accepting a cache hit."""
    artifacts = MethodBenchmarkArtifacts()
    setup_path = Path(paths[str(artifacts.setup_events)])
    solve_path = Path(paths[str(artifacts.solve_attempts)])
    summary_path = Path(paths[str(artifacts.summary)])
    config_path = Path(paths[str(artifacts.method_config)])
    setup = read_setup_events(setup_path)
    solve = read_solve_attempts(solve_path, setup_events_path=setup_path)
    summary = tomllib.loads(summary_path.read_text(encoding="utf-8"))
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config != dict(method_config):
        raise ValueError("method config artifact does not match requested workload")
    _validate_stored_tables(paths, artifacts, method=method, system_ids=system_ids)
    method_summary = summary.get("method")
    workload_summary = summary.get("workload")
    expected_ids = list(system_ids)
    if not isinstance(method_summary, dict) or not isinstance(workload_summary, dict):
        raise TypeError("method summary is missing identity fields")
    if method_summary.get("name") != method or method_summary.get("config_digest") != config_digest:
        raise ValueError("method summary identity does not match requested workload")
    if workload_summary.get("membership_digest") != ordered_workload_digest(
        system_ids, system_identities
    ):
        raise ValueError("method summary membership does not match requested workload")
    if workload_summary.get("system_ids") != expected_ids:
        raise ValueError("method summary system membership does not match requested workload")
    _validate_event_methods(setup, solve, method)
    solved_ids = tuple(solve.get_column("system_id").to_list())
    if len(set(solved_ids)) != len(solved_ids) or not set(solved_ids).issubset(system_ids):
        raise ValueError("solve artifact membership does not match requested workload")
    _validate_summary(summary, setup, solve, expected_ids=system_ids, solved_ids=solved_ids)


def _validate_stored_tables(
    paths: Mapping[str, str],
    artifacts: MethodBenchmarkArtifacts,
    *,
    method: str,
    system_ids: tuple[str, ...],
) -> None:
    """Validate the planned systems, presentation and convergence tables of a method run."""
    planned = read_planned_systems(Path(paths[str(artifacts.planned_systems)]))
    if tuple(str(system.tag.system_id) for system in planned) != system_ids:
        raise ValueError("planned systems do not match requested workload")
    stored = read_presentation(Path(paths[str(artifacts.presentation)]))
    if str(stored.method.prepared.method_id) != method:
        raise ValueError("stored presentation does not belong to requested method")
    if {system.tag.dataset for system in planned} != set(stored.datasets):
        raise ValueError("stored dataset presentations do not match planned systems")
    convergence = read_convergence(Path(paths[str(artifacts.convergence)]))
    if not set(convergence.get_column("system_id").to_list()).issubset(system_ids):
        raise ValueError("convergence membership does not match requested workload")
    if {str(value) for value in convergence.get_column("method").to_list()} - {method}:
        raise ValueError("convergence method does not match requested workload")


def _validate_non_negative_number(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


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
    _validate_summary_memory(summary.get("memory"))
    for key in ("planned_rhs", "attempted_rhs", "converged"):
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"summary {key} must be a non-negative integer")
    for key in ("setup_s", "solve_s", "workload_s"):
        _validate_non_negative_number(online.get(key), f"summary {key}")
    expected_missing = [system_id for system_id in expected_ids if system_id not in solved_ids]
    if expected_missing:
        raise ValueError("incomplete method workload is not reusable")
    if counts.get("planned_rhs") != len(expected_ids):
        raise ValueError("summary planned count does not match workload membership")
    if counts.get("attempted_rhs") != len(solved_ids):
        raise ValueError("summary attempted count does not match solve events")
    converged = sum(value == "converged" for value in solve.get_column("solve_status").to_list())
    if counts.get("converged") != converged:
        raise ValueError("summary converged count does not match solve events")
    if "missing_system_ids" in counts:
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


def _validate_summary_memory(memory: object) -> None:
    """Validate the workload-lifetime memory discriminant independently of event peaks."""
    if not isinstance(memory, dict):
        raise TypeError("summary memory is missing")
    status = memory.get("status")
    peak_bytes = memory.get("peak_bytes")
    if status == "unknown":
        if peak_bytes is not None:
            raise ValueError("unknown summary memory must not carry peak bytes")
        return
    if status != "measured":
        raise ValueError("unknown summary memory status")
    _validate_non_negative_int(peak_bytes, "summary peak memory bytes")


def _require_close(value: object, expected: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"summary {label} is missing")
    if not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"summary {label} does not match event frames")
