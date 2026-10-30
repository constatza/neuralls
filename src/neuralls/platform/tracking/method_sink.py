"""Persistence of one reduced method workload into its already-active MLflow run."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mlflow

from neuralls.domain.benchmarking.selection import (
    ReferenceSelection,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import (
    MeasuredPeakMemory,
    SetupFailed,
    SolveFailed,
)
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    write_method_benchmark_artifacts,
)
from neuralls.shared.types import DatasetId, SystemId


@dataclass(frozen=True, slots=True)
class MLflowBenchmarkMethodSink:
    """Write one reduced method workload into the already-active child run.

    ``datasets`` presents each dataset a workload ran on. ``solver_contexts`` and
    ``references`` give the settings and reference state each planned system was timed
    under; they are persisted with the raw events.

    Run ownership deliberately stays in ``comparison_batch``.  Calling this
    sink while no run is active is a caller error, just like calling
    ``mlflow.log_metric`` directly; the sink never silently creates a nested
    system or preconditioner run.
    """

    datasets: Mapping[DatasetId, DatasetPresentation]
    solver_contexts: Mapping[SystemId, SolverTimingContext]
    references: Mapping[SystemId, ReferenceSelection]

    def record_method(
        self,
        workload: MethodWorkload,
        summary: MethodWorkloadSummary,
        *,
        root: Path | None = None,
    ) -> None:
        """Persist canonical event artifacts and one copy of each selected scalar."""
        temporary: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            temporary = tempfile.TemporaryDirectory(prefix="neuralls-method-")
            root = Path(temporary.name)
        try:
            systems = [
                SystemProvenance(
                    tag, self.solver_contexts[tag.system_id], self.references[tag.system_id]
                )
                for tag in workload.planned
            ]
            artifacts = write_method_benchmark_artifacts(
                workload, summary, root, systems=systems, datasets=self.datasets
            )
            self._log_method_metadata(workload, summary)
            for relative in artifacts.paths():
                mlflow.log_artifact(str(root / relative), artifact_path=METHOD_ARTIFACT_DIR)
        finally:
            if temporary is not None:
                temporary.cleanup()

    @staticmethod
    def _log_method_metadata(workload: MethodWorkload, summary: MethodWorkloadSummary) -> None:
        """Log only reducer-owned searchable scalars, never event-loop aggregates."""
        mlflow.log_param("method_id", str(workload.method))
        mlflow.log_param("method_config_digest", str(workload.config_digest))
        n_setup_failed = sum(
            isinstance(event.outcome, SetupFailed) for event in workload.setup_events
        )
        n_solve_failed = sum(
            isinstance(attempt.outcome, SolveFailed) for attempt in workload.solve_attempts
        )
        metrics = {
            "n_planned_rhs": float(summary.n_planned_rhs),
            "n_attempted_rhs": float(summary.n_attempted_rhs),
            "n_converged": float(summary.n_converged),
            "success_rate": (
                summary.n_converged / summary.n_planned_rhs if summary.n_planned_rhs else 0.0
            ),
            "setup_total_s": summary.setup_total_s,
            "solve_total_s": summary.solve_total_s,
            "workload_online_s": summary.workload_online_s,
            "n_setup_failed": float(n_setup_failed),
            "n_solve_failed": float(n_solve_failed),
            "iterations_total": float(
                sum(attempt.outcome.iterations for attempt in workload.solve_attempts)
            ),
        }
        mlflow.set_tag(
            "workload_outcome",
            "completed_with_failures" if n_setup_failed or n_solve_failed else "completed",
        )
        if summary.online_s_per_attempted_rhs is not None:
            metrics["online_s_per_attempted_rhs"] = summary.online_s_per_attempted_rhs
        if isinstance(summary.workload_peak_memory, MeasuredPeakMemory):
            metrics["workload_peak_memory_bytes"] = float(summary.workload_peak_memory.bytes)
        if workload.generation_cost is not None:
            metrics["generation_s"] = workload.generation_cost.wall_time_seconds
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
