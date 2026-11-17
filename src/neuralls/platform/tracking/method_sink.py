"""Persistence of one reduced method workload into its method run."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from neuralls.domain.benchmarking.selection import (
    ReferenceSelection,
)
from neuralls.domain.presentation.provenance import DatasetPresentation
from neuralls.domain.solver.models.benchmark_events import (
    MeasuredPeakMemory,
    SetupFailed,
    SolveFailed,
)
from neuralls.domain.solver.models.cost import recorded_seconds
from neuralls.domain.solver.models.group_presentation import SolverTimingContext
from neuralls.domain.solver.models.system_provenance import SystemProvenance
from neuralls.domain.solver.models.workload import MethodWorkload, MethodWorkloadSummary
from neuralls.platform.reporting.benchmark_events import (
    METHOD_ARTIFACT_DIR,
    write_method_benchmark_artifacts,
)
from neuralls.platform.tracking.run_writer import RunWriter
from neuralls.shared.types import DatasetId, SystemId


@dataclass(frozen=True, slots=True)
class MLflowBenchmarkMethodSink:
    """Write one reduced method workload into its method run, ``run_id``.

    ``datasets`` presents each dataset a workload ran on. ``solver_contexts`` and
    ``references`` give the settings and reference state each planned system was timed
    under; they are persisted with the raw events. The sink writes into the run it is given
    and never creates one: run ownership stays with the comparison session.
    """

    writer: RunWriter
    run_id: str
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
                self.writer.log_artifact(self.run_id, root / relative, METHOD_ARTIFACT_DIR)
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _log_method_metadata(
        self, workload: MethodWorkload, summary: MethodWorkloadSummary
    ) -> None:
        """Log only reducer-owned searchable scalars, never event-loop aggregates."""
        self.writer.log_params(
            self.run_id,
            {
                "method_id": str(workload.method),
                "method_config_digest": str(workload.config_digest),
            },
        )
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
            "workload_setup_s": summary.workload_setup_s,
            "workload_setup_amortized_s": summary.workload_setup_amortized_s,
            "workload_solve_s": summary.workload_solve_s,
            "workload_online_s": summary.workload_online_s,
            "workload_online_amortized_s": summary.workload_online_amortized_s,
            "n_setup_failed": float(n_setup_failed),
            "n_solve_failed": float(n_solve_failed),
            "iterations_total": float(
                sum(attempt.outcome.iterations for attempt in workload.solve_attempts)
            ),
        }
        outcome = "completed_with_failures" if n_setup_failed or n_solve_failed else "completed"
        self.writer.set_tags(self.run_id, {"workload_outcome": outcome})
        if isinstance(summary.workload_peak_memory, MeasuredPeakMemory):
            metrics["workload_peak_memory_bytes"] = float(summary.workload_peak_memory.bytes)
        offline = workload.offline_cost
        for key, stage in (
            ("generation_s", offline.charged_generation()),
            ("fit_s", offline.fit),
            ("offline_s", offline.combined()),
        ):
            seconds = recorded_seconds(stage)
            if seconds is not None:
                metrics[key] = seconds
        self.writer.log_metrics(self.run_id, metrics)
