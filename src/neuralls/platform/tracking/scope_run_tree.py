"""The nested MLflow runs of a report: one run per aggregation scope, created on first use."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from neuralls.domain.solver.models.report_scope import ReportScope, parent_scope
from neuralls.domain.solver.reporting.report_levels import level_of_scope
from neuralls.platform.tracking.run_writer import RunWriter
from neuralls.platform.tracking.scope_params import scope_params

REPORT_SCOPE_PHASE = "report_scope"
"""Phase tag of every non-root run of a report tree; the root carries ``report``."""


class RunTree(Protocol):
    """Creates the child runs of a run tree; the only MLflow call the scope tree needs."""

    def child_run(
        self, parent_run_id: str, name: str, tags: Mapping[str, str], params: Mapping[str, str]
    ) -> str:
        """Create a finished run under ``parent_run_id`` and return its id."""
        ...


@dataclass(frozen=True, slots=True)
class MlflowRunTree:
    """Creates the child runs of a report in its experiment, ``experiment_id``."""

    writer: RunWriter
    experiment_id: str

    def child_run(
        self, parent_run_id: str, name: str, tags: Mapping[str, str], params: Mapping[str, str]
    ) -> str:
        """Create, parameterize and finish a run named ``name`` under ``parent_run_id``."""
        run_id = self.writer.create_run(self.experiment_id, name, tags, parent_run_id=parent_run_id)
        self.writer.log_params(run_id, params)
        self.writer.finish(run_id, "FINISHED")
        return run_id


@dataclass(slots=True)
class ScopeRunTree:
    """Maps each report scope to its run, creating ancestors before descendants.

    The root scope is the run the caller already has open, so it is never created here.
    ``name_of`` renders the scientific text that names the run of a scope.
    """

    runs: RunTree
    name_of: Callable[[ReportScope], str]
    _run_ids: dict[ReportScope, str] = field(default_factory=dict)

    @property
    def created_scopes(self) -> tuple[ReportScope, ...]:
        """The scopes whose runs were created, in creation order (parents first)."""
        return tuple(self._run_ids)

    def run_id(self, scope: ReportScope, root_run_id: str) -> str:
        """The run of ``scope``, creating it (and its ancestors) under its parent first."""
        parent = parent_scope(scope)
        if parent is None:
            return root_run_id
        known = self._run_ids.get(scope)
        if known is not None:
            return known
        run_id = self.runs.child_run(
            self.run_id(parent, root_run_id),
            self.name_of(scope),
            {"phase": REPORT_SCOPE_PHASE, "report_level": level_of_scope(scope).value},
            scope_params(scope),
        )
        self._run_ids[scope] = run_id
        return run_id
