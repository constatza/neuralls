# Comparison Failure Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve recoverable setup/solve failures as explicit benchmark outcomes while making every blocking comparison-workflow failure visible in the terminal and in the exact MLflow run where it occurred.

**Architecture:** Method child runs will own execution as well as persistence, so a blocking method failure creates a failed child instead of only a failed parent. Recoverable scientific failures remain the existing closed `SetupOutcome`/`SolveOutcome` values; blocking Python exceptions cross typed failure boundaries that record context and traceback, then re-raise the original exception unchanged.

**Tech Stack:** Python 3.14, frozen/slotted dataclasses, structural Protocols, Typer, Loguru, MLflow, pytest.

**Spec:** Approved design in the 2026-10-26 conversation; this file is its durable repository copy.

## Global Constraints

- Do not convert blocking exceptions into ordinary result values.
- Do not let diagnostic logging replace or mask the original exception.
- Keep recoverable failures explicit in canonical setup/solve events and warnings.
- Keep MLflow run ownership in composition; platform sinks never open runs.
- Add no dependency and no configurable failure-policy framework.
- Update the Markdown reference in every changed module directory.

## Review Focus

- A method exception before persistence must create a failed method child and failed parent with the same exact cause.
- A failure while recording diagnostics must not mask the workflow exception.
- Setup and solve failures must remain persisted data and must not abort unrelated methods.
- A report-finalization exception must leave completed method children intact but fail the parent.
- Both `neuralls compare` and `neuralls run` must emit a non-empty exact error and exit non-zero.

---

### Task 1: Typed comparison outcomes

**Files:**
- Modify: `src/neuralls/composition/comparison/models.py`
- Modify: `src/neuralls/composition/assignments/comparison_batch.py`
- Modify: `src/neuralls/composition/assignments/case_pipeline.py`
- Modify: `src/neuralls/cli/compare.py`
- Modify: `src/neuralls/cli/run.py`
- Test: `tests/cli/test_compare_logging.py`
- Test: `tests/composition/test_comparison_batch_session_gating.py`
- Test: `tests/workflows/test_comparison_workflow.py`

**Interfaces:**
- Produces: `ComparisonCompleted | ComparisonReused | ComparisonEntryFailed`, aliased as `ComparisonOutcome`.
- Consumes: existing `ComparisonResult`, identity fields, and warning strings.

- [x] Write tests proving each state contains only state-valid data and callers handle the union exhaustively.
- [x] Run the focused tests and confirm they fail against the boolean/optional model.
- [x] Replace `success/error/payload` with frozen, slotted state-specific variants and update constructors/callers.
- [x] Run the focused tests and type checker.

### Task 2: Typed workflow-failure context and MLflow adapter

**Files:**
- Create: `src/neuralls/shared/workflow_failures.py`
- Create: `src/neuralls/platform/tracking/failure_reporting.py`
- Create: `tests/platform/test_failure_reporting.py`
- Modify: `src/neuralls/shared/README.md`
- Modify: `src/neuralls/platform/README.md`

**Interfaces:**
- Produces: closed batch/method/matrix failure-context types and `WorkflowFailureSink.record(context, exc)`.
- Produces: `MLflowWorkflowFailureSink`, which records `failure/context.json` and `failure/traceback.txt` in the active run.

- [x] Write tests for exact type/message/traceback/context persistence and diagnostic-sink failure isolation.
- [x] Run the tests and confirm missing interfaces fail.
- [x] Implement the types, narrow port, and MLflow adapter without serializing locals.
- [x] Run the focused tests and type checker.

### Task 3: Method children own execution and persistence

**Files:**
- Modify: `src/neuralls/composition/assignments/comparison_batch.py`
- Modify: `tests/composition/test_comparison_batch_scoping.py`
- Modify: `tests/composition/test_comparison_batch_session_gating.py`
- Modify: `tests/workflows/test_comparison_workflow.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Consumes: `BenchmarkMethodSink` and `WorkflowFailureSink`.
- Produces: `_execute_method_run` as the single owner of one method child; `_execute_all` schedules and collects only.

- [x] Write tests showing execution happens inside the child, failures re-raise the identical exception, and sink failures cannot mask it.
- [x] Run the tests and confirm the current execute-then-persist topology fails them.
- [x] Move method child creation ahead of execution, persist before closing it, and remove the separate persistence pass.
- [x] Add parent boundaries around warm-up/method execution, projection, reporting, and finalization.
- [x] Run focused composition/workflow tests.

### Task 4: Recoverable failures warn and remain data

**Files:**
- Modify: `src/neuralls/composition/comparison/comparison_run.py`
- Modify: `src/neuralls/domain/solver/comparison.py`
- Modify: `src/neuralls/platform/tracking/comparison_tracking.py`
- Modify: `tests/workflows/test_comparison_event_lifecycle.py`
- Modify: `tests/platform/test_benchmark_method_sink.py`
- Modify: `src/neuralls/composition/comparison/README.md`

**Interfaces:**
- Consumes: existing `SetupFailed`, `SolveFailed`, and incomplete-workload summaries.
- Produces: contextual warnings plus MLflow `n_setup_failed`, `n_solve_failed`, and `workload_outcome` metadata.

- [x] Write tests for contextual warnings and persisted failure counts.
- [x] Run the tests and confirm warnings/metadata are absent.
- [x] Emit warnings at the setup/solve conversion boundaries and derive method metadata from canonical events.
- [x] Run focused lifecycle and sink tests.

### Task 5: Loud CLI boundary

**Files:**
- Create: `src/neuralls/cli/errors.py`
- Modify: `src/neuralls/cli/compare.py`
- Modify: `src/neuralls/cli/run.py`
- Modify: `tests/cli/test_main_cli.py`
- Modify: `tests/cli/test_compare_logging.py`
- Modify: `src/neuralls/cli/README.md`

**Interfaces:**
- Produces: one CLI helper that prints operation, exception type, exact reason, notes, and logs the traceback before raising `typer.Exit`.

- [x] Write CLI tests injecting `RuntimeError("injected workload failure")` into both commands.
- [x] Confirm both tests fail because output is empty or incomplete.
- [x] Implement the shared CLI rendering boundary, catching `Exception` but never `BaseException`.
- [x] Run focused CLI tests.

### Task 6: End-to-end failure topology

**Files:**
- Modify: `tests/workflows/test_task12_reporting_e2e.py`
- Modify: `tests/workflows/test_comparison_workflow.py`

**Interfaces:**
- Verifies all interfaces produced by Tasks 1-5 against a real local MLflow store.

- [x] Add an integration test for a blocking method exception: failed child and parent, exact diagnostic artifacts, no manifest.
- [x] Add/extend the report-finalization test: finished children, failed parent, parent diagnostics.
- [x] Add a recoverable setup/solve test: warning, finished child, canonical failure event, later work continues.
- [x] Run the integration tests with `-m integration`.

### Task 7: Full verification

- [x] Run `uv run pytest`.
- [x] Run `uv run ty check src/`.
- [x] Run `uv run ruff check src/`.
- [x] Review the final diff against the failure-classification table and SOLID constraints.

## SOLID constraints

- **SRP:** computation, orchestration, persistence, failure recording, and CLI rendering stay in separate units.
- **OCP:** orchestration consumes narrow sink protocols instead of branching on tracking backends.
- **LSP:** any failure sink may be substituted; the boundary always preserves the original workflow exception.
- **ISP:** the failure sink exposes only `record(context, exc)`.
- **DIP:** composition depends on `BenchmarkMethodSink` and `WorkflowFailureSink`; MLflow remains a platform adapter.
