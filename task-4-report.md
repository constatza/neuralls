# Task 4 report: method-major comparison execution

## Outcome

Task 4 now executes one resolved method at a time, groups its work by matrix,
and solves the ordered RHS siblings with the exact prepared state for that
matrix. A method-owned checkpoint store spans matrices; matrix state is
released before the next matrix and the method resources are released before
the next method.

Setup and solve events are the canonical hand-off. Setup failures produce a
failed setup event and no solve attempts for that matrix. Solve failures retain
their measured attempt. Repeat samples remain attached to events while the
representative value alone contributes to totals. Workload peak memory is
observed without nesting allocator peak resets.

## Authoritative cost-model correction

The user ruling is explicit: checkpoint and data I/O are outside online timing
and are not tracked as any cost or metric. The only offline cost in
`CompletedMethodWorkload.offline_costs` is the optional historical generation
cost.

Accordingly, this task removes the checkpoint-load `StageCost`, the
`checkpoint_load_s` batch metric, `CheckpointStore.load_seconds`, the added
checkpoint-load cost test, and the obsolete assertion/documentation. The
focused lifecycle trace still proves checkpoint/data reads occur before setup
timing; it does not assert an I/O cost.

## Inherited RED/GREEN evidence

The typed event and normalized aggregation work inherited by Task 4 remains
green:

- `tests/domain/test_benchmark_events.py`
- `tests/domain/test_event_aggregation.py`
- `tests/domain/test_profiles.py`
- `tests/domain/test_classify_level.py`
- Result: 60 passed.

The preceding reporting commits (`85c7df3`, `b0c85eb`, `eb45ac7`,
`f4f1bcb`, `daf78d8`, `d289fe5`, and `99365f9`) supplied the typed events,
matrix-first reductions, legacy-cost removal, and diagnostic regressions that
Task 4 consumes.

## Task 4 verification

- Focused lifecycle/cache/scoping suite: 17 passed.
- Adjacent setup, grouping, batch-scoping, and comparison-workflow suite: 51 passed.
- Targeted `ty`: passed for composition, solver, and shared packages.
- Targeted `ruff check`: passed.
- Targeted `ruff format --check`: passed.
- Architecture check (`tach check`): passed.
- Focused collection smoke: 17 tests collected.
- Import smoke for comparison batch, method execution, solver, and device modules: passed.
- `git diff --check`: passed.

The comparison workflow regression also verifies the production staged matrix
normalization and preserves distinct RHS draws for each planned system.

## Remaining scope

This report covers Task 4 execution and its required adjacent verification.
Later artifact persistence, method-level MLflow topology, and report-level
figure migrations remain separate tasks.
