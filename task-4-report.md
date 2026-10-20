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

## Scope boundary

Method-level MLflow topology and report-level figure migrations remain later
tasks; this task preserves the existing per-comparison persistence contract
while method-major execution supplies its canonical diagnostics.

## Fix Round 1

The review's seven findings were addressed with focused RED/GREEN tests and
minimal production changes:

- Mixed same-name configuration identities, matrix normalization, solver/timing
  parameters, and sibling method membership are rejected before scheduling;
  this prevents sharing transformed matrices or first-entry solve semantics.
- Method-major execution now projects its canonical diagnostics back through
  the retained nested comparison run, input/config/figure artifacts, run
  parameters, child runs, and cost recorder. The SQLite artifact integration
  test runs the production workflow and verifies the comparison phase run.
- Setup failure creates a breakdown diagnostic for every planned RHS, while
  retaining the canonical failed setup event and zero solve attempts.
- Matrix bundles are yielded lazily and each bundle is released in a
  ``finally`` block; host arrays are not retained for the rest of a method.
- Parameter arrays are staged as copied NumPy values outside setup timing;
  Torch construction, device/dtype conversion, and binding occur in the timed
  setup callback.
- Checkpoint-store implementation and tests describe the method-workload
  lifetime. Checkpoint/data I/O remains untracked; generation is the only
  offline cost.

The initial focused RED characterization had 6 failures and 17 passing tests.
After the fixes, the focused lifecycle/scoping/group suite passes 21 tests;
the collection smoke reports 23 tests. Adjacent comparison workflow, run,
session-gating, and scoping tests pass 50 tests. The retained SQLite MLflow
artifact test passes, and the inherited domain/checkpoint/MLflow suite passes
75 tests with one integration deselection.

Additional verification passes: targeted ``ty``, Ruff lint and format,
``tach check``, import smoke, and ``git diff --check``. The compatibility
workflow spy now materializes the lazy iterator only inside the spy, leaving
production bundle lifetime bounded while preserving the original provenance
and RHS assertions.

Self-review: method-major solver execution remains the sole source of measured
diagnostics; the retained public result/artifact projection performs no solve
and all its I/O/reporting occurs after online timing. Rejection is explicit for
semantics or membership that cannot be represented by one method workload.

## Fix Round 2

The re-review's three accepted findings are resolved:

- Compatibility `CGComparisonResult` projections now copy the linked canonical
  `MatrixSetupEvent` representative setup seconds, repeat samples, and measured
  setup peak. Canonical aggregation remains event-backed; the copied fields are
  only for the retained per-comparison payload and MLflow compatibility sink.
- `_load_matrix_workloads` delegates each bundle to a helper scope. Prepared
  linear-system temporaries therefore die before the generator resumes to load
  the next matrix; a weak-reference regression proves the first prepared object
  is collectible before that next load starts.
- Compatibility figure styles now use config/family labels, the configured
  notation policy, existing POD dataset/weighting keys, and provenance
  contextualization without reconstructing or solving a preconditioner. A POD
  mathtext regression covers both label notation and style metadata.

Fix Round 2 verification:

- Focused Task 4/cache/lifecycle/group/setup/checkpoint/style suite: 48 passed.
- Retained SQLite MLflow artifact integration: 1 passed.
- Adjacent comparison workflow/run/session/scoping suite: 52 passed.
- Targeted Ruff, format, and `ty` checks pass; architecture, collection/import
  smoke, and `git diff --check` are rerun before commit.

Self-review: setup values are copied only after canonical event production and
never used to derive group aggregates; lazy producer locals no longer overlap
matrix loads; projected labels intentionally expose stable configured facts
when released runtime objects cannot provide realized facts. Generation remains
the sole offline cost and checkpoint/data I/O remains untracked and untimed.
