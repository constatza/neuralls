# Comparison Benchmark Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace denormalized per-system comparison timing with scientifically correct method workloads whose setup and solve events reconcile exactly with summaries, MLflow runs, figures, and documentation.

**Architecture:** Preserve the existing planner's `(matrix, RHS)` identities, but execute them method-major. Each method records one setup event per matrix and one linked solve attempt per attempted RHS; pure Polars reductions derive all totals and per-system costs. Method child runs own raw data and method artifacts, while the session parent owns cross-method reporting and one typed figure manifest.

**Tech Stack:** Python 3.13+, frozen/slotted dataclasses, Pydantic v2, Polars, Matplotlib, MLflow, TOML, Parquet, pytest.

**Spec:** `docs/correction-plan.md`

## Global Constraints

- The independent workload unit is a matrix with its associated RHS bundle; RHS observations are nested within matrices.
- Every method receives one timed `setup(matrix)` event per distinct matrix, including a measured near-zero event for static/no-op setup.
- A setup event is referenced by solve attempts and is never copied into solve records.
- `solve_s` includes only PCG/FCG execution and preconditioner applications. `online_s` is setup plus solve only.
- Dataset/checkpoint reads, reference solves, staging, plotting, artifact writes, MLflow calls, and historical generation/training stay outside online timing.
- Required computation after I/O, including matrix-specific binding, belongs inside setup or solve; no untimed compute gap is allowed.
- Representative repeat statistics contribute to totals; warmups and individual repeat samples are retained but never summed into online time.
- Planned RHS membership is an explicit aggregation input. Missing attempts must not shrink the planned denominator invisibly.
- Only one candidate method and its method-owned resources may be resident while that method's workload executes.
- No observed result may select a reference. Reference-relative output requires an explicitly configured, executed method.
- ECDFs and Dolan-More performance profiles use `(matrix, RHS)` instances; failures stay censored in their denominators.
- Any uncertainty interval resamples whole matrix clusters with all sibling RHS observations.
- COCO-style ERT, recurring cost, setup-sharing policy, and ambiguous total wall-time terminology are removed from this workflow.
- Profiles/ECDFs are recipe-specific. Medium/global reports are explicitly named workload mixtures and emit aggregate bars only.
- Titles contain the core metric only; short scope goes in the subtitle and full context goes in `figure_manifest.toml`.
- Dataset/preconditioner legend semantics are unchanged.
- Every affected module README is edited in place. `AGENTS.md` and `docs/coupling_graph.dot` change only if ownership or dependency edges change.

## Review Focus

- **Two matrices with three RHS values each:** Task 2 and Task 4 prove two setup events and six solve attempts per method, with exactly one allocated setup charge.
- **Setup failure or interrupted method:** Task 2 proves planned/attempted/converged counts remain distinct and incomplete summaries are not ranking candidates.
- **Configured reference fails most systems:** Task 7 proves asymmetric failures remain visible and ratios use only both-solved pairs.
- **Several recipes share one matrix:** Task 2 proves recipe aggregation allocates the matrix setup across its complete planned RHS bundle once; Task 8 proves profiles remain separate.
- **Repeated timing and MLflow duration:** Task 4 and Task 6 prove totals use representative event values and never use process/run elapsed time.

---

### Task 1: Add typed benchmark identities and raw events

**Files:**
- Create: `src/neuralls/domain/solver/models/benchmark_events.py`
- Modify: `src/neuralls/shared/types.py`
- Modify: `src/neuralls/domain/solver/models/__init__.py`
- Create: `tests/domain/test_benchmark_events.py`
- Modify: `src/neuralls/domain/solver/README.md`

**Interfaces:**
- Produces semantic `SetupEventId`, `SystemId`, and `PreconditionerName` `NewType`s in `shared/types.py`.
- Produces `PeakMemory = MeasuredPeakMemory | UnknownPeakMemory`; unknown memory is never encoded as zero or `None`.
- Produces `SetupOutcome = SetupCompleted | SetupFailed` and `SolveOutcome = SolveConverged | SolveFailed`.
- Produces `MatrixSetupEvent`, `SolveAttempt`, and moves `SystemTag` from `reporting/metrics_frame.py` into this model module.
- `SolveAttempt.setup_event_id` is required; no setup seconds, convergence Boolean, or state-dependent optional field exists on it.

- [ ] **Step 1: Write event-construction tests and fixtures**

  Use reusable fixtures with `DatasetId("ds")`, `MatrixId(7)`, `SystemId("ds-7-0")`, `PreconditionerName("amg")`, and `SetupEventId("amg:ds:7")`. Assert complete payload retention, frozen/slotted behavior, and `TypeError` for omitted required identity/timing/sample/memory arguments.

- [ ] **Step 2: Write disjoint-state tests**

  Assert `SolveConverged(iterations=4, final_relative_residual=1e-9)` rejects `reason=...`; assert `SolveFailed(iterations=100, reason="iteration limit")` rejects `final_relative_residual=...`. Add equivalent `SetupCompleted`/`SetupFailed(reason=...)` checks.

- [ ] **Step 3: Verify RED**

  Run: `uv run pytest tests/domain/test_benchmark_events.py -q`

  Expected: import failure for `benchmark_events` and the three semantic IDs.

- [ ] **Step 4: Implement the exact domain shapes**

  ```python
  @dataclass(frozen=True, slots=True)
  class MatrixSetupEvent:
      event_id: SetupEventId
      dataset: DatasetId
      matrix_id: MatrixId
      method: PreconditionerName
      setup_s: float
      samples_s: tuple[float, ...]
      peak_memory: PeakMemory
      outcome: SetupOutcome

  @dataclass(frozen=True, slots=True)
  class SolveAttempt:
      setup_event_id: SetupEventId
      system_id: SystemId
      dataset: DatasetId
      recipe: str
      matrix_id: MatrixId
      rhs_ordinal: int
      method: PreconditionerName
      solve_s: float
      samples_s: tuple[float, ...]
      peak_memory: PeakMemory
      outcome: SolveOutcome
  ```

  Validate non-negative times, samples, ordinals, iterations, residuals, and measured bytes in `__post_init__`. Keep rendered recipe text open; generated RHS values do not always have a physical `RhsId`.

- [ ] **Step 5: Export and document ownership**

  Export the public types from `models/__init__.py`. Update the solver README to state that setup events and solve attempts are canonical timing data; `CGComparisonResult` remains a diagnostic solver DTO during migration.

- [ ] **Step 6: Verify GREEN and types**

  Run: `uv run pytest tests/domain/test_benchmark_events.py -q`

  Run: `uv run ty check src/neuralls/domain/solver src/neuralls/shared`

  Expected: all pass.

- [ ] **Step 7: Commit**

  ```text
  feat(benchmarking): add typed setup and solve events

  - model benchmark identities, memory availability, and setup/solve outcomes explicitly
  - document raw-event ownership and online timing boundaries
  ```

### Task 2: Build normalized Polars frames and matrix-first summaries

**Files:**
- Create: `src/neuralls/domain/solver/reporting/event_frames.py`
- Rewrite: `src/neuralls/domain/solver/reporting/aggregation.py`
- Modify: `src/neuralls/domain/solver/models/convergence.py`
- Create: `src/neuralls/domain/solver/models/workload.py`
- Create: `tests/domain/test_event_aggregation.py`
- Modify: `src/neuralls/domain/solver/README.md`

**Interfaces:**
- Produces `build_setup_frame(events: Collection[MatrixSetupEvent]) -> pl.DataFrame`.
- Produces `build_solve_frame(attempts: Collection[SolveAttempt]) -> pl.DataFrame`.
- Produces `build_system_cost_frame(planned: Sequence[SystemTag], setup_events: Collection[MatrixSetupEvent], attempts: Collection[SolveAttempt]) -> pl.DataFrame`.
- Produces `build_matrix_bundle_frame(planned: Sequence[SystemTag], setup_events: Collection[MatrixSetupEvent], attempts: Collection[SolveAttempt]) -> pl.DataFrame`.
- Produces `summarize_method_workload(method: PreconditionerName, planned: Sequence[SystemTag], setup_events: Collection[MatrixSetupEvent], attempts: Collection[SolveAttempt], workload_peak_memory: PeakMemory) -> MethodWorkloadSummary`.
- Produces `ranking_costs(summaries: Sequence[MethodWorkloadSummary]) -> Mapping[PreconditionerName, float]`; incomplete variants are rejected rather than silently omitted or ranked.
- Produces explicit `CompleteWorkloadSummary` and `IncompleteWorkloadSummary`; only the complete variant is accepted by ranking helpers.

- [ ] **Step 1: Add the canonical two-matrix fixture**

  In `tests/domain/test_event_aggregation.py`, fixtures create six planned tags, setup costs `6.0` and `12.0`, and solve costs `(1, 2, 3)` and `(4, 5, 6)`. Expected totals are setup `18`, solve `21`, workload online `39`, and online per attempted RHS `6.5`.

- [ ] **Step 2: Add linkage and allocation tests**

  Assert setup-frame height `2`, solve-frame height `6`, and system costs `(3, 4, 5, 8, 9, 10)`. Assert matrix denominators are both `3`, allocated setup sums to `18`, an unknown setup ID raises `ValueError`, and a mismatched dataset/matrix/method link raises `ValueError`.

- [ ] **Step 3: Add recipe and incomplete-workload tests**

  Split one matrix's RHS tags across two recipes. Assert recipe-group allocated setup still sums to one matrix charge. For six planned systems with four attempts and three convergences, assert the result is `IncompleteWorkloadSummary`, contains the two missing `SystemId`s, reports `(6, 4, 3)`, and is rejected by `ranking_costs(...)`.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/domain/test_event_aggregation.py -q`

  Expected: missing frame builders and summary variants.

- [ ] **Step 5: Implement schemas and referential validation**

  Setup columns: `setup_event_id`, `dataset`, `matrix_id`, `method`, `setup_s`, `samples_s`, `peak_memory_bytes`, `memory_status`, `setup_status`, `failure_reason`.

  Solve columns: `setup_event_id`, `system_id`, `dataset`, `recipe`, `matrix_id`, `rhs_ordinal`, `method`, `solve_s`, `samples_s`, `peak_memory_bytes`, `memory_status`, `solve_status`, `iterations`, `final_relative_residual`, `failure_reason`.

  Nullable columns are permitted only at this Polars serialization boundary; domain construction stays variant-based.

- [ ] **Step 6: Implement matrix-first derivations**

  Count planned RHS membership by `(dataset, matrix_id)` before joining attempts. Derive `allocated_setup_s = setup_s / planned_rhs_count`, `amortized_online_s = solve_s + allocated_setup_s`, `matrix_online_s = setup_s + sum(solve_s)`, and workload totals. Never group raw setup rows by recipe.

- [ ] **Step 7: Implement summary variants**

  Both summary variants carry method, planned/attempted/converged counts, setup/solve/workload totals, attempted-RHS average when defined, and peak memory. `IncompleteWorkloadSummary` additionally carries missing system IDs and no rankable cost.

- [ ] **Step 8: Verify GREEN**

  Run: `uv run pytest tests/domain/test_event_aggregation.py -q`

  Run: `uv run ty check src/neuralls/domain/solver`

  Expected: all pass.

- [ ] **Step 9: Commit**

  ```text
  feat(reporting): derive workload costs from normalized events

  - validate setup-to-solve linkage and allocate each matrix setup once
  - distinguish complete and incomplete workload summaries with planned counts
  ```

### Task 3: Remove recurring-time, sharing, ERT, and projected-cost models

**Files:**
- Modify: `src/neuralls/domain/solver/models/outcomes.py`
- Modify: `src/neuralls/domain/solver/models/group_outcomes.py`
- Modify: `src/neuralls/domain/solver/models/statistics.py`
- Modify: `src/neuralls/domain/solver/reporting/classification.py`
- Modify: `src/neuralls/domain/benchmarking/costs.py`
- Modify: `src/neuralls/domain/benchmarking/profiles.py`
- Modify: `src/neuralls/domain/benchmarking/paired.py`
- Modify: `src/neuralls/domain/benchmarking/uncertainty.py`
- Remove: `src/neuralls/domain/solver/reporting/metrics_frame.py`
- Remove: `src/neuralls/domain/solver/reporting/solve_time.py`
- Remove: `src/neuralls/domain/solver/models/sharing.py`
- Remove: `src/neuralls/domain/solver/models/projection.py`
- Remove: `src/neuralls/domain/benchmarking/breakeven.py`
- Rewrite: `tests/domain/test_aggregate_projected_results.py`
- Remove: `tests/domain/test_recurring_time.py`
- Modify: `tests/domain/test_profiles.py`
- Modify: `tests/domain/test_classify_level.py`
- Remove: `tests/domain/test_breakeven.py`
- Remove: `tests/domain/test_projected_comparison_result.py`
- Remove: `tests/domain/test_setup_sharing_policy.py`

**Interfaces:**
- `Solved` carries `matrix_id`, `iterations`, `solve_s`, and `amortized_online_s`.
- `Failed` carries `matrix_id`, `iterations`, and `reason`; it never receives an imputed finite time.
- `costs.solve_seconds` and `costs.amortized_online_seconds` are distinct profile cost functions.
- `ConvergenceSummary` becomes counts plus converged-iteration statistics; it contains no ERT field.
- `PairedComparison` uses `speed_ratios`, `median_speed_ratio`, `median_speed_ratio_ci`, `n_both_solved`, `n_reference_only`, and `n_method_only`.
- Matrix-cluster bootstrap helpers accept matrix-keyed samples and resample matrix bundles, not RHS rows.

- [ ] **Step 1: Rewrite profile tests first**

  Preserve failure plateaus and fastest-per-problem Dolan-More normalization. Run every profile test once with `solve_seconds` and once with `amortized_online_seconds`. Assert failed observations stay in the denominator and have no numeric time.

- [ ] **Step 2: Add clustered uncertainty tests**

  Use two matrices with unequal RHS counts. Patch or seed the resampler and prove siblings are selected as a cluster; add a regression whose interval differs from independent RHS resampling.

- [ ] **Step 3: Add classification tests against raw events**

  Change `classify_level` to consume planned tags, setup events, solve attempts, and the report level. Assert planned/attempted/converged and matrix counts independently, and assert no result exposes `ert_seconds`, `recurring_s`, or setup-sharing fields.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/domain/test_profiles.py tests/domain/test_classify_level.py tests/domain/test_event_aggregation.py -q`

  Expected: failures on old `Solved.recurring_time_s`, sharing arguments, and ERT/projection fields.

- [ ] **Step 5: Implement outcome and profile migration**

  Build `Solved`/`Failed` from the system-cost frame. Preserve measured failure solve time only in the raw attempt/frame for workload totals. Rename every ratio field and docstring atomically.

- [ ] **Step 6: Delete obsolete models and callers**

  Remove sharing-policy derivation, `recurring_seconds`, projected scaling laws, break-even values, and `ert_interval`. Replace `CGComparisonResult.total_time_seconds` with explicit diagnostic `solve_time_seconds` and offline generation access; aggregate online reporting must no longer read `CGComparisonResult`.

- [ ] **Step 7: Verify domain suite and stale imports**

  Run: `uv run pytest tests/domain -q`

  Run: `rg -n "SetupSharingPolicy|share_setup|recurring_s|ert_seconds|ProjectedComparisonResult|total_time_seconds" src/neuralls/domain tests/domain`

  Expected: tests pass; search has no reachable comparison-reporting use.

- [ ] **Step 8: Commit**

  ```text
  refactor(reporting): remove recurring and ERT cost models

  - use explicit solve and amortized-online outcomes with censored failures
  - remove setup-sharing projections and cluster uncertainty by matrix
  ```

### Task 4: Execute one method over matrix/RHS bundles

**Files:**
- Modify: `src/neuralls/composition/assignments/comparison_batch.py`
- Modify: `src/neuralls/composition/comparison/comparison_run.py`
- Modify: `src/neuralls/composition/comparison/_preconditioner_setup.py`
- Modify: `src/neuralls/composition/comparison/models.py`
- Modify: `src/neuralls/domain/solver/comparison.py`
- Modify: `src/neuralls/shared/device.py`
- Modify: `tests/composition/test_preconditioner_sweep_cache.py`
- Modify: `tests/composition/test_comparison_batch_scoping.py`
- Create: `tests/workflows/test_comparison_event_lifecycle.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Produces frozen/slotted `CompletedMethodWorkload(method, planned, setup_events, solve_attempts, diagnostics, offline_costs, workload_peak_memory)`.
- Produces `execute_method_workload(method: ResolvedMethod, bundles: Sequence[MatrixWorkload], *, params: SolverParams, settings: NeurallsSettings, notation_policy: NotationPolicy) -> CompletedMethodWorkload`.
- A `MatrixWorkload` contains one matrix identity and its ordered planned RHS members; loaded tensors are scoped to one bundle.
- A method-owned context shares checkpoint objects across matrices, but prepared matrix state only across sibling RHS values.

- [ ] **Step 1: Characterize the old order with failing expectations**

  Replace old cache assertions with a trace for two methods, two matrices, and two RHSs: `method -> setup matrix -> solve RHSs -> release matrix -> ... -> release method`. Assert the old system-major trace fails.

- [ ] **Step 2: Add setup-count and state-reuse tests**

  For static Identity and a matrix-dependent spy, assert exactly two measured setup events each, four attempts each, and exact object identity reuse only across sibling RHSs. Static setup may reuse loaded weights but must not reuse a setup event across matrices.

- [ ] **Step 3: Add timing-boundary and failure tests**

  Spy on checkpoint/data reads, extra-input conversion/binding, setup, reference solve, solver call, artifact writing, and MLflow. Assert reads/reference/staging/reporting are outside timers; conversion/binding is inside setup; solver is inside solve. A setup failure records `SetupFailed`, zero attempts for that matrix, and an incomplete workload.

- [ ] **Step 4: Add repeat and memory tests**

  Assert events retain all samples but totals use each representative once. Add a workload-memory observer test proving one peak value is recorded over the method resource lifetime while per-operation measurements remain valid.

- [ ] **Step 5: Verify RED**

  Run: `uv run pytest tests/composition/test_preconditioner_sweep_cache.py tests/composition/test_comparison_batch_scoping.py tests/workflows/test_comparison_event_lifecycle.py -q`

  Expected: failures show system-major scheduling, cross-matrix static reuse, and post-setup binding.

- [ ] **Step 6: Split I/O from timed preparation**

  Replace `_load_and_bind_extra_inputs` with an untimed loader returning staged tensors and a timed binder/preparer. Ensure `PreconditionerService.preload` remains outside setup and method-owned checkpoint storage is released after the method.

- [ ] **Step 7: Implement method-major orchestration**

  Group planned systems by method, then `(dataset, matrix_id)`. Load one matrix bundle, measure one setup, solve ordered RHS members, release matrix state, then release the method before constructing the next method.

- [ ] **Step 8: Implement non-resetting workload memory measurement**

  Do not nest two CUDA peak-reset contexts. Add a workload observer compatible with `repeat_timed`; document CPU `ru_maxrss` as a process high-water lower bound when exact scoping is unavailable.

- [ ] **Step 9: Verify GREEN and types**

  Run the focused tests above, then: `uv run ty check src/neuralls/composition src/neuralls/domain/solver src/neuralls/shared`

  Expected: all pass.

- [ ] **Step 10: Commit**

  ```text
  refactor(comparison): execute method workloads by matrix

  - measure one setup per method and matrix and reuse state across sibling RHS solves
  - bound method residency and separate online computation from loading and reporting
  ```

### Task 5: Persist versioned event artifacts and summaries

**Files:**
- Create: `src/neuralls/platform/reporting/benchmark_events.py`
- Modify: `src/neuralls/platform/reporting/artifacts.py`
- Create: `tests/platform/test_benchmark_event_artifacts.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Defines `BENCHMARK_EVENT_SCHEMA_VERSION = 1`.
- Produces `write_method_benchmark_artifacts(workload: CompletedMethodWorkload, summary: MethodWorkloadSummary, root: Path) -> MethodBenchmarkArtifacts`.
- Produces schema-checking `read_setup_events(path)` and `read_solve_attempts(path)` helpers used by tests and reuse validation.
- Writes `setup_events.parquet`, `solve_attempts.parquet`, and `summary.toml`; raw Parquet is canonical and the summary is derived only from Task 2 reducers.

- [ ] **Step 1: Write Parquet round-trip tests**

  Include completed/failed setup outcomes, converged/failed solves, unknown/measured memory, and nontrivial sample lists. Assert exact schema, values, order, and setup links after round-trip.

- [ ] **Step 2: Write summary reconciliation and version tests**

  Recompute every summary timing/count from returned frames. Assert the TOML contains schema version, method identity/config digest, counts, setup/solve/online totals, optional attempted-RHS average, peak memory, and separately named offline costs. Reject missing/wrong schema versions and corrupt links.

- [ ] **Step 3: Verify RED**

  Run: `uv run pytest tests/platform/test_benchmark_event_artifacts.py -q`

  Expected: missing serializer module and artifact DTO.

- [ ] **Step 4: Implement the serialization boundary**

  Keep Polars and TOML conversion in platform code. Store stable artifact-relative paths in `MethodBenchmarkArtifacts`; never expose temporary-directory paths.

- [ ] **Step 5: Verify GREEN**

  Run: `uv run pytest tests/platform/test_benchmark_event_artifacts.py -q`

  Run: `uv run ty check src/neuralls/platform/reporting`

- [ ] **Step 6: Commit**

  ```text
  feat(reporting): persist versioned benchmark events

  - round-trip linked setup and solve events through schema-versioned Parquet
  - derive human-readable method summaries from canonical event frames
  ```

### Task 6: Replace system runs with method-level MLflow runs

**Files:**
- Modify: `src/neuralls/domain/solver/ports.py`
- Modify: `src/neuralls/platform/tracking/comparison_tracking.py`
- Modify: `src/neuralls/platform/tracking/mlflow_client.py`
- Modify: `src/neuralls/composition/assignments/comparison_batch.py`
- Rewrite: `tests/platform/test_comparison_sink.py`
- Modify: `tests/workflows/test_mlflow_artifact_locations.py`
- Modify: `tests/workflows/test_comparison_workflow.py`

**Interfaces:**
- Replaces `CostRecorder` with `BenchmarkMethodSink.record_method(workload, summary) -> None`.
- `MLflowBenchmarkMethodSink` opens no runs itself; the batch owns one nested run per method and calls it after aggregation/artifact staging.
- Reuse identity includes phase `comparison_method`, schema version `1`, method config/checkpoint digest, and complete workload membership.

- [ ] **Step 1: Add no-loop-logging tests**

  Record MLflow calls around a fake event emitter. Assert zero aggregate metric calls during setup/solve emission; after `record_method`, assert each permitted scalar exactly once: planned/attempted/converged, success rate, setup total, solve total, workload online total, optional online per attempted RHS, and workload peak memory.

- [ ] **Step 2: Add hierarchy and artifact-location tests**

  With the real SQLite fixture, assert one session parent and one direct child per method; no system runs and no method grandchildren. Method children own config, two Parquet files, summary, and method figures. Parent owns only shared provenance and cross-method reporting.

- [ ] **Step 3: Add reuse-version regression**

  Seed an otherwise matching legacy run without schema tag and assert execution occurs. Seed a complete version-1 method run with required raw artifacts and assert valid reuse.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/platform/test_comparison_sink.py tests/workflows/test_mlflow_artifact_locations.py tests/workflows/test_comparison_workflow.py -q`

- [ ] **Step 5: Implement the port and topology atomically**

  Remove per-system parent metrics, nested preconditioner runs, `best_preconditioner`, ERT, recurring, and per-loop aggregate logs. Keep residual histories as method-owned artifacts rather than MLflow grandchildren.

- [ ] **Step 6: Verify GREEN**

  Run the focused tests above and `uv run ty check src/neuralls/domain/solver src/neuralls/platform/tracking src/neuralls/composition`.

- [ ] **Step 7: Commit**

  ```text
  feat(tracking): record one MLflow run per method workload

  - store raw method events and selected post-aggregation scalars on method children
  - keep cross-method reporting on the benchmark session and version reuse identities
  ```

### Task 7: Require and propagate an explicit reference

**Files:**
- Modify: `src/neuralls/platform/config/models/experiments.py`
- Modify: `src/neuralls/platform/config/models/comparison.py`
- Modify: `src/neuralls/composition/comparison/config_assembler.py`
- Modify: `src/neuralls/composition/assignments/comparison_batch.py`
- Modify: `src/neuralls/domain/solver/models/config.py`
- Modify: `src/neuralls/domain/benchmarking/selection.py`
- Modify: `src/neuralls/domain/benchmarking/reference_comparison.py`
- Modify: `src/neuralls/domain/solver/reporting/classification.py`
- Create: `tests/configuration/test_comparison_reference.py`
- Modify: `tests/domain/test_reference_comparison.py`
- Modify: `tests/domain/test_profiles.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Adds `reference_preconditioner: PreconditionerName | None = None` to `ComparisonGeneral`; case baseline and method override resolve into that one domain field.
- Produces `ReferenceSelection = ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable`.
- `compare_to_reference(outcomes, *, selection, seed)` never chooses from observed outcomes.
- Post-resolution validation checks the final executed method names, including assignment filters and skipped model resolution.

- [ ] **Step 1: Add config tests**

  Cover valid explicit reference, omitted reference, unknown name, method override filtering it out, and runtime model-resolution filtering it out. Unknown/static-filter cases raise a focused `ValueError` naming the configured and available methods; a runtime resolution skip produces `ReferenceUnavailable` with the skipped-method reason and no fallback.

- [ ] **Step 2: Add asymmetric reference tests**

  Configure `amg`; make it fail a system that another method solves. Assert reference-only/method-only counts remain visible, ratios use only both-solved systems, and no selection changes to the more successful method. With no reference, assert empty comparisons and `NoReferenceConfigured`.

- [ ] **Step 3: Preserve performance-profile semantics**

  Add a test proving performance profiles still normalize by fastest method per problem even when no configured reference exists; this fastest method is never returned as `ReferenceSelection`.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/configuration/test_comparison_reference.py tests/domain/test_reference_comparison.py tests/domain/test_profiles.py tests/domain/test_classify_level.py -q`

- [ ] **Step 5: Implement propagation and validation**

  Merge the method override by checking `model_fields_set`; reject references absent from the statically resolved method tuple. If runtime checkpoint/model resolution later skips that otherwise-valid method, emit `ReferenceUnavailable` and never select a fallback. A method that executes but fails remains the configured reference.

- [ ] **Step 6: Rename reference-facing schema**

  Rename speedup fields/keys to speed ratio atomically in domain DTOs, TOML/MLflow adapters, and tests. Direction is always `reference / method`; below one is slower, above one is faster.

- [ ] **Step 7: Verify GREEN and commit**

  Run the focused tests and `uv run ty check src/`.

  ```text
  fix(benchmarking): require an explicit comparison reference

  - validate the configured comparator against final executed methods
  - preserve asymmetric failures and rename reference ratios unambiguously
  ```

### Task 8: Separate recipe profiles from workload-mixture summaries

**Files:**
- Modify: `src/neuralls/domain/solver/reporting/report_levels.py`
- Modify: `src/neuralls/domain/solver/reporting/classification.py`
- Modify: `src/neuralls/domain/solver/models/group_outcomes.py`
- Modify: `src/neuralls/composition/comparison/group_reporting.py`
- Modify: `src/neuralls/domain/solver/ports.py`
- Modify: `tests/domain/test_classify_level.py`
- Modify: `tests/composition/test_group_reporting.py`
- Modify: `tests/workflows/test_comparison_sweep_projected_metrics.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Produces a closed report policy: `RecipeProfileReport` for `FineLevel`; `WorkloadMixtureReport` for medium/global.
- Only recipe reports carry per-system profile/ECDF inputs. Mixture reports carry matrix/workload summaries and complete recipe membership.
- Counts are `n_matrices`, `n_planned_rhs`, `n_attempted_rhs`, and `n_converged`, never rendered as `matrices x RHS`.

- [ ] **Step 1: Add two-frequency fixtures and policy tests**

  Use masked sinusoidal recipes with two omega values. Assert two fine recipe reports, mixture membership containing both recipes at medium/global, and no profile data on mixture variants.

- [ ] **Step 2: Add sink-call tests**

  Assert `report_all_levels` schedules profiles/ECDFs only for fine variants and schedules aggregate bars for all levels. One-recipe medium/global reports remain mixture bars only to avoid duplicate profiles.

- [ ] **Step 3: Verify RED**

  Run: `uv run pytest tests/domain/test_classify_level.py tests/composition/test_group_reporting.py tests/workflows/test_comparison_sweep_projected_metrics.py -q`

- [ ] **Step 4: Implement variants and exhaustive dispatch**

  Remove presentation decisions based on label strings. Use `match` plus `assert_never` for report variants; preserve ordered datasets/recipes and explicit weighting metadata.

- [ ] **Step 5: Verify GREEN and commit**

  Run focused tests and `uv run ty check src/neuralls/domain/solver src/neuralls/composition`.

  ```text
  fix(reporting): separate recipe profiles from workload mixtures

  - emit per-system profiles only for homogeneous recipe groups
  - retain explicit medium and global workload summaries with independent counts
  ```

### Task 9: Replace plan-order scatter with setup-aware bars and linear paired plots

**Files:**
- Modify: `src/neuralls/platform/reporting/outcome_plots.py`
- Modify: `src/neuralls/platform/tracking/comparison_tracking.py`
- Modify: `tests/workflows/test_outcome_plots.py`
- Modify: `tests/platform/test_comparison_sink.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Produces `plot_online_time_bars(summaries, *, scenario, style, save_path)` using Task 2 aggregate DTOs.
- Produces separate solve-only and amortized-online per-system figures plus workload-online total bars.
- `plot_paired_time_scatter(outcomes: Outcomes, *, reference: ConfiguredReference, style: GroupStyle, save_path: Path) -> Path | None` uses linear equal axes and never imports selection logic.

- [ ] **Step 1: Write bar-artist tests**

  Assert `Axes.patches` contains method bars for solve and setup-plus-solve, no integer-plan-order scatter collection exists, widths equal aggregate values, incomplete methods are marked/not ranked, and convergence `n/N` is visible. Error bars appear only when a matrix-cluster interval is supplied.

- [ ] **Step 2: Write paired-axis tests**

  Pass a configured reference that is neither fastest nor most successful. Assert both axes are linear, limits equal and non-negative, aspect is equal, the `y=x` line spans the shared limits, and reference name appears in the x label or subtitle. Add zero/near-zero values to pin the padding policy.

- [ ] **Step 3: Write failure-visibility and ratio-copy tests**

  Assert failed points remain censored/success-annotated rather than imputed. Assert the speed-ratio plot and legend say `reference / method`, `slower` below one, and `faster` above one.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/workflows/test_outcome_plots.py tests/platform/test_comparison_sink.py -q`

- [ ] **Step 5: Implement plot APIs and update stable filenames**

  Replace `system_time_scatter_*` with scenario-specific bar filenames. Do not relabel the old projected-scaling plot; remove it and its TOML artifact if still reachable.

- [ ] **Step 6: Verify GREEN and commit**

  Run focused tests.

  ```text
  fix(plots): show setup-aware bars and linear paired times

  - replace plan-order timing scatters with explicit workload and per-system bars
  - use configured-reference linear paired axes and failure-aware annotations
  ```

### Task 10: Add typed figure planning and one session manifest

**Files:**
- Create: `src/neuralls/platform/reporting/figure_metadata.py`
- Modify: `src/neuralls/platform/reporting/outcome_plots.py`
- Modify: `src/neuralls/platform/reporting/plots.py`
- Modify: `src/neuralls/composition/comparison/group_reporting.py`
- Modify: `src/neuralls/domain/solver/ports.py`
- Modify: `src/neuralls/platform/tracking/comparison_tracking.py`
- Create: `tests/platform/test_figure_metadata.py`
- Modify: `tests/workflows/test_plot_features.py`
- Modify: `tests/workflows/test_outcome_plots.py`
- Modify: `tests/platform/test_comparison_sink.py`
- Modify: `src/neuralls/composition/README.md`

**Interfaces:**
- Produces closed `FigureKind`, `TimingScenario`, and `FailureTreatment` enums.
- Produces frozen/slotted `FigureMetadata` with figure kind, metric definition, scenario, report level, short scope, ordered datasets/recipes, independent counts, `ReferenceSelection`, failure treatment, timing policy, and stable source artifact names.
- Produces `FigureManifest.add(filename, metadata)` with duplicate/collision rejection and `write_figure_manifest(path)`.
- Extends the session report sink with `finalize()`, called exactly once while the session parent is active.

- [ ] **Step 1: Write metadata round-trip and collision tests**

  Assert deterministic TOML ordering, complete typed fields, duplicate filename rejection, sanitized-name collision rejection, and stable relative sources (`setup_events.parquet`, `solve_attempts.parquet`, `summary.toml`).

- [ ] **Step 2: Write title/subtitle/filename tests**

  Assert the exact expected title and subtitle for representative fine and global figures: titles are core metrics such as `Online Time`, `Performance Profile`, or `Speed Ratio`; subtitles are concise scopes such as `Dataset X - sinusoidal omega=...` or `All datasets`. Do not add separate tests for strings removed by this change. Assert filenames include kind, level, group slug, and scenario.

- [ ] **Step 3: Write manifest lifecycle tests**

  Draw multiple groups through one sink, call `finalize()`, and assert exactly one parent-level `figure_manifest.toml`; its keys equal all logged PNG basenames and omitted figures have no entry.

- [ ] **Step 4: Verify RED**

  Run: `uv run pytest tests/platform/test_figure_metadata.py tests/workflows/test_plot_features.py tests/workflows/test_outcome_plots.py tests/platform/test_comparison_sink.py -q`

- [ ] **Step 5: Implement a typed figure plan before drawing**

  The plan decides eligibility, filename, visible text, scenario, and metadata once. Plotters receive it and do not reconstruct scientific context from titles or group IDs.

- [ ] **Step 6: Retain one sink through finalization**

  Remove per-group manifest writes and temporary-path metadata. Finalize/upload while the session run is active; define an empty-manifest behavior for sessions with no figures.

- [ ] **Step 7: Verify GREEN and commit**

  Run focused tests and `uv run ty check src/`.

  ```text
  feat(reporting): add concise figures and session metadata

  - plan figure eligibility, filenames, titles, and scientific context once
  - upload one complete caption-ready manifest on the session parent
  ```

### Task 11: Rewrite benchmark terminology and configuration examples

**Files:**
- Modify: `src/neuralls/domain/solver/reporting/__init__.py`
- Modify: `src/neuralls/domain/solver/README.md`
- Modify: `src/neuralls/composition/README.md`
- Modify: `README.md`
- Modify: `src/neuralls/domain/benchmarking/*.py`
- Modify: `src/neuralls/platform/reporting/outcome_plots.py`
- Modify: affected `configs/**/*.toml`
- Create: `tests/documentation/test_comparison_metric_terms.py`

**Interfaces:**
- Documents canonical definitions used by code, artifacts, MLflow, and captions.
- Example cases name a reliable `reference_preconditioner`; Identity is chosen only where the example actually establishes it as reliable.

- [ ] **Step 1: Write documentation acceptance tests**

  Assert canonical terms exist and user-facing `recurring cost`, `shared one-time setup`, `recurring_s`, ambiguous `total wall time`, and ordinary-sweep `ERT` definitions do not. Permit `ERT` only in the explanation of non-applicability/future restart experiments.

- [ ] **Step 2: Verify RED**

  Run: `uv run pytest tests/documentation/test_comparison_metric_terms.py -q`

  Run: `rg -n "recurring cost|shared one-time setup|recurring_s" README.md docs src/neuralls`

  Expected: tests/search identify current stale text.

- [ ] **Step 3: Rewrite documentation in place**

  Define generation, checkpoint/data load, setup, solve, workload online, amortized online/RHS, success rate, empirical cumulative distribution function (ECDF), Dolan-More profile, paired time, and speed ratio. Explain matrix clustering, failures, configured reference semantics, timing exclusions, and why heterogeneous deterministic systems are not COCO restarts.

- [ ] **Step 4: Update examples and architecture reference**

  Add explicit references to applicable configs. Update module READMEs after their code changes. Modify root `AGENTS.md` only if the final module ownership differs from the existing architecture map.

- [ ] **Step 5: Verify GREEN and commit**

  Run the documentation test and acceptance `rg`; require no stale user-facing definitions.

  ```text
  docs(comparison): define benchmark costs and reference semantics

  - document setup-aware online timing, clustered failures, and profile populations
  - remove ERT, recurring-cost, and automatic-reference terminology
  ```

### Task 12: Reconcile an end-to-end benchmark and finish migration

**Files:**
- Create/modify shared fixtures under `tests/workflows/` and `tests/conftest.py`
- Create: `tests/workflows/test_comparison_benchmark_reporting.py`
- Modify: remaining fixtures/snapshots under `tests/domain`, `tests/platform`, and `tests/workflows`
- Modify: `AGENTS.md` only if module ownership changed
- Modify: `docs/coupling_graph.dot` only if dependency edges changed

**Interfaces:**
- Produces one deterministic two-matrix benchmark with two sinusoidal frequencies, multiple RHSs, measured near-zero static setup, a matrix-dependent method, one configured reliable reference, and one nonconverging Identity case.
- Every timing scalar and plotted value is reproducible from versioned method Parquet artifacts.

- [ ] **Step 1: Build shared deterministic fixtures**

  Reuse the sparse-ID dataset with physical matrix IDs `5` and `9`. Put all data creation in fixtures under `tmp_path`. Use timing/resource spies for exact representative values and samples; force Identity nonconvergence through a deterministic solver seam, not accidental numerics.

- [ ] **Step 2: Add raw-count and arithmetic assertions**

  Per method, assert two setup rows and all planned solve rows, exact setup links, near-zero setup presence, planned/attempted/converged counts, and summary totals recomputed from Parquet.

- [ ] **Step 3: Add reporting assertions**

  Assert configured-reference metadata and paired counts, two frequency-specific profile sets, no medium/global pooled profiles, workload-mixture bars, linear paired axes, independent matrix/RHS counts, and every figure in exactly one manifest entry.

- [ ] **Step 4: Add MLflow topology assertions**

  Query SQLite for one session parent and one direct child per method, no grandchildren, correct method/session artifact placement, schema tags, and no aggregate timing emission during execution.

- [ ] **Step 5: Verify focused end-to-end RED/GREEN**

  Run: `uv run pytest tests/workflows/test_comparison_benchmark_reporting.py -q`

  Expected after implementation: pass with exact artifact reconciliation.

- [ ] **Step 6: Run full verification**

  Run: `uv run pytest`

  Run: `uv run ty check src/`

  Run: `uv run ruff check src/ tests/`

  Run the terminology acceptance `rg` from Task 11.

- [ ] **Step 7: Generate and visually inspect representative artifacts**

  Inspect titles, subtitles, axes, reference labels, grouped bars, failure annotations, and recipe separation. Record the command and inspected artifact directory in the task report; do not treat MLflow duration as algorithm time.

- [ ] **Step 8: Commit**

  ```text
  test(comparison): verify setup-aware benchmark reporting

  - reconcile raw events, summaries, MLflow hierarchy, figures, and manifest end to end
  - migrate remaining fixtures and verify the complete reporting workflow
  ```

## Final review checklist

- [ ] Generate a whole-branch review package from the branch merge base through `HEAD`.
- [ ] Have a fresh reviewer check every Global Constraint and Review Focus item, especially setup failures, method residency, old-artifact reuse, reference asymmetry, and matrix-cluster intervals.
- [ ] Fix every Critical/Important finding with a failing test first, rerun the full suite/type/lint gates, and re-review the fix diff.
- [ ] Confirm `git status --short` contains no unintended files and every code-bearing module README is current.
