# Plan: split comparison into compute and a separate report workflow

Paths relative to `src/neuralls/` unless prefixed `tests/` or `docs/`.
Branch: `refactor/comparison-report-split`. Pause and report after each step.

## Progress

All steps are implemented. Remaining work and the structural follow-ups live in `docs/plans/report-split-completion.md`.

- [x] Step 1: legacy per-system chain deleted from `src/` with its tests. Also removed what became unused: `_presentation.py`, `comparison_inputs.py`, `build_comparison_run_spec`, the child-run tag types, the tolerance-ladder and per-DOF cost helpers, `render_session_title`. Full suite, ruff, ty and tach green. Tests not preserved: three `compare_preconditioners` end-to-end tests, legacy logging and child-run-name tests, the old AMG "rebuilt for another matrix" case.
- [x] Step 2: `ComparisonCompleted.payload` -> `summary: str`; `_outcomes_from_workloads` formats the summary directly. Also removed what became unused: `build_comparison_result_from_diagnostics`, `_plots.py`, `ComparisonResult`, `ComparisonPaths`, `COST_PLOT_STYLE` and its test.
- [x] Cleanup after step 2: removed the unused `DatasetPreconditionerCache`, `SystemPreconditionerCache` and cache-side `DatasetScope` (method-major execution already builds once per matrix and loads checkpoints once per method; dataset-wide reuse of static setups was judged not worth a new setup-event type), and the unused `summarize_best_combinations`, `ComparisonRecommendations`, `RankedRecommendation` and `PlotPaths`. AGENTS.md and READMEs updated.
- [x] In-memory test for `_outcomes_from_workloads` (d837d3e): full summary string for a completed item; missing presentation yields `ComparisonEntryFailed`.
- [x] Step 3a (dd26619): `planned_systems.parquet` and `presentation.toml` writers/readers, `iterations_total`, artifact set defined once by `MethodBenchmarkArtifacts` (writer, reuse gate and `required_artifacts` tag derive from it), reuse validators for the new tables, no figure requirement in the compare reuse gate; `realized_facts.json` and `method_diagnostics.json` removed. `facts_codec` output verified TOML-safe by round trip. Full suite (including `-m integration`), ruff, ty, tach green. Note: `addopts` deselects integration tests, so run `pytest -m "integration or not integration"` before each commit.
- [x] Step 3b: the method residual and energy PNGs are no longer drawn in `record_method`; the report workflow draws them from the stored convergence table and registers them with `MLflowReportSink.register_method_figures` (uploaded under `method_figures/`). `MLflowBenchmarkMethodSink` needs no `notation`.
- [x] Step 4: `platform/reporting/stored_benchmark.py` (`StoredMethod`, `StoredBenchmark`, `read_stored_benchmark`) cross-validates each method's artifacts and merges dataset presentations; `platform/tracking/session_artifacts.py` (session -> FINISHED method children -> downloaded artifact directories, `MethodArtifactsMissingError`) is shared with the reuse gate. `pod2g_style_keys` moved beside `preconditioner_family`. The stored config holds content digests for dataset and checkpoint inputs, so the round-trip test asserts family and marker equality and that the colour key groups by fit-dataset content, not that it equals the compute-time path. Convergence is read eagerly per method; switching to `scan_parquet` is left to step 5 where the aggregation consumes it.
- [x] Steps 5 and 6: `compare` no longer reports; `neuralls report` (`composition/assignments/comparison_report.py::run_comparison_report`) reads the stored session through `read_stored_benchmark`, rebuilds typed events (`composition/comparison/postprocessing/report_inputs.py`, `decode_setup_frame`/`decode_solve_frame`), and feeds `report_all_levels` and `write_system_figures` (both in `composition/comparison/postprocessing/`). Output is a tree of nested runs in the `Reports` experiment (`names.report`): `ReportScope` (`domain/solver/models/report_scope.py`) with `.parent`, the `ReportSink` Protocol (`postprocessing/ports.py`: `report`, `report_system`, `finalize`) and `MLflowReportSink` (`platform/tracking/comparison_tracking.py`, also `register_method_figures`), which creates a scope's ancestors first. Metric keys are `{metric}/{method}`; each scope logs its identity as run params. Every figure has a same-stem card (`<stem>.toml`, group figures also `<stem>.parquet`) in the run of its scope, and `figure_index.toml` on the root run maps figures to runs, paths and cards (`platform/reporting/figure_card.py`, `figure_index.py`, `figure_table.py`). The root is gated by the source-session, notation-digest and schema-version tags and by a valid `figure_index.toml`; `run` chains `report` after `compare`. Architecture tests enforce the two import rules. The drift guard compares each method run's `method_config_digest` tag with the `config_digest` in its stored `summary.toml`.
- [ ] Remaining: Phase A2/A3 and Phases B to D of `docs/plans/report-split-completion.md`; per-system run count has not been measured.

Design decisions kept:

- **One definition owns all seven artifacts** (`MethodBenchmarkArtifacts`, including `method_config.toml` and `presentation.toml`); `convergence.parquet` is required for reuse.
- **`SystemProvenance`** (`domain/solver/models/system_provenance.py`) carries a system's tag, `SolverTimingContext` and `ReferenceSelection`; the sink takes per-system `solver_contexts` and `references` maps built from the prepared batch.
- **Reuse rejections are logged** at info level with the reason.
- **Per-step checklist:** run `uv run pytest -m "integration or not integration"`, `ruff`, `ty` and `tach`; the default `addopts` deselects integration tests.

## 1. Context and decision

Before the split, Phase B (`report_all_levels`, `register_method_figures`, `write_system_figures`, `_outcomes_from_workloads`) consumed in-memory `CompletedMethodWorkload`/`CGComparisonResult` and figures were drawn inside the timed method child run's `record_method` and inside the session run before it closed.

Decision: **two first-class workflows with the same mechanics as generate/train/compare.**
- `compare` (compute): timed solves, persists *raw, durable results* and primitive scalars. Draws nothing.
- `report` (postprocess): reads compare's stored artifacts, aggregates with polars, draws, logs. Never solves, never times. Can be re-run (new notation, new figure code) without re-solving.
They share no in-memory types. Contract between them = stored artifacts (section 4).

Rejected: keeping postprocessing inside `run_comparison_batch` and also exposing it standalone. Reason: two entry paths to one function means the in-run path keeps receiving in-memory objects and drifts from the from-disk path; the standalone path then rots. One path (read artifacts) is the only way to keep them equal.

## 2. Legacy chain to delete (and why)

Delete: `comparison_batch.py::{_execute_prepared_comparison, _run_comparison_from_config, _execute_comparison_in_run, _run_and_log_comparison, _run_comparison_with_resolved_specs, _build_child_runs}`; `composition/comparison/comparison_run.py::{compare_preconditioners, _run_preconditioner, _evaluate_preconditioner}`; `platform/reporting/artifacts.py::write_comparison_artifacts`; `platform/tracking/comparison_tracking.py::{MLflowCostRecorder, create_cost_recorder, log_comparison_result_metrics, log_linear_system_params, log_comparison_run_params}` and other per-system logging helpers no live path calls; `domain/solver/ports.py::CostRecorder`. Confirm each with `ast-grep run -p '<name>($$$)' -l python src tests` before deleting.

Why: it is the earlier one-MLflow-run-per-system design. Verified: `run_comparison_batch` (callers: `cli/compare.py:97`, `composition/assignments/case_pipeline.py:81`) composes `_prepare_comparison_entry` + method-major execution itself and never reaches the chain; only ~50 tests do. It is a second implementation of compute + logging + plotting. Leaving it means every postprocessing change must be mirrored or silently diverges, and its tests give false coverage. Deleting removes ~700 lines and the only consumers of the per-system plot code.
Live-path leftover: `_outcomes_from_workloads` re-resolves inputs and draws ~14 plots into a discarded `TemporaryDirectory` for a `ComparisonCompleted.payload` nobody consumes. Replace payload with `summary: str` (`format_results_summary`, text only).
Test porting table: keep the existing table in `docs/plans/compute-postprocess-split.md` step 1 (re-target preparation/cache/checkpoint tests at `_prepare_comparison_entry`/`_build_scheduled_and_bind`; delete tests that assert legacy payloads/logging; delete `test_comparison_artifact_persistence.py`).

## 3. MLflow layout (the end state)

Per-system compute results are **rows**, not runs: compare is method-major, one child per method covering all its systems (verified: `_execute_method_run`). Per-system compute runs would re-introduce the loop inversion the redesign removed.

Two experiments, two lifecycles, mirroring `Train` vs `Comparisons` (`ExperimentNamesConfig`, `names.training` / `names.comparison`; add `names.report`, default `"Reports"`):
- `Comparisons` = expensive, timed, immutable evidence (compute). Unchanged shape: session -> method children.
- `Reports` = derived, disposable, regenerable analyses. Hierarchy follows the **aggregation axis**, most aggregated at the top, least aggregated (per system) at the deepest level.

```
Comparisons                                   (compute; compare writes, report only reads)
└─ SESSION  phase=session_comparison
   ├─ METHOD phase=comparison_method   params: configured facts
   │     metrics (primitives): setup_total_s, solve_total_s, workload_online_s, n_planned_rhs, n_attempted_rhs,
   │             n_converged, success_rate, n_setup_failed, n_solve_failed, online_s_per_attempted_rhs,
   │             workload_peak_memory_bytes, generation_s, + iterations_total (new)
   │     artifacts method_workload/: setup_events / solve_attempts / convergence / planned_systems .parquet,
   │             summary.toml, method_config.toml, presentation.toml
   └─ METHOD ...

Reports                                       (derived; report writes)
└─ ROOT report_level=all_matrix_sets   "Report | <session title> | <timestamp> | notation <digest8>"
   │   tags: source_session_run_id, source_experiment, notation_digest, report_schema_version
   │   metrics : mixture-level aggregates, one key per metric and method:  <metric>/<method>
    │   artifacts: figure_index.toml, mixture figures with cards (`<stem>.png`, `<stem>.toml`, `<stem>.parquet`), method_figures/<method>/*.png
   └─ MATRIX SET   report_level=matrix_set        "<dataset presentation>"
      │   metrics + figures aggregated over the whole matrix set
      └─ RHS TYPE   report_level=matrix_set_rhs_type    "<dataset>, <RHS recipe>"
         │   metrics + figures aggregated over one (matrix set, RHS type) cohort (the primary analysis level)
         └─ SYSTEM   report_level=system          "matrix 3, RHS 7"
               metrics : per-method values for this one system (iterations/<method>, solve_s/<method>, ...)
                artifacts: residual_history.png, energy_history.png (all methods on this system) with cards
```

Concrete browse example (`Reports` experiment):
```
Reports
└─ Report | Gaussian CG Residuals (k=20) | 2026-10-30 14:35 | notation a1b2      [all_matrix_sets] FINISHED
   ├─ Gaussian CG Residuals (k=20)                                               [matrix_set]
   │  └─ Gaussian CG Residuals (k=20), RHS: random (seed 0)                      [matrix_set_rhs_type]
   │     ├─ matrix 0, RHS 0                                                      [system]  residual.png energy.png
   │     ├─ matrix 0, RHS 1                                                      [system]
   │     └─ ...
   └─ ...                                  (a second notation produces a second ROOT, same shape)
```

Why this shape (answers "does nesting by aggregation level make sense"): yes, and it is what the data already is. The three existing report levels (`MatrixSetRhsTypeLevel`, `MatrixSetLevel`, `AllMatrixSetsLevel`, `domain/solver/reporting/report_levels.py`) form a containment chain, and `ReportLevelName.SYSTEM` is the missing deepest rung (exists, unused). A run per level gives each level its own metrics and artifacts; today the group id is baked into metric names (`_group_metric_key`: `{metric}/{group}/{method}` flattened on one run), which nesting removes: the group is the run, the key is `{metric}/{method}`. Reading top-down in the UI goes mixture -> matrix set -> cohort -> single system. Method figures (one method across all systems) are not on the aggregation axis; they live as artifacts of the root run.

Why `Reports` is a separate experiment (revises the earlier "same experiment" idea): with one run per level and per system, a single report creates `1 + sets + (set,RHS) cohorts + systems` runs, and each re-render creates another full tree. Left in `Comparisons` that swamps the run list that holds the expensive, immutable evidence, and clutters MLflow's native compare view and every `search_runs` over the experiment. Separate experiments give an independent lifecycle: wipe/regenerate all reports without touching compute evidence (same reason training and comparison are separate). Trade-off accepted: the link to the session is by tag (`source_session_run_id`, `source_experiment`), not by MLflow nesting; MLflow only nests within one experiment.

Compute runs are never written to; report runs are additive. Re-rendering with a different notation creates a new ROOT tree (file names derive from rendered presentation, so reuse-in-place would leave stale figures); metrics logged twice under one key would form ambiguous history.

What compare logs vs report logs: compare = anything measured or known at that moment that is a scalar/primitive or a raw table. Report = anything that aggregates across methods/systems or is a picture.

## 4. Persistence contract (compare -> report)

Rule (one sentence, goes in `platform/README.md`): **parquet holds tabular, row-per-entity, numerical or categorical data that polars aggregates; TOML holds nested, per-run documents (identity, configuration, presentation); JSON is not used for new artifacts; nothing rendering-specific (colours, markers, labels) is persisted.**
Strings in parquet are fine when they are identifiers or closed categories (`system_id`, `dataset`, `recipe` machine key, `*_status`, `stopping_criterion`); the existing tables already do this. They are not fine for rendered text or free-form payload blobs.

| Artifact | Format | Content | Source in code |
|---|---|---|---|
| `setup_events.parquet`, `solve_attempts.parquet`, `convergence.parquet` | parquet | unchanged schemas (`event_frames.py`) | exist |
| `planned_systems.parquet` (new) | parquet | one row per planned system of the method: `system_id, dataset, recipe (machine key), matrix_id, rhs_ordinal`; solver context as typed columns `rtol, atol, max_iterations, stopping_criterion, m_max, reference_precision_margin, timing_repeats, timing_warmup, timing_statistic, normalization`; reference selection as `reference_kind` (`configured`/`none`/`unavailable`), `reference_method`, `reference_reason` | `SystemTag`, `SolverTimingContext` (`domain/solver/models/group_presentation.py`), `ReferenceSelection` (`domain/benchmarking/selection.py`) |
| `presentation.toml` (new) | TOML | configured method presentation + compact fact selection; realized facts per system; presentations of the datasets the method ran on; all through `platform/storage/facts_codec.py` encoders (types and keys only, no rendered text) | `PreparedMethodPresentation`, `ExecutedMethod`, `DatasetPresentation` |
| `method_config.toml`, `summary.toml` | TOML | unchanged content (reuse-identity contract); `method_config.toml` is now written by `write_method_benchmark_artifacts` | exist |

The artifact set (all seven files) is defined once by `platform/reporting/benchmark_events.py::MethodBenchmarkArtifacts`; the writer, the reuse gate and the `required_artifacts` tag derive from it. `planned_systems` rows are built from `SystemProvenance` (tag + `SolverTimingContext` + `ReferenceSelection`).

Why `planned_systems` is needed: `solve_attempts` only has *attempted* systems; censoring/classification needs planned systems whose setup failed (`n_planned_rhs`). Why solver context and reference selection are persisted (compute provenance): they are the conditions under which the numbers were produced and `ReferenceUnavailable` is only known at run time; re-deriving from a case config edited later would describe different numerics than were timed. They are numeric/categorical per system, so they fit parquet; the reference sum type is stored in its flat form and decoded in exactly one reader (reader validates "`reference_method` present iff kind != none, `reference_reason` present iff unavailable", then builds `ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable`).

Not persisted (derived at report time):
- Plot style (`family`, `color_key`, `marker_key`): pure function of the method config (`platform/config/models/preconditioner_family.py::preconditioner_family`, `_pod2g_style_keys`). Report parses `method_config.toml` through the existing preconditioner config model and applies the same function. Keeps rendering rules owned by the report; a changed style rule re-renders old runs correctly. Required test: `style(parse(dump(cfg))) == style(cfg)` for every config kind. Move `_pod2g_style_keys` out of `comparison_run.py` to sit beside `preconditioner_family`.
- `NotationPolicy`: from the case config at report time (cosmetic; the codec deliberately never persists it).
- Rendered names, slugs, figure file names, figure cards and the figure index.

Reader boundary: `platform/reporting/stored_benchmark.py` (read side) decodes parquet/TOML, validates (extend `event_frames.py` validators and `validate_setup_solve_frame_links`), and returns one typed immutable `StoredBenchmark` (frames + `tuple[StoredMethod, ...]` + per-system `SolverTimingContext`/`ReferenceSelection` + dataset presentations). Malformed data is rejected here; downstream operates on validated types only.

Open verification (do first, step 3): that `facts_codec` payloads are TOML-safe (no `None`, homogeneous where required). If any kind needs null, change the codec to omit the key.

## 5. How the workflows connect

`neuralls report <case_config.toml> [--force] [--session-run-id ID]`
- Needs the case config: yes. For (1) tracking topology via `_resolve_comparison_topology` (tracking URI, experiment) — same resolution as compare, so both always target the same store; (2) `NotationPolicy`; (3) settings. Same loading mechanic as other commands: `load_case_settings` -> one composition entry -> `fail_workflow`.
- Run ids needed from the user: none. Session lookup: newest FINISHED run in the experiment with `phase=session_comparison` and `case_config_path` equal to the config (both tags already stamped by `build_session_run_spec`). `--session-run-id` overrides for explicit pinning. Method children: `mlflow.parentRunId == session` and `phase=comparison_method`; require each to have the full artifact set, else error naming the missing run.
- Parquet location: downloaded from each method child's `method_workload/` artifact dir via `MlflowClient.download_artifacts` into a temp dir, then `pl.scan_parquet` + lazy concat. Same download mechanics as `find_reusable_benchmark_method`. A shared helper `platform/tracking/session_artifacts.py` (session -> method runs -> artifact paths) serves both reuse and report so run-discovery exists once.
- Output: reads the session from the `Comparisons` experiment, writes only to the `Reports` experiment (`ensure_experiment` as `setup_comparison_tracking` does; `names.report`). `MLflowReportSink` opens the ROOT run, then opens nested runs lazily top-down as scopes are first used (parents always before children). It logs all figures with their cards, level metrics, scope params, the figure index and tags. Compute runs are never written to.
- Drift guard: report recomputes nothing from the case config except notation/topology, so config drift cannot corrupt numerics. `read_stored_benchmark` asserts each method run's `method_config_digest` tag equals the `config_digest` in its stored `summary.toml`.
- Run count: ROOT + one run per matrix set + per (set, RHS type) + per system. Creation is roughly tens of ms per run on local SQLite; thousands of systems means about a minute and a long deepest list in the UI. Measure in step 6; if too heavy, make the SYSTEM level opt-in via `ReportParams.system_runs` (deferred, not built now).
- Chaining: `case_pipeline.py` calls `run_comparison_batch` then `run_comparison_report`; `neuralls compare` alone is compute-only; `neuralls report` alone needs a prior compare. New Typer command `report`; add to AGENTS.md command lists.
- Failure boundaries: compare keeps WARMUP, METHOD_WORKLOADS, RESULT_PROJECTION (+ method PERSISTENCE). `REPORTING`/`FINALIZATION` phases and a report-scoped failure sink move to the report workflow (recorded on the report run).

## 6. Reuse and idempotence

- Compare gate (`_has_complete_reusable_session`, moved to `platform/tracking/session_reuse.py`): all method children reusable under one FINISHED session parent; there is no figure requirement (figures are a report concern). `required_artifacts` tag and `_METHOD_ARTIFACTS` updated to the new set (drop `method_diagnostics.json`, add `planned_systems.parquet`, `presentation.toml`, `convergence.parquet`); replace `_validate_diagnostics` by validation of the new tables.
- Report gate: skip when a FINISHED ROOT report run in `Reports` exists with `source_session_run_id` equal to the session, the same `notation_digest` and `report_schema_version`, and a valid `figure_index.toml` (`validate_figure_index`); `--force` overrides. Changing notation therefore yields a new report tree automatically. A crash mid-report leaves the ROOT not FINISHED (failure boundary marks it FAILED), so it never satisfies the gate; stale failed trees are deleted by the user (disposable by design).
- Report is deterministic given stored artifacts + notation; re-running without `--force` is a no-op.

## 7. Module layout and layering (cli -> composition -> {application, platform, domain} -> shared)

- `cli/report.py` (thin; imports composition + shared only, per tach).
- `composition/assignments/comparison_report.py::run_comparison_report(case_config_path, params, settings) -> ComparisonReportOutcome` - shell: topology, session lookup, failure boundaries, sink wiring. Has no figure or aggregation logic.
- `composition/comparison/postprocessing/` (registered in `tach.toml`): `group_reporting.py` (`report_all_levels`), `system_figures.py`, `report_inputs.py` (stored-artifact inputs), `ports.py::ReportSink` (consumer-layer Protocol: `report` for group levels, `report_system`, `finalize`). `ReportScope` (`domain/solver/models/report_scope.py`) is a closed sum type (`AllMatrixSets | MatrixSet | MatrixSetRhsType | SystemScope`) with `.parent`; exhaustive `match`, no registry. The sink knows only scopes; it maps a scope to a nested run, so nothing else in `postprocessing` knows about MLflow runs.
- `platform/tracking/comparison_tracking.py::MLflowReportSink` implements `ReportSink` and also exposes `register_method_figures`; it owns the scope -> nested-run mapping, is the sole allocator of figure slugs/file names, and writes the cards, tables and `figure_index.toml` (`platform/reporting/figure_card.py`, `figure_index.py`, `figure_table.py`, `figure_metadata.py`). Moving it to its own module is Phase B1 of `docs/plans/report-split-completion.md`.
- `platform/reporting/benchmark_events.py`: write side (`write_planned_systems`, `write_presentation`) used by compare; `platform/reporting/stored_benchmark.py::read_stored_benchmark`: read side used by report; both share schemas with `event_frames.py`.
- `platform/tracking/session_artifacts.py` (session -> method runs -> artifact directories) serves both reuse and report.
- Dependency direction: compare modules never import `postprocessing`; `postprocessing` never imports compute-time types (`CompletedMethodWorkload`, `CGComparisonResult`). Add an architecture test enforcing both.
- Plotting inside `BenchmarkMethodSink`/`record_method` and the `MethodWorkload`-mixed-with-sinks shape of `domain/solver/ports.py` are removed or split (Protocols to consuming layers); `GroupReportSink` and `MLflowGroupReportSink` no longer exist.

## 8. Steps (stop and report after each; each step green on pytest, ty, ruff, tach, and committed on its own with scoped conventional commit + bullet body, no AI mention)

1. Delete legacy chain; port tests (section 2).
2. `ComparisonCompleted.payload` -> `summary: str`; drop temp-dir plotting and input re-resolution in `_outcomes_from_workloads`.
3. Compare persistence: `planned_systems.parquet`, `presentation.toml` (verify codec TOML-safety), `iterations_total`; delete `realized_facts.json` and `method_diagnostics.json` writers; update `required_artifacts`, `_METHOD_ARTIFACTS`, validators, reuse gate. Read-side + round-trip tests (tests/platform). (Done as 3a; removing figure drawing from `record_method` moved to step 5.)
4. `read_stored_benchmark` + `session_artifacts.py`; style-derivation move + parse/dump round-trip test; `StoredBenchmark` types.
5. Remove `_write_method_figures` from `record_method` together with the port below. `postprocessing/` port onto `StoredBenchmark`: levels, per-system figures (registered with `ReportLevelName.SYSTEM`), method figures (root-run artifacts under `method_figures/<slug>`), the deepest `System` scope (per-system metrics + figures), `ReportSink`, `MLflowReportSink`. Remove from `run_comparison_batch` all reporting/finalization.
6. `comparison_report.py`, `cli/report.py`, report gate, `case_pipeline.py` chaining, failure sink for report, architecture tests, AGENTS.md/READMEs/tach/`docs/coupling_graph.dot`.
7. Final doc pass: remove stale text from READMEs/AGENTS.md (no deprecation notes). Deferred (not in scope): by-matrix level, dataset display-name wording, `execution/` package move, converting `method_config.toml`/`summary.toml` to another format.

## 9. Testing (per project rules)

- Fixtures in `conftest.py`, seeded; `tmp_path` only; tests isolated from repo configs.
- Pure aggregation/figure-plan logic on in-memory frames: `tests/domain`, `tests/composition` (no IO, no `deferred_io`).
- Parquet/TOML write+read round trips: `tests/platform/` (file is the subject). Real MLflow: `tests/integration` with the `mlflow_tracking_uri` fixture (one compare-then-report E2E on tiny data; assert run tree, tags, artifact names, metric keys, not pixels). Compare-batch tests mocking MLflow reuse `tests/workflows/comparison_support.py`.
- Never test matplotlib images or specific config values; assert structured objects, not substrings.
- Add: `test_timed_regions_do_no_io` stays green (compare writes only after workload ends); new architecture tests for the two import rules above; text-IO encoding lint (`encoding="utf-8"` on every new text read/write).

## 10. Risks

- Metrics move from session run to report run (breaking; documented in READMEs, no aliases).
- Reports need compare to have run; error message must say so.
- `convergence` volume: use `scan_parquet`/lazy aggregation, never hold histories as Python objects.
- TOML null-safety of the codec; reference-selection flat encoding (single decode point).
- Nested-run creation order and failure marking in MLflow: verify on a tiny real store in step 6.
- Report-time colour keys are content digests, not paths: the stored method config holds a digest for dataset and checkpoint inputs, so `pod2g_style_keys` groups by fit-dataset content and the key is not human-readable. Legends and labels must come from `presentation.toml`, never from the key. If `canonical()` digests a dataset directory sensitively to file order or metadata, identical datasets would split into separate colours; verify before step 5 relies on it.
- The reader lives in `platform/reporting/stored_benchmark.py`; the write side stays in `benchmark_events.py`.

## 11. Compliance review (against the engineering rules and AGENTS.md)

| Rule | Finding | Resolution in plan |
|---|---|---|
| Domain state as sum types, no nullable-by-state fields | `planned_systems` flat `reference_*` columns are a nullable encoding | Confined to storage form; one reader validates then constructs `ConfiguredReference \| NoReferenceConfigured \| ReferenceUnavailable`; documented (section 4) |
| Boundary data validated before core logic | Report reads downloaded artifacts | Single `read_stored_benchmark` boundary returning `StoredBenchmark`; downstream trusts it |
| `@dataclass(slots=True)`, frozen value types | New types `StoredBenchmark`, `StoredMethod`, `ComparisonReportOutcome` | Declared `frozen=True, slots=True` |
| Distinct domain types for ids | session/method run ids, report ids | Use `NewType`/existing `SystemId`, `DatasetId`, `PreconditionerName`; add `SessionRunId` rather than raw `str` |
| Exhaustive closed-set handling | reference kind decode, report levels | `match` with `assert_never`; no registry for levels/figure kinds |
| Single source of truth | style rules, run discovery, artifact name set | Style derived once from config; discovery in `session_artifacts.py`; artifact names from one `MethodBenchmarkArtifacts` definition used by writer, reuse gate, reader; `required_artifacts` tag derived from it |
| No duplicated logic | three duplicated `resolve_comparison_input` blocks | Not touched here; out of scope, noted |
| Named constants | artifact dir names, phase tags, schema version | Constants in `shared/constants.py` / module constants (`REPORT_PHASE`, `REPORT_SCHEMA_VERSION`) |
| Config never hardcoded; MLflow URI resolution | report topology | Reuses `_resolve_comparison_topology` (env > case config > default SQLite) |
| New orchestration = Typer command, not script | report | `cli/report.py` + AGENTS.md command list |
| Metrics/artifacts through MLflow path | report outputs | Via `MLflowReportSink`; diagnostic logs stay separate (loguru) |
| No import-in-function decoupling | `comparison_tracking.py` has function-local imports (`tomli_w`, `MeasuredPeakMemory`) in code we touch | Hoist to module level in files touched in step 3 |
| No assert-based narrowing; match/case | readers | `match` on kind; no `assert x is not None` |
| No planning-doc refs / plan codes in code or commits | steps named "step N" | Code/comments/commits use descriptive names; plan numbering only in docs |
| No theoretical analysis in src | none added | n/a |
| No substring asserts, no image tests, no config-content tests | testing section | Explicit in section 9 |
| IO only in io/integration/platform/architecture test dirs; `deferred_io` markers | new tests | Placement per section 9; drop stale markers on files that stop doing IO |
| Docs updated per change, in place, no deprecation notes | READMEs, AGENTS.md | Per step 6/7: update module `.md` files and AGENTS.md in the same commit as code |
| Layer rule, tach `layers_explicit_depends_on` | new packages | Register `composition.comparison.postprocessing` and new platform modules in `tach.toml`; `cli/report.py` imports composition/shared only |
| Domain modeling of report hierarchy | levels as ad-hoc strings/ids would be a stringly-typed hierarchy | `ReportScope` closed sum type with `.parent`; exhaustive `match` |
| Config values not hardcoded | new experiment name | `names.report` in `ExperimentNamesConfig` with default, resolved like `names.comparison` |
| Pause at milestones | | Step list ends each step with stop-and-report |

Fixes made during review: added `SessionRunId` type; made artifact-name single-source; hoisted function-local imports in touched files; moved `_pod2g_style_keys` beside `preconditioner_family`; stated the commit-footer conflict resolution; made architecture tests explicit; separated failure phases between workflows.

## 12. Evaluation against the current state: what we gain, what we lose, is it worth it

| Aspect | Today | After | Verdict |
|---|---|---|---|
| Failure coupling | Figure/aggregation code runs inside the open session run. A plotting or reporting exception leaves the session not FINISHED with no figures, so `_has_complete_reusable_session` is false and **the next `compare` re-solves everything** | Compute session is FINISHED before any figure code runs; a report failure only fails a report run; re-run `report` alone | **Biggest win**: hours of solves no longer hostage to a plotting bug |
| Iterating on figures/notation/aggregation | Requires re-running solves (or `--force` surgery) | `neuralls report` only; seconds to minutes; can run on a CPU machine | **Big win** for research workflow |
| Responsibilities of `comparison_batch.py` | 1968 lines, ~8 reasons to change (config, runs, naming, reuse, schedule, execution, projection, reporting) | Compute-only shell; reporting, figure sinks, system convergence gone; legacy chain gone; target well under 1000 lines | Clear win |
| Code paths for reports | In-memory objects only; `read_convergence` unused outside tests; legacy chain duplicates logging | One path: stored artifacts -> `StoredBenchmark` -> `build_report` | Win (single path, no drift) |
| Testability | Reporting tests need `CompletedMethodWorkload`/`CGComparisonResult` builders | Reporting tests build small polars frames; reader tested by round trip | Win |
| Timing hygiene | Figure drawing and artifact upload happen in the same process/run around timed work (outside timers, enforced by lint) | Compare process never imports plotting; no GPU memory or allocator interaction with plotting | Modest win |
| Number of concepts | 1 command, 2 run phases, 3 sink protocols, JSON + TOML + parquet | +1 command (`report`), +1 run phase, 1 sink protocol, TOML + parquet only (JSON gone) | Net neutral: 2 additions, 3 removals |
| Lines of code | ~1968 + legacy ~700 in play | Delete ~700 (legacy) and ~400 (reporting wiring in batch/sinks); add ~600 (reader, `planned_systems`/`presentation` writers, report workflow, `MLflowReportSink`, CLI) and move (not add) ~700 (`group_reporting`, `system_figures`, plotting) | Net roughly -500; not the point, the coupling is |
| MLflow layout | Session -> method children; all report metrics/figures flattened on the session with group id baked into metric keys | `Comparisons`: unchanged shape, compute only. `Reports`: tree by aggregation level (all sets -> set -> cohort -> system); metric keys `{metric}/{method}` | Cleaner browsing and provenance; more runs (see next row) |
| Run count | 1 + methods | Comparisons same; Reports adds 1 + sets + cohorts + systems per report tree | **Loss**: can reach thousands for large sweeps; isolated in its own experiment, system level can be made opt-in if measured heavy |
| Operational steps | `compare` does everything | `compare` then `report` (pipeline command runs both; standalone `compare` yields no figures) | **Loss**: one more thing to remember; mitigated by `case_pipeline` chaining and a clear message from `compare` ("run `neuralls report`") |
| Old sessions | Reusable | Sessions produced before this change lack `planned_systems.parquet`/`presentation.toml`: not reusable, not reportable; re-run `compare` | **Loss** (one-time) |
| Storage/IO | Merged convergence uploaded once | Report downloads N small method artifact sets; presentation duplicated per method (small) | Minor loss; sweeps with hundreds of methods mean hundreds of small downloads (acceptable; batch if measured slow) |
| Metric keys | On session run, `{metric}/{group}/{method}` | On the level's own report run, `{metric}/{method}` | Breaking, allowed by you, no aliases |
| Risk of drift between what was timed and what is reported | Config read at compute time | Stored provenance (solver context, reference, realized facts) read back; only notation from live config | Neutral to win |

Verdict: **worth it.** It does not make the code dramatically smaller; it removes the structural problem: a result-producing step (expensive, timed, GPU, must not fail late) is currently fused to a presentation step (cheap, iterated constantly, bug-prone). The measurable payoffs are (1) a report bug can no longer invalidate a completed compute session, (2) figures/notation iterate without re-solving, (3) one verified data path replaces two. Costs are a second command, a one-time loss of old sessions, and a moderate amount of persistence plumbing (two small artifacts plus a validated reader). The scope is deliberately limited to keep that plumbing minimal: no new registry, no by-matrix level, no execution-package move, no change to the reuse-identity TOMLs.

If you want a cheaper variant: skip `presentation.toml` by re-deriving presentations from the case config at report time. Not recommended: realized facts (known only after setup) cannot be re-derived, and a config edited after the run would silently relabel the numbers.
