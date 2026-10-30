# Plan: finish the nested-report split, then fix SOLID violations in priority order

First implementation step: copy this file to `docs/plans/report-split-completion.md` (project rule: plans live in the project) and replace the stale progress text in `docs/plans/compute-postprocess-split.md` with a pointer. Branch `refactor/comparison-report-split`. Pause and report after each step; each step green on `uv run pytest -m "integration or not integration"`, `ruff check src tests`, `ty check src/`, `tach check`; one scoped conventional commit per step, bullet body, no AI mention in the message.

## Progress

- [x] A1 drift guard (run `method_config_digest` tag vs `summary.toml` `config_digest`) and report gate validating `figure_index.toml`
- [ ] A2, A3, A4 (docs, tach, coupling graph in progress), then Phases B to D

## Context

`docs/plans/compute-postprocess-split.md` split `compare` (compute) from `report` (postprocess, nested run tree in the `Reports` experiment). Commit 142d7c1 implemented the nested tree, `neuralls report`, figure cards and `figure_index.toml`, and `docs/plans/figure-provenance.md` is complete. The mechanism works end to end. What is left is (a) a few unimplemented plan items and stale plan/doc text, and (b) structural debt. The SOLID review shows the debt concentrates in two god modules that the split made worse: `platform/tracking/comparison_tracking.py` (1001 lines, `MLflowReportSink` inside) and `composition/assignments/comparison_batch.py` (1598 lines).

## Status of the original plan (verified against code)

Done: `neuralls report` command and flags, `case_pipeline` chaining, report gate on source session + notation digest + schema version, `--force`, newest-FINISHED-session lookup, report-scoped failure boundaries (REPORTING/FINALIZATION only in report), `names.report`, frozen/slots on new types, method figures drawn from stored frames, `GroupReportSink` removed, `_write_method_figures` removed, architecture test for the compute/postprocessing import rules, AGENTS.md lists `report`.

Not done or deviating:

| Item | State |
|---|---|
| `ComparisonTopology.report_experiment_name`, `REPORT_PHASE` constant, `SessionRunId` type | missing (workflow reads `names.report`; `"phase": "report"` literal; raw `str` ids) |
| Report workflow reaches private helpers of `comparison_batch` (`_failure_boundary`, `_load_master_config`, `_resolve_comparison_topology`) | layering smell |
| `comparison_batch.py` under 1000 lines | 1598 |
| `MLflowReportSink` in its own module | still inside `comparison_tracking.py` |
| `docs/coupling_graph.dot` refresh | missing |
| `cli/README.md` has no `report`; `platform/README.md` ~116/147 and `composition/comparison/README.md` mention old shapes | stale |
| Plan text: `figure_manifest.toml` (sections 3, 5, 6, 8, "Remaining"), `MLflowGroupReportSink` reuse, step 3b notes, sink API names, file locations | stale after 142d7c1 |
| Per-system run count measurement / opt-in `system_runs` | not measured; deferred by design |
| Failure marking of a crashed report run is implicit and untested | add one test |

## Phase A: close the split (small, no redesign)

A1. **Drift guard and gate.** In `read_stored_benchmark` (`platform/reporting/stored_benchmark.py`) compare each method run's `method_config_digest` tag with the `config_digest` in its stored `summary.toml`; mismatch raises the existing stored-benchmark error type. In the report gate (`composition/assignments/comparison_report.py` `_finished_report`) additionally require `read_figure_index` + `validate_figure_index` to pass on the root run; a FINISHED root with an invalid index is not reusable. Reuse `validate_figure_index` (`platform/reporting/figure_index.py`). Tests: stored-benchmark drift case in `tests/platform/test_stored_benchmark.py`; gate rejects invalid index (fixture, no real MLflow if the gate takes a lookup port, else `tests/integration`).

A2. **Names and types.** Add `REPORT_PHASE` beside the other phase constants and use it for tag write and gate query; add `SessionRunId` (`NewType`) in `shared/types.py` and use it in `session_artifacts.py`, `StoredBenchmark`, `ComparisonReportOutcome`; add `report_experiment_name` to the topology type and resolve it in `_resolve_comparison_topology` (so report and compare resolve the store from one place).

A3. **Failure marking test.** One test: an exception inside the report body leaves the root run FAILED and `_finished_report` does not match it (`tests/integration`, `mlflow_tracking_uri` fixture).

A4. **Docs.** Edit in place (no deprecation notes): `docs/plans/compute-postprocess-split.md` (manifest -> cards + index, sink API names, file locations, remove step 3b/decision bullets that are done, mark complete), `cli/README.md` (add `report`, fix `compare` line), `platform/README.md`, `composition/comparison/README.md`, AGENTS.md if any line still says compare reports; regenerate `docs/coupling_graph.dot` (`dot -Tsvg` render check). Register `neuralls.composition.comparison.postprocessing` explicitly in `tach.toml`.

## Phase B: SOLID fixes on the report path (high severity, directly touched by the split)

B1. **Split `comparison_tracking.py` (SRP).** Pure moves first, no behaviour change, tests unchanged:
- `platform/tracking/report_sink.py`: `MLflowReportSink` only (run tree, upload, finalize).
- `platform/tracking/method_sink.py`: `MLflowBenchmarkMethodSink`.
- `platform/tracking/metrics_logging.py`: metric-key helpers (`parent_run_metric_key`, scalar logging) and `log_configured_method_facts`.
- `platform/reporting/figure_planning.py`: `_FigureSpec`, `_figure_plans`, `_planned_figure_specs`, `_figure_metadata`.
- `platform/reporting/group_plots.py` and `method_figures.py`: `_write_*_plots`, `_GroupStyler`, `_draw_method_figure`, `_system_legend`.
Drop `notation`/`datasets` from the method sink constructor if unused after the move.

B2. **Figure-kind registry (OCP, DRY).** One frozen `FigureRecipe` per `FigureKind` (stem, scenario, title, metric definition, failure treatment, draw callable) in `figure_planning.py`; `make_plan`, `_figure_metadata`, `_planned_figure_specs` and the group plot dispatch iterate it. This removes the five parallel per-kind switches, the `.get(kind, kind.value)` fallback, `next(...)` lookups that raise bare `StopIteration`, the duplicated profile/mixture bar block, and the non-exhaustive `case _: raise` in method figures. Registry keys are a closed enum, so a test asserts every `FigureKind` has a recipe.

B3. **Decompose `MLflowReportSink` (SRP, DIP).** Extract `ScopeRunTree` (scope -> nested run id, creation order, params via `scope_params`) and `FigureUploader` (staging, per-run stem collisions, card/table upload, uploaded-record list, index write). Both take narrow Protocols defined in `platform/tracking` (`RunTree.child_run`, `ArtifactSink.log`) with the MLflow adapter beside them; the sink becomes a coordinator and one `MlflowClient` is built once, not per child run. Replace `_finalized: bool` by two types (`OpenReport` -> `finalize()` -> `FinalizedReport`) and the `table: DataFrame | None` argument by `StagedFigure | StagedTabledFigure` (table present exactly when the card names one). Keep the `ReportSink` Protocol in `postprocessing/ports.py` unchanged.

B4. **Typed figure metadata.** Split `FigureMetadata` into `GroupFigureMetadata | MethodFigureMetadata` (no empty-tuple "not applicable" defaults); `timing_policy` / `aggregation_rule` become enums; `FigureSource.run_id` -> `RunId`, `dataset_ids` -> `tuple[DatasetId, ...]`. Card encode/decode updated; bump `REPORT_SCHEMA_VERSION` to `"3"` only if the card layout changes.

B5. **Report workflow shell.** Move `_failure_boundary`, `_load_master_config`, `_resolve_comparison_topology` out of `comparison_batch.py` into `composition/assignments/comparison_topology.py` (public names, shared by compare and report). Replace direct `mlflow.start_run` / `log_params` in `run_comparison_report` with a `ReportRunSession` in `platform/tracking`; build the notation digest with `shared.digest.canonical_digest`, not `sha256(repr(notation))`. `ComparisonReportOutcome.reused: bool` -> `ReportReused | ReportCreated`.

## Phase C: split `comparison_batch.py` (high severity, same layer)

Mechanical, one commit per extracted module, `run_comparison_batch` stays the thin orchestrator (<150 lines):
`comparison_topology.py` (from B5), `comparison_resolution.py` (spec/checkpoint resolution, ~lines 176-620), `comparison_prepare.py` (`_prepare_comparison_entry`, ~636-990, itself 131 lines: split by concern), `comparison_schedule.py` (~1007-1260), `comparison_execute.py` (~1269-1535, failure boundaries). Replace the four `_batch_*` `isinstance` comprehensions by one TypeGuard partition. Replace the non-exhaustive `case _:` in the job-kind dispatch (`comparison_batch.py` ~419-438) by explicit cases + `assert_never`. Update `tests/architecture/test_report_compute_separation.py` path allow-lists.

## Phase D: remaining SOLID backlog (do only after A-C; each independent, none blocks the report work)

Ranked; take as separate small commits when touching the area:
1. DIP: `TrackingServices` factory Protocol (identity store, lease manager, run lookup) injected from the CLI instead of inline `MlflowClient(...)` in `comparison_batch`, `evaluation`, `training`, `training_batch`, `_training_artifacts`; one `reuse_lookup` helper replaces the four repeated gate blocks.
2. `_execute_matrix_bundle` (191 lines, 13 params, 4 out-parameter lists): return a `MatrixBundleResult`, extract `_record_setup_failure`.
3. Domain/shared I/O: inject readers in `domain/generation/providers.py` and `archive_files.py` (pattern already used by `file_sources.py`); move `toml_digest`/`_file_digest` from `shared/digest.py` to `platform/storage`.
4. Config sum types: `LoggedModelRefConfig` / `RegisteredModelRefConfig` (`latest: bool | None` + alternative ids) -> `ByRunId | LatestMatching`, `ByAlias | ByVersion | Latest`; `provide_rhs_option: bool | str | Path | None` -> `NoRhs | RhsFromFile | RhsFromSolutions`; `AssignmentResult` success/failure variants instead of nullable `error`/`mlflow_run_id`.
5. Split `platform/config/models/preconditioner.py` (1434) into checkpoint refs, coarsening, weighting, slugs; per-config `slug()` instead of match chains.
6. Hoist function-local imports (`assignments/inference.py`, `_weighting.py`, mlflow client repeats); break config import cycles with a leaf module.
7. One `slugify` in `shared` replacing the five regex copies; `ForcePolicy` instead of threaded `force_*` bools; shared Typer option aliases in `cli/options.py`.
8. Move the event-frame codec out of `domain/solver/reporting/event_frames.py`; `ConvergenceSeries` dataclass replacing `Any`/`Mapping` in `platform/reporting/plots.py`.
9. Measure per-system run creation on a large store; add opt-in system level only if slow.

Not planned: training/evaluation skeleton unification (finding 5 of the composition review) - it is a large redesign with no second concrete requirement yet (YAGNI).

## Reuse

`validate_figure_index`, `read_figure_index` (`platform/reporting/figure_index.py`); `scope_params` (`platform/tracking/scope_params.py`); `download_session_methods`, `latest_session_run_id` (`platform/tracking/session_artifacts.py`); `canonical_digest` (`shared/digest.py`); `ReportScope`/`ReportSink` (`domain/solver/models/report_scope.py`, `postprocessing/ports.py`); `FigureSlugs`, `FigurePlan` (`platform/reporting/figure_metadata.py`); fixtures in `tests/platform/conftest.py`, `mlflow_tracking_uri`.

## Verification

Per step: full pytest incl. integration, ruff, ty, tach (project commands). B1 and C are pure moves: the existing suite must pass unmodified apart from import paths. End of Phase A/B: run `neuralls compare` then `neuralls report` on a tiny case; confirm re-running `report` is a no-op, `--force` makes a second tree, a tampered `figure_index.toml` or `method_config.toml` is rejected, and one figure card resolves from `figure_index.toml`. Tests follow project rules: fixtures, `tmp_path`, structured asserts, no image or substring asserts, IO only in io/integration/platform dirs.

## Out of scope

By-matrix level, `execution/` package move, converting `method_config.toml`/`summary.toml` format, caption generation, training/evaluation unification.


## Status (after commit 9f8c4e3)

Done: A1-A4, B1-B5, C, and Phase D items 1, 2, 4, 5, 6, 7 (ForcePolicy, shared CLI options) and the `MatrixReader` dedupe from item 3. Full suite, ruff, ty (src, tests, scripts) and tach green.

## Open todos (workflow works without them)

- Domain file I/O: inject readers for `np.loadtxt` in `domain/generation/providers.py` and the `.npy` header read in `archive_files.py`. Needs a `readers` parameter on about 20 strategy `generate()` methods and the `interfaces.py` protocols, plus `runner.run_generation`'s caller.
- `shared/digest.py`: `toml_digest`/`_file_digest` do file I/O. `canonical` uses them for `InputData`/`InputConfig` markers, so moving them to `platform/storage` needs a digester passed into `canonical` (or a context variable like `digest_memo`) to keep `shared` from importing `platform`.
- One `slugify` in `shared` replacing the regex copies (`figure_metadata`, `platform/storage/filesystem.py`, `platform/tracking/mlflow.py`).
- Move the event-frame codec out of `domain/solver/reporting/event_frames.py`; `ConvergenceSeries` dataclass replacing `Any`/`Mapping` in `platform/reporting/plots.py`.
- Measure per-system run creation on a large store; add an opt-in system level only if slow.
