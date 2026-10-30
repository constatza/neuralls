# Plan: every report figure carries its own parameters (figure cards), with clear artifact / provenance / index roles

First implementation step: copy this file to `docs/plans/figure-provenance.md` (project rule: plans live in the project). Branch `refactor/comparison-report-split`. Pause and report after each step; each step green on `uv run pytest -m "integration or not integration"`, ruff, ty, tach; one scoped conventional commit per step, bullet body, no AI mention.

## Context

After the nested report tree, figures live in many runs (root, matrix set, cohort, system) but the only record of their parameters is one root `figure_manifest.toml` keyed by filename. Problems: system figures are absent from it; an entry cannot be located (no run, no path); parameters sit far from the figure and must be mapped back by filename; sources are pseudo-paths; nothing proves every figure has a record. Goal: the hyperparameters that produced a figure, and that its filename cannot or should not encode (rtol 1e-6, max iterations, which methods, failure treatment), are next to the figure, unambiguously attributed to it, readable by humans and agents without any join.

## Roles (no overlap: each fact has exactly one home)

| Role | What it is | Where it lives | Holds |
|---|---|---|---|
| **Artifact** | the figure and the files that describe it | the report run of the figure's scope, same directory, same stem: `<stem>.png`, `<stem>.toml` (figure card), optional `<stem>.parquet` (compact plotted table) | the figure; its parameters; its small plotted numbers |
| **Provenance** | where the numbers came from | `[source]` table in every card, plus tags on every report run (`source_session_run_id`, `source_experiment`, `notation_digest`, `report_schema_version`) | session run id, method child run ids, method config digests; nothing about the figure's own parameters |
| **Index** | a derived directory of the figures | `figure_index.toml` on the root run (replaces `figure_manifest.toml`) | per figure only: `run_id`, `artifact_path`, `card`, `kind`, `level`, scope display. No parameters. Generated from the cards, never hand-edited; exists so an agent can list every figure with one read instead of walking the run tree |
| **Run params** (optional) | scope identity of a run | MLflow params on each scope run | `report_level`, dataset id, recipe key, matrix id, RHS ordinal, `n_matrices`, `n_planned_rhs`: facts constant over every figure of that run |

Attribution rule that prevents conflicts: a fact lives at the narrowest place where it is constant. Constant over a whole run (scope identity) can be a run param; varying between figures of the same run (kind, scenario, failure treatment, metric definition) lives only in that figure's card. Solver settings (`rtol`, `atol`, `max_iterations`, `m_max`, timing policy, normalization) can differ between systems pooled in one mixture scope, so they live in cards, not run params. A card never repeats what only the index or provenance owns, and the index never repeats card parameters.

## Figure card (`<stem>.toml`)

One table per figure, written next to the PNG in the same artifact directory:

- `figure`: file name of the PNG it describes (checked equal to the sibling's name)
- `kind`, `scenario`, `title`, `metric_definition`, `level`, `failure_treatment`, `aggregation_rule`, `timing_policy`
- `scope`: encoded scope (`encode_figure_scope`) plus rendered `display`
- `methods`: array of tables, one per method drawn: `method_id`, encoded prepared presentation, `config_digest`
- `counts`: `n_matrices`, `n_planned_rhs`, recipe counts and weights
- `reference`: selection kind, reference method id, reason
- `parameters`: the distinguishing hyperparameters, one table per solver context (rtol, atol, max_iterations, stopping_criterion, m_max, reference_precision_margin, timing repeats/warmup/statistic, normalization)
- `source`: session run id, and the method child runs (run id plus the artifact paths of the tables used)

Cards are TOML (nested per-run document); the plotted-table sidecar is parquet. Neither stores colours, markers or other rendering choices. Convergence histories and any other heavy series stay only in the compare artifacts and are referenced through `source`.

## Design

1. **`SystemScope`** as a fourth `FigureScope` variant (dataset, recipe, matrix id, RHS ordinal); extend `render_figure_scope`, `encode_figure_scope`, `decode_figure_scope` with exhaustive `match`. System identity is a scope, not optional fields on the metadata.
2. **`FigureMethod`** (method id, presentation, config digest) and `FigureMetadata.methods`; **`FigureSource`** (run id, artifact paths) replaces the string `sources`. `StoredMethod` gains `run_id` (set in `read_stored_benchmark` from `download_session_methods`, which already iterates the runs).
3. **`FigureCard`**: the immutable, validated value written to `<stem>.toml`, built from `FigureMetadata` plus `FigureSource`; `write_figure_card` / `read_figure_card` with `validate_figure_card` (the current `validate_figure_manifest` generalised to one entry). Single source of the card's fields: `FigureMetadata` encodes to the card and the card decodes back; the index is derived by projecting cards.
4. **Sink**: `MLflowReportSink` uploads PNG and card (and parquet when present) together in one call per figure, into the figure's scope run. File stems only need to be unique inside one run directory, so system figures keep short stems (`residual_history.png`) and the global filename collision rule (`FigureManifest._collision_key`) becomes a per-run rule. The sink records `(run_id, artifact_path, card)` per figure and writes `figure_index.toml` once at the root in `finalize`.
5. **Completeness invariant** before `finalize`: every uploaded PNG has a same-stem card whose `figure` equals it, and the index lists exactly the uploaded figures. Violations raise (a missing card is a bug).
6. **Optional run params**: log scope identity params when entering a scope run (create path in `_scope_run_id`).
7. **Compact plotted table** (`<stem>.parquet`, last step, group figures only): one row per method with the numbers the figure shows (e.g. converged count, planned count, workload seconds); no histories, no per-system series. Schema defined once beside `FigureKind`.
8. Bump `REPORT_SCHEMA_VERSION` to `"2"` (`composition/assignments/comparison_report.py`) so reports with the old manifest are not reused. Reuse gate validates `figure_index.toml`.

## Reuse

`FigurePlan`, `make_plan`, `FigureSlugs`, `encode_figure_scope`, `encode_method_presentation` (`platform/reporting/figure_metadata.py`, `platform/storage/facts_codec.py`); `MLflowReportSink._scope_run_id/_in_scope/register_method_figures` (`platform/tracking/comparison_tracking.py`); `download_session_methods` (`platform/tracking/session_artifacts.py`); `write_system_figures`, `SystemFigure` (`composition/comparison/postprocessing/system_figures.py`); `ReportSink` Protocol (`composition/comparison/postprocessing/ports.py`).

## Status

Steps 1 to 6 implemented: `SystemScope`; `FigureCard`/`FigureMethod`/`FigureSource` and `figure_index.toml` (the manifest is gone); cards, index and completeness check in `MLflowReportSink`; scope identity run params (`platform/tracking/scope_params.py`); group-figure parquet tables (`platform/reporting/figure_table.py`); `REPORT_SCHEMA_VERSION` is `"2"`.

## Steps

1. **Scope and codec**: `SystemScope`, renderer, encode/decode, exhaustive tests.
2. **Card types**: `FigureMethod`, `FigureSource`, `FigureCard` read/write/validate, `StoredMethod.run_id`; round-trip test (encode, validate, decode).
3. **Sink writes cards beside figures** (group, method and system figures all register); per-run stems; completeness invariant; `figure_index.toml` replaces the manifest; schema version bump; update reuse validation.
4. **Run params** for scope identity (optional, small).
5. **Compact plotted table** for group figures.
6. **Docs**: platform README (roles table, card schema), composition README, AGENTS.md note on the three roles; mark plan done.

## Tests (project rules: fixtures, `tmp_path`, structured asserts, no image or substring asserts)

- Unit: card for each of the four scope kinds contains scope, methods, parameters and source run ids; round trips through TOML.
- Unit: completeness invariant raises for a figure without a card, a card naming another file, and an index entry without a figure.
- Integration (`tests/integration/test_mlflow_artifact_locations.py`, real SQLite store, `figure_placeholders`): walk the report run tree; every PNG has a same-directory card with matching `figure` and the index entry's `run_id`/`artifact_path` resolves to it; every `source` run id is a method child of the session; two systems each have a card with different coordinates but the same short stem.
- Update existing manifest and sweep tests for the index and cards.

## Verification

Full pytest including integration, `ruff check src tests`, `ty check src/`, `tach check`. Then run `neuralls compare` and `neuralls report` on a tiny case: open one cohort figure's card and one system figure's card straight from their own runs, confirm the parameters answer "what rtol, which methods, which matrix/RHS", and confirm `figure_index.toml` resolves every figure.

## Out of scope

Caption generation, rendering choices in cards, convergence histories or per-system series as data, by-matrix level.
