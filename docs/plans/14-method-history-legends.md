# Per-method history legends

## Goal

Make residual and energy history legends describe each distinct comparison matrix set and RHS recipe once. Keep every system curve and its stored per-system identity. No legend entry repeats a full comparison scope or names a matrix/RHS curve that cannot be reliably matched by its style.

## Scope and design

`draw_method_figure` in `platform/reporting/method_figures.py` draws a method over all its planned systems. `MethodScope` guarantees a shared method, but neither a shared dataset nor a shared matrix: `platform/tracking/report_sink.py::_register_method` registers these figures at the root with every dataset and recipe of the stored method. Per-system figures are scoped separately. Group the systems that have a history for the selected method figure by the typed `(DatasetId, RecipeKey)` pair, not by rendered strings. Render one label per group: recipe when the dataset is shared, dataset when the recipe is shared, and both when both vary. The one-group case retains one recipe label. Distinct groups whose presentations render alike receive local ordinal qualifiers. Machine ids never become labels.

Use a frozen, slotted `HistoryLegend` value in `method_figures.py` with `legend: Legend` (one entry per group) and `group_for_system: Mapping[str, str]` (system id to the first system id in its group). Change `_system_legend` to `_history_legend(systems, plotted_ids, notation, datasets) -> HistoryLegend`. Preserve the missing-dataset error. For residual and energy, construct the grouped value only from systems with nonempty history for that metric.

Pass `group_for_system: Mapping[str, str] | None = None` through `plot_convergence_comparison` to `plot_convergence_curves`. With it, only the first curve of a group gets `legend.label(group_id)`; the other curves get Matplotlib's no-legend label. Existing callers without it retain per-series labels. `Legend.layout` needs no change because the supplied `Legend` already has one entry per group. Keep line families keyed by recipe; key method-history colours by `(DatasetId, RecipeKey)` so distinct groups have distinct base colours. Do not change the shared style resolver or the report artifact format.

## Step 1 — grouped value, drawing path, and reference

1. Add fixture-based, in-memory tests for `_history_legend` values: one group, recipe variation, dataset variation, both variations, and a planned system without a history. Test missing dataset and colliding rendered presentations. Tests assert `Legend.entries` and `group_for_system`; they do not create, save, mock, or inspect figures or axes.
2. Run the focused tests and confirm the new behavior fails against the current code.
3. Implement `HistoryLegend`, `_history_legend`, metric-specific history selection in `draw_method_figure`, and the `group_for_system` parameter in `MethodPlot`, `plot_convergence_comparison`, and `plot_convergence_curves`. Keep all existing non-method callers working.
4. Edit the history-figure description in `src/neuralls/platform/README.md` in place.
5. Run the focused tests, `uv run pytest -m "integration or not integration"`, `uv run ruff check src tests`, `uv run ty check src/`, and `uv run tach check`. Review the diff for accidental changes to per-system plots and stored records. Stop and report this step.

No case config, export, dependency, or fixture IO is needed. Existing sessions need a report re-run to regenerate figures; they do not need a comparison re-run.
