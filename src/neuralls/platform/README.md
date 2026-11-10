# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, registry resolution, TOML loaders, lower-case job metadata readers, and the thin DLKit job loader adapter. `loaders.py` separates reading from validating: `parse_case_config(raw, ctx, read_dataset_config, materialize_generated)` and `parse_data_config(raw, ctx)` take already-read mappings and touch no file unless a callback does (the default callbacks read nothing for sweeps and read dataset TOMLs from disk only for entries without an id), and `load_case_config` / `load_data_config` are their disk edges. The `ConfigContext.config_path` only anchors relative paths, so an in-memory caller passes a path that need not exist
- `config/models/`: Pydantic models for TOML sections. `comparison.py`'s RHS source models share `RhsSourceModelBase`, whose `mask: RhsMask` (`NoMask | RandomMask`, discriminated by `kind`) is available on every kind. The source models only hold configuration; `composition/comparison/recipes.py::rhs_recipe` turns one into its typed `RhsRecipe`. `OutputConfig.matrix_format` (`MatrixFormat`, default `csr`) selects the storage format of the system matrix for a dataset; it is the single declaration of that default. `OutputConfig.sparsity_pattern` (`SparsityPattern`, `shared` or `ragged`, default `ragged`) selects the CSR layout; it is part of the digest. The preconditioner configs are split by concern: `preconditioner.py` (the preconditioner variants and `PreconditionerType`), `checkpoint_refs.py` (`NeuralCheckpointRef` plus the `model_ref` kinds; `RegisteredModelRefConfig.selector` is `ByAlias | ByVersion | LatestVersion` and `LoggedModelRefConfig.selector` is `ByRunId | LatestMatching`, built from the flat TOML keys `alias`/`version`/`latest` and `run_id`/`latest`/filters by a before-validator), `coarsening.py`, `weighting.py`, `slugs.py` (`slug_number`/`slug_text`) and `fact_roles.py`. Every config renders its own identity key through `slug()` (`checkpoint_slug()` for anything with a checkpoint ref); `BasePreconditionerConfig.name` is that slug.
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `storage/array_store.py`: `ZarrArrayStore` and `Hdf5ArrayStore`, the dense backends of the `ArrayStore` protocol (defined in `domain/generation/ports.py`, so the domain depends on it without importing platform). Arrays are created at full shape; `close()` raises `ValueError` naming any array with unwritten rows. A one-dimensional array is a single chunk, so row kinds and matrix indices do not create one file per row.
- `storage/errors.py`: `_raise_storage_error`, the OSError translation every storage module shares.
- `storage/dense_stream.py`: streamed dense persistence for `zarr` and `hdf5`. `open_dense_array_store()` opens the store that the streamed writer fills, and `save_dense_stream_manifest()` writes the manifest last, from the shapes the writer reports.
- `storage/matrix_readers.py`: suffix-keyed registry `MATRIX_READERS` (`.npy`, `.txt`, `.npz`, `.mtx`, `.mtx.gz`) returning `SystemMatrix` via `read_matrix`; `to_dense` is the explicit densify helper. Unknown suffixes raise; `.npz` must hold a sparse matrix.
- `tracking/`: MLflow run helpers, naming/query policy, workflow topology resolution, client adapters, and best-effort blocking-failure diagnostics (`failure_reporting.py`). Comparison tracking is split by concern: `report_sink.py` (`MLflowReportSink`, the `ReportSink` adapter over an `OpenReport` that `finalize()` turns into a `FinalizedReport`), `scope_run_tree.py` (`ScopeRunTree`: scope to nested run, ancestors first; `RunTree` Protocol with the `MlflowRunTree` adapter holding the one shared client), `figure_uploader.py` (`FigureUploader`: staging, per-run stem collision rule, card/table upload, uploaded-record list, index write; `ArtifactSink` Protocol with the `MlflowArtifactSink` adapter), `method_sink.py` (`MLflowBenchmarkMethodSink`: one method workload's events and scalars), `metrics_logging.py` (`setup_comparison_tracking`, parent-run metric keys, convergence/comparison scalars, configured method facts). `runtime_environment.py` (`RuntimeEnvironment`, `collect_runtime_environment`, `environment_params`: the execution environment every timing of a session depends on). The matplotlib drawing and figure planning they call live in `reporting/` (`figure_recipes.py`, `figure_summaries.py`, `figure_planning.py`, `group_plots.py`, `method_figures.py`) and never import `tracking/`. Failure diagnostics record typed context and the exact traceback in the active run without replacing the workflow exception.
- `reporting/`: plotting, artifact staging, versioned benchmark-event Parquet
  and summary artifacts, and inference output adapters
- `dlkit/`: DLKit-backed adapters for solver preconditioners and batch inference

## Semantic Difference

Platform code owns concrete integrations and side effects. A module belongs
here when it speaks a third-party API, reads or writes files, configures
runtime services, or serializes artifacts. It should not decide workflow order
or contain domain algorithms.

Platform also owns translation between repository-facing configuration data and
external runtimes. That includes path and tracking resolution, the thin DLKit
job loader adapter, and MLflow client operations. DLKit itself owns job
composition and schema validation; platform does not reconstruct DLKit sections
locally.

Preconditioner configuration uses the `PreconditionerType` discriminator
strictly. Removed spellings such as `none` and `null` are invalid at Pydantic
validation time; there is no alias normalization or deprecation path. Config
`name` values remain opaque result identifiers and never determine
preconditioner behavior.

AMG-family preconditioner config keeps coarsening controls explicit at the TOML
boundary. Classical smoothed aggregation exposes both the strength-of-connection
threshold `theta`, which controls aggregate and coarse-grid size, and the
prolongation smoothing damping `omega`; composition passes those values through
unchanged to the concrete `torchalg` coarsening strategy.

The current DLKit bridge is intentionally thin. It rejects removed uppercase
neuralls job manifests and forwards lower-case job TOMLs to DLKit's native
`load_job()` entrypoint. DLKit resolves `run.model`, `run.data`,
`run.training`, and `run.tracking` profile references itself; platform does not
re-implement that merge policy.

Lower-case job identity lookup is also centralized here. Platform config
helpers read job/model metadata from one job plus its referenced model
profile, and storage/tracking code reuse that single reader instead of
duplicating TOML traversal logic in multiple modules.

Case-config auto naming also lives at this boundary, for ids only: auto-generated
assignment ids are job-first (`{job_id}-{dataset_id}`) and comparison ids derive from
the dataset ids. A missing `display_name` stays absent; it is an explicit editorial
title or nothing, and a title is never rebuilt from registry ids (composition
composes it from the dataset's recorded presentation instead).
Preconditioner descriptions are structured values built in
`composition/preconditioners/descriptions.py` (see the composition README):
every algorithmic parameter is shown, including defaults, and neural variants
are distinguished by a validated `architecture` name (set on the `[[jobs]]` entry,
`JobEntry.architecture`, separate from its free-text training `display_name`),
never by assignment identity, job title or class name. A POD-2G method carries
its fit dataset's recorded presentation (never an id or path), which the renderer
appends once after ``|``. The plotting API takes the typed presentation, not
text: `platform/reporting/figure_text.py` renders legend entries, bar labels,
titles and subtitles at the moment a figure is drawn. Training and evaluation bar charts label each bar with `training_legend` (the rendered `TrainingPresentation`) through `plot_method_bars`; `plot_metric_comparison` only draws labels its caller rendered. `scope_label` renders a figure scope as one plain line for logs. `method_legend` renders
every method's compact presentation under the case's `NotationPolicy`; methods
whose texts would collide only because decimals were rounded are re-rendered
with more digits, and a method with no entry raises
`MissingMethodPresentationError` instead of falling back to its key. The length
limit (`LEGEND_PROFILE.max_length`) controls layout only: lines break at the
renderer's safe boundaries (`wrap_at_boundaries`), and a legend that wraps or has
more than a handful of entries moves outside the axes in as many columns as its
lines need (`Legend.layout`); no fact is dropped. Result keys and plot styles stay
keyed by the stable result key. Horizontal metric plots place the first result
at the top, matching convergence-legend and configuration order while keeping
each value paired with the same result key. The non-method-major projection uses
the `PreparedComparisonPresentation` built at preflight, never the configs again.
Every comparison run writes two convergence figures with identical styling:
relative residual `||r||/||b||` (`plot_convergence_comparison`) and relative
energy-norm error `||e_k||_A/||e_0||_A` (`plot_error_convergence_comparison`,
reading `CGComparisonResult.error_history_a_rel`). The method residual figure draws a dashed
stopping-threshold line (`method_figures.residual_threshold`, from the stored `rhs_norm` and the
planned `rtol`/`atol`) only when every plotted system shares one threshold.
Condition-number figures and MLflow metrics are deliberately absent from the
comparison workflow because iterative spectral estimation can dominate or
stall otherwise healthy solver runs.
Plotting defaults keep markers compact so convergence and diagnostic figures
stay readable when several methods or dense prediction samples are shown.
Comparison-plot styling (`reporting/series_style.py`) resolves three independent
visual axes instead of deriving everything from one attribute: linestyle
always follows preconditioner family (lowest-cardinality attribute, matching
matplotlib's own default solid/dashed/dashdot/dotted order), while marker and
color follow explicit per-entry keys when the composition layer supplies them
— e.g. POD-2G comparisons key color on fit dataset and marker on snapshot
weighting scheme (`preconditioner_family.py::pod2g_style_keys`) — falling back to
family when a caller doesn't supply one. Colors come from the Okabe-Ito
colorblind-safe palette for up to 8 distinct keys, extended with
procedurally generated evenly-spaced hues beyond that so the color axis
never depletes into repeats. Markers are two-tiered (solid shapes first,
line-drawn ones as overflow) with a per-shape size multiplier so the pool
renders at a visually balanced weight. Any residual group of entries sharing
an identical (linestyle, marker, color) triple — the case when no override
keys are given at all — still gets spread across a lightness ramp so it
stays distinguishable.

The group figures use that same resolution through two modules: `series_style.method_palette` (the one
way to get a method's legend label, row order, colour, marker and linestyle from the recorded families and
colour/marker keys; callers that want a method to look the same whatever subset a figure shows pass the names
sorted) and `figure_writer.finish_group_figure` (title, subtitle, save, close). A preconditioner thus
looks the same in every plot. The history curves (`convergence_plots.py`) call `series_style.resolve_styles`
directly, and batch training and evaluation metric bars live in `metric_bars.py`. Their legends are rendered from the group's method presentations
(`GroupStyle.legend`) and append explicit success counts; a `GroupStyle` (`group_style.py`) is the fixed
title, the typed `scope` (rendered as the subtitle), the figure's `setup_sharing` record, the legend and the series
style maps. `GroupStyler.style` copies `setup_sharing` from the plan's metadata, the record the card stores as `r_m`,
and the subtitle adds `figure_text.setup_sharing_note` of it under the scope: for every `online_s` figure (bars, ECDF,
performance profile, paired scatter, speed ratio, means, distribution) "each matrix's setup is shared by its r_m
planned RHS, r_m = 4", with the distinct values ("r_m = 2 or 4 (varies by matrix)") or, past four values, their range
when matrices differ, and nothing for a metric that shares no setup, so the drawn text and the card cannot disagree. The per-run convergence plots also come in solve-only variants
(`solve_only=True`) whose curves all start at t=0. `MLflowReportSink.report`
takes the aggregated group plus a `GroupPresentation` and logs it into the nested run of the
group's report scope; figure file names are slugs of the rendered scope (see the figure-card
paragraph below), never group ids.

Report figures are drawn inside `figure_style.paper_style()`, one `rc_context` (seaborn `paper` context and `ticks` style, serif font, top and right spines removed, thin lines, fixed font sizes as named constants). It is entered only by the code that calls a drawer: `group_plots.write_outcome_plots` around `recipe.draw`, `method_figures.draw_method_figure`, and `composition/comparison/postprocessing/system_figures.write_system_figures`; drawers never enter it, and an architecture test allows `rc_context`, `set_theme` and `style.use` only in `figure_style.py`. `figure_style.rotate_x_labels(ax, degrees)` rotates x labels anchored at their right end. The settings are `FigureStyleConfig.label_rotation_deg` (`platform/config/models/figure_style.py`; `[comparison_baseline.figure_style]`, default `DEFAULT_LABEL_ROTATION_DEG` = 30, validated 0 to 90), `min_box_samples` (default `DEFAULT_MIN_BOX_SAMPLES` = 5) and `min_violin_samples` (default `DEFAULT_MIN_VIOLIN_SAMPLES` = 30), validated `1 <= min_box_samples <= min_violin_samples`; the section is optional like `[comparison_baseline.presentation]`: `composition/assignments/assignment_subjects.case_figure_style` reads it from the case, `MLflowReportSink` carries it to `GroupStyler`, and `GroupStyler.style` copies it into `GroupStyle.label_rotation_deg` and the two limits into `GroupStyle.row_limits` (`distribution_mode.RowLimits`), the only way a drawer sees them (the matrix heatmap's method labels and the x axis of every vertical figure use the rotation; distribution figures use the limits).

Bar figures (means, time per iteration, workload totals, cost breakdown) are vertical: one column per method, method labels on the x axis rotated by `label_rotation_deg` through `rotate_x_labels`, the metric on the y axis, stacked segments stacked upward. The text above a bar ("k/n solved", workload counts, the cost total or "total unknown: ...") is a `figure_annotations.ColumnAnnotation` standing on the top of the column's highest mark (whisker end or point) and placed by `annotate_columns`, which wraps it to 12 characters per line (the alternatives around " / " stay on one line) and extends the value axis from 0 through `value_axis_top`, whose headroom is the largest `base / (1 - lines needed)` over the annotations, so a long note over a short bar does not stretch the axis for the others; annotations never widen the figure. The stacked-bar legends sit outside the axes on the right so they do not cover the rotated labels. Distribution figures and the convergence-factor figure are vertical the same way, sized by `bar_columns_figure_size` and annotated by `annotate_columns`. `annotate_rows` serves only the remaining horizontal figures.

Every figure that draws summary marks explains them with one legend built by `mark_legend.py`: a drawer returns the set of `Mark`s it drew (`MEAN_BAR`, `RATIO_BAR`, `CONFIDENCE_INTERVAL`, `VIOLIN`, `IQR`, `MEDIAN`, `MEDIAN_POINT`, `QUANTILE_WHISKER`, `MEAN_POINT`, `SYSTEM_POINTS`, `MATRIX_MEAN_POINTS`) and `draw_mark_legend` places one proxy artist per mark, in the order of `Mark`, outside the axes on the right (`figure_text.LEGEND_ANCHOR`). A mark that was not drawn has no entry: a mean bar has an interval entry only when some method has an interval (`MIN_INTERVAL_MATRICES` matrices). Legend text is plain language ("mean", "95% matrix-cluster interval", "density of systems" (only when a violin was drawn), "IQR (q25-q75)", "median", "whiskers within q05-q95", "systems", "matrix means") and never a count; counts are the "k/n" annotations above the columns. Per-system points are white with a dark edge (`mark_legend.SYSTEM_POINT_STYLE`), so they stay visible on any method colour; the convergence-factor figure keeps method-coloured points with a white edge, draws the mean as a white-filled ring and the median as a black diamond (`MEDIAN_POINT`), distinct from the interval caps.

Every report figure is written by `figure_writer.finish_group_figure` (an architecture test forbids `savefig`
elsewhere). `figure_writer.use_headless_backend` selects matplotlib's off-screen Agg backend; it is called where
figures are created (`write_outcome_plots`, `plot_convergence_curves`, `plot_metric_comparison`), never when a
plotting module is imported, and does nothing in a process that is already headless, once per `FigureFormat` (`reporting/figure_formats.py`: `png`, the raster a browser shows). `FigureFormat` is the one place that lists the written formats:
`figure_files(stem)` derives a figure's file names from it, `split_figure_file` reads a stem and format back from a
name, and a `FigureTarget(directory, stem)` is what every plot function and `GroupDrawer` receives and returns, so
no caller spells a suffix. Adding a format is one enum member and a report schema bump; batch training and evaluation
metric bars written through the same function upload their files the same way. Seaborn figures take their data through
`seaborn_figures.plot_frame`, the one polars-to-pandas conversion, because seaborn raises on a polars frame.

The group outcome plots are drawn from one frame each by `profile_plots.py` (performance profile, ECDF),
`workload_bars.py` (workload totals), `paired_plots.py` (paired scatter, speed-ratio CDF); `figure_sizing.py` holds the sizes they share (`bar_columns_figure_size`: width follows the number of methods plus `rotated_label_overhang_in`, the reach of the leftmost rotated label past the first column, height adds `rotated_labels_height_in`; labels with no fact boundary are wrapped by `wrap_axis_label`, applied by `rotate_x_labels` and by the sizes, so both agree; `heatmap_figure_size` shares them; `paired_scatter_figure_size` and `with_outside_legend` widen a figure by its outside legend so the axes are not squeezed) and `stacked_segments.py` the segment key
shared with the cost breakdown. Planning (`figure_summaries.figure_summaries`)
computes, once per report, the frame every planned variant draws and stores (`summarize`, `ecdf_series`,
`performance_series`, `paired_systems`, `speed_ratio_summary`, `break_even_curves` or the workload totals, by the variant's `Statistic`;
at a mixture level the profile, ECDF, pairing and speed-ratio frames are computed once on the pooled `observations` of all the group's systems (the ECDF and profile denominators stay every planned system, so failures keep the plateau below 1), while the break-even curve and the heatmap stay recipe-level, and
the three reference figures only when the group's configured reference is among its methods). The sink wraps each
frame as a `FigureData` (the group's `observations`, the `summary` frame, the `GroupStyle`, the variant's `metric`
and the group's `ReferenceSelection`) and the kind's drawer in `group_drawers.py`, `(FigureData, Path) -> Path | None`,
hands it to a plot function; the drawn frame is uploaded beside the figure unchanged. A profile drawer draws the
curves from `summary` and reads the observations only for what the curve rows cannot carry (the planned count of a
method that solved nothing, the systems a performance profile excluded). Every plot function takes a required
`save_path` and returns it, or `None` when nothing was drawn, and a `GroupStyle` (fixed title, typed scope,
notation, legend and the family/colour/marker mappings). A failed run is never given a time:

- `plot_workload_bars` (`solve_time_bars_*`, `online_time_bars_*`): one stacked workload total per method from
  the workload-totals table, the `WorkloadTotal` (`solve_total_s` or `workload_online_s`) chosen by the kind's drawer
  in `group_drawers.py` ("Solve time [s]", "Total online workload time [s]"). `workload_segments` names the columns
  stacked: `converged_solve_s` then `failed_solve_s` (hatched) for solve time, preceded by `allocated_setup_s` for
  online time. Setup is its own segment because one setup event serves every RHS of a matrix, converged or not. The
  label beside each bar (`workload_bar_label`) reads "k/n converged" of the planned RHS, followed by "a/n attempted"
  when fewer were attempted, so a method that fails fast is not read as cheap. These are totals, not per-system
  amortized shares.
  Numeric bar positions pair each bar with its method; the legend guarantees distinct texts.
- `plot_paired_time_scatter` (`paired_time_scatter_*`): draws a `paired_systems` frame of the amortized
  online time of each method against the reference on equal log-log axes, labelled with the catalog label of the
  metric, with the diagonal; `paired_log_axes` spans the positive times with a factor `FAILED_AXIS_MARGIN` either
  side, and hollow markers (not clipped, as they sit on the axis limit) sit on the upper edge of the axis of the run that failed (`method_only` and
  `reference_only` rows). A system neither side solved is not in the frame, and a time of 0, which a log axis
  cannot place, is not drawn.
- `plot_performance_profile` (`performance_profile_solve_*`, `performance_profile_online_*`): Dolan-Moré profiles for solve-only and amortized-online time
  (Math. Program. 91, 2002), drawn from `performance_series`. When systems were left out because the
  best value on them is 0, the subtitle ends with "N systems excluded: best value is 0"
  (`excluded_zero_best_note`); the legend sits outside right and each curve's legend is the method name alone, the solved and planned counts being in the card and table.
- `plot_ecdf_profile` (`ecdf_iterations_*`, `ecdf_time_solve_*`, `ecdf_time_online_*`): unnormalised iteration, solve-only, and amortized-online ECDFs
  from `ecdf_series` over an absolute budget (Hansen et al. 2016, COCO), not the n_p+1-normalised
  Moré-Wild data profile. Step curves share one right edge so each plateau is
  its success rate.
- `plot_speed_ratio_cdf` (`speed_ratio_cdf_*`): ECDF of per-system reference/method
  ratios (`speed_ratios`) with a line at 1; the legend (`speed_ratio_legend`, from the
  `speed_ratio_summary` row) gives the median and its matrix-clustered interval, for example "median 1.23 [0.98, 1.50]"; the x label names the reference once ("Speed ratio, <reference> (reference) / method"); the legend sits outside right; the system and matrix counts the interval rests on are on the card. Reference-only and method-only counts are logged separately; no Wilcoxon
  p-value is reported.
- `seaborn_cost_curves.plot_break_even_curves` (`breakeven_*`, `RecipeProfileReport` only, skipped
  without a configured reference among the group's methods; online cost only, so it excludes the
  offline stages the `cost_breakdown` figure adds): the cumulative cost of solving $n$
  RHS of one matrix, one solid line per compared method and a dashed line of the same colour for the
  reference on that pair, drawn by seaborn from the `break_even_curves` table (`COST_CURVE_SCHEMA`),
  with a star at the break-even count $n^*$ read from the same table. Observation unit: the
  matrix for setup, the system for solve. Population: per pair, the matrices both set up and the
  systems both solved (`break_even`'s common population, so the reference's line can differ between
  pairs). Statistic: the means $\bar{t}^{\mathrm{setup}}$ and $\bar{t}^{\mathrm{solve},\mathrm{sys}}$,
  no interval. Time scenario: $C(n) = \bar{t}^{\mathrm{setup}} + n\,\bar{t}^{\mathrm{solve},\mathrm{sys}}$
  for $n = 1..N$, $N$ the largest planned RHS count of a matrix extended to $\lceil n^* \rceil$ (at
  most `BREAK_EVEN_HORIZON_LIMIT`). Failure treatment: omitted (a system either side failed is
  outside the common population). Every compared method keeps a legend entry
  (`break_even_legend`): "breaks even at n* = ... RHS", "never breaks even (not faster per RHS)" or
  "no systems in common with the reference", the `BreakEvenState` stored with its rows.
- `seaborn_cost_curves.plot_convergence_factors` (`convergence_factor_*`, every report level): the
  residual reduction factor per iteration $\gamma_{p,s}$ of the metric catalog
  (`CONVERGENCE_FACTOR`, read from each system's stored `convergence.parquet` history into the
  observation view), one vertical column per method (labels rotated like the bars) for the one cohort the figure draws, read from the
  cohort's rows of its `COHORT_SUMMARY` table (`by_cohort` of `summarize`; one figure per cohort,
  as for the mean bars below). Observation unit: the system. Population: own (every system of the
  method with a factor, within the cohort). Statistic: each cohort row gives
  the drawn mean over systems with its 95% matrix-cluster bootstrap whisker (the card's
  `[interval]` is `bootstrap_of_means`) and the median diamond ($q_{0.5}$); seaborn strips the
  per-system points, which are the observation rows of the cohort the summary was taken
  over and are not stored again. Time scenario: none, it is not a time, so the card records no timing evidence.
  Failure treatment: omitted; only `converged` systems with at least one stored iteration and a
  nonzero initial residual have a factor, and the card's outcome counts state how many of each method's
  planned systems do, and the axis label is the method name alone. The legend is the mark legend
  (see below) of the marks drawn. The card cites
  `convergence.parquet` among the sources.

`seaborn_means.py` draws the bar figures of means and of time per iteration with
`sns.barplot(..., errorbar=None)` through `seaborn_figures.plot_frame`: seaborn's own error bars
resample rows independently, wrong when the RHS of one matrix share it, so every bar, whisker and
"k/n" annotation is read from the figure's cohort table (`bar_panel` turns the one cohort it
holds into a `BarPanel`, each method a `MethodBar` whose `height` is a `Bar(value, Whisker |
NoWhisker)` or `NoBar` when nothing is defined, never a zero bar). A figure has one axes and draws
one cohort; there are no side-by-side panels (see "Figures per cohort" below), and the recipe, or
the mixture, it draws is named in the subtitle (`figure_text.cohort_note`). The count words (`count_label`: "solved", "set up", "returned an iterate") and the row annotations
(`annotate_columns` for bars, `annotate_rows` for distribution rows) are in `figure_annotations.py`, and the observation rows behind a panel
(`cohort_systems`, `panel_points`, of observations already narrowed to the cohort) in
`cohort_points.py`. The five points of each kind, with the formulas in
`domain/solver/README.md`:

| Kind (stems) | Observation unit | Population | Statistic (bar; whisker) | Time scenario | Failure treatment |
|---|---|---|---|---|---|
| `iteration_means` (`iteration_means_*`) | system | own | $\bar{x}^{\mathrm{sys}}_p$ (`mean_over_systems`); matrix-cluster interval `mean_over_systems_ci_*` | none (iteration count) | omitted: only `converged` systems enter, "k/n solved" from `n_defined`/`n_planned` |
| `time_means` (`time_means_{solve,online,standalone}_{own,common}_*`) | system | own, and common to every method of the group | $\bar{x}^{\mathrm{sys}}_p$; `mean_over_systems_ci_*` | $t^{\mathrm{solve}}$, $t^{\mathrm{online}} = t^{\mathrm{solve}} + t^{\mathrm{setup}}/r_m$, $t^{\mathrm{standalone}} = t^{\mathrm{solve}} + t^{\mathrm{setup}}$ | omitted: only `converged` systems, "k/n solved" |
| `setup_means` (`setup_means_*`) | matrix | own | $\bar{x}^{\mathrm{mat}}_p$ (`mean_over_matrices`); `mean_over_matrices_ci_*` | $t^{\mathrm{setup}}_{p,m}$ once per matrix | omitted: only matrices whose setup completed (`converged`, `unconverged`, `solver_error`), "k/n set up" over planned matrices |
| `time_per_iteration` (`time_per_iteration_*`) | system | own | $\Theta_p = \sum t^{\mathrm{solve}} / \sum k$ (`ratio_of_sums`); no whisker, per-system $\theta_{p,s}$ drawn as points from the observations | solve only | omitted: `converged` and `unconverged` systems with at least one iteration enter, "k/n returned an iterate" |

All four are drawn at every report level. Their cards state `[interval]` as the
`MatrixClusterBootstrap` of the whiskers (`figure_evidence.bootstrap_of_means`: `matrices` is the
number of distinct matrices that give any method a defined value, and the interval is absent below
`MIN_INTERVAL_MATRICES`, where no whisker exists); `time_per_iteration` records none.

**Cohort tables and figures per cohort.** The tables of the means, time per iteration,
distributions and convergence-factor figures are cohort tables (`TableKind.COHORT_SUMMARY`,
`TableKind.COHORT_RATIO_OF_SUMS`): the statistic's own schema behind one leading `cohort` column
(`domain/solver/reporting/cohorts.py::by_cohort`), one block of rows per recipe (the recipe's
lossless key) and, only at a mixture level, a last block over all recipes labelled `mixture`.
`figure_summaries` computes the whole cohort table once per variant. Each cohort is then its own
figure (`cohort_variants.py`, pure): which cohorts get a figure is decided by the report's
`CohortLevel`, matched exhaustively.

| Report level | Figures of a cohort kind | Stem |
|---|---|---|
| matrix set and RHS type | one, the report's only recipe | `<stem>_<slug>` (unchanged) |
| matrix set | the pooled one only: a recipe cohort would repeat the matrix-set-and-RHS-type report of that recipe row for row, since both take the statistic over that recipe's systems of that matrix set | `<stem>_<slug>__mixture` |
| all matrix sets | one per recipe, then the pooled one | `<stem>_<slug>__<recipe key slug>`, `<stem>_<slug>__mixture` |

The ECDFs, performance profiles, paired scatter and speed-ratio CDF (`CohortSplit.POOLED_AT_MIXTURE`:
tables `series`, `paired`, `speed_ratio`) have no per-recipe row blocks. At recipe level they are the
one unsuffixed figure; at matrix-set and all-matrix-sets levels each variant is one figure pooled
over all of the group's systems and all RHS types, stem suffix `__mixture`, card cohort `mixture`,
a subtitle line stating that it is a mixture of recipes, and a table that is that figure's own
series, pairs or ratio frame (never `cohort_rows` of anything). A recipe's pairs and curves are
unaffected by the other recipes' systems because system ids are unique per session and every
frame is keyed by `(dataset, system_id)`; the speed-ratio interval clusters by `(dataset,
matrix_id)`, so recipes sharing a matrix share one cluster. The break-even curve and the heatmap
(a cumulative-cost curve and a per-matrix cell have no meaning pooled) stay recipe-level only.
Figures whose table is neither (workload bars, cost breakdown, the break-even curve and the
heatmap) are one figure of the whole group at every level they apply to, with no suffix. A cohort figure has its own card and its own sibling table `<its
stem>.parquet`, which holds exactly `cohort_rows(table, cohort)`; `table_of(stem, kind)` and the
uploader's stem-collision check work on the full suffixed stem, and a stem is mapped back to its
variant through the card's kind, metric and population, never by parsing the stem. The card's
`cohort` entry (absent for a figure of the whole group) is the recipe key or `mixture`, and its
`recipes`, counts, weights and evidence describe the systems of that cohort alone, while `scope`
stays the report group's; `figure_index.toml` entries differ by card path. `group_plots.write_outcome_plots`
hands each drawer the cohort's rows and the observations of the cohort's systems
(`cohort_observations`), so a drawer draws one axes and never filters by recipe.

`seaborn_distributions.py` draws the distribution figures and the matrix heatmap. A
distribution figure has one column per method (methods on the x axis, the metric on y). The pure
`distribution_mode.row_mode(n_defined, RowLimits)` picks each column's mode from the method's number
of defined systems: below `min_box_samples`, `PointsOnly` (`sns.swarmplot` of the systems and a short
median tick read from the summary row); from `min_box_samples`, `BoxAndSwarm` (`sns.boxplot(whis=(5, 95),
showfliers=False, fill=False)` over the swarm: box = IQR q25 to q75, line = median); from
`min_violin_samples`, `Violin` (`sns.violinplot(inner=None, cut=0)` with the same box and the per-matrix
means as strip points). Seaborn recomputes the box from the systems; linear interpolation is also the
interpolation of the stored `q25`/`q50`/`q75` (`summarize`), and a test pins that the two agree. A
whisker is drawn to the most extreme system inside q05 to q95 (matplotlib clips it to a data point),
hence its legend text "whiskers within q05-q95". The swarm, the violin density and the matrix points
have no table counterpart and are not persisted. The median tick of a points-only column and the "k/n
solved" annotation are read from the method's row of the figure's cohort summary table
(`distribution_panel` gives the `DistributionPanel` of the one cohort the table holds, each method a
`MethodDistribution` whose `quantiles` is `Quantiles` or `NoQuantiles` when nothing is defined).
The whiskers are a distribution summary of the systems, not a confidence interval: the card records
no `[interval]` (`NoInterval`) and its `metric_definition` says so. Mixture levels draw
one figure per cohort, like the mean bars. The heatmap's rows are the matrices of a
recipe level (labelled `matrix <index>`; the dataset is the figure's scope) and its columns the
methods; `sns.heatmap` colours the cells from the pivoted `MATRIX_BREAKDOWN_SCHEMA` table, and every
cell's annotation is read from its row (`heatmap_cells`, `cell_label`): the mean, "k/r" under it
when only some RHS converged, or a hatched cell labelled "0/r solved" when none did. The figure grows
in height with the number of matrices.

| Kind (stems) | Observation unit | Population | Statistic (marks) | Time scenario | Failure treatment |
|---|---|---|---|---|---|
| `iteration_distribution` (`iteration_distribution_*`) | system; points are per-matrix means | own | quantiles $q_{0.05}, q_{0.25}, q_{0.5}, q_{0.75}, q_{0.95}$ of `SUMMARY_SCHEMA` (`Statistic.QUANTILES`); violin density of $k_{p,s}$ | none (iteration count) | omitted: only `converged` systems enter, "k/n solved" |
| `time_distribution` (`time_distribution_{solve,online}_*`) | system; points are per-matrix means | own | the same quantiles of $t^{\mathrm{solve}}$ or $t^{\mathrm{online}}$ | $t^{\mathrm{solve}}$, $t^{\mathrm{online}} = t^{\mathrm{solve}} + t^{\mathrm{setup}}/r_m$ (card lists `r_m`) | omitted: only `converged` systems, "k/n solved" |
| `cost_breakdown` | total cost: offline generation, training, setup, solve (reported as online) | own | workload total per method, offline stages stacked on setup and solve (`Statistic.OFFLINE_ONLINE_TOTAL`, `build_cost_breakdown_table`) | none: failed attempts keep their time; an unavailable offline stage draws no segment and the bar states that the total is unknown |
| `matrix_heatmap` (`matrix_heatmap_iterations_*`) | matrix (one cell per method and matrix) | own, per cell | $\bar{x}_{p,m}$, the mean of $k_{p,s}$ over the matrix's converged RHS (`Statistic.MEAN_PER_MATRIX`, `matrix_breakdown`) | none (iteration count) | omitted from the cell mean, "k/r" when some RHS failed; a cell with none converged is hatched and labelled "0/r solved" |

The distributions are drawn at every report level and write `TableKind.COHORT_SUMMARY`; the heatmap
is drawn only at the matrix set and RHS type level, since a mixture would pool recipes inside one
cell and the recipe levels already draw every matrix, and writes `TableKind.MATRIX_BREAKDOWN`.
Iterations are the heatmap's only metric: they are what differs between matrices for a
preconditioner, while per-matrix times are shown by the time distributions' points.

`MLflowReportSink` logs planned, attempted, and converged counts, the iteration statistics and the
failure-aware paired fields, each into the nested run of the group's report scope. The paired fields
come from `speed_ratio_summary` of the recipe report's `observations` against the configured
reference on amortized online time (nothing is logged without a configured reference among the
group's methods), under `<metric>/<method>` keys. The iteration statistics are `iterations_converged_mean`, `iterations_converged_median` and `iterations_converged_std` (`metrics_logging.metric_key`, which names a statistic after the outcomes the metric catalog defines the metric on, so the key says only converged systems entered it) next to `iterations_n_defined` and `iterations_n_planned` (`count_key`). The paired fields are: `n_both_solved`, `n_reference_only`,
`n_method_only`, `n_ratio` (both-solved systems with a positive ratio), `n_matrices` (distinct
matrices among them, the cluster count the interval rests on), `speed_ratio_median`,
`speed_ratio_ci_low` and `speed_ratio_ci_high`; a null statistic is not logged. The `ReportSink` port (`composition/comparison/postprocessing/ports.py`) declares `register_method_figures` beside `report`, `report_system` and `finalize`. Its typed plans (`FigurePlan`: editorial title plus the figure's metadata) are also the single source of
figure scope, stable filenames, metric and statistic, and card
metadata. `figure_recipes.py` owns the recipes and their variants and lookups; `figure_summaries.py` computes the frame of each planned figure (one exhaustive `match` over `Statistic`); `figure_planning.py` builds the plans, card metadata and `table_of`; `cohort_variants.py` fans a cohort kind out into the figures its report level gets. `figure_recipes.FIGURE_RECIPES` holds one recipe per `FigureKind`, the single registry a new figure is added to: a `GroupFigureRecipe` (the kind's `FigureVariant`s, failure treatment, the group report types it applies to, its `draw` function, a public function of `group_drawers.py` named after the figure (`group_drawers.speed_ratio_cdf`) and the `TableKind` it writes) or a `HistoryFigureRecipe` (one `HistoryVariant`; drawn from a stored method by `method_figures.METHOD_PLOTS`). A `FigureVariant` is one figure of a kind: file stem, title, metric definition and the `metric` (`MetricId`), `population` (`Population`) and `statistic` (`Statistic`: workload total, ECDF, performance profile, paired values, speed-ratio CDF, mean over systems, mean over matrices, ratio of sums, cumulative cost, mean and median, quantiles, mean per matrix, offline plus online total; the values of the bar statistics are the table columns their bars are read from) it draws; the variants of a kind differ in metric or population (`time_means` has both populations of each time), which is how `GroupFigureRecipe.variant(metric, population)` selects one and how `write_outcome_plots` hands the right metric to the kind's drawer (it draws each planned figure that has a summary and stages it with that summary as its table). `TableKind` (`summary`, `series`, `paired`, `speed_ratio`, `workload_totals`, `cohort_summary`, `cohort_ratio_of_sums`, `cost_curves`, `matrix_breakdown`, `cost_breakdown`) names the table a recipe writes and `figure_metadata.TABLE_SCHEMAS` maps each kind to its columns and dtypes, taken from the domain schema of the frame it stores. Metadata is `GroupFigureMetadata` (cohort, metric, population, statistic, failure treatment, `GroupAggregationRule`, and the evidence records of `figure_evidence.py`: `interval`, `setup_sharing`, `outcome_counts`, `timing`) or `MethodFigureMetadata` (`MethodAggregationRule`; no metric or failure treatment), sharing `TimingPolicy`, `DatasetId` dataset ids and `RunId` sources. The evidence fields are closed variants (`figure_evidence.py`), present on a card only when they apply: `[interval]` (`method = "matrix_cluster_bootstrap"`, `level`, `resamples`, `matrices`) is the `MatrixClusterBootstrap` of the speed-ratio CDF, built from the constants `speed_ratio_summary` resamples with and the distinct matrices that give any method a ratio, or of the whiskers of a mean figure and of the convergence factors (`bootstrap_of_means`, above), and is absent from every figure that draws no interval; `r_m` lists the distinct planned RHS counts per matrix and exists only for a metric that divides a matrix's setup over them (`online_s`, the `SetupSharedOverRhs` variant); `[outcome_counts]` is `summary.outcome_counts` per method; `[timing_spread]` holds each method's median relative spread of its timing repeats (`summary.timing_noise`) and exists only on a figure of a metric measured in seconds, with `timing_warning` added when the presentation's solver contexts show a single timing repeat, so the figure's times carry noise it cannot show. A group figure is uploaded as a `StagedTabledFigure` (figure plus its table frame), a history figure as a `StagedFigure` (its plan and the `FigureTarget` its files were written to; `FigurePlan.stem` and the target's stem must agree). Every figure (group, method and system) is uploaded with its **figure card**
(`reporting/figure_card.py`: `<stem>.toml` beside `<stem>.png`, same run directory, same stem) into the
run of its own scope; stems only need to be unique inside one run directory, so system figures keep the
short stems `residual_history` and `energy_history`. A card is the single record of a figure's parameters: `stem`
(the file stem its files share) and `files` (those files, one per `FigureFormat`; a card whose `files` are not the ones its stem implies is rejected), `kind`, `title`, `metric_definition`, `level`, `aggregation_rule`, `timing_policy`, and for group figures `metric`, `population`, `statistic`, `failure_treatment`, `[outcome_counts]` (per method, the number of its planned systems in each `Outcome`), `table` and `table_kind`, and, only where meaningful, `[interval]`, `r_m` and `[timing_spread]` with `timing_warning`, `[scope]` (encoded plus `display`), `[[methods]]` (method id, encoded
presentation, config digest), `[counts]`, `[reference]`, `[[parameters]]` (one table per solver context:
tolerances, iteration limits, timing repeats, normalization) and `[source]` (the session run id and, per
method child run, the artifact paths its numbers came from). `FigureCard` is the validated immutable value;
`write_figure_card`, `read_figure_card` and `validate_figure_card` round-trip it. Group figures (`figure_metadata.GROUP_FIGURE_KINDS`, every kind but the two histories) also carry a **table** `<stem>.parquet` in the same directory: the frame the figure draws (`SUMMARY_SCHEMA` rows for the bars of converged counts, `PROFILE_SERIES_SCHEMA` rows for ECDFs and performance profiles, `PAIRED_SCHEMA` rows for the paired scatter, `SPEED_RATIO_SCHEMA` rows for the speed-ratio CDF, `COST_CURVE_SCHEMA` rows for the break-even curves, `COHORT_SUMMARY_SCHEMA` rows for the mean bars, the convergence factors and the distributions and `COHORT_RATIO_OF_SUMS_SCHEMA` rows for time per iteration (both by recipe cohort, with the pooled `mixture` block at a mixture level), `MATRIX_BREAKDOWN_SCHEMA` rows, exactly the cells drawn, for the matrix heatmap, and for the sum bars `reporting/figure_table.py::build_group_table` rows of workload totals with `method_id`, `n_planned_rhs`, `n_attempted_rhs`, `n_converged`, `allocated_setup_s`, `converged_solve_s`, `failed_solve_s`, `solve_total_s`, `workload_online_s`; converged plus failed is the solve total, and adding the setup gives the online total, and for the cost breakdown `build_cost_breakdown_table` rows with `method_id`, `generation_s` and `training_s` (null unless the stage's status is `known`, with `generation_status` and `training_status` of `not_applicable`, `known` or `unavailable`), `fit_origin`, `allocated_setup_s`, `solve_total_s` and `total_s` (`total_cost_seconds`, null with `total_status = offline_unavailable` when an applicable stage is unavailable, never a total that reads it as zero)), so a paper table and the figure cannot disagree. The card names it in its `table` and `table_kind` entries; the recipe alone decides whether one exists and of which kind (`figure_planning.table_of` returns `SiblingTable(name, kind)` or `NoTable`, so the card cannot disagree), the uploader refuses a table whose schema is not its kind's (`figure_table.check_table_kind`), `read_group_table(path, kind)` reads a stored table back only as that kind and raises `InvalidFigureTableError` otherwise, and `verify_figure_records` additionally requires a parquet for exactly the figures whose card names one. `tracking/scope_params.py::scope_params` logs each non-root scope run's identity as MLflow params (`report_level`; `dataset_id` from matrix-set level down; `recipe_key` (the lossless `recipe_key`) from cohort level down; `matrix_id` and `rhs_ordinal` for a system), and the report workflow logs `report_level` on the root run; solver settings and anything else varying between figures of one run stay in the cards. `figure_index.toml`
(`reporting/figure_index.py`, written once to the root run's artifact root in `finalize`) lists only
`run_id`, `artifact_paths` (every file of the figure), `card`, `kind`, `level` and `scope` per figure, derived from the cards, and cites in a top-level `splits` key the `splits.toml` copied beside it (`FigureUploader.write_index` uploads both, so train/test separation is checkable from the report's own data; `validate_figure_index` rejects an index without the citation); before it is
written `verify_figure_records` raises unless every uploaded figure file belongs to a figure that has all its format files and a same-stem card naming that stem, and the index
lists exactly the uploaded figure files, each entry's files being the ones its card names. The card stores each scope as keys and typed values (`facts_codec.encode_figure_scope`, whose recipe is `encode_recipe`: kind, parameters and mask) beside its plain rendered `display` text; file names are slugs of rendered text, never ids: a group figure `{kind}_{slug}` (written as `.png`) takes `slug` from the plain rendered scope (`scope_label`, which prefixes dataset, recipe, and system scopes with "Comparison matrix set") and a method figure `residual_history_{slug}` from the rendered method label (`POD-2G (r=10)` becomes `POD-2G_r_10`); `figure_metadata.FigureSlugs` allocates one stem per subject (<= 80 characters) and suffixes `_2`, `_3` in encounter order when two subjects render alike. Machine ids live only in metric keys (`{metric}/{method}`, one run per report scope) and in `*_id` fields (`dataset_ids`, encoded `recipes`, `method_id`; `[reference]` holds the selection kind plus the method id and reason when it has them). Card counts retain every dataset, recipe population/weight, and
the actual solver, normalization, and timing-repeat settings; method-child
history figures are registered with unique method-qualified ownership. Raw
benchmark events and method summaries become
the persistence source of truth; the removed scaling-law TOML is not emitted.

**Figure kinds.** Every group figure answers five questions, which its card records and which `FIGURE_RECIPES` fixes per kind; formulas and the per-metric failure rule are in `domain/solver/README.md`, "Observation view and metric catalog". Observation unit is a system unless stated; "own" population keeps the systems on which each method's metric is defined, "common" those defined for every method. Online time is the amortized scenario $t^{\mathrm{solve}} + t^{\mathrm{setup}}/r_m$, standalone charges the whole setup, solve ignores it.

| Kind | Metric (time scenario) | Population | Statistic drawn | Failure treatment |
|---|---|---|---|---|
| `solve_time`, `online_time` | solve, online | own | workload total over attempted systems, stacked setup, converged and failed solve time with k/n | none: failed attempts keep their time |
| `paired_time` | online, method against reference | own | per-system pairs, log-log | censored on the axis for an asymmetric failure; both-failed pairs omitted |
| `speed_ratio` | online, reference/method | own | CDF of the ratio with a matrix-cluster interval | omitted |
| `performance_profile` | solve, online | own | profile series | censored |
| `ecdf_iterations`, `ecdf_time` | iterations, solve, online | own | ECDF with failures at infinity, so the plateau stays below 1 | censored |
| `breakeven` | standalone | common | cumulative cost against the RHS count | omitted |
| `iteration_means`, `time_means` | iterations; solve, online, standalone | own and common | mean over systems with a matrix-cluster interval | omitted; the card states k/n |
| `setup_means` | setup (one value per matrix) | own | mean over matrices | omitted |
| `time_per_iteration` | solve time / iterations | own | ratio of sums | omitted; unconverged runs with a real iteration count enter |
| `iteration_distribution`, `time_distribution` | iterations; solve, online | own | points, box (quartiles, whiskers inside q05-q95) or violin by sample size, a distribution summary and not a confidence interval | omitted |
| `matrix_heatmap` | iterations | own | mean per matrix | omitted: a cell with no converged RHS is hatched "0/r solved" |
| `convergence_factor` | residual reduction per iteration | own | mean and median | omitted |

**Cost breakdown.** Offline cost is a property of the method artifact, so the figure reads it from each method's stored `summary.toml` (`StoredMethod.offline`, passed to `figure_summaries`) and the online segments from the report group's workload summaries; at a sub-workload level the full offline cost stands beside only that level's online seconds, which is the cost of serving that workload from a fresh artifact. The figure is stacked vertical bars where colour encodes the stage, not the method: one fixed Okabe-Ito colour per stage (`cost_breakdown_plot.segment_color`; dataset generation, offline fit / training, matrix setup, solves), no hatches, and the method is the rotated x tick label; one legend of stages (`stacked_segments.segment_key` with the stage colour) sits outside the axes on the right. The text above each bar is the total (or "total unknown: ...") and, on a second line, the fit origin from `OfflineStages.fit_origin()`: "fit job" (a training stage applies; its seconds are the wall time of the MLflow run that produced the checkpoint, a `run.type = "fit"` POD job included), "fit inline, counted in setup" (no training stage but a dataset to generate: the fit runs inside the measured setup segment, so there is no offline fit segment) or nothing (no fit at all). The table states the same in its `fit_origin` column (`none`, `inline`, `job`). Training seconds are the wall time of the whole training job; `tracking/run_duration.py::run_wall_seconds_from_run` reads a record already fetched during method preparation, so cost attribution does not fetch the same run again (see `domain/solver/README.md`, "Comparison vocabulary").

**Memory scope.** Peak memory is an operation- or method-lifetime reading, never a comparison across platforms. On CUDA it is PyTorch's allocator peak (`torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`) for the timed operation; it excludes the CUDA context, library workspaces and non-PyTorch allocations, so it is not total device memory. On CPU it is a process RSS high-water mark (`ru_maxrss`, `peak_wset` on Windows) read as a delta: the process has no reset, so it is a lower bound when an earlier workload set a higher peak, and a CPU peak must not be compared across methods. `shared/device.py` holds both readings.

**Run names.** `tracking/run_names.py::render_run_name` is the only place a typed
`RunName` (`domain/presentation/run_names.py`) becomes text: plain Unicode under the case's
notation, one line, no wrapping. `MlflowRunConfig` carries a `NamedRun`; the
text is produced where MLflow (or dlkit, which opens the training and evaluation runs) is
handed a name (`run_name_text`), so a run cannot be named from an id, a slug, a path or a
class name. `ComparisonRunName` renders its dataset as "Comparison matrix set: ..."
after any editorial title. A comparison session parent applies that prefix to a
dataset title too; training and evaluation session parents retain their dataset
titles. Method-child names and assignment training/evaluation subjects keep
their own typed labels. Tags keep machine ids in `*_id` fields (`method_id`, `assignment_id`,
`comparison_id`) and rendered text in `*_label` fields (`method_label`,
`assignment_label`, `comparison_label`, and the `dataset` tag/param holding the rendered dataset beside the machine `dataset_id` lookup tag); metric keys stay machine-safe and are never labels.
The method-child tag `method_id` is what reuse queries.

**Fact logging.** `facts_codec.encode_method_parameters` turns a prepared method's configured
facts into stable parameter names (`method.<field key>`, `training_dataset.strategy_<n>.<field
key>`) with typed values and no rendered text; `log_configured_method_facts` logs them when
the method child opens, before it executes. Facts known only after setup are written by
`MLflowBenchmarkMethodSink` into `method_workload/presentation.toml`, one entry per system,
because a realized fact can differ between matrices and must never become a run-wide
parameter.

**Artifact format rule.** Parquet holds tabular, row-per-entity numerical or categorical data
that polars aggregates; TOML holds nested per-run documents (identity, configuration,
presentation); JSON is not used for new artifacts; nothing rendering-specific (colours,
markers, labels, rendered text) is persisted. Strings in Parquet are identifiers or closed
categories only.

`MLflowBenchmarkMethodSink` persists one completed method workload into the
already-active method child. The artifact set is defined once by
`benchmark_events.MethodBenchmarkArtifacts`; the writer, the reuse gate and the child's
`required_artifacts` tag all derive from it:

- `setup_events.parquet`, `solve_attempts.parquet`, and `convergence.parquet` (one row per
  method, system and iteration with the relative residual and the nullable relative energy
  error). `solve_attempts.parquet` also carries the extra residual measurements as flat
  `residual_status` (`measured`/`unmeasured`), `rhs_norm`, `true_residual_rel` and
  `reference_status` (`solved`/`unavailable`), `reference_true_residual_rel`, `reference_reason`
  columns, and the timing-repeat agreement as `repeat_status`
  (`single`/`unchecked`/`agreed`/`diverged`) with `repeat_first_iterations` and
  `repeat_other_iterations`; a payload is null exactly when its status says so (the iteration
  counts are set only for `diverged`). A table without them is rejected,
  so the reuse gate recomputes such a run.
- `planned_systems.parquet`: one row per planned system, including systems whose setup
  failed, with its dataset, recipe key, matrix and RHS ordinal, the solver and timing
  settings it ran under, and its reference state as flat `reference_kind`
  (`configured`/`none`/`unavailable`), `reference_method` and `reference_reason` columns.
  `event_frames.decode_planned_systems_frame` is the only place those columns become a
  `ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable`. These are compute
  provenance: a case config edited later would describe different numerics than the ones
  timed.
- `summary.toml` (counts, online totals, memory, and the method's `[offline.generation]` and
  `[offline.training]` stage tables, each a `status` of `not_applicable`, `known` (with `seconds`,
  `provenance` and, when recorded, `peak_bytes`) or `unavailable` (with a `reason` and no
  number); `offline_cost_codec.py` is the one reader and writer, so a missing block is an error
  and an unavailable stage is never a zero), `method_config.toml` (the reuse identity), and
  `presentation.toml` (the
  method's configured presentation, compact fact selection, per-system realized facts, and
  the presentation of each comparison matrix set it ran on, all through `facts_codec`: keys and typed values
  only). Its `[data_origin]` table records one closed status: `no_fit_data`, `inline_fit`,
  `checkpoint_verified` with the attested training-dataset digest, or
  `checkpoint_unavailable` with a reason. The strict reader rejects old or malformed evidence;
  method reuse compares it with the newly resolved origin before accepting a run.

It also draws the method's residual and energy figures, then logs reducer-owned scalars
(including `iterations_total`, and `generation_s`, `training_s` and `offline_s` only for the
stages whose cost is recorded) exactly once. The sink never opens an MLflow run;
`comparison_batch` owns the session parent and its one direct child per method.
The parent receives only shared provenance and cross-method reports, while raw
method artifacts remain on the child. Reuse requires phase
`comparison_method`, the method config/checkpoint
digest, complete workload membership, and all canonical raw artifacts; legacy
per-system runs are executed again. Reuse parses the required
configuration, planned systems, presentation and convergence tables and reconciles event
methods, membership, and summary totals before accepting a hit, and requires the `[offline]`
block, so a run stored without it is executed again; the session's figure
figure index is not part of the gate. The persisted configuration is the
canonical identity representation: checkpoint inputs are content digests and
runtime lease paths are omitted.

**Runtime environment.** `tracking/runtime_environment.py::collect_runtime_environment(device)` is the one impure reader (platform, os, torch, `importlib.metadata`; the CPU model comes from `/proc/cpuinfo`, then `platform.processor()`, and any fact the platform cannot report is the explicit `Unreported.UNKNOWN`, never a raise); `environment_params` flattens the frozen `RuntimeEnvironment` to the stable `env.*` MLflow params `env.device` (CUDA device name or `cpu`), `env.cpu_model`, `env.cpu_logical_count`, `env.torch_threads`, `env.torch_version`, `env.torchalg_version` and `env.python_version`. They are constant for a session, so the comparison session run logs them once and no parquet table or per-row record repeats them. No dtype is recorded: the stored datasets are pinned to float64 by the readers and no session-level dtype setting exists.

The comparison session run also carries `splits.toml` (`tracking/split_records.py`): one table per dataset id with the pinned split's content digest, train/validation/test row counts, the matrix ids of the fit (train and validation) and test folds, and the separation (`single_matrix_shared`, whose one matrix is in every fold by design, or `matrix_disjoint`). `download_session_splits` reads it back and rejects a session that predates it. Writing it is the composition layer's `mlflow.log_text`; `neuralls report` reads it with `download_session_splits` and the report sink copies it to the root report run beside `figure_index.toml`.

A report reads a session back through two modules. `tracking/session_artifacts.py` finds
the session's FINISHED `comparison_method` children (by `mlflow.parentRunId`), checks each
holds the full artifact set, and downloads it; the reuse gate lists artifacts through the
same helper, so run discovery exists once. `reporting/stored_benchmark.py::read_stored_benchmark`
reads one `MethodDirectory` (local path, the method child run id and its `method_config_digest` tag, as `download_session_methods` returns them) per method into a `StoredBenchmark` of `StoredMethod`s (run id, config digest, parsed method
config, presentation, offline cost, planned systems, and the setup, solve and convergence tables) and
rejects anything inconsistent: a presentation of another method, events for systems that
were not planned, a summary whose digest differs from the run's tag, a method stored twice, or datasets presented differently by two methods.
The stored config holds content digests for checkpoint and dataset inputs, so a report-time
`pod2g_style_keys` colour key groups methods by fit-dataset content, not by path.

The DLKit dataset bridge stays generic. Platform helpers construct and patch
DLKit-native dataset entries with the names supplied by composition, but
platform does not own the canonical runtime naming policy itself. Storage-layer
artifact families such as `rhs.zarr`/`rhs.npy`, `solutions.zarr`/`solutions.npy`,
and `matrix.zarr`/`matrix.npy` stay in platform storage, while the composition
dataset contract decides which runtime entry names those artifacts map to.
Training artifact resolution preserves those on-disk sources as path-backed
dataset inputs, and platform adapters translate resolved entry specs into
concrete `NpyEntry` / `ZarrEntry` objects instead of eagerly converting whole
datasets into `ValueEntry` payloads.

That change improves separation of concerns and keeps format-specific loading
policy inside DLKit, including `NpyEntry` support for `mmap_mode`. It does not
by itself guarantee fully lazy `.npy` training because DLKit's current
`FlexibleDataset` still materializes non-lazy path entries during dataset
construction.

The `dlkit/` package is the runtime adapter boundary. It hides predictor and
inference integration details behind local abstractions so solver and
application code depend on structural contracts rather than DLKit return-shape
quirks or registry helpers.
Solver-side DLKit predictors must preserve fitted checkpoint transforms during
load so transform-aware models such as PCA-preprocessed preconditioners receive
inputs in the feature space they were trained on.
The adapter layer also owns compatibility shims for checkpoint inference when
DLKit metadata serializes constructor hyperparameters under a nested `params`
object; that flattening stays local to platform code rather than leaking into
solver or comparison orchestration.
The same adapter boundary owns lifecycle translation: solver-facing predictors
implement torchalg's idempotent `cleanup()` port by delegating to DLKit's
`CheckpointPredictor.unload()`/context-manager release path exactly once. This
keeps comparison orchestration free of DLKit-specific cleanup calls while still
making one-model-at-a-time evaluation deterministic for GPU memory.

Neural model input is prepared by `predictor_adapter.py`'s `_prepare_model_input`,
which takes a `torch.Tensor` and converts it to float64 with a leading batch
axis, which DLKit's predictor expects. It never sees a `csr_array`: a sparse
system matrix is densified before it reaches this layer, in `composition`'s
builder dispatch (`_DENSIFY_FOR_CSR`, one `O(n^2)`-cost INFO log line per
call), not here. Keeping the densify out of the adapter means this layer
stays a plain tensor-shape step with no format branching of its own.

`DLKitPredictor` exposes a `required_inputs: tuple[str, ...]` property so that
the solver layer can derive which extra arrays a neural model needs without
consulting the comparison TOML. The DLKit model config is the single source of
truth; the comparison TOML's `extra_input_names` field is an optional override
kept for backward compatibility.

When extra inputs are bound, `DLKitPredictor.apply()` resolves the primary
(residual) tensor's `forward()` kwarg name from the checkpoint's own
`CheckpointPredictor.feature_names` (training-order truth persisted at save
time) instead of assuming it is always named `x`. This lets multi-input
architectures like DeepONet (`forward(branch, trunk)`) resolve correctly: the
primary name is whichever declared feature isn't already bound as an extra
input. It falls back to `x` only for legacy checkpoints with no persisted
`feature_names`, and raises a clear error if the checkpoint's declared inputs
and the bound extra inputs don't leave exactly one candidate.
DLKit itself also validates the resolved name against the checkpoint's
persisted `forward_arg_map` before calling the model, raising
`dlkit.common.errors.ForwardContractError` on a mismatch; the adapter's error
boundary translates that into `RuntimeError` alongside every other DLKit
framework failure.

## Boundary

Platform code may depend on domain protocols and domain data structures, but it
should not own business rules or assignment orchestration.

Tracking helpers treat DLKit as the authoritative checkpoint artifact logger.
Workspace uploads therefore exclude the local `checkpoints/` tree and only
forward staged diagnostics/config artifacts, avoiding duplicate MLflow artifact
layouts such as `checkpoints/checkpoints/...`.
Checkpoint discovery canonicalizes byte-identical duplicate artifacts and uses
explicit filename-role pattern matching for preferred candidates; currently only
a unique `best.ckpt` is preferred, while interval-style checkpoint names remain
ambiguous.

MLflow-specific policy also belongs here: safe metric-key sanitization, search
filter escaping, workflow tracking-environment resolution, artifact path
selection, lease-backed artifact access, and comparison-run metric logging all
stay under `platform.tracking` so orchestration code does not reimplement
third-party rules.
Reuse lookup lives here too: `mlflow_store.py::MlflowIdentityStore` (the
`domain/identity_ports.py::IdentityStore` implementation) finds the newest
FINISHED run in one experiment carrying a `StageIdentity`'s stage and key tags
(escaped filters, deterministic newest-first order, paginated, optional
real-checkpoint check via dlkit). Its client comes from an injectable
`mlflow_client.py::MlflowClientFactory` (default: the real `MlflowClient`), the
same factory shape model resolution and evaluation take. Composition derives the identities and tags
the runs (`StageIdentity.tags()`); platform only owns the MLflow filter/search
mechanics. Dataset identity lives in `storage/dataset_digest.py`
(`current_dataset_digest`: O(1) from the manifest while the stat snapshot is
unchanged, otherwise a recompute of the logical array content).
Checkpoint choice is pure: `tracking/checkpoint_selection.py::choose_checkpoint` decides over `CheckpointCandidate`s (path plus content digest), and `find_single_checkpoint` is the thin shell that globs and hashes.
MLflow artifact recovery follows the same boundary. Platform tracking helpers
resolve and validate checkpoints, split JSON, and staged config artifacts
through an `ArtifactLeaseManager` protocol with explicit abstract methods.
Local MLflow artifact stores are borrowed in place; remote stores are
materialized into scoped scratch storage owned by the lease manager. Composition
decides which assignment or model ref should be evaluated and how those local
paths are wired into DLKit, but it does not choose persistent download
directories. INFO logs bracket remote download, returned-path resolution, and
materialized-artifact validation so a timestamped CLI trace distinguishes
MLflow transfer work from local filesystem work.
When runtime `MLFLOW_TRACKING_URI` or `MLFLOW_ARTIFACT_URI` values are already
exported, platform tracking helpers preserve them verbatim instead of
re-normalizing them against the local operating system.

Storage validation owns concrete dataset-layout checks. Comparison matrix/RHS
preflight belongs under `platform.storage` because it depends on manifest and
artifact-layout knowledge rather than workflow sequencing.
Generated-RHS workflows therefore validate only the matrix artifact they
actually consume, while dataset-backed RHS workflows validate both matrix and
RHS artifacts.
The same boundary owns filesystem replacement, stale artifact cleanup, and
write-failure enrichment for dataset artifacts so CLI callers receive
operation- and path-specific diagnostics without importing storage policy into
composition. Same-format generation rewrites replace the previous artifact set;
cross-format rewrites remain blocked by the manifest guard before persistence.

Dataset storage is split by responsibility:
- `storage/manifest.py`: typed dataset manifest dataclasses and JSON serialization
- `storage/generation_formats.py`: the manifest vocabulary the streamed writers share. Manifest
  assembly is `build_dense_manifest(...)`, which takes array shapes and the run's normalization.
  The per-format differences are carried by a `ManifestLocations` DTO (one `ArtifactLocation(path,
  key)` per artifact), and `zarr`/`hdf5` read the physical matrix shape back from the written
  container. No generation writer buffers samples here. `npy` is not a generation format; the
  build entry point refuses it (see `composition/README.md`), and npy datasets stay readable
  through the reader modules.
- `storage/csr_layout.py`: the CSR-in-zarr schema (member names, dtypes, per-sample and
  shared-pattern layouts) documented in its docstring, plus re-exports of `sparse_io` under storage names.
- `sparse_io/`: storage-agnostic CSR conversions, member layouts, the sparse reader/writer protocols,
  and the zarr and hdf5 backends selected by format name through `registry.backend_for`. See
  `sparse_io/README.md`.
- `storage/csr_storage.py`: `open_csr_matrix_stream`, the streamed CSR matrix writer used by generation
  (zarr or hdf5, one sample at a time), `write_csr_matrix_group`, a whole-group zarr writer kept for
  reader fixtures (rejects dense or mixed samples; it is not on the generation path), and
  `describe_csr_matrix_group`, which reads the manifest descriptor back from a zarr group.
- `storage/dataset_readers.py`: manifest-driven read helpers and explicit resolved dataset contracts;
  `load_matrix_sparse_sample` reads one CSR sample through the backend registered for the artifact's
  format (dense datasets are converted to CSR); `load_matrix_dense_sample` densifies CSR datasets via
  `.toarray()`.
  `open_resolved_array` opens any artifact lazily (memmap/zarr/h5py) as an axis-0 sliceable
- `storage/dataset_digest.py`: `dataset_content_digest` (logical-content sha256 over every
  manifest artifact, identical across npy/hdf5/zarr and independent of location),
  `stat_digest` (relative path + size + mtime_ns of every reachable file) and
  `current_dataset_digest` (returns the manifest's stamped `content_digest` in O(1) when the
  stored `stat_digest` still matches, otherwise recomputes; never writes). Documented blind
  spot: a same-size edit with a restored mtime passes the fast path (`--force` covers it).
  The manifest carries `content_digest`, `stat_digest`, `identity_key`, `identity_components`
  (all optional; legacy manifests load them as None).

`storage/manifest_io.py::load_dataset_manifest` and
`storage/dataset_readers.py::load_matrix_dense_sample` are `functools.lru_cache`-memoized
(keyed on `dataset_dir`, and `(dataset_dir, sample_index)` respectively). Every other
manifest-driven reader (`resolve_dataset_artifacts`, `list_available_matrix_indices`,
`load_dense_training_arrays`, etc.) routes through these two, so a batch of comparison
entries that share one `matrix_dataset` reads its manifest and matrix samples once per
process instead of once per entry. `save_dataset_manifest` clears the manifest cache as
it writes, so a read after a write in the same process sees the manifest just written —
which is what lets the generate stage stamp its identity and digests into the manifest it
has only just saved. Caveat: because the cache is keyed only on the path, mutating a
dataset's *array* files on disk mid-process (rare — datasets are normally
write-once/read-many) is still not picked up without a process restart.

Safe comparison selection relies on manifest-declared metadata artifacts stored
in the dataset's native format. Mature datasets may expose:

- `row_kind`
- `matrix_sample_index`

Readers derive safe RHS candidates from `row_kind`; they do not trust ad hoc
filenames or stored allowlists. Storage owns persistence and artifact
resolution, while the compact integer encoding/decoding boundary lives in
shared pure codecs so domain and composition code do not depend on platform.
When datasets expose `matrix_sample_index`, the canonical triplet resolver uses
that binding to load the matching matrix for a selected `(rhs, solution)` row.

Case-driven comparison sample selection stays explicit and deterministic, and
is decided once by the composition planner (`composition/comparison/plan.py`)
from `dataset_shape.py` — `classify_dataset` (single vs multi matrix, from the
unique ids in `matrix_sample_index`), `load_held_out_fold` (the pinned
`split.json` test rows grouped by the matrix id each is bound to) and
`system_dimension` (the system order, from the manifest's RHS shape). A matrix
id is the sample id the generator bound rows to; the physical row passed to
`load_system_arrays`/`load_matrix_dense_sample` is a different number, and the
planner hands readers the row. Platform itself performs no leakage inference
and has no notion of a test position. Dataset-backed `rhs_source` entries
resolve a canonical manifest-backed triplet for the one planned row;
`raw_rhs` sources point at concrete RHS vector files; `raw_lhs` sources point
at concrete solution-side vector files that comparison transforms into an RHS
with `A @ vector` and the configured scale. Every system solves on a matrix of
its own dataset, so an external vector is accepted only when its length equals
that dataset's `system_dimension` (`storage/comparison.py::external_vector_length`
reads it, memory-mapping `.npy` files, with the same row-selection and
`sample_index` bounds rules as `load_system_arrays`).

Training artifact persistence also stays generic: platform storage writes the
already-normalized numpy payload it receives from composition without
reintroducing DLKit prediction-key fallback logic.
