# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, registry resolution, TOML loaders, lower-case job metadata readers, and the thin DLKit job loader adapter. `loaders.py` separates reading from validating: `parse_case_config(raw, ctx, read_dataset_config, materialize_generated)` and `parse_data_config(raw, ctx)` take already-read mappings and touch no file unless a callback does (the default callbacks read nothing for sweeps and read dataset TOMLs from disk only for entries without an id), and `load_case_config` / `load_data_config` are their disk edges. Path validators call the pure `context.anchor(value, ctx)` (also `expand_config_path`, `expand_config_glob` which anchors only the prefix, `expand_path_field`): it substitutes `${NEURALLS_*}`, takes the path of a `file:` / `sqlite:` URI (`file://` with a host is rejected), expands a leading `~` from `ConfigContext.home_dir` (`~user` is rejected), rejects a Windows absolute path on a posix host, a posix-absolute path on Windows and a drive-relative path (`C:x`) everywhere, joins a relative path onto `ConfigContext.config_dir` and normalizes lexically with `os.path.normpath` (`..` collapses also in absolute paths, so `link/../x` does not follow `link`). The `tracking_uri` of `MlflowTopologyConfig` only has its placeholders substituted (`substitute_placeholders`), never anchored. It never reads the filesystem, the working directory or `$HOME`, so symlinks below the config directory are kept. The disk edges (`load_case_config`, `load_data_config`, `load_comparison_config`, the method-override read in `composition/comparison/config_assembler.py`) build their context with `ConfigContext.for_file(path, settings)`, which resolves `config_path` and reads `Path.home()`; an in-memory caller passes any `config_path` and `home_dir`, even relative ones that do not exist, and gets relative results. A validator anchors whenever a context is given; a model built without a context is taken as already anchored. `resolution.py` helpers (`resolve_local_path`, `expand_user_path`) serve non-validator callers only. Path-origin rule for everything outside the validators: a path typed on the command line (and `NEURALLS_CASE_CONFIG` / `NEURALLS_ENV_FILE`, which stand in for flags) anchors at the working directory, and only the CLI does that (`resolve_user_path`, `profile.resolve_typed_profile`); a value written in a file anchors at that file's directory (case values at the case directory, `configs/tracking.toml` values at its directory: `tracking/mlflow.build_workflow_environment(config_dir=...)`, with `shared_or_default_environment(default_output_root)` for a workflow that has no case config (it also serves `build_runtime_environment` after the `MLFLOW_TRACKING_URI` env branch), `resolution.normalize_tracking_uri(base_dir=...)` and siblings take the base as a required argument, never a working-directory fallback); a root from an environment variable, an env file or the stored profile must be absolute after `~` expansion (`resolution.require_absolute_user_path`, applied by `settings.get_settings`, `load_case_settings` and `profile.load_profile`; the error names the variable or profile field), and so must a local `MLFLOW_ARTIFACT_URI` and a local artifact URI MLflow stores for a run (`tracking/artifact_access.py`). `NeurallsSettings` and `ProfileConfig` are pure: a model built in memory keeps its paths as given; `resolution.resolve_path_context` only picks roots (an override wins) and keeps them as given, since every input is already absolute where it comes from disk
- `config/models/`: Pydantic models for TOML sections. `comparison.py`'s RHS source models share `RhsSourceModelBase`, whose `mask: RhsMask` (`NoMask | RandomMask`, discriminated by `kind`) is available on every kind. The source models only hold configuration; `composition/comparison/recipes.py::rhs_recipe` turns one into its typed `RhsRecipe`. `OutputConfig.matrix_format` (`MatrixFormat`, default `csr`) selects the storage format of the system matrix for a dataset; it is the single declaration of that default. `OutputConfig.sparsity_pattern` (`SparsityPattern`, `shared` or `ragged`, default `ragged`) selects the CSR layout; it is part of the digest. The preconditioner configs are split by concern: `preconditioner.py` (the preconditioner variants and `PreconditionerType`), `checkpoint_refs.py` (`NeuralCheckpointRef` plus the `model_ref` kinds; `RegisteredModelRefConfig.selector` is `ByAlias | ByVersion | LatestVersion` and `LoggedModelRefConfig.selector` is `ByRunId | LatestMatching`, built from the flat TOML keys `alias`/`version`/`latest` and `run_id`/`latest`/filters by a before-validator), `coarsening.py` (`PODCoarseningConfig` has no checkpoint fields: its basis is fitted once per method inside the comparison; its `inner_product` is an `InnerProductKind`, the fact `W`, energy weighing the fit by the train-fold mean matrix and adding `_energy` to the slug; its slug ends with `_data_<fit dataset directory name>`, so equal-rank POD methods on two datasets keep distinct keys), `slugs.py` (`slug_number`/`slug_text`) and `fact_roles.py`. Every config renders its own identity key through `slug()` (`checkpoint_slug()` for anything with a checkpoint ref); `BasePreconditionerConfig.name` is that slug.
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `storage/array_store.py`: `ZarrArrayStore` and `Hdf5ArrayStore`, the dense backends of the `ArrayStore` protocol (defined in `domain/generation/ports.py`, so the domain depends on it without importing platform). Arrays are created at full shape; `close()` raises `ValueError` naming any array with unwritten rows. A one-dimensional array is a single chunk, so row kinds and matrix indices do not create one file per row.
- `storage/errors.py`: `_raise_storage_error`, the OSError translation every storage module shares.
- `storage/dense_stream.py`: streamed dense persistence for `zarr` and `hdf5`. `open_dense_array_store()` opens the store that the streamed writer fills, and `save_dense_stream_manifest()` writes the manifest last, from the shapes the writer reports.
- `storage/matrix_readers.py`: suffix-keyed registry `MATRIX_READERS` (`.npy`, `.txt`, `.npz`, `.mtx`, `.mtx.gz`) returning `SystemMatrix` via `read_matrix`; `to_dense` is the explicit densify helper. Unknown suffixes raise; `.npz` must hold a sparse matrix.
- `tracking/`: MLflow run helpers, naming/query policy, workflow topology resolution, client adapters, and best-effort blocking-failure diagnostics (`failure_reporting.py`). Every tracking write goes through `run_writer.py`: `MlflowRunWriter` (the `RunWriter` Protocol's MLflow adapter) writes on a client bound to one store into a run named by id, and `open_run` creates a run and finishes it, as failed when its block raises. The package never imports the top-level `mlflow` module, whose fluent API reads the process-wide active run and tracking URI and, with neither set, starts a run on `./mlflow.db`; `tests/architecture/test_io_boundary.py` enforces it. Comparison tracking is split by concern: `comparison_runs.py` (`MlflowComparisonRuns`, the adapter of composition's `ComparisonRuns` port: a session run whose `MlflowSessionRun` records splits and environment and opens one `MlflowMethodRun` child per method, each with its own failure sink), `report_sink.py` (`MLflowReportSink`, the `ReportSink` adapter over an `OpenReport` that `finalize()` turns into a `FinalizedReport`; it writes every metric and artifact into the run of its scope through a `RunWriter`, a system's into its RHS type's run under `systems/<system_key>/` and `system/<system_key>/` metric keys, `system_key` being its planned system id made MLflow-safe, since a system has no run), `scope_run_tree.py` (`ScopeRunTree`: scope to nested run, ancestors first; `RunTree` Protocol with the `MlflowRunTree` adapter creating children in the report's experiment), `figure_uploader.py` (`FigureUploader`: staging under a directory the caller owns and removes, the report's own temporary directory, so a failed report leaves nothing behind; per-run stem collision rule, card/table upload, uploaded-record list, index write into the root run; `ArtifactSink` Protocol, which `MlflowRunWriter` implements), `method_sink.py` (`MLflowBenchmarkMethodSink`: one method workload's events and scalars, into its method run), `metrics_logging.py` (parent-run metric keys; the pure `convergence_metrics` and `comparison_metrics`). `runtime_environment.py` (`RuntimeEnvironment`, `collect_runtime_environment`, `environment_params`: the execution environment every timing of a session depends on). The matplotlib drawing and figure planning they call live in `reporting/` (`figure_recipes.py`, `figure_summaries.py`, `figure_planning.py`, `group_plots.py`, `method_figures.py`) and never import `tracking/`. Failure diagnostics record typed context and the exact traceback in the run of the scope that failed, without replacing the workflow exception.
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
never by assignment identity, job title or class name. A POD method carries
its fit dataset's recorded presentation (never an id or path), which the renderer
appends once after ``|``. The plotting API takes the typed presentation, not
text: `platform/reporting/figure_text.py` renders legend entries, bar labels,
titles and subtitles at the moment a figure is drawn. Training and evaluation bar charts label each bar with `training_legend` (the rendered `TrainingPresentation`) through `plot_method_bars`; `plot_metric_comparison` only draws labels its caller rendered. `scope_label` renders a figure scope as one plain line for logs. `method_legend` renders
every method's compact presentation under the case's `NotationPolicy`; methods
whose texts would collide only because decimals were rounded are re-rendered
with more digits, and a method with no entry raises
`MissingMethodPresentationError` instead of falling back to its key. The length
limit (`LEGEND_PROFILE.max_length`) controls layout only: lines break at the
renderer's safe boundaries (`wrap_at_boundaries`). Every legend sits inside its
axes at matplotlib's `best` location (`figure_text.LEGEND_LOCATION`), in as many
columns as its lines need (`Legend.columns`); no fact is dropped. Result keys and plot styles stay
keyed by the stable result key. Horizontal metric plots place the first result
at the top, matching convergence-legend and configuration order while keeping
each value paired with the same result key. The non-method-major projection uses
the `PreparedComparisonPresentation` built at preflight, never the configs again.
Every comparison run writes two convergence figures with identical styling:
relative residual `||r||/||b||` (`plot_convergence_comparison`) and relative
energy-norm error `||e_k||_A/||e_0||_A` (`plot_error_convergence_comparison`,
reading `CGComparisonResult.error_history_a_rel`). A per-method history figure draws every
attempted system of the method's stored workload at the root report level. Its legend has one
entry per plotted dataset and RHS recipe pair, with only the varying dataset or recipe context
shown; matrix and RHS positions stay in the per-system records rather than repeating across an
oversized legend. The method residual figure draws a dashed
stopping-threshold line (`method_figures.residual_threshold`, from the stored `rhs_norm` and the
planned `rtol`/`atol`) only when every plotted system shares one threshold.
Condition-number figures and MLflow metrics are deliberately absent from the
comparison workflow because iterative spectral estimation can dominate or
stall otherwise healthy solver runs.
Plotting defaults keep markers compact so convergence and diagnostic figures
stay readable when several methods or dense prediction samples are shown.
Comparison-plot styling (`reporting/series_style.py`) resolves its
visual axes instead of deriving everything from one attribute: linestyle
and marker follow preconditioner family (lowest-cardinality attribute, matching
matplotlib's own default solid/dashed/dashdot/dotted order), while color
follows an explicit per-entry key when the composition layer supplies one
— e.g. POD comparisons key color on fit dataset
(`preconditioner_family.py::pod_color_key`) — falling back to
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
colour keys; callers that want a method to look the same whatever subset a figure shows pass the names
sorted) and `figure_writer.finish_group_figure` (title, subtitle, save, close). A preconditioner thus
looks the same in every plot. The history curves (`convergence_plots.py`) call `series_style.resolve_styles`
directly, and batch training and evaluation metric bars live in `metric_bars.py`. Their legends are rendered from the group's method presentations
(`GroupStyle.legend`) and append explicit success counts; a `GroupStyle` (`group_style.py`) is the fixed
title, the typed `scope` (rendered as the subtitle), the figure's `setup_sharing` record, the legend and the series
style maps. `GroupStyler.style` copies `setup_sharing` from the plan's metadata, the record the card stores as `r_m`,
and the subtitle adds `figure_text.setup_sharing_note` of it under the scope: for every `online_s` figure (bars, ECDF,
performance profile, paired scatter, speed ratio, means, distribution) "each matrix's setup is shared by its r_m
planned RHS, r_m = 4", with the distinct values ("r_m = 2 or 4 (varies by matrix)") or, past four values, their range
when matrices differ, and nothing for a metric that shares no setup, so the subtitle (drawn when `show_titles` is on) and the card's `setup_sharing` record cannot disagree. The per-run convergence plots also come in solve-only variants
(`solve_only=True`) whose curves all start at t=0. `MLflowReportSink.report`
takes the aggregated group plus a `GroupPresentation` and logs it into the nested run of the
group's report scope; figure file names are slugs of the rendered scope (see the figure-card
paragraph below), never group ids.

Report figures are drawn inside `figure_style.paper_style()`, one `rc_context` (seaborn `paper` context and `ticks` style, serif font, top and right spines removed). Every font size and line width is a fixed ratio of `BASE_FONT_SIZE` (9 pt): ticks, legend and subtitle at the base, axis labels one point above, the title two; strokes are multiples of a 0.5 pt unit, `THIN_LINE_WIDTH` (grid, edges) 0.5, `AXES_LINE_WIDTH` (spines, ticks) 0.8, `REFERENCE_LINE_WIDTH` (thresholds, diagonals, whiskers, outlines) 1.0, `DATA_LINE_WIDTH` (series) 1.5 and `EMPHASIS_LINE_WIDTH` (median ticks) 2.0, so drawers name a weight and never a number. It is entered only by the code that calls a drawer: `group_plots.write_outcome_plots` around `recipe.draw`, `method_figures.draw_method_figure`, and `composition/comparison/postprocessing/system_figures.write_system_figures`; drawers never enter it, and an architecture test allows `rc_context`, `set_theme` and `style.use` only in `figure_style.py`. `figure_style.rotate_x_labels(ax, degrees)` rotates x labels anchored at their right end. The settings are `FigureStyleConfig.label_rotation_deg` (`platform/config/models/figure_style.py`; `[comparison_baseline.figure_style]`, default `DEFAULT_LABEL_ROTATION_DEG` = 30, validated 0 to 90), `min_box_samples` (default `DEFAULT_MIN_BOX_SAMPLES` = 5) and `min_violin_samples` (default `DEFAULT_MIN_VIOLIN_SAMPLES` = 30), validated `1 <= min_box_samples <= min_violin_samples`; the section is optional like `[comparison_baseline.presentation]`: `composition/assignments/assignment_subjects.case_figure_style` reads it from the case, `MLflowReportSink` carries it to `GroupStyler`, and `GroupStyler.style` copies it into `GroupStyle.label_rotation_deg` and the two limits into `GroupStyle.row_limits` (`distribution_mode.RowLimits`), the only way a drawer sees them (the matrix heatmap's method labels and the x axis of every vertical figure use the rotation; distribution figures use the limits). `show_titles` (default `false`) is the fourth setting: `GroupStyler.style` copies it into `GroupStyle.show_titles`, every drawer passes it to `finish_group_figure`, `write_system_figures` takes it as an argument, and `draw_method_figure` takes it from the sink's `figure_style`, so the default draws every report figure (group, per-system and per-method) without suptitle or scope subtitle (a paper's caption carries them) and `true` restores both. The figure card keeps its `title` because it is the index record, not the drawing; the training and evaluation bar charts are outside the report and keep their titles. Sizes are the named constants of `figure_style.py` and `figure_text.py`: tick and legend entries 9 pt (`TICK_FONT_SIZE`, `LEGEND_FONT_SIZE`), axis labels 11 pt (`AXIS_LABEL_FONT_SIZE`), subtitle 9 pt, title 13 pt; drawers take their text sizes from these constants rather than literals.

Bar figures (means, time per iteration, workload sums, full-cost breakdown) are vertical: one column per method, method labels on the x axis rotated by `label_rotation_deg` through `rotate_x_labels`, the metric on the y axis, stacked segments stacked upward. The text above a bar ("k/n solved", workload counts, the cost total or "total unknown: ...") is a `figure_annotations.ColumnAnnotation` standing on the top of the column's highest mark (whisker end or point) and placed by `annotate_columns`, which wraps it to `COLUMN_TEXT_WIDTH_IN` (0.58 in, 8 characters at the 9 pt legend size, derived through `LABEL_CHAR_WIDTH_IN`) per line (the alternatives around " / " stay on one line) and extends the value axis from 0 through `value_axis_top`, whose headroom is the largest `base / (1 - lines needed)` over the annotations, so a long note over a short bar does not stretch the axis for the others; annotations never widen the figure. Distribution figures and the convergence-factor figure are vertical the same way, sized by `bar_columns_figure_size` and annotated by `annotate_columns`. `annotate_rows` serves only the remaining horizontal figures.

Every figure that draws summary marks explains them with one legend built by `mark_legend.py`: a drawer returns the set of `Mark`s it drew (`MEAN_BAR`, `RATIO_BAR`, `MATRIX_CLUSTER_INTERVAL`, `TWO_LEVEL_INTERVAL`, `VIOLIN`, `IQR`, `MEDIAN`, `MEDIAN_POINT`, `QUANTILE_WHISKER`, `MEAN_POINT`, `SYSTEM_POINTS`, `MATRIX_MEAN_POINTS`) and `draw_mark_legend` places one proxy artist per mark, in the order of `Mark`. A mark that was not drawn has no entry: a mean bar has an interval entry only when some method has an interval (`MIN_INTERVAL_MATRICES` matrices). An interval whisker is the mark of the bootstrap its metric is resampled by (`interval_mark(timed_values.bootstrap_kind_of(metric))`), so its text comes from that `BootstrapKind`. Legend text is plain language ("mean", "95% two-level interval" for a time or "95% matrix-cluster interval" otherwise, "density of systems" (only when a violin was drawn), "IQR (q25-q75)", "median", "whiskers within q05-q95", "systems", "matrix means") and never a count; counts are the "k/n" annotations above the columns. Per-system points are white with a dark edge (`mark_legend.SYSTEM_POINT_STYLE`), so they stay visible on any method colour; the convergence-factor figure keeps method-coloured points with a white edge, draws the mean as a white-filled ring and the median as a black diamond (`MEDIAN_POINT`), distinct from the interval caps.

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
`workload_bars.py` (workload sums), `paired_plots.py` (paired scatter, speed-ratio CDF); `figure_sizing.py` holds the sizes they share (`bar_columns_figure_size`: width follows the number of methods plus `rotated_label_overhang_in`, the reach of the leftmost rotated label past the first column, height adds `rotated_labels_height_in`; labels with no fact boundary are wrapped by `wrap_axis_label`, applied by `rotate_x_labels` and by the sizes, so both agree; `heatmap_figure_size` shares them; `paired_scatter_figure_size` keeps the paired scatter's axes square) and `stacked_segments.py` the segment key
shared with the cost breakdown. Planning (`figure_summaries.figure_summaries`)
computes, once per report, the frame every planned variant draws and stores (`summarize`, `ecdf_series`,
`performance_series`, `paired_systems`, `speed_ratio_summary`, `break_even_curves` or the workload sums, by the variant's `Statistic`;
at a mixture level only the profile, ECDF, pairing and speed-ratio frames are computed, once on the pooled `observations` of all the group's systems (the ECDF and profile denominators stay every planned system, so failures keep the plateau below 1), every raw-scale frame being class-level only, and
the three reference figures only when the group's configured reference is among its methods). The sink wraps each
frame as a `FigureData` (the group's `observations`, the `summary` frame, the `GroupStyle`, the variant's `metric`
and the group's `ReferenceSelection`) and the kind's drawer in `group_drawers.py`, `(FigureData, Path) -> Path | None`,
hands it to a plot function; the drawn frame is uploaded beside the figure unchanged. A profile drawer draws the
curves from `summary` and reads the observations only for what the curve rows cannot carry (the planned count of a
method that solved nothing, the systems a performance profile excluded). Every plot function takes a required
`save_path` and returns it, or `None` when nothing was drawn, and a `GroupStyle` (fixed title, typed scope,
notation, legend and the family/colour/marker mappings). A failed run is never given a time:

- `plot_workload_bars` (`workload_solve_bars_*`, `workload_online_bars_*`, `workload_online_amortized_bars_*`):
  one stacked workload sum per method from the workload table (names from the cost vocabulary of
  `domain/solver/README.md`), the `WorkloadTotal` (`workload_solve_s`, `workload_online_s` or
  `workload_online_amortized_s`) read from the variant's metric by `workload_bars.workload_total`, and the y label
  from `workload_ylabel`. `workload_segments` names the columns stacked: `workload_converged_solve_s` then
  `workload_failed_solve_s` (hatched) for solve time, preceded by `workload_setup_s` (every pair charged its
  matrix's setup) or `workload_setup_amortized_s` (one setup per matrix) for online time. Setup is its own segment
  because one setup event serves every RHS of a matrix, converged or not. The label beside each bar
  (`workload_bar_label`) reads "k/n converged" of the planned RHS, followed by "a/n attempted" when fewer were
  attempted, so a method that fails fast is not read as cheap. These are sums, not per-pair shares.
  Numeric bar positions pair each bar with its method; the legend guarantees distinct texts.
- `plot_paired_time_scatter` (`paired_online_scatter_*`, `paired_online_amortized_scatter_*`): draws a
  `paired_systems` frame of the variant's metric (online time per pair, or its amortized variant) of each method against the reference on equal log-log axes, labelled with the catalog label of the
  metric, with the diagonal; `paired_log_axes` spans the positive times with a factor `FAILED_AXIS_MARGIN` either
  side, and hollow markers (not clipped, as they sit on the axis limit) sit on the upper edge of the axis of the run that failed (`method_only` and
  `reference_only` rows). A system neither side solved is not in the frame, and a time of 0, which a log axis
  cannot place, is not drawn.
- `plot_performance_profile` (`performance_profile_solve_*`, `performance_profile_online_*`,
  `performance_profile_online_amortized_*`): Dolan-Moré profiles for solve-only, online and amortized online time per pair
  (Math. Program. 91, 2002), drawn from `performance_series`. When systems were left out because the
  best value on them is 0, the subtitle, when `show_titles` is on, ends with "N systems excluded: best value is 0"
  (`excluded_zero_best_note`), and the card always records the count as `excluded_zero_best`; the legend sits outside right and each curve's legend is the method name alone, the solved and planned counts being in the card and table.
- `plot_ecdf_profile` (`ecdf_iterations_*`, `ecdf_solve_*`, `ecdf_online_*`, `ecdf_online_amortized_*`): unnormalised
  iteration, solve-only, online and amortized online ECDFs
  from `ecdf_series` over an absolute budget (Hansen et al. 2016, COCO), not the n_p+1-normalised
  Moré-Wild data profile. Step curves share one right edge so each plateau is
  its success rate.
- `plot_speed_ratio_cdf` (`speed_ratio_online_cdf_*`, `speed_ratio_online_amortized_cdf_*`): ECDF of per-system reference/method
  ratios (`speed_ratios`) with a line at 1; the legend (`speed_ratio_legend`, from the
  `speed_ratio_summary` row) gives the median and its two-level bootstrap interval (matrices, then timing repeats), for example "median 1.23 [0.98, 1.50]"; the x label names the metric and the reference once ("Speed ratio of <metric label>, <reference> (reference) / method"); the legend sits outside right; the system and matrix counts the interval rests on are on the card. Reference-only and method-only counts are logged separately; no Wilcoxon
  p-value is reported.
- `seaborn_cost_curves.plot_break_even_curves` (`breakeven_online_amortized_*`, `RecipeProfileReport` only, skipped
  without a configured reference among the group's methods; multi-RHS online cost only, so it excludes the
  offline stages the `full_cost_breakdown` figure adds): the cumulative cost of solving $n$
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
- `breakeven_full_plot.plot_breakeven_full` (`breakeven_full_*`, matrix set and RHS type level only,
  no configured reference needed, not drawn when no method has a known offline cost): one row per
  method $m$ with a known offline cost against every other method $h$ whose offline cost is known
  or does not apply, a dot at $N^\star$ in pairs and a whisker over the finite part of its
  interval, with the state beside it ("cheaper from N* = ... pairs", "cheaper until N* = ...
  pairs", "always cheaper (...)", "never cheaper (...)" or "no pair both converged") and then
  "cheaper from: 95% CI [low, high]" when $C_m \ge C_h$ ("inf": never) or "cheaper until: 95% CI
  [low, high]" otherwise ("inf": always), drawn from the `breakeven_full` table
  (`BREAKEVEN_FULL_SCHEMA`, `domain/solver/README.md`). Observation unit: the pair, charged its
  matrix's whole setup. Population: the pairs both converged. Failure treatment: omitted. The card's
  `[interval]` is `kind = "per_method_pair"` with one `[[interval.pairs]]` entry per row (a method
  entry's fields plus `reference_method`), each the two-level bootstrap over that pair's matrices
  or `too_few_matrices`.
- `seaborn_cost_curves.plot_convergence_factors` (`convergence_factor_*`, matrix set and RHS type level only): the
  residual reduction factor per iteration $\gamma_{p,s}$ of the metric catalog
  (`CONVERGENCE_FACTOR`, read from each system's stored `convergence.parquet` history into the
  observation view), one vertical column per method (labels rotated like the bars), read from its
  `COHORT_SUMMARY` table (`by_cohort` of `summarize`, the group's one recipe). Observation unit:
  the system. Population: own (every system of the method with a factor). Statistic: each row gives
  the drawn mean over systems with its 95% matrix-cluster bootstrap whisker, the factor being no time (the card's
  `[interval]` is `intervals_of_means`) and the median diamond ($q_{0.5}$); seaborn strips the
  per-system points, which are the observation rows the summary was taken over and are not
  stored again. Time scenario: none, it is not a time, so the card records no timing evidence.
  Failure treatment: omitted; only `converged` systems with at least one stored iteration and a
  nonzero initial residual have a factor, and the card's outcome counts state how many of each method's
  planned systems do, and the axis label is the method name alone. The legend is the mark legend
  (see below) of the marks drawn. The card cites
  `convergence.parquet` among the sources.

`seaborn_means.py` draws the bar figures of means and of time per iteration with
`sns.barplot(..., errorbar=None)` through `seaborn_figures.plot_frame`: seaborn's own error bars
resample rows independently, wrong when the RHS of one matrix share it, so every bar, whisker and
"k/n" annotation is read from the figure's cohort table (`bar_panel` turns the one recipe block it
holds into a `BarPanel`, each method a `MethodBar` whose `height` is a `Bar(value, Whisker |
NoWhisker)` or `NoBar` when nothing is defined, never a zero bar). A figure has one axes; there are
no side-by-side panels. The count words (`count_label`: "solved", "set up", "returned an iterate") and the row annotations
(`annotate_columns` for bars, `annotate_rows` for distribution rows) are in `figure_annotations.py`, and the observation rows behind a panel
(`cohort_systems`, `panel_points`) in `cohort_points.py`. The five points of each kind, with the formulas in
`domain/solver/README.md`:

| Kind (stems) | Observation unit | Population | Statistic (bar; whisker) | Time scenario | Failure treatment |
|---|---|---|---|---|---|
| `iteration_means` (`iteration_means_*`) | system | own | $\bar{x}^{\mathrm{sys}}_p$ (`mean_over_systems`); matrix-cluster interval `mean_over_systems_ci_*` | none (iteration count) | omitted: only `converged` systems enter, "k/n solved" from `n_defined`/`n_planned` |
| `time_means` (`time_means_{solve,online,online_amortized}_{own,common}_*`) | system | own, and common to every method of the group | $\bar{x}^{\mathrm{sys}}_p$; `mean_over_systems_ci_*` | $t^{\mathrm{solve}}$, $t^{\mathrm{online}} = t^{\mathrm{setup}} + t^{\mathrm{solve}}$, $t^{\mathrm{online,amortized}} = t^{\mathrm{setup}}/r_m + t^{\mathrm{solve}}$ | omitted: only `converged` systems, "k/n solved" |
| `setup_means` (`setup_means_*`) | matrix | own | $\bar{x}^{\mathrm{mat}}_p$ (`mean_over_matrices`); `mean_over_matrices_ci_*` | $t^{\mathrm{setup}}_{p,m}$ once per matrix | omitted: only matrices whose setup completed (`converged`, `unconverged`, `solver_error`), "k/n set up" over planned matrices |
| `time_per_iteration` (`time_per_iteration_*`) | system | own | $\Theta_p = \sum t^{\mathrm{solve}} / \sum k$ (`ratio_of_sums`); no whisker, per-system $\theta_{p,s}$ drawn as points from the observations | solve only | omitted: `converged` and `unconverged` systems with at least one iteration enter, "k/n returned an iterate" |

All four are drawn at the matrix set and RHS type level only (see "Report levels" below). Their cards state `[interval]` per method
(`figure_evidence.intervals_of_means`): each method's `matrices` is the number of distinct matrices
that give that method a defined value, the clusters `summarize` resamples for its whisker, so it
equals `n_matrices` of the method's row in the figure's table; below `MIN_INTERVAL_MATRICES`, where
that method has no whisker, its entry is `too_few_matrices`. `time_per_iteration` records
`not_drawn_by_kind`.

**Report levels: raw-scale aggregates at one class, scale-free views pooled.** A number that
pools several RHS classes or matrix sets depends on the arbitrary mixing weights (Dolan & Moré
2002; COCO, Hansen et al.), so `cohort_variants.py` (pure) gives every `TableKind` one of two
closed `CohortSplit`s, matched exhaustively, and `cohort_level` reads the group's label:
`CohortLevel.RECIPE` for a matrix-set-and-RHS-type label, `MATRIX_SET` or `ALL_MATRIX_SETS`
otherwise. A reported mixture group pools several recipes or several matrix sets: one whose systems
are exactly a narrower group's is never reported (`report_all_levels`).

| `CohortSplit` | Tables | Matrix set and RHS type | Matrix set, all matrix sets |
|---|---|---|---|
| `CLASS_LEVEL_ONLY` (raw scale) | `summary`, `workload`, `cost_curves`, `cohort_summary`, `cohort_ratio_of_sums`, `matrix_breakdown`, `full_cost_breakdown`, `breakeven_full` | one figure, `<stem>_<slug>`, `WholeGroup` | none |
| `POOLED_AT_MIXTURE` (scale-free, per pair) | `series`, `paired`, `speed_ratio` | one figure, `<stem>_<slug>`, `WholeGroup` | one figure pooled over every system of the group, `<stem>_<slug>__mixture`, `MixtureCohort` |

`drawn_variants` skips a kind its level does not draw, so its frame is neither summarized, planned
nor carded, and a mixture level's run holds only its pooled figures (at least the ECDFs and
profiles, so no reported mixture run is empty). A pooled figure's card `cohort` names what it pools
(`cohorts.mixture_label`: "equal-weight mixture of <recipe keys>" for several recipes, "pooled over
<k> matrix sets" for one recipe over the card's several `dataset_ids`, both joined by "; " for
several of each; each pair counts once, whatever its class), and the decoder rebuilds that exact
string from the card's recipes and dataset ids; its subtitle line (drawn when `show_titles` is on,
`figure_text.cohort_note`) names the rendered recipes the same way, and its table is its own series, pairs or ratio frame of the pooled systems. A recipe's
pairs and curves are unaffected by the other recipes' systems because system ids are unique per
session and every frame is keyed by `(dataset, system_id)`; the speed-ratio interval clusters by
`(dataset, matrix_id)`, so recipes sharing a matrix share one cluster. The cohort tables
(`TableKind.COHORT_SUMMARY`, `TableKind.COHORT_RATIO_OF_SUMS`) keep the statistic's schema behind
a leading `cohort` column naming the group's one recipe
(`domain/solver/reporting/cohorts.py::by_cohort`). Every figure has its own card and sibling table
`<its stem>.parquet`; `table_of(stem, kind)` and the uploader's stem-collision check work on the
full stem, and a stem is mapped back to its variant through the card's kind, metric and
population, never by parsing the stem. `group_plots.write_outcome_plots` hands each drawer the
group's observations and its whole frame.

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
`[interval]` `kind = "not_drawn_by_kind"` and its `metric_definition` says so. Mixture levels draw
no distribution, like the mean bars. The heatmap's rows are the matrices of a
recipe level (labelled `matrix <index>`; the dataset is the figure's scope) and its columns the
methods; `sns.heatmap` colours the cells from the pivoted `MATRIX_BREAKDOWN_SCHEMA` table, and every
cell's annotation is read from its row (`heatmap_cells`, `cell_label`): the mean, "k/r" under it
when only some RHS converged, or a hatched cell labelled "0/r solved" when none did. The figure grows
in height with the number of matrices.

| Kind (stems) | Observation unit | Population | Statistic (marks) | Time scenario | Failure treatment |
|---|---|---|---|---|---|
| `iteration_distribution` (`iteration_distribution_*`) | system; points are per-matrix means | own | quantiles $q_{0.05}, q_{0.25}, q_{0.5}, q_{0.75}, q_{0.95}$ of `SUMMARY_SCHEMA` (`Statistic.QUANTILES`); violin density of $k_{p,s}$ | none (iteration count) | omitted: only `converged` systems enter, "k/n solved" |
| `time_distribution` (`time_distribution_{solve,online,online_amortized}_*`) | system; points are per-matrix means | own | the same quantiles of $t^{\mathrm{solve}}$, $t^{\mathrm{online}}$ or $t^{\mathrm{online,amortized}}$ | $t^{\mathrm{solve}}$, $t^{\mathrm{setup}} + t^{\mathrm{solve}}$, $t^{\mathrm{setup}}/r_m + t^{\mathrm{solve}}$ (the amortized card lists `r_m`) | omitted: only `converged` systems, "k/n solved" |
| `full_cost_breakdown` | full cost: offline generation, fit, setup charged to each pair, solve | own | pair workload sum per method, offline stages stacked on setup and solve (`Statistic.FULL_COST`, `build_full_cost_breakdown_table`) | none: failed attempts keep their time; an unavailable offline stage draws no segment and the bar states that the full cost is unknown |
| `matrix_heatmap` (`matrix_heatmap_iterations_*`) | matrix (one cell per method and matrix) | own, per cell | $\bar{x}_{p,m}$, the mean of $k_{p,s}$ over the matrix's converged RHS (`Statistic.MEAN_PER_MATRIX`, `matrix_breakdown`) | none (iteration count) | omitted from the cell mean, "k/r" when some RHS failed; a cell with none converged is hatched and labelled "0/r solved" |

The distributions write `TableKind.COHORT_SUMMARY` and the heatmap `TableKind.MATRIX_BREAKDOWN`;
both are drawn only at the matrix set and RHS type level.
Iterations are the heatmap's only metric: they are what differs between matrices for a
preconditioner, while per-matrix times are shown by the time distributions' points.

`MLflowReportSink` logs planned, attempted, and converged counts, the iteration statistics and the
failure-aware paired fields, each into the nested run of the group's report scope. The paired fields
come from `speed_ratio_summary` of the recipe report's `observations` against the configured
reference on `REFERENCE_METRIC`, the online time per pair (nothing is logged without a configured reference among the
group's methods), under `<metric>/<method>` keys. The iteration statistics are `iterations_converged_mean`, `iterations_converged_median` and `iterations_converged_std` (`metrics_logging.metric_key`, which names a statistic after the outcomes the metric catalog defines the metric on, so the key says only converged systems entered it) next to `iterations_n_defined` and `iterations_n_planned` (`count_key`). The paired fields are: `n_both_solved`, `n_reference_only`,
`n_method_only`, `n_ratio` (both-solved systems with a positive ratio), `n_matrices` (distinct
matrices among them, the cluster count the interval rests on), `speed_ratio_median`,
`speed_ratio_ci_low` and `speed_ratio_ci_high`; a null statistic is not logged. The `ReportSink` port (`composition/comparison/postprocessing/ports.py`) declares `register_method_figures` beside `report`, `report_system` and `finalize`. Its typed plans (`FigurePlan`: editorial title plus the figure's metadata) are also the single source of
figure scope, stable filenames, metric and statistic, and card
metadata. `figure_recipes.py` owns the recipes and their variants and lookups, and `drawn_variants` is the one decision of which group figures a report gets: a kind its report level does not draw (`cohort_variants`) is left out, and a variant whose statistic `compares_to_reference` is left out when the group has no configured reference among its methods, so it is neither planned, carded nor summarized; `figure_summaries.py` computes the frame of each variant `drawn_variants` gives (one exhaustive `match` over `Statistic`); `figure_planning.py` builds the plans, card metadata and `table_of`; `cohort_variants.py` gives each kind the figure its report level gets, or none. `figure_recipes.FIGURE_RECIPES` holds one recipe per `FigureKind`, the single registry a new figure is added to: a `GroupFigureRecipe` (the kind's `FigureVariant`s, failure treatment, its `draw` function, a public function of `group_drawers.py` named after the figure (`group_drawers.speed_ratio_cdf`; a kind with a pair and an `_amortized` variant has one drawer that reads the metric from the variant) and the `TableKind` it writes, which also decides the report levels it is drawn at) or a `HistoryFigureRecipe` (one `HistoryVariant`; drawn from a stored method by `method_figures.METHOD_PLOTS`). A `FigureVariant` is one figure of a kind: file stem, title, metric definition and the `metric` (`MetricId`), `population` (`Population`) and `statistic` (`Statistic`: workload sum, ECDF, performance profile, paired values, speed-ratio CDF, mean over systems, mean over matrices, ratio of sums, cumulative cost, mean and median, quantiles, mean per matrix, full cost, offline break-even; the values of the bar statistics are the table columns their bars are read from) it draws; the variants of a kind differ in metric or population (`time_means` has both populations of each time; the online kinds have a pair and an `_amortized` variant), which is how `GroupFigureRecipe.variant(metric, population)` selects one and how `write_outcome_plots` hands the right metric to the kind's drawer (it draws each planned figure, whose summary planning guarantees, and stages it with that summary as its table). `TableKind` (`summary`, `series`, `paired`, `speed_ratio`, `workload`, `cohort_summary`, `cohort_ratio_of_sums`, `cost_curves`, `matrix_breakdown`, `full_cost_breakdown`, `breakeven_full`) names the table a recipe writes and `figure_metadata.TABLE_SCHEMAS` maps each kind to its columns and dtypes, taken from the domain schema of the frame it stores. Metadata is `GroupFigureMetadata` (cohort, metric, population, statistic, failure treatment, `GroupAggregationRule`, and the evidence records of `figure_evidence.py`: `interval`, `setup_sharing`, `outcome_counts`, `timing`) or `MethodFigureMetadata` (`MethodAggregationRule`; no metric or failure treatment), sharing `TimingPolicy`, `DatasetId` dataset ids and `RunId` sources. The evidence fields are closed variants (`figure_evidence.py`), present on a card only when they apply, except `[interval]`, which every group card has: `kind = "not_drawn_by_kind"` when the statistic never draws one, `kind = "per_method_pair"` for the offline break-even (one `[[interval.pairs]]` entry per ordered method pair, a method entry's fields plus `reference_method`), else `kind = "per_method"` with one `[[interval.methods]]` entry per method in name order, because each method's interval resamples only that method's matrices (the speed-ratio CDF's compared methods, `intervals_of_speed_ratios`; the whiskers of a mean figure and of the convergence factors, `intervals_of_means`, above). An entry has `method`, `kind` and `matrices`, equal to `n_matrices` of that method's row in the figure's table: a drawn interval's `kind` is its `BootstrapKind`, `two_level_bootstrap` for a metric in seconds (matrices, then each event's timing repeats) and `matrix_cluster_bootstrap` otherwise (`timed_values.bootstrap_kind_of`), and adds `resampled` (the levels in words, "two-level bootstrap (matrices, timing repeats)" or "matrix-cluster bootstrap (matrices)"), `level` and `resamples`; `too_few_matrices` adds `minimum` (fewer than `MIN_INTERVAL_MATRICES` matrices, so that method has no interval; `domain/benchmarking/uncertainty.py::cluster_bootstrap` is the one decision, and the list is empty only for a cohort with no systems; the decoder rejects a value out of range or an entry whose kind its count contradicts; it checks the card's own shape and does not re-read the sibling table, whose `n_matrices` the writer took the counts from); `r_m` lists the distinct planned RHS counts per matrix and exists only for a metric that divides a matrix's setup over them (`online_amortized_s`, the `SetupSharedOverRhs` variant); `[outcome_counts]` is `summary.outcome_counts` per method; `[timing_spread]` holds, per method, `median_spread` (the median relative spread of its timing repeats) and `n_diverged` (its events whose repeats gave different fingerprints, kept in every statistic) from `summary.timing_noise`, and exists only on a figure of a metric measured in seconds, with `timing_warning` added when the presentation's solver contexts show a single timing repeat, so the figure's times carry noise it cannot show. A group figure is uploaded as a `StagedTabledFigure` (figure plus its table frame), a history figure as a `StagedFigure` (its plan and the `FigureTarget` its files were written to; `FigurePlan.stem` and the target's stem must agree). Every figure (group, method and system) is uploaded with its **figure card**
(`reporting/figure_card.py`: `<stem>.toml` beside `<stem>.png`, same run directory, same stem) into the
run of its own scope; stems only need to be unique inside one run directory, so system figures keep the
short stems `residual_history` and `energy_history`. A card is the single record of a figure's parameters: `stem`
(the file stem its files share) and `files` (those files, one per `FigureFormat`; a card whose `files` are not the ones its stem implies is rejected), `kind`, `title`, `metric_definition`, `level`, `aggregation_rule`, `timing_policy`, and for group figures `metric`, `population`, `statistic`, `failure_treatment`, `[outcome_counts]` (per method, the number of its planned systems in each `Outcome`), `table` and `table_kind`, and, only where meaningful, `[interval]`, `r_m` and `[timing_spread]` with `timing_warning`, `[scope]` (encoded plus `display`), `[[methods]]` (method id, encoded
presentation, config digest, and for a method with a seed fact `setup_seed`, stating that each matrix's setup seed is derived from that seed), `[counts]`, `[reference]`, `[[parameters]]` (one table per solver context:
tolerances, iteration limits, timing `timing_repeats`, `timing_warmup` and `timing_statistic`, `system_scaling` as a table (the scaling's `model_dump`) and `rhs_scaling`) and `[source]` (the session run id; `[source.environment]`, the device, CPU model, logical CPUs, torch threads,
torch and torchalg versions, Python version and the neuralls and torchalg git revisions of the session the timings
were taken in, each `unknown` when not recorded; and, per method child run, the artifact paths its numbers came from). `FigureCard` is the validated immutable value;
`write_figure_card`, `read_figure_card` and `validate_figure_card` round-trip it. Group figures (`figure_metadata.GROUP_FIGURE_KINDS`, every kind but the two histories) also carry a **table** `<stem>.parquet` in the same directory: the frame the figure draws (`SUMMARY_SCHEMA` rows for the bars of converged counts, `PROFILE_SERIES_SCHEMA` rows for ECDFs and performance profiles, `PAIRED_SCHEMA` rows for the paired scatter, `SPEED_RATIO_SCHEMA` rows for the speed-ratio CDF, `COST_CURVE_SCHEMA` rows for the break-even curves, `BREAKEVEN_FULL_SCHEMA` rows for the offline break-even, `COHORT_SUMMARY_SCHEMA` rows for the mean bars, the convergence factors and the distributions and `COHORT_RATIO_OF_SUMS_SCHEMA` rows for time per iteration (both behind a `cohort` column naming the group's one recipe), `MATRIX_BREAKDOWN_SCHEMA` rows, exactly the cells drawn, for the matrix heatmap, and for the sum bars `reporting/figure_table.py::build_group_table` rows of workload sums (`WORKLOAD_SCHEMA`, names from the cost vocabulary of `domain/solver/README.md`) with `method_id`, `n_planned_rhs`, `n_attempted_rhs`, `n_converged`, `workload_setup_s`, `workload_setup_amortized_s`, `workload_converged_solve_s`, `workload_failed_solve_s`, `workload_solve_s`, `workload_online_s`, `workload_online_amortized_s`; converged plus failed is the solve sum, and adding either setup gives the matching online sum, and for the full-cost breakdown `build_full_cost_breakdown_table` rows with `method_id`, `generation_s` and `fit_s` (null unless the stage's status is `known`, with `generation_status` and `fit_status` of `not_applicable`, `known` or `unavailable`), `dataset_generation_s`, `generation_share` and `generation_folds` (the whole generation, the share of it charged as `generation_s`, and the folds of the pinned split that share covers: `train` for a POD fit inside the comparison, `train_and_validation` for neural training; null with `generation_s`), `fit_origin`, `workload_setup_s`, `workload_solve_s` and `full_s` (`full_cost_seconds`, null with `full_status = offline_unavailable` when an applicable stage is unavailable, never a full cost that reads it as zero)), so a paper table and the figure cannot disagree. The card names it in its `table` and `table_kind` entries; the recipe alone decides whether one exists and of which kind (`figure_planning.table_of` returns `SiblingTable(name, kind)` or `NoTable`, so the card cannot disagree), the uploader refuses a table whose schema is not its kind's (`figure_table.check_table_kind`), `read_group_table(path, kind)` reads a stored table back only as that kind and raises `InvalidFigureTableError` otherwise, and `verify_figure_records` additionally requires a parquet for exactly the figures whose card names one. `tracking/scope_params.py::scope_params` logs each non-root scope run's identity as MLflow params (`report_level`; `dataset_id` from matrix-set level down; `recipe_key` (the lossless `recipe_key`) from cohort level down; a system has no run, so no params), and the report workflow logs `report_level` on the root run; solver settings and anything else varying between figures of one run stay in the cards. `figure_index.toml`
(`reporting/figure_index.py`, written once to the root run's artifact root in `finalize`) lists only
`run_id`, `artifact_paths` (every file of the figure), `card`, `kind`, `level` and `scope` per figure, derived from the cards, and cites in a top-level `splits` key the `splits.toml` copied beside it (`FigureUploader.write_index` uploads both, so train/test separation is checkable from the report's own data; `validate_figure_index` rejects an index without the citation); before it is
written `verify_figure_records` raises unless every uploaded figure file belongs to a figure that has all its format files and a same-stem card naming that stem, and the index
lists exactly the uploaded figure files, each entry's files being the ones its card names. The card stores each scope as keys and typed values (`facts_codec.encode_figure_scope`, whose recipe is `encode_recipe`: kind, parameters and mask) beside its plain rendered `display` text; file names are slugs of rendered text, never ids: a group figure `{kind}_{slug}` (written as `.png`) takes `slug` from the plain rendered scope (`scope_label`, which prefixes dataset, recipe, and system scopes with "Comparison matrix set") and a method figure `residual_history_{slug}` from the rendered method label (`POD-CG (c=10, k=20, N=4500)` becomes `POD-CG_c_10_k_20_N_4500`); `figure_metadata.FigureSlugs` allocates one stem per subject (<= 80 characters) and suffixes `_2`, `_3` in encounter order when two subjects render alike. Machine ids live only in metric keys (`{metric}/{method}`, one run per report scope; a system's on its RHS-type run as `system/<system id>/{metric}/{method}`, its figures under `systems/<system id>/`, the planned system id made MLflow-safe) and in `*_id` fields (`dataset_ids`, encoded `recipes`, `method_id`; `[reference]` holds the selection kind plus the method id and reason when it has them). Card counts retain every dataset, recipe population/weight, and
the actual solver, scaling, and timing-repeat settings; method-child
history figures are registered with unique method-qualified ownership. Raw
benchmark events and method summaries become
the persistence source of truth; the removed scaling-law TOML is not emitted.

**Figure kinds.** Every group figure answers five questions, which its card records and which `FIGURE_RECIPES` fixes per kind; formulas and the per-metric failure rule are in `domain/solver/README.md`, "Observation view and metric catalog". Observation unit is a system unless stated; "own" population keeps the systems on which each method's metric is defined, "common" those defined for every method. Online time charges a pair its matrix's whole setup, $t^{\mathrm{setup}} + t^{\mathrm{solve}}$; online amortized shares it, $t^{\mathrm{setup}}/r_m + t^{\mathrm{solve}}$; solve ignores it (names from the cost vocabulary of `domain/solver/README.md`).

| Kind | Metric (time scenario) | Population | Statistic drawn | Failure treatment |
|---|---|---|---|---|
| `solve_time`, `online_time` | solve; online, online amortized | own | workload sum over planned pairs, stacked setup, converged and failed solve time with k/n | none: failed attempts keep their time; a failed setup is charged to each planned pair of its matrix |
| `paired_time` | online, online amortized; method against reference | own | per-system pairs, log-log | censored on the axis for an asymmetric failure; both-failed pairs omitted |
| `speed_ratio` | online, online amortized; reference/method | own | CDF of the ratio with a two-level interval | omitted |
| `performance_profile` | solve, online, online amortized | own | profile series | censored |
| `ecdf_iterations`, `ecdf_time` | iterations, solve, online, online amortized | own | ECDF with failures at infinity, so the plateau stays below 1 | censored |
| `breakeven` | online amortized (setup and solve means) | common | cumulative cost against the RHS count | omitted |
| `iteration_means`, `time_means` | iterations; solve, online, online amortized | own and common | mean over systems with a matrix-cluster interval (iterations) or a two-level one (times) | omitted; the card states k/n |
| `setup_means` | setup (one value per matrix) | own | mean over matrices | omitted |
| `time_per_iteration` | solve time / iterations | own | ratio of sums | omitted; unconverged runs with a real iteration count enter |
| `iteration_distribution`, `time_distribution` | iterations; solve, online, online amortized | own | points, box (quartiles, whiskers inside q05-q95) or violin by sample size, a distribution summary and not a confidence interval | omitted |
| `matrix_heatmap` | iterations | own | mean per matrix | omitted: a cell with no converged RHS is hatched "0/r solved" |
| `convergence_factor` | residual reduction per iteration | own | mean and median | omitted |
| `full_cost_breakdown` | online plus offline generation and fit | own | pair workload sum per method, offline stages stacked on setup and solve | none: failed attempts keep their time |
| `breakeven_full` | offline difference over the online saving per pair, every ordered method pair | common to the pair | $N^\star$ with a two-level bootstrap interval | omitted |

**Full-cost breakdown.** Offline cost is a property of the method artifact, so the figure reads it from each method's stored `summary.toml` (`StoredMethod.offline`, passed to `figure_summaries`, which the offline break-even reads too) and the online segments from the report group's workload summaries; at a sub-workload level the full offline cost stands beside only that level's online seconds, which is the cost of serving that workload from a fresh artifact. The figure is stacked vertical bars where colour encodes the stage, not the method: one fixed Okabe-Ito colour per stage (`cost_breakdown_plot.segment_color`; dataset generation, offline fit, matrix setup charged to each pair, solves), no hatches, and the method is the rotated x tick label; one legend of stages (`stacked_segments.segment_key` with the stage colour) sits inside the axes. The text above each bar is the full cost (or "full cost unknown: ...") and, on a second line, the fit origin from `OfflineStages.fit_origin()`: "fit job" (a historical or unavailable fit stage: the wall time of the MLflow run that produced the checkpoint), "fit in comparison, measured" (a POD basis fitted once inside the comparison before the timed setups; its measured seconds are the offline fit segment) or nothing (no fit at all). The table states the same in its `fit_origin` column (`none`, `inline`, `job`). Fit seconds are the measured POD fit or the wall time of the whole training job; `tracking/run_duration.py::run_wall_seconds_from_run` reads a record already fetched during method preparation, so cost attribution does not fetch the same run again (see `domain/solver/README.md`, "Cost vocabulary").

**Memory scope.** Peak memory is an operation- or method-lifetime reading, never a comparison across platforms. On CUDA it is PyTorch's allocator peak (`torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`) for the timed operation; it excludes the CUDA context, library workspaces and non-PyTorch allocations, so it is not total device memory. On CPU it is a process RSS high-water mark (`ru_maxrss`, `peak_wset` on Windows) read as a delta: the process has no reset, so it is a lower bound when an earlier workload set a higher peak (an event's CPU peak is the larger of its warm-up's and first timed execution's, since the warm-up raises the mark first), and a CPU peak must not be compared across methods. `shared/device.py` holds both readings.

**Run names.** `tracking/run_names.py::render_run_name` is the only place a typed
`RunName` (`domain/presentation/run_names.py`) becomes text: plain Unicode under the case's
notation, one line, no wrapping. `MlflowRunConfig` carries a `NamedRun`; the
text is produced where MLflow (or dlkit, which opens the training and evaluation runs) is
handed a name (`run_name_text`), so a run cannot be named from an id, a slug, a path or a
class name. `ComparisonRunName` renders its dataset as "Comparison matrix set: ..."
after any editorial title. A session root run renders as
`"{stage} · {YYYY-MM-DD HH:MM} · {revision}"`, the revision cut to 8 characters with
any `-dirty` suffix kept and omitted when unreported; a report root (`tracking/report_run.py::root_report_run`, opened under its session) renders the same with stage `report`; `render_session_title` renders
its scientific title for the `title` tag and run description, applying the
"Comparison matrix set: ..." prefix to a dataset title only for a comparison
session. Method-child names and assignment training/evaluation subjects keep
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
  counts are set only for `diverged`). `setup_events.parquet` carries the same agreement of
  each setup's timed repeats as `repeat_status` with `repeat_first_fingerprint` and
  `repeat_other_fingerprint` (the built preconditioner's fingerprint, set only for
  `diverged`). A table without them is rejected, so the reuse gate recomputes such a run.
- `planned_systems.parquet`: one row per planned system, including systems whose setup
  failed, with its dataset, recipe key, matrix and RHS ordinal, the solver and timing
  settings it ran under, and its reference state as flat `reference_kind`
  (`configured`/`none`/`unavailable`), `reference_method` and `reference_reason` columns.
  `event_frames.decode_planned_systems_frame` is the only place those columns become a
  `ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable`. These are compute
  provenance: a case config edited later would describe different numerics than the ones
  timed.
- `summary.toml` (counts, the `[online]` workload sums `workload_setup_s`,
  `workload_setup_amortized_s`, `workload_solve_s`, `workload_online_s` and
  `workload_online_amortized_s` of the cost vocabulary in `domain/solver/README.md`, memory, and the
  method's `[offline.generation]` and `[offline.fit]` stage tables, each a `status` of `not_applicable`, `known` (with `seconds`,
  `provenance` and, when recorded, `peak_bytes`) or `unavailable` (with a `reason` and no
  number); a known generation stores the whole generation's `dataset_seconds`, the `train_share`
  of them charged and the `fit_folds` (`train` or `train_and_validation`) that share covers, in
  place of `seconds`; `offline_cost_codec.py` is the one reader
  and writer, so a missing block or a known generation without its folds is an error and an
  unavailable stage is never a zero), `method_config.toml` (the reuse identity), and
  `presentation.toml` (the
  method's configured presentation, compact fact selection, per-system realized facts, and
  the presentation of each comparison matrix set it ran on, all through `facts_codec`: keys and typed values
  only). Its `[data_origin]` table records one closed status: `no_fit_data`, `inline_fit`,
  `checkpoint_verified` with the attested training-dataset digest, or
  `checkpoint_unavailable` with a reason. The strict reader rejects old or malformed evidence;
  method reuse compares it with the newly resolved origin before accepting a run.

It also draws the method's residual and energy figures, then logs reducer-owned scalars
(including the five workload sums under their `summary.toml` names, `iterations_total`, and
`generation_s`, `fit_s` and `offline_s` only for the stages whose cost is recorded) exactly once. The sink never opens an MLflow run;
`comparison_batch` owns the session parent and its one direct child per method.
The parent receives only shared provenance and cross-method reports, while raw
method artifacts remain on the child. Reuse requires phase
`comparison_method`, the method config/checkpoint
digest, complete workload membership, and all canonical raw artifacts; legacy
per-system runs are executed again. Reuse parses the required
configuration, planned systems, presentation and convergence tables and reconciles event
methods, membership, and summary totals before accepting a hit, and requires the `[offline]`
block with a known generation's charge, so a run stored without either is executed again; the session's figure
figure index is not part of the gate. The persisted configuration is the
canonical identity representation: checkpoint inputs are content digests and
runtime lease paths are omitted.

**Runtime environment.** `shared/runtime_environment.py` holds the frozen `RuntimeEnvironment` and its string codec (`encode_environment`, `decode_environment`), shared by the MLflow layer and the figure cards. `tracking/runtime_environment.py::collect_runtime_environment(device)` is the one impure reader (platform, os, torch, `importlib.metadata`, `git describe --always --dirty --exclude=*` of the neuralls checkout (a commit hash, never a tag name) and the commit in torchalg's `direct_url.json`; the CPU model comes from `/proc/cpuinfo`, then `platform.processor()`, and any fact the platform cannot report is the explicit `Unreported.UNKNOWN`, never a raise); `environment_params` flattens it to the stable `env.*` MLflow params `env.device` (CUDA device name or `cpu`), `env.cpu_model`, `env.cpu_logical_count`, `env.torch_threads`, `env.torch_version`, `env.torchalg_version`, `env.python_version`, `env.revision` and `env.torchalg_revision`; `read_environment_params` reads them back, with any missing key unknown, so a session stored before a fact was recorded stays reportable. `neuralls report` reads the session run's params and every figure card cites them in `[source.environment]`. They are constant for a session, so the comparison session run logs them once and no parquet table or per-row record repeats them. No dtype is recorded: the stored datasets are pinned to float64 by the readers and no session-level dtype setting exists.

The comparison session run also carries `splits.toml` (`tracking/split_records.py`): one table per dataset id with the pinned split's content digest, train/validation/test row counts, the matrix ids of the fit (train and validation) and test folds, and the separation (`single_matrix_shared`, whose one matrix is in every fold by design, or `matrix_disjoint`). `download_session_splits` reads it back and rejects a session that predates it. The session run writes it (`MlflowSessionRun.record_splits`); `neuralls report` reads it with `download_session_splits` and the report sink copies it to the root report run beside `figure_index.toml`.

A report reads a session back through two modules. `tracking/session_artifacts.py` picks the
newest FINISHED `session_comparison` run of the config's experiment (`latest_session_run_id`, no
path tag: the experiment already is the config) and finds
the session's FINISHED `comparison_method` children (by `mlflow.parentRunId`; the report root,
also a child of the session, has phase `report` and is never among them), checks each
holds the full artifact set, and downloads it; the reuse gate lists artifacts through the
same helper, so run discovery exists once. `reporting/stored_benchmark.py::read_stored_benchmark`
reads one `MethodDirectory` (local path, the method child run id and its `method_config_digest` tag, as `download_session_methods` returns them) per method into a `StoredBenchmark` of `StoredMethod`s (run id, config digest, parsed method
config, presentation, offline cost, planned systems, and the setup, solve and convergence tables) and
rejects anything inconsistent: a presentation of another method, events for systems that
were not planned, a summary whose digest differs from the run's tag, a method stored twice, or datasets presented differently by two methods.
The stored config holds content digests for checkpoint and dataset inputs, so a report-time
`pod_color_key` colour key groups methods by fit-dataset content, not by path.

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
Both DLKit entry points take the device from their caller: `DLKitAdapter(device)` and
`create_inference_predictor(checkpoint_path, device)` pass it to `load_model` and record it on
the predictor. DLKit never chooses a device (no `device="auto"`) and the adapters do not read
it back from the loaded parameters, so there is one decider, in composition. That decider is torchalg's
`resolve_device` (CUDA when available, else CPU), so an Apple MPS host runs the models on CPU,
where DLKit's own choice would have picked MPS; the predictors run in float64
(`PrecisionStrategy.FULL_64`), which MPS does not support, and the matrices, residuals and memory
observer were already on the composition device.

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
  assembly is `build_dense_manifest(...)`, which takes array shapes and the run's `DatasetScaling` (scaling, matrix norm, one `alpha` per matrix sample id). `dataset_readers.try_read_matrix_scaling(path, matrix_row)` maps the stored row to its `MatrixSampleId` through `matrix_sample_index` and returns that matrix's `PersistedMatrixScaling`; a dataset without `matrix_sample_index` raises `MissingMatrixSampleIndexError` (the canonical triplet reader likewise), so a row is never read as an id. `dataset_readers.matrix_row` finds a row's stored matrix copy from the copy count and `matrix_sample_index`, never from a layout flag: one copy per row is read at the row, one copy shared by rows bound to one id at 0, and any other combination raises `InconsistentMatrixStorageError` (stored copies, rows, distinct ids) instead of reading a wrong matrix. `read_dataset_manifest` reports an undecodable `scaling` or `provenance` block (a manifest from before system scaling) as `InvalidDatasetManifestError(dataset_dir, field)`, a `ValueError`, so reuse checks treat the dataset as stale and `--force` regenerates it.
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

`storage/manifest_io.py::load_dataset_manifest` and `storage/dataset_readers.py::load_matrix_dense_sample` read from disk on every call: no process-wide cache, so a read after a write or a regeneration in the same process always sees the files as they are now (a manifest is small JSON, and a dense sample is read once per caller).

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
`load_system_arrays`/`load_matrix_sparse_sample` is a different number, and the
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
