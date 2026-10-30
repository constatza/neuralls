# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, registry resolution, TOML loaders, lower-case job metadata readers, and the thin DLKit job loader adapter
- `config/models/`: Pydantic models for TOML sections. `comparison.py`'s RHS source models share `RhsSourceModelBase`, whose `mask: RhsMask` (`NoMask | RandomMask`, discriminated by `kind`) is available on every kind. The source models only hold configuration; `composition/comparison/recipes.py::rhs_recipe` turns one into its typed `RhsRecipe`. `OutputConfig.matrix_format` (`MatrixFormat`, default `csr`) selects the storage format of the system matrix for a dataset; it is the single declaration of that default. `OutputConfig.sparsity_pattern` (`SparsityPattern`, `shared` or `ragged`, default `ragged`) selects the CSR layout; it is part of the digest.
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `storage/array_store.py`: `ZarrArrayStore` and `Hdf5ArrayStore`, the dense backends of the `ArrayStore` protocol (defined in `domain/generation/ports.py`, so the domain depends on it without importing platform). Arrays are created at full shape; `close()` raises `ValueError` naming any array with unwritten rows. A one-dimensional array is a single chunk, so row kinds and matrix indices do not create one file per row.
- `storage/errors.py`: `_raise_storage_error`, the OSError translation every storage module shares.
- `storage/dense_stream.py`: streamed dense persistence for `zarr` and `hdf5`. `open_dense_array_store()` opens the store that the streamed writer fills, and `save_dense_stream_manifest()` writes the manifest last, from the shapes the writer reports.
- `storage/matrix_readers.py`: suffix-keyed registry `MATRIX_READERS` (`.npy`, `.txt`, `.npz`, `.mtx`, `.mtx.gz`) returning `SystemMatrix` via `read_matrix`; `to_dense` is the explicit densify helper. Unknown suffixes raise; `.npz` must hold a sparse matrix.
- `tracking/`: MLflow run helpers, naming/query policy, workflow topology resolution, client adapters, and best-effort blocking-failure diagnostics (`failure_reporting.py`). Failure diagnostics record typed context and the exact traceback in the active run without replacing the workflow exception.
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
reading `CGComparisonResult.error_history_a_rel`).
Condition-number figures and MLflow metrics are deliberately absent from the
comparison workflow because iterative spectral estimation can dominate or
stall otherwise healthy solver runs; that analysis remains available only as
an explicit standalone operation.
Plotting defaults keep markers compact so convergence and diagnostic figures
stay readable when several methods or dense prediction samples are shown.
Comparison-plot styling (`reporting/plots.py`) resolves three independent
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

The failure-aware functions in `outcome_plots.py` use that same resolution through
two public helpers in `plots.py`: `group_series_styles` (per-preconditioner
colour, marker and linestyle from the recorded families and colour/marker keys)
and `finish_group_figure` (title, subtitle, save, close). A preconditioner thus
looks the same in every plot. Their legends are rendered from the group's method presentations
(`GroupStyle.legend`) and append explicit success counts; a `GroupStyle` is the fixed
title, the typed `scope` (rendered as the subtitle), the legend and the series style maps. The per-run convergence plots also come in solve-only variants
(`solve_only=True`) whose curves all start at t=0. `MLflowGroupReportSink.report`
takes the aggregated group plus a `GroupPresentation` and names artifacts by the
sanitized group id.

The per-run work-precision diagram is solve-only: its x-axis is measured
`solve_time_seconds`, independent of whether historical offline generation cost
is available.

`outcome_plots.py` draws the per-system outcome plots from
`domain.benchmarking.profiles` and `domain.benchmarking.paired`; every function takes a
required `save_path` and returns it, or `None` when nothing was drawn, and a
`GroupStyle` (fixed title, typed scope, notation, legend and the family/colour/marker mappings). A failed run is never given a time:

- `plot_online_time_bars` (`solve_time_bars_*`, `online_time_bars_*`): method
  workload totals with separate solve-only and setup-plus-solve scenarios.
  Numeric bar positions pair each bar with its method; the legend guarantees distinct texts.
- `plot_paired_time_scatter` (`paired_time_scatter_*`): amortized online times of each method against the
  reference; hollow markers sit on the axis of the run that failed, and a
  reference run that solved without a comparable time is not drawn as failed.
  Zero-time solved coordinates remain valid on these linear axes.
- `plot_performance_profile` (`performance_profile_solve_*`, `performance_profile_online_*`): Dolan-Moré profiles for solve-only and amortized-online time
  (Math. Program. 91, 2002).
- `plot_ecdf_profile` (`ecdf_iterations_*`, `ecdf_time_solve_*`, `ecdf_time_online_*`): unnormalised iteration, solve-only, and amortized-online ECDFs
  over an absolute budget (Hansen et al. 2016, COCO), not the n_p+1-normalised
  Moré-Wild data profile. Step curves share one right edge so each plateau is
  its success rate.
- `plot_speed_ratio_cdf` (`speed_ratio_cdf_*`): ECDF of per-system reference/method
  ratios with a line at 1; the legend gives the median, its matrix-clustered
  interval, and the both-solved count. Reference-only and method-only counts
  remain available separately; no Wilcoxon p-value is reported.

`MLflowGroupReportSink` logs planned, attempted, and converged counts plus the
failure-aware paired fields. Its typed plans (`FigurePlan`: editorial title plus the `FigureMetadata` scope) are also the single source of
figure scope, stable filenames, timing scenarios, and manifest
metadata. The figure manifest stores each scope as keys and typed values (`facts_codec.encode_figure_scope`, whose recipe is `encode_recipe`: kind, parameters and mask) beside its plain rendered `display` text; file names are slugs of rendered text, never ids: a group figure `{kind}_{slug}.png` takes `slug` from the plain rendered scope (`scope_label`, e.g. `aggregate_bars_Gaussian_CG_Residuals_k_20_N_100_-_Gaussian.png`) and a method figure `residual_history_{slug}.png` from the rendered method label (`POD-2G (r=10)` becomes `POD-2G_r_10`); `figure_metadata.FigureSlugs` allocates one stem per subject (<= 80 characters) and suffixes `_2`, `_3` in encounter order when two subjects render alike. Machine ids live only in metric keys (`{metric}/{group_id}/{method}`) and in `*_id` fields (`dataset_ids`, `recipes` (each recipe's machine key), `reference_method_id`; `reference_selection` is the kind alone). Global metadata retains every dataset, recipe population/weight, and
the actual solver, normalization, and timing-repeat settings; method-child
history figures are registered with unique method-qualified ownership. Raw
benchmark events and method summaries become
the persistence source of truth; the removed scaling-law TOML is not emitted.

**Run names.** `tracking/run_names.py::render_run_name` is the only place a typed
`RunName` (`domain/presentation/run_names.py`) becomes text: plain Unicode under the case's
notation, one line, no wrapping. `MlflowRunConfig` carries a `NamedRun`; the
text is produced where MLflow (or dlkit, which opens the training and evaluation runs) is
handed a name (`run_name_text`), so a run cannot be named from an id, a slug, a path or a
class name. Tags keep machine ids in `*_id` fields (`method_id`, `assignment_id`,
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
  error).
- `planned_systems.parquet`: one row per planned system, including systems whose setup
  failed, with its dataset, recipe key, matrix and RHS ordinal, the solver and timing
  settings it ran under, and its reference state as flat `reference_kind`
  (`configured`/`none`/`unavailable`), `reference_method` and `reference_reason` columns.
  `event_frames.decode_planned_systems_frame` is the only place those columns become a
  `ConfiguredReference | NoReferenceConfigured | ReferenceUnavailable`. These are compute
  provenance: a case config edited later would describe different numerics than the ones
  timed.
- `summary.toml`, `method_config.toml` (the reuse identity), and `presentation.toml` (the
  method's configured presentation, compact fact selection, per-system realized facts, and
  the presentation of each dataset it ran on, all through `facts_codec`: keys and typed values
  only).

It also draws the method's residual and energy figures, then logs reducer-owned scalars
(including `iterations_total`) exactly once. The sink never opens an MLflow run;
`comparison_batch` owns the session parent and its one direct child per method.
The parent receives only shared provenance and cross-method reports, while raw
method artifacts remain on the child. Reuse requires phase
`comparison_method`, the method config/checkpoint
digest, complete workload membership, and all canonical raw artifacts; legacy
per-system runs are executed again. Reuse parses the required
configuration, planned systems, presentation and convergence tables and reconciles event
methods, membership, and summary totals before accepting a hit; the session's figure
manifest is not part of the gate. The persisted configuration is the
canonical identity representation: checkpoint inputs are content digests and
runtime lease paths are omitted.

A report reads a session back through two modules. `tracking/session_artifacts.py` finds
the session's FINISHED `comparison_method` children (by `mlflow.parentRunId`), checks each
holds the full artifact set, and downloads it; the reuse gate lists artifacts through the
same helper, so run discovery exists once. `reporting/stored_benchmark.py::read_stored_benchmark`
reads one directory per method into a `StoredBenchmark` of `StoredMethod`s (parsed method
config, presentation, planned systems, and the setup, solve and convergence tables) and
rejects anything inconsistent: a presentation of another method, events for systems that
were not planned, a method stored twice, or datasets presented differently by two methods.
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
