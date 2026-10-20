# Composition Module

The composition package owns wiring and config-driven assembly.

## Package Map

- `assignments/`: registry loading, assignment wiring, training, eval, comparison, inference
- `comparison/`: single-run comparison assembly around application/domain logic
- `generation/`: config-driven dataset orchestration, dataset persistence wiring, and default tracing services
- `inference/`: inference data-loading composition helpers
- `preconditioners/`: config-to-`torchalg` preconditioner factory wiring; one `(PreconditionerType, MatrixFormat)` table selects the dense or sparse CSR builder (Identity, Jacobi, ILU, IC0, AMG, ADAPTIVE_SA_AMG, BOOTSTRAP_AMG, NeuralAMG, Neural)
- `solvers/`: config/workflow-to-`torchalg` solver runner adapters; `run_traced_pcg` accepts a dense ndarray or a `csr_array` and converts CSR once into a sparse CSR tensor
- `tracking/`: tracking tag and run-spec assembly

## Semantic Difference

Composition is the only layer allowed to know both the abstract workflow shape
and the concrete adapter set needed to run it. If a module mostly wires config
models, ports, and runtime collaborators together, it belongs here. If it
starts doing reusable numerical work, it belongs in `domain`. If it starts
owning concrete IO or MLflow mechanics, that code belongs in `platform`.

Comparison references are explicit configuration. The case
`[comparison_baseline].reference_preconditioner` is inherited by each entry;
an override may set `general.reference_preconditioner`, including `null` to
clear the baseline. Composition validates that value against the final method
tuple after assignment filters and auto-generated methods are applied. If
runtime checkpoint resolution skips that configured method, reporting keeps
`ReferenceUnavailable` with the resolution reason and never chooses a fallback
from observed outcomes. Paired outputs use the reference/method speed ratio;
performance profiles still normalize independently by the fastest method on
each problem.

## Matrix format dispatch

`preconditioners/factory.py` selects a builder by `(PreconditionerType,
MatrixFormat)` from one module-level table. A missing pair raises `ValueError`
naming both halves; nothing falls back to another format. Every sparse-mapped
type has a CSR entry backed by the torchalg sparse sibling. The sparse AMG
preset supports aggregation coarsening only, so other coarsening kinds are
rejected for CSR input. Neural types have a dense entry only: CSR input is
densified in this layer, with one INFO log line per call stating the O(n^2)
cost, and the dense builder then runs. The predictor and its model-input
adapter never densify.

## Dense streamed dataset commit

`generation/dense_streaming.py` writes a dense dataset batch by batch into a sibling
staging directory, `<name>.partial`, through
`platform/storage/staged_commit.py::commit_staged_directory`, so the final name never holds
a partial dataset. `SampleWriter.finalize` closes the array
store (the HDF5 file handle, or the zarr group) before the commit runs, because Windows
refuses to rename a directory with an open file. The commit then renames the staging
directory to the final name with `os.replace`; only after that rename does `dataset_builder`
stamp the identity key and content digest into the manifest, so a crash before the stamp
leaves a dataset that reuse checks reject. Every rename goes through one helper that never
targets an occupied name and retries only `PermissionError` (transient Windows locks) a
bounded number of times, then reports the failure through `_raise_storage_error`.

When the final directory already exists (a forced regeneration), it is first renamed aside
to `<name>.old`, the staged directory is renamed into place, and the aside copy is deleted
last. If the staged rename fails, the aside copy is renamed back before the error propagates,
so the previous dataset is never lost. At the start of a commit, a leftover `.old` is
restored to the final name when the final name is missing (it is then the only copy of the
previous dataset) and deleted when the final name exists. Stale `.partial` directories are
removed before a new commit starts.

## Streamed dataset commit (the only generation path)

`dataset_builder.build_dataset` is the single generation entry point. Every build streams
batches from the generator into a staging directory and renames it to the final name only
after the manifest is written. Dense and CSR matrices, and zarr and hdf5 storage, all take
this path. Nothing buffers the whole dataset: no accumulator, no whole-payload build, no
second writer. The entry point refuses `npy` with a ValueError that names zarr and hdf5 as
the supported generation formats (`npy` datasets stay readable).

- `generation/dense_streaming.py` writes dense zarr and hdf5 datasets, batch by batch, through
  `SampleWriter`.
- `generation/csr_streaming.py` writes CSR datasets. The dense arrays (rhs, solutions, row kind,
  matrix sample index, parameters) use the same `SampleWriter`; the CSR matrix goes through
  `open_csr_matrix_stream` at the manifest's matrix address (`dataset.zarr/matrix` or the
  `matrix` group of `dataset.h5`).
- `generation/streamed_write.py` holds what both of the above share: planning the batch
  stream and rejecting an inexact row count (`open_prepared_stream`), closing a
  partially-written store without masking the real error (`release_after_failure`), and
  building the manifest facts common to both formats except `layout` (`manifest_facts`).

The CSR layout is the configured `[output].sparsity_pattern`, passed through `process_config`,
`_execute_plan` and the executors into `build_dataset`. `shared` maps to `SHARED_PATTERN` (one
indptr/indices, data of shape (N, nnz)); the stream raises on the first sample whose pattern
differs from sample 0, before the dataset is committed. `ragged` (the default) maps to
`MANY_MATRICES`. The layout is never inferred from the data, so the generation path never calls
`choose_layout`. The manifest is written by `save_csr_stream_manifest`; for zarr the shape and
layout come from `describe_csr_matrix_group`.

A plan whose strategy counts are open-ended (for example a generated strategy with `samples = -1`
and no file list to size it) has no file grid to pre-size from. Archive sources are always sized
from their file lists, so they never reach this refusal. `BatchPlan.require_exact` raises a ValueError
that names the open-ended strategies; there is no fallback. The identity stamp runs after the
commit, so an interrupted or refused run never leaves a dataset that looks reusable.

## Boundary

Composition is where config models, platform adapters, workflow DTOs, and
domain services are connected. Entry modules that still assemble concrete
collaborators belong here rather than under `application`.

Composition assumes that platform has already resolved configuration and
runtime concerns into concrete values and adapters. Its job is to select the
right collaborators, construct workflow-local DTOs, and hand execution to
application or platform entrypoints without re-implementing path, environment,
or client policy.

Preconditioner assembly follows the same boundary. Platform config validates
shared schedule fields such as `start_iter`, `limit_iters`, and `fallback`;
composition maps those values into `torchalg` preconditioner objects without
owning the algorithms or iteration-switching policy. This adapter is an
anti-corruption layer between neuralls config/IO concepts and torchalg runtime
objects; it must not reimplement solver or preconditioner APIs.

Solver runner assembly follows the same ownership split. Composition may adapt
NumPy-backed workflow archives into tensors before calling `torchalg`, and may
reshape `torchalg` outputs for existing dataset/reporting DTOs, but it does not
own CG/PCG/FCG algorithms.

For DLKit training and eval, composition now consumes already-loaded lower-case DLKit
jobs. DLKit owns TOML parsing, profile references under `[run]`, section merge
order, and typed validation. Composition owns only runtime materialization:
dataset entry injection, workspace path patching, and tracking enablement on
top of the validated job object.

That materialization is split on purpose. `job_loader.py` is the only local
boundary that asks platform to load one typed DLKit job. `assembler.py` then
applies only the minimal mode-specific startup patching needed to create a
runnable assignment workspace, while `job_materializer.py` owns the later
training-only runtime patch sequence for dataset entries, split pinning,
dataloader defaults, workspace callback wiring, and tracking enablement.
Composition no longer duplicates that patch policy by re-parsing TOML or by
open-coding workspace and tracking mutations in multiple call sites.

Train/validation/test splits belong to the dataset, and generation owns them.
`generation/split_pinning.py::pin_dataset_split` derives a `split.json` from the
dataset's `matrix_sample_index` and its config's `[split]` section
(`test_ratio`, `val_ratio`, `seed`) right after the dataset is written or
found reusable, replaces it only when its content actually changes, and logs
when it does. `[split]` is excluded from the dataset's generation identity
because it changes no generated byte; training and comparison identities
instead include the split file's content digest, so a changed split
invalidates exactly the models and comparisons that depended on it. Training,
search, fit, evaluation and comparison only read the file (via DLKit's
`ExternalFileSplitStrategy` or `load_split_indices`) and never create or alter
it, so two runs against the "same" dataset cannot land on different
partitions. For a many-matrix dataset (`matrix_sample_index` has more than one
unique value) the split is computed at the matrix-group level first and then
expanded back to rows, so every row belonging to one matrix lands in the same
fold. `assignments/runtime_split_patcher.py::patch_split_settings` points
`settings.data.splits.filepath` at the pinned file for every
`TrainableJobConfig` kind and raises `StageInputsStale` (naming
`neuralls generate`) when none exists; the job's own `data.splits` ratios are
not consulted. `job_materializer.py::materialize_training_job` runs this stage
right after dataset entries are injected.

Comparison consumes hashes: `comparison/input_gate.py::require_current_inputs`
runs before any planning and checks that each `matrix_dataset` is reusable
under `generation_identity(cfg)` and that its `split.json` equals what the
dataset's `[split]` config yields now. A missing or stale input raises
`StageInputsStale` naming the stage to run; comparison never generates, splits
or trains. A held-out evaluation dataset declares its own fold in its dataset
TOML (e.g. `test_ratio = 1.0`).

For DLKit-supervised training, composition is also the only layer allowed to
translate repository storage names into runtime dataset-entry names. That
translation is owned by a composition-level runtime dataset contract. The
current supervised bridge maps dataset `rhs` artifacts to feature entry `x`,
dataset `solutions` artifacts to target entry `y`, and the dataset `matrix`
artifact to auxiliary feature entry `matrix`. The storage format (`zarr` or
`npy`) is resolved in platform storage before composition turns those artifacts
into format-neutral resolved dataset-entry specs. Platform adapters then
translate those specs into concrete DLKit entries such as `NpyEntry` or
`ZarrEntry`. Domain terms such as `solutions` remain valid on disk, but
composition does not expose `solutions` as a runtime target alias.

Additional runtime model inputs are opt-in and come from exactly one source:
extra `[[data.features]]` declarations in the DLKit job or data profile. Composition
preserves their declaration order and binds them positionally to persisted
`parameters_0`, `parameters_1`, ... dataset artifacts. Model hyperparameters in
`[model]` are never treated as dataset inputs.

This bridge is intentionally declarative at the config boundary. DLKit job and
profile TOMLs remain the source of truth for runtime entry names, transforms,
and supported entry-routing metadata, while composition only patches in the
resolved on-disk paths and neutral runtime semantics at execution time.
DLKit-specific entry construction and optional metadata application stay in
platform code.

Training assembly keeps feature specs path-backed instead of eagerly opening
dataset arrays inside `neuralls`, but that does not guarantee fully lazy `.npy`
training. DLKit's current
`FlexibleDataset` still materializes non-lazy path entries during dataset
construction, even though `NpyEntry` can forward format-specific load kwargs
such as `mmap_mode`.

Eval-only assembly reuses the same supervised dataset contract as training but
does not recreate split policy. For each assignment it resolves the latest
finished training run, asks platform tracking for lease-backed local paths to
that run's checkpoint and `splits/*.json` artifact, and patches the resolved
split file into `data.splits.filepath` — producing one fully-resolved,
assignment-specific `InferenceJobConfig` per child. Missing or ambiguous split
artifacts are hard failures because regenerating ratios would evaluate a
different test set. The lease scope spans both settings preparation and the
DLKit sweep, so remote artifact scratch files stay alive only for the execution
window and no persistent `_downloads` tree is created under eval outputs.

Eval assembly delegates sweep orchestration to DLKit the same way training
does: every prepared assignment becomes one DLKit multirun `RunSpec`, and the
whole case config's selected assignments dispatch as a single
`run_multirun_spec()` sweep. DLKit's `MultiRunOrchestrator` owns the parent
run's lifecycle, per-child MLflow run creation and `mlflow.parentRunId`
tagging, and per-child failure isolation; composition's only remaining job is
building each child's settings ahead of dispatch and finalizing (saving
metrics/figures, tagging bookkeeping params) each successful `ChildOutcome`
afterward. Resolving a distinct `InferenceJobConfig` per assignment before
dispatch — rather than sharing one settings object across the sweep — is what
lets one job evaluated across several datasets correctly evaluate each child
against the dataset it was actually trained on, instead of the exotic
mixed-job-batch case.

Prediction payload normalization follows the same rule. Composition accepts
DLKit's raw boundary output once, normalizes it into the canonical prediction
key `y_pred`, and then hands the normalized payload to diagnostics and artifact
staging. Downstream reporting code does not keep a fallback list of prediction
aliases. Training diagnostics are scalar `eval/*` MLflow metrics (MAE, MSE, RMSE,
R², prediction/target std ratio, normalized mean shift); parity, residual and
error-histogram figures come only from DLKit's `[plots]` callbacks (training)
and `evaluate()` (eval), never from neuralls plotting code.

Training, eval, inference, comparison, and generation assembly all follow the same
rule: composition may decide which collaborators participate in a workflow, but
it must not absorb low-level IO mechanics, config normalization policy, or
tensor-level runtime behavior.

For MLflow naming, composition only propagates the resolved case-config names.
Training uses `names.training` and comparison uses `names.comparison`, with the
defaults owned by the case-config Pydantic models rather than composition-layer
constants.

Batch training adds one session-scoped parent run per case-config invocation.
That parent is identified by case-config path plus launch time and exists only
to group the per-assignment child runs. Comparison workflows do not add an
equivalent wrapper: their parent run identity stays defined by the dataset
selection encoded in each comparison entry.

Comparison parent-run summary metrics sanitize each preconditioner label into an
MLflow-safe metric-key segment before appending it under namespaced keys such as
`iterations/<preconditioner>`. Composition does not implement that MLflow policy
itself: it delegates metric-key sanitization, comparison-run metric logging, and
nested child-run writes to `platform.tracking`, while only assembling child tag
payloads and deciding when the logging happens.

Comparison plot legends are a separate concern from that MLflow metric-key
policy. `platform.reporting.preconditioner_labels` builds a display name as a
`ScientificDescription` rendered under the shared `LEGEND_PROFILE` and the case's
`notation_policy` (`[comparison_baseline.presentation]`, threaded from
`ComparisonConfig` through `compare_preconditioners`): the family abbreviation plus
facts of two kinds. *Realized* facts come from the *constructed* `torchalg` object
(POD-2G's fitted basis rank, an aggregation threshold's realized coarse dimension,
an adaptive hierarchy's realized levels, a neural model's architecture class name)
and are never dropped for length. *Configured* facts come from the config (grid
levels, an IC0 drop tolerance, smoothing steps, aSA/BAMG settings) and cover CSR
preconditioners too; those that merely tell variants apart are lowest priority and
dropped first, anything at its config default is omitted, and seeds are never shown.
Composition supplies only workflow provenance: for POD-2G it appends the fit
dataset id, producing labels such as ``POD-2G (c=10) | gaussian-cg10-...``.
`composition.comparison._plots` keeps solver results and style metadata keyed by
their original preconditioner key and passes display labels in a separate
mapping; it never re-keys results by presentation text. Assignment-generated
preconditioners use the unique assignment id as that key, never the optional
display name. The batch boundary rejects duplicate keys before checkpoint
resolution, and execution rejects missing, duplicate, or unexpected evaluation
entries as well as a solver result carrying the wrong key. If two constructed
methods still produce identical legend text, only
their presentation labels are suffixed with the stable keys. This keeps labels
truthful to what was built, distinguishes equal-rank bases fitted on different
datasets, and preserves a one-to-one mapping even when generation/training is
skipped or one model cannot be resolved.
Comparison participants are exactly those declared by the resolved typed
configuration. Neither composition nor the domain solver injects a baseline or
infers preconditioner semantics from a config `name`; names are opaque result
identifiers, while `PreconditionerType` is the sole semantic discriminator.
This leaves baseline selection open to future solver-specific configuration
without adding type-specific filters to shared execution code.
The same composition wrapper derives one canonical scientific context per
comparison (`matrix=... | rhs=...`). It logs that context once at comparison
start alongside the execution-only `device=...` field, and reuses only the
matrix/RHS context as the title of every diagnostic plot, with the loaded
linear-system size on a second `N=...` line. Per-preconditioner progress logs
stay concise because those values cannot change within a comparison run.

Comparison execution deliberately excludes raw and preconditioned condition-number
estimation. Spectral estimation is an optional standalone analysis concern, not a
prerequisite for CG: iterative eigensolvers may require thousands of expensive
preconditioner applications on large or clustered systems and must not block the
solver-comparison workflow.
The loaded matrix and right-hand side are moved to the selected solver device once,
before any preconditioner is constructed, so stateful preconditioners and CG vectors
share a device without repeated host-to-device transfers.

Plot *styling* (marker/color, as opposed to the label text above) is a
further separate concern, driven by `PreconditionerComparisonEntry.color_key`/
`.marker_key` alongside `.family`. `comparison_run.py::_evaluate_preconditioner`
populates them via `_pod2g_style_keys`, which reads the raw (unresolved)
`PreconditionerConfig` — POD-2G's fit dataset (`coarsening.dataset_dir`)
becomes the color key and its snapshot weighting scheme
(`coarsening.weighting.method`) becomes the marker key, so a dense POD-2G
sweep (many dataset x weighting combinations in one plot) separates into
distinct color/marker combinations instead of collapsing onto one
family-wide marker with only color-shade differences. Non-POD-2G configs
yield `(None, None)` and fall back to family-based styling in
`platform.reporting.plots._resolve_styles`.

Case-driven comparison selection is made once, by the planner (below): the
runner receives one `PlannedSystem` per linear system and never decides which
matrix or row it solves. Generated and raw sources load the planned matrix row
of the configured matrix dataset, while a `dataset` source delegates to platform
storage to resolve one canonical `(matrix, rhs, lhs)` triplet for its planned
held-out row. `gaussian` and `sinusoidal` are the two *generated* RHS
kinds (`composition/comparison/rhs_generation.py`, dispatched through the one
`GeneratedSourceHandler`): each synthesizes its vector on the fly from `dim`
alone, independent of any dataset row, and is therefore never rescaled by a
matrix-derived normalization scale (the `generated` trait of its source model,
`platform/config/models/comparison.py`, read by `_linear_system.py::_rhs_is_generated`).
`sinusoidal`'s `omega` is total phase swept across the vector (e.g. `2*pi*k`
for `k` full cycles), normalized by vector length the same way as the
`sinusoidal_field` generation strategy, so it reads as a cycle count
independent of system size rather than a per-step frequency.

**Planning.** Comparison consumes hashes and never touches datasets: after
`input_gate.require_current_inputs` proves every dataset is generated and
pinned under today's config, `composition/comparison/plan.py::plan_comparisons`
turns every `[[comparisons]]` and `[[comparison_sweeps]]` entry into concrete
`PlannedSystem` values (`composition/comparison/models.py`). Each carries a
concrete `ComparisonRegistryEntry` (unique id, derived RHS seed), a
`MatrixRef(matrix_id, row)`, the `rhs_row` for the `dataset` kind, and the
matrix's leakage provenance. A *matrix id* is the sample id the generator bound
rows to (`matrix_sample_index`); the *row* is the physical storage position
that matrix and parameter-array readers take. They coincide only by accident, so
the planner never derives one from the other: it reads the pinned test fold
(`platform/storage/dataset_shape.py::load_held_out_fold`, matrix id -> held-out
rows) and takes the matrix's first held-out row.

The matrix shape of each dataset (`classify_dataset`, the only place this is
decided) selects the config array:

- `[[comparisons]]` requires a single-matrix dataset. Its one matrix is in both
  the train and the test fold, so provenance is `SharedSingleMatrix`; the
  pinned split holds out only the RHS rows used by the `dataset` kind.
- `[[comparison_sweeps]]` requires a multi-matrix dataset. Its `selector`
  (`FirstNSelector(n)` or `FractionSelector(p, seed)`; `p = 1.0` is every
  held-out matrix) chooses among the *held-out matrices*: `FirstNSelector`
  takes the first `n` in ascending matrix-id order, `FractionSelector` samples
  `round(p * count)` matrices with `numpy.random.default_rng(seed)`; an empty
  selection is an error. Provenance is `HeldOutMatrix(matrix_id, ...)`.

Every system solves on a matrix of its own `matrix_dataset`, so
`PlannedSystem.provenance` is always `SharedSingleMatrix | HeldOutMatrix`. An
external RHS (`raw_rhs` / `raw_lhs`) is applied to that native matrix and must
match its dimension: the raw handlers' `validate` compares the file's vector
length (and bounds-checks `sample_index`) with
`platform.storage.dataset_shape.system_dimension` and raises a `ValueError`
naming the file, the length found and the size expected. That fails only that
system, as a failed `ComparisonOutcome` logged at error level and counted by the
report levels' unmeasured-system warning; the rest of the batch continues.

A shape mismatch raises a `ValueError` naming the dataset and the rule. Only
same-size systems are compared, so a case whose datasets have different system
dimensions (read from the manifests) is rejected.

`rhs_count` is the number of RHS draws per matrix. Systems get ids `<id>`
(`<id>-<k>` when `rhs_count > 1`; sweeps use `<id_prefix>-<matrix id>` plus the
same suffix). What a kind does with the draw count is declared by class-level
traits on its source model (`RhsSourceModelBase` in
`platform/config/models/comparison.py`), which the planner and the
normalization step read instead of keeping lists of kinds: `generated` (the
vector is synthesized, never matrix-scaled), `seeded` (the source consumes the
per-system seed, so each system gets an independent
`plan.derive_seed(entry.seed, matrix_id, k)`, a `SeedSequence` of the triple,
and neighbouring systems never share an RHS vector) and `single_draw`
(`rhs_count` must be 1). `gaussian` and `sinusoidal` are generated and seeded;
`raw_lhs`/`raw_rhs` are one fixed vector (single-draw, not generated); `dataset`
has none of the three. A new kind needs one source model declaring its traits
plus one handler registered by kind. The `dataset` kind uses rows of the
entry's own `matrix_dataset` — the first `rhs_count` held-out rows of the
matrix; fewer is an error. Duplicate planned ids are rejected.

**RHS mask.** Every source model carries `mask`, a sum type `NoMask | RandomMask(density)`
(`platform/config/models/comparison.py`; `density` in (0, 0.5], default 0.1), e.g.
`rhs_source = { kind = "gaussian", mask = { kind = "random", density = 0.1 } }`. The mask is
applied in exactly one place, `source_handlers.resolve_comparison_source`, after any handler
produced the vector, using `domain/masking.py` (`random_support_mask`, `apply_mask`). Its random
stream is `SeedSequence([system seed, MASK_STREAM_TAG])`, separate from the RHS draw, so changing
the density never changes the underlying draw and each system's support is reproducible. A random
mask makes `RhsSourceModelBase.needs_own_seed` true even for an unseeded kind (`dataset`), so the
`rhs_count` systems get different supports. Masking changes `b`, so a masked `raw_lhs`/`dataset`
system drops its known solution (`ResolvedComparisonInput.lhs = None`): the Ax = b check against
`lhs` and the error-vs-time plots, which are skipped when no error history exists, then do not apply.
The mask is part of the dumped source params, so it is part of the comparison identity.

`config_assembler.resolve_comparison_config` consumes a `PlannedSystem`: it
builds `ComparisonData(matrix_index=planned.matrix.row, selection_seed=...)` and,
for the `dataset` kind, injects `sample_index=planned.rhs_row` into the source
params. `comparison_batch.py::_run_comparison_with_resolved_specs` attaches
`planned.provenance` (`domain/solver/models/provenance.py`: `SharedSingleMatrix`,
`HeldOutMatrix`) to the resolved input, which requires it. Directory
resolution for a `matrix_dataset` id is shared between the planner and
`resolve_comparison_config` via `config_assembler.resolve_dataset_dir`.

**Report levels.** Every executed system is tagged once
(`comparison/recipes.py::system_tag` -> `domain.solver.models.benchmark_events.SystemTag`)
with its dataset, RHS recipe label, matrix id, RHS ordinal and entry id.
Each source model's `recipe_label` is the single definition of a recipe: the
kind plus the parameters that make problems differ (`gaussian`,
`sinusoidal ω=6.28319`, `raw_lhs`, `dataset`, each followed by ` · mask ρ=0.1` when masked), so two
omegas, or a masked and an unmasked run of one kind, are two recipes and entries with identical parameters share one.
`PlannedSystem.rhs_ordinal` is the draw's position among the matrix's
`rhs_count` draws.

Domain classification consumes canonical setup events and solve attempts,
never `CGComparisonResult` rows. The preconditioner is always part of the key;
the level chooses the other columns:

| Level | Keys besides the preconditioner | Group id (`render_group_id`) |
|---|---|---|
| `FineLevel` | dataset, recipe | `fine/<dataset>/<recipe slug>` |
| `MediumLevel` | dataset | `medium/<dataset>` |
| `GlobalLevel` | none | `global` |

`classify_level(planned, setup_events, solve_attempts, level)` returns
`GroupAggregated` when at least one method has a setup event and
`GroupSkippedAllCacheHit` when the group has no benchmark event. Aggregated
groups carry planned matrix/RHS membership, per-method planned/attempted/
converged counts, converged-iteration statistics, and failure-aware per-system
outcomes. A setup failure therefore remains a zero-attempt incomplete method,
not an artificially cheap empty result.

`comparison_batch._method_major_schedule` groups execution as method, matrix,
then ordered RHS members. `comparison_run.execute_method_workload` emits one
`MatrixSetupEvent` per matrix and one linked `SolveAttempt` per attempted RHS;
`group_reporting.py` reduces only these completed workloads. Prepared method
state, including an Identity/no-op setup, is reused across sibling RHS values
but never across matrices. This makes every matrix setup visible instead of
copying it onto RHS rows or treating a cross-matrix static-state reuse as a
missing setup event.

Completed method workloads are persisted by
`platform.reporting.benchmark_events.write_method_benchmark_artifacts` as the
versioned, artifact-relative `setup_events.parquet`, `solve_attempts.parquet`,
and `summary.toml` files. The Parquet frames are canonical; the TOML summary is
reconciled from the Task 2 reducers and records the method name/config digest,
planned/attempted/converged counts, setup/solve/online timings, attempted-RHS
average, and workload peak memory. If an offline value is present, it is named
`generation_s`; checkpoint and data I/O are not serialized as costs or metrics.
The frames remain canonical for realized setup/solve events and their timing,
outcomes, repeats, per-event memory, and attempted/converged counts. The
summary remains canonical for planned membership, missing system IDs, and the
method-lifetime peak, which are not reconstructible from only those event
frames.

The comparison batch opens one `session_comparison` parent and one direct
`comparison_method` child per executed `CompletedMethodWorkload`. The batch,
not the platform sink, owns those MLflow contexts. A method child owns its
required configuration, the two canonical Parquet event files, `summary.toml`,
method histories, and method diagnostic figures; the session parent owns shared
provenance and cross-method figures. No matrix/RHS system runs or method
grandchildren are created. Method reuse is accepted only for schema-v1 children
whose identity contains the method config/checkpoint digest and complete
workload membership and whose required raw artifacts pass schema/link and
summary validation. Reuse also compares the canonical method config (checkpoint
inputs are content digests, not lease-local paths), checks
diagnostic membership and event-frame methods, and reconciles summary totals
with the frames; an older no-schema comparison run is not a cache hit.

**Checkpoint loading and memory.** Loading a checkpoint is file I/O that a
deployment pays once, not algorithmic setup cost, so no kind's measured setup
includes it (`execute_method_workload` is the contract). Before each matrix
setup, `PreconditionerService.preload` ensures every checkpoint the
config depends on into the method's `CheckpointStore`
(`preconditioners/checkpoint_store.py`); the timed build then only borrows the
loaded object. Which checkpoints a config needs comes from the
`CheckpointRefBearing` protocol, and how each kind loads is the mapping in
`preconditioners/preloading.py` (fitted POD-2G basis, neural and neural-AMG
predictors, neural POD-2G's inference predictor). A fitted POD-2G basis does not
depend on the matrix (only the coarse operator `Phi^T A Phi` does), so it is read
once and shared by every matrix of the dataset. The store owns the loaded
predictors and builds only borrow them: torchalg's `NeuralPreconditioner` takes an
already-loaded predictor and never releases it, so a discarded timing repeat cannot
unload a model the kept build still uses. A neural preconditioner built without a
preload loads its own predictor and owns it
(`OwningDLKitNeuralPreconditioner`). Checkpoint and data loading are
excluded from benchmark cost metrics. The device is
also warmed once per process before the first system (`shared.device.warm_up_device`),
so no preconditioner's timing absorbs CUDA/cuSPARSE first-call costs; this is
separate from the per-run `timing.warmup`, which stays opt-in.

The method-owned checkpoint store spans its matrices, so loaded weights are
shared without sharing a matrix setup event. Each matrix's tensors and prepared
state are dropped after its last RHS; the checkpoint store is released before
the next method starts. `WorkloadMemoryObserver` combines operation peaks over
that lifetime without resetting CUDA peak counters around an already-timed
setup or solve. CPU `ru_maxrss` is process-wide and monotonic, so a method's CPU
peak is documented and reported as a lower bound when an earlier workload set a
higher process high-water mark.

One check runs before anything executes:
`spec_consistency.validate_dataset_specs_consistent` rejects a dataset whose
systems define the same preconditioner name with different configs (the derived
name ignores scheduling fields such as `start_iter`, yet a cached,
schedule-wrapped build would be wrongly served to the other system).

Multi-matrix and single-matrix cases answer genuinely different questions;
`[[comparison_sweeps]]` addresses the former and `[[comparisons]]` the latter,
and the planner enforces that split per dataset. Every distinct matrix incurs
one measured setup. Increasing sibling RHS count amortizes that matrix setup;
cross-matrix reuse is limited to method-owned checkpoint objects, never the
prepared hierarchy or factorization.

Comparison input preflight also stays out of composition. Workflow code invokes
platform-owned validation for matrix/RHS inputs rather than inspecting dataset
manifests directly. Preflight validates only the concrete artifacts required by
the chosen comparison mode: matrix-only validation for generated-RHS runs, and
matrix-plus-RHS validation for raw or dataset-backed RHS runs.
Likewise, enriched infrastructure failures from storage and tracking propagate
through composition unchanged; composition does not reformat low-level I/O
errors into user-facing strings.

Resolved comparison inputs are treated as first-class artifacts. Composition
stages the final `(A, b)` pair plus provenance metadata and delegates MLflow
logging of those artifacts to platform tracking helpers so future runs can
reconstruct the exact system that was compared.

Auto-generated comparison stubs for an assignment reference its model through
`TrainedAssignmentRefConfig`, resolved by `_resolve_trained_assignment_ref`
(`assignments/model_resolution.py`) — never a tag search. It asks the single
MLflow `IdentityStore` (`platform/tracking/mlflow_store.py`) for the newest
run in the case's training experiment that is FINISHED, holds a real
checkpoint, and carries exactly the assignment's derived *training identity*
(`identity/training.py::training_identity`: dataset content and effective job
settings; `AssignmentModelContext.training`, a `TrainingLookup`). There is
no fallback to an older or differently-trained
run: no match raises, and the comparison skips only that model (with a logged
warning) while the rest run.

Comparison model resolution treats one resolved MLflow `run_id` as the hard
boundary for checkpoint discovery — but that scan-and-select contract applies
to run references (`LoggedModelRefConfig` and the strict assignment ref above,
which both end in `_resolve_checkpoint_for_run`). When resolved run artifacts contain multiple
`.ckpt` files, composition canonicalizes byte-identical duplicate copies,
prefers a unique `best.ckpt`, and raises on remaining ambiguity instead of
silently picking the first path.

Registered model versions (`RegisteredModelRefConfig`) do not go through that
scan: `register_logged_model` (`platform/tracking/model_registry.py`) pins one
unambiguous checkpoint file once, at registration time, recording it in the
`checkpoint_artifact_path` version tag alongside the version's `source`.
Resolution then leases that exact artifact directly — an O(1) lookup with
no scanning, deduping, or best-checkpoint fallback. A version registered
before this pinning existed has no `checkpoint_artifact_path` tag and cannot
be resolved; it must be re-registered. Registration itself is a deliberate,
manual action (e.g. called ad hoc or via the MLflow UI for alias management)
— not an automatic side effect of training.

## Full case pipeline: three stage functions, one orchestrator above them

There are exactly three pipeline-stage functions, each independently
invocable via its own CLI command, and `assignments/case_pipeline.py::run_case_pipeline()`
(the composition entry point behind the CLI's `run` command) does nothing but
sequence them — no stage's logic is duplicated into another or into the
orchestrator itself:

1. **Generate** — `generation/multi_generation.py::generate_batch()`, also
   called directly by `neuralls generate`.
2. **Train** — `assignments/training_batch.py::run_assignment_sweep()`, also
   called directly by `neuralls train` (renamed from `run_assignment_matrix`
   since it dispatches dlkit's own "multirun"/sweep mechanism, not a 2D
   matrix). This is the *only* training implementation — the older,
   generate-and-train-in-one `multi_training.py::train_batch` was retired in
   favor of this shared function.
3. **Compare** — `assignments/comparison_batch.py::run_comparison_batch()`,
   also called directly by `neuralls compare`, only when the case config
   declares `[[comparisons]]` or `[[comparison_sweeps]]`.

All three stages, and every fit-job child within the training sweep, run
sequentially in one process with no CUDA context reset between them —
`run_case_pipeline` calls `shared.device.release_device_memory()` between
the training and comparison stages, and `run_assignment_sweep` registers it
as a dlkit `LifecycleHooks.on_child_completed`/`on_child_failed` callback so
it fires after every sweep child, success or failure. Without this, a
CUDA-heavy fit-job OOM (or even just a high memory peak) leaves the caching
allocator holding those blocks for every child and stage that follows,
turning one real failure into a cascade of identical-looking ones.

Training assumes generation already ran and never generates a dataset
itself — it only resolves the already-generated dataset's directory
(`data_cfg.output.data_dir / data_cfg.id`, the same path
`generation/_context_builder.py` computes) and fails cleanly with a
"re-run the generation pipeline" error if nothing is there. This is a real
constraint, not just tidiness: it's what makes stage 1 the single place
dataset generation happens, so `neuralls train`'s original "assumes data
already generated" contract holds whether you call it standalone or through
`run`. Each stage keeps its own independent `force_generate`/`force_train`/
`force_compare` flag and its own MLflow-backed reuse check.

Three reuse checks compose into one cascade, so rerunning `run` after any
change only redoes the affected work:

1. Dataset generation derives a `StageIdentity` (`identity/generation.py::generation_identity`:
   the validated dataset config, minus cosmetic fields such as output dir and id,
   plus the content digest of every raw source file the config declares) *before*
   loading any source data (`generation/process_data.py`). It skips
   regeneration iff the manifest's `identity_key` equals that key *and*
   `platform.storage.dataset_digest.current_dataset_digest` equals the manifest's
   `content_digest` (O(1) while the stat snapshot is unchanged; re-hashes the
   logical arrays after a copy), unless `force_generate=True`
   (`generation/dataset_builder.py::is_dataset_reusable`). A fresh write stamps
   `identity_key`, `identity_components`, `content_digest` and `stat_digest`
   into the manifest as its last step; a manifest without `identity_key`
   (legacy, or a crash between write and stamp) is regenerated, never trusted
   by existence. Declared sources that do not exist raise `FileNotFoundError`.
2. Training's reuse check (`run_assignment`, inside `run_assignment_sweep`)
   derives the assignment's training identity
   (`identity/training.py::training_identity`) from three inputs: the
   dataset's *content* digest (`platform.storage.dataset_digest
   .current_dataset_digest`) and the job hashed after DLKit loads it (defaults
   filled, data profile inlined, minus the `tracking`/`plots`/`experiment`
   sections). `MlflowIdentityStore` then looks for a FINISHED run with a real
   checkpoint carrying that key. The key is tagged on the new run, so a
   changed hyperparameter or a dataset regenerated by stage 1 since the last
   training run never reads as "already trained" — retraining happens
   automatically the next time `run_assignment_sweep` is invoked (via `run` or
   standalone `train`). After training, the dataset is re-checked against the
   digest the key was built from; a dataset that changed mid-training marks
   the run FAILED. There is no CLI override for job config content (e.g.
   epochs) — a smoke test points at its own job TOML instead, so the identity
   never needs to reconcile a config value with a flag that silently replaced it.
3. Comparison's reuse check mirrors this one level up
   (`identity/comparison.py::comparison_identity`): the key covers the
   resolved solver/data/RHS config, every preconditioner's settings, the
   matrix/RHS data and the *content digest of each resolved checkpoint* — so
   a static-only comparison is keyed too, a retrained model with different
   weights invalidates exactly the comparisons that used it, and an identical
   rerun does not. INFO logs bracket checkpoint artifact resolution, each
   checkpoint-content hash, matrix/dataset identity construction, and the
   MLflow reuse lookup so normal log timestamps expose slow preparation phases
   without a separate timing mechanism. `run_comparison_batch` wraps the whole
   batch in `shared.digest.digest_memo()`, so each checkpoint file is hashed
   once per `(resolved path, size, mtime_ns)` and each dataset's input digest
   once per directory, however many systems share them (the memo ends with the
   batch, so a file rewritten between runs is never served stale). All entries
   of a batch also share one `MlflowArtifactLeaseManager`
   (`_ComparisonResolutionContext.artifact_leases`): one download per run id,
   kept until the batch ends. Download and hashing happen in the prepare phase,
   outside every measured walltime, and so is checkpoint *loading* for every
   checkpoint-backed kind (see "Checkpoint loading and memory" below), so setup
   costs stay comparable across kinds.

Every key is derived by the pipeline from the inputs themselves (never
authored in a config or bumped by hand), and user ids stay pure labels. All
three checks share `shared/digest.py` (canonical digests, `array_digest` over
logical array content, `Cosmetic`/`InputData`/`InputConfig` field markers) and
one lookup, `platform/tracking/mlflow_store.py::MlflowIdentityStore`; tags are
written through `StageIdentity.tags()` (`domain/identity.py`).

This is deliberately pure Python composing existing MLflow-tag lookups — no
new orchestration dependency. Each function stays a plain, side-effect-free-
until-called composable (no CLI coupling), so wrapping them as tasks in an
external orchestrator later, if ever needed, is additive rather than a
rewrite.
