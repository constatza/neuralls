# Composition Module

The composition package owns wiring and config-driven assembly.

## Package Map

- `assignments/`: registry loading, assignment wiring, training, eval, comparison, inference
- `comparison/`: single-run comparison assembly around application/domain logic
- `generation/`: config-driven dataset orchestration, dataset persistence wiring, and default tracing services
- `inference/`: inference data-loading composition helpers
- `preconditioners/`: config-to-`torchalg` preconditioner factory wiring (Identity, Jacobi, ILU, IC0, ICholesky, AMG, NeuralAMG, Neural)
- `solvers/`: config/workflow-to-`torchalg` solver runner adapters
- `tracking/`: tracking tag and run-spec assembly

## Semantic Difference

Composition is the only layer allowed to know both the abstract workflow shape
and the concrete adapter set needed to run it. If a module mostly wires config
models, ports, and runtime collaborators together, it belongs here. If it
starts doing reusable numerical work, it belongs in `domain`. If it starts
owning concrete IO or MLflow mechanics, that code belongs in `platform`.

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
training-only runtime patch sequence for dataset entries, dataloader defaults,
workspace callback wiring, and tracking enablement. Composition no longer
duplicates that patch policy by re-parsing TOML or by open-coding workspace
and tracking mutations in multiple call sites.

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
aliases.

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
policy. `platform.reporting.preconditioner_labels` derives method abbreviations
and structural detail from the *constructed* `torchalg` preconditioner object's
own attributes (AMG grid levels, aggregation theta, smoother/coarsening omega,
and POD-2G's actual fitted basis rank) rather than re-reading the TOML config.
Composition supplies only workflow provenance: for POD-2G it appends the fit
dataset id, producing labels such as ``POD-2G (c=10) | gaussian-cg10-...``.
`composition.comparison._plots` keeps solver results and style metadata keyed by
their original preconditioner key and passes display labels in a separate
mapping; it never re-keys results by presentation text. This keeps labels
truthful to what was built, distinguishes equal-rank bases fitted on different
datasets, and prevents duplicate display labels from dropping curves.
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

Case-driven comparison selection is intentionally simple: one comparison entry
loads one system from its required `rhs_source`. Generated and raw sources use
the configured matrix dataset plus `matrix_index`, while dataset sources
delegate to platform storage to resolve one canonical `(matrix, rhs, lhs)`
triplet. When a dataset source omits `sample_index`, platform selects the first
STANDARD row and follows persisted `matrix_sample_index` to load the matching
matrix. Comparison does not inspect training/test splits or infer held-out
semantics from MLflow artifacts.

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
   declares `[[comparisons]]`.

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
   rerun does not.

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
