# Platform Module

The platform package isolates external integrations and side-effecting helpers.

## Package Map

- `config/`: settings, config-validation context, registry resolution, TOML loaders, lower-case job metadata readers, and the thin DLKit job loader adapter
- `storage/`: filesystem, workspaces, dataset I/O, and storage validation helpers
- `tracking/`: MLflow run helpers, naming/query policy, workflow topology resolution, and client adapters
- `reporting/`: plotting, artifact staging, and inference output adapters
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

Case-config auto naming also lives at this boundary. Auto-generated assignment
ids are job-first (`{job_id}-{dataset_id}`), and auto-generated assignment
display names follow the same order (`{job_label} | {dataset_label}`).
Auto-generated comparison display names stay dataset-defined: they resolve to
the matrix dataset label alone when matrix and RHS come from the same dataset,
or to `{matrix_label} | {rhs_label}` when they differ. Explicitly configured
comparison display names still override the generated label.
Reporting labels derive the method abbreviation and measured structural detail
from the constructed preconditioner. Workflow provenance is added separately:
POD-2G labels append the fit-dataset id after ``|`` so equal-rank bases fitted
on different datasets remain explicit, while the plotting API keeps result
keys and display labels as separate mappings. Human-readable labels are never
used as result identity, so duplicate display text cannot overwrite curves.
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
weighting scheme (`comparison_run.py::_pod2g_style_keys`) — falling back to
family when a caller doesn't supply one. Colors come from the Okabe-Ito
colorblind-safe palette for up to 8 distinct keys, extended with
procedurally generated evenly-spaced hues beyond that so the color axis
never depletes into repeats. Markers are two-tiered (solid shapes first,
line-drawn ones as overflow) with a per-shape size multiplier so the pool
renders at a visually balanced weight. Any residual group of entries sharing
an identical (linestyle, marker, color) triple — the case when no override
keys are given at all — still gets spread across a lightness ramp so it
stays distinguishable.

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
real-checkpoint check via dlkit). Composition derives the identities and tags
the runs (`StageIdentity.tags()`); platform only owns the MLflow filter/search
mechanics. Dataset identity lives in `storage/dataset_digest.py`
(`current_dataset_digest`: O(1) from the manifest while the stat snapshot is
unchanged, otherwise a recompute of the logical array content).
MLflow artifact recovery follows the same boundary. Platform tracking helpers
resolve and validate checkpoints, split JSON, and staged config artifacts
through an `ArtifactLeaseManager` protocol with explicit abstract methods.
Local MLflow artifact stores are borrowed in place; remote stores are
materialized into scoped scratch storage owned by the lease manager. Composition
decides which assignment or model ref should be evaluated and how those local
paths are wired into DLKit, but it does not choose persistent download
directories.
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
- `storage/generation_formats.py`: generation-time `zarr`, `npy`, and `hdf5`
  writers/accumulators plus backend-neutral artifact replacement helpers.
  Manifest assembly is shared: each writer performs only its format-specific
  array I/O, then calls the pure `_build_manifest(payload, locations, matrix_shape, params)`
  helper. The per-format differences are carried by a `ManifestLocations` DTO
  (one `ArtifactLocation(path, key)` per artifact) plus the physical matrix shape —
  `zarr`/`hdf5` read that shape back from the written container, `npy` derives it
  from the payload layout.
- `storage/dataset_readers.py`: manifest-driven read helpers and explicit resolved dataset contracts;
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

- `rhs_kind`
- `target_kind`
- `matrix_sample_index`

Readers derive safe RHS candidates from `rhs_kind`; they do not trust ad hoc
filenames or stored allowlists. Storage owns persistence and artifact
resolution, while the compact integer encoding/decoding boundary lives in
shared pure codecs so domain and composition code do not depend on platform.
When datasets expose `matrix_sample_index`, the canonical triplet resolver uses
that binding to load the matching matrix for a selected `(rhs, solution)` row.

Case-driven comparison sample selection stays explicit and deterministic.
`ComparisonRegistryEntry.matrix_index` applies to generated and raw sources;
dataset-backed `rhs_source` entries resolve a canonical manifest-backed triplet
instead. Dataset sources use `sample_index` when provided and otherwise select
the first STANDARD row. `raw_rhs` sources point at concrete RHS vector files;
`raw_lhs` sources point at concrete solution-side vector files that comparison
transforms into an RHS with `A @ vector` and the configured scale. Platform does
not infer held-out semantics from training runs or MLflow split artifacts.

Training artifact persistence also stays generic: platform storage writes the
already-normalized numpy payload it receives from composition without
reintroducing DLKit prediction-key fallback logic.
