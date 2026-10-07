# Generation Module

The generation package turns matrices and optional archives into processed
training datasets. Most users should interact with it through `process-data`
first and only then drop into the package internals.

## Handling Arbitrarily-Named Matrix Files

When a colleague provides matrices with parameter-encoded filenames (no sequential
integer ID), set `enumerate_by` in the `[source]` block of the dataset TOML:

```toml
[source]
matrix_path = "/data/matrices/E1_*_E2_*.txt"
enumerate_by = "name"   # lexicographic — deterministic across runs
```

`enumerate_by` sorts the glob results by the chosen criterion and assigns sequential
IDs 0, 1, 2, …  No renaming or modification of the source files is required.

| Value | Sort criterion | When to use |
| --- | --- | --- |
| `"name"` | Lexicographic filename | **Default choice** — fully reproducible |
| `"ctime"` | File creation timestamp | Files arrived in a known order |
| `"mtime"` | Last-modified timestamp | Files were last touched in a known order |

`enumerate_by` and `sample_id_regex` are mutually exclusive; specifying both raises
a validation error.

Config-driven generation and the public composition entrypoint
`neuralls.composition.generation.dataset_builder.build_dataset(...)` both honor
`enumerate_by` and pass it through to the glob source streams.

Each strategy in a mixture draws from its own stream: unless a strategy sets its own
`seed` override, its seed is derived from the mixture seed and the strategy name
(`helpers.derive_strategy_seed`). A shared seed would make two strategies on the same
matrix emit identical RHS vectors. Each binding (one matrix) derives its own mixture seed
from the mixture seed and its binding index (`helpers.derive_seed`), so two matrices with
equal counts never draw the same stream. The derivation is deterministic for a given mixture seed.

For multi-matrix sources, a generated strategy's count is split with the remainder
rule (`allocation.split_remainder`). Each matrix gets `count // matrix_count` samples,
and the `count % matrix_count` leftover samples go to distinct matrices chosen by a
generator seeded from the mixture seed under the label `matrix-split`. Each matrix's share is
split the same way across its bindings, with a generator seeded under `binding-split` and the
matrix index, so the two levels never reuse one stream. Nothing is dropped, so the counts always sum to the request, and no matrix
gets more than one leftover sample. Every strategy accepts any number of matrices, so a
mixture may combine archive and generated strategies. `replacement = true` is rejected
(config validation, `GenerationConfig._reject_replacement`, and `_reject_replacement_request`).

Trajectory strategies (`residuals`, `gaussian_residuals`,
`smoother_filtered_probes`) take a row budget. They differ in where their starting systems come from:
`residuals` is archive-backed (a `solutions_glob` or the explicit solution rows, each file used once),
while `gaussian_residuals` and `smoother_filtered_probes` draw random probe vectors and have no file pool,
so no glob caps their sample count. Each base system (one solve) yields
K_rows rows, where K_rows = `len(window.resolve_indices(stop + 1))`, the number of kept steps. For a
positive start this is `ceil((stop - start + 1) / step)`. It is fixed, independent of the initial guess,
and registered per strategy in `runner.py` as `rows_per_base_system`. The budget becomes
B = ceil(R / K_rows) base systems, and the same split above (or the archive map) is applied
to base systems, not rows. The overshoot O = B * K_rows - R is trimmed from one matrix
that owns a base system: the lowest-index one for generated strategies, the owner of the
last unit for archives. Trimming never removes a whole base system, so every matrix keeps
at least one row.

Archive strategies (`solution_archive`, `rhs_archive`, `scaled_solutions`, `validated_archive`)
draw from a grid of `M` matrices by `K` files and emit (matrix, file) pairs through
`allocation.archive_units`. Unit `t` goes to matrix `t % M`, file
`(matrix + t // M) % K`, so no pair repeats. A request is capped at `M*K` base systems with
one warning naming the strategy, the rows and the base systems. `samples = -1` means all `M*K` pairs, each once.
The orchestrator writes each binding's files into `file_indices`, and the provider loads
exactly those files. A single matrix with several bindings is rejected for archive
strategies before any glob read, because every binding would draw the same files.

Each archive strategy's glob sets its pool: `solutions_glob` for `solution_archive`,
`scaled_solutions` and `validated_archive`, and `rhs_glob` for `rhs_archive`. The
per-binding `solution_path` source and an archive glob are competing pools, so a
strategy that has both is rejected with a ValueError instead of letting files repeat.

**Python API:**

```python
from neuralls.domain.generation.source_streams import GlobMatrixStream, EnumerateBy

stream = GlobMatrixStream("data/E1_*_E2_*.txt", enumerate_by=EnumerateBy.NAME)
# stream.sample_ids → (0, 1, 2, …)
```

## Holding Out Matrices From a Parametric Family

When a `[source]` glob spans a parametric matrix family (many distinct matrices,
e.g. randomized material parameters) and you need some of them excluded from
training/POD-snapshot generation — so a separate comparison dataset can evaluate
against matrices nothing has been fit on — set `include_indices` or
`exclude_indices` in `[source]` (mutually exclusive):

```toml
[source]
matrix_path = "/data/matrices/E1_*_E2_*.txt"
enumerate_by = "name"
exclude_indices = [85, 86, 87]   # or: include_indices = [85, 86, 87]
```

Both are plain lists of the ids `enumerate_by`/`sample_id_regex` assigned — no
computation needed, just decide which ids to keep or drop. The same filter is
applied uniformly to every glob-based stream opened from that `[source]` block
(`matrix_path`, `rhs_path`, `solution_path`, `parameters_paths`), so a
multi-matrix source's existing id-matching validation (`bind_sources`) keeps
holding. Referencing an id that doesn't exist in a given stream raises
immediately rather than silently doing nothing.

**`matrix_index` is a row position, not a raw family id.** Code that later reads
a generated dataset by `matrix_index` (e.g. `[[comparisons]]` in a case TOML) is
indexing that dataset's own stored matrix array — physically laid out as one
block of rows per matrix binding, in ascending raw-id order — not the original
`enumerate_by` id. Two consequences:

- After filtering, `matrix_index = 0` means "the first *included* matrix," not
  "raw id 0."
- `matrix_index` values only address genuinely distinct matrices when the
  dataset's generation strategy emits **exactly one row per included matrix**
  (set `[[generation.strategy]].samples` equal to the number of included
  matrices). If a strategy pools more samples than matrices across the family
  (the common case for training data), small `matrix_index` values can all fall
  inside the same matrix's row block and resolve to the identical physical
  matrix — see `_resolve_binding_strategy_counts` in `binding_allocation.py`.

`configs/cases/45x15randomE/default.toml` and its
`configs/datasets/{train,test}/45x15randomE/*.toml` datasets are a worked
example of both: train datasets `exclude_indices` a held-back subset, and
`gaussian-eval.toml` `include_indices`s the same subset with `samples` set to
its exact size so every comparison `matrix_index` is a distinct matrix.

## User Path

### Basic

Build one dataset from one config:

```bash
uv run process-data /path/to/dataset.toml \
  --case-config /path/to/case.toml
```

### Intermediate

Build every dataset declared in one case config:

```bash
uv run generate-all /path/to/case.toml
```

### Advanced

Import generation internals when you need to extend strategy behavior:

```python
from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.generation import open_batch_stream, run_generation
```

## Strategy Progression

Start with the simplest family that matches the model target.

| Level | Strategy | Output |
| --- | --- | --- |
| Basic | `normal` | `(b, x)` |
| Basic | `krylov` | `(b, x)` |
| Intermediate | `rhs_archive` | `(b, x)` |
| Intermediate | `solution_archive` | `(b, x)` |
| Advanced | `residual_traces` | `(r_k, x_k)` |
| Advanced | `residuals` | `(r_k, x_true - x_k)` |
| Advanced | `gaussian_residuals` | `(r_k, x_true - x_k)` |
| Advanced | `smoother_filtered_probes` | trace pairs, `x` = random probes damped by weighted-Jacobi sweeps |

## Smoother-Filtered Probes

`smoother_filtered_probes` (`strategies/smoother_probes.py`) synthesizes
"algebraically smooth" error snapshots directly, instead of deriving them
incidentally from a CG trajectory: `samples` random probe vectors (Gaussian
or Rademacher, via `probe_distribution`) are passed through `window.stop`
sweeps of the weighted-Jacobi error-propagation map
`v <- v - omega * D^-1 A v` (`omega` defaults to unset, matching
`torchalg.preconditioners.implementations.amg.smoothers.JacobiSmoother`'s own
default of auto-computing the damping per-matrix via its spectral-radius rule).
Directions the smoother handles well are quickly attenuated, so what
survives after `window.stop` sweeps is, by construction, smoother-resistant
— the directions a POD-2G coarse space needs to cover. Reuses torchalg's
`apply_jacobi_damping_trajectory` for the damping sweep itself (a brief
numpy/torch round-trip, the one exception to this package's otherwise
numpy-only strategies) so "smooth" means the exact same thing here as it
does in `composition/preconditioners/_weighting.py`'s
`smoother_persistence` snapshot weighting, which targets the same operator
— that function returns every intermediate sweep (not just the final one),
so `window` (see "Step Selection" below) can pick any sweep depth or range
of depths, not only the fully-damped result. Output is a `ResidualTraceSamples`
(the same container `residuals.py` populates) — one `(A @ x, x)`
row per kept sweep per probe, always 2D regardless of how many sweeps are
kept.

## Step Selection

`StepWindow` (`step_window.py`) is the one shared abstraction for "which
steps of a trajectory to run and keep," used by `residuals.py`,
and `smoother_probes.py` — every strategy that
harvests snapshots from a bounded iterative trajectory of a single system
(`krylov.py` is not a consumer: its samples are random basis combinations,
not sequential iterates). It replaced the `cg_iters`/`every_n`/`steps`
fields those strategies used to express this independently.

Mirrors Python's own `slice`/`range` vocabulary — `stop`/`start`/`step`:

- **`stop`** (required, no default) is the hard iteration/sweep cap handed
  to the solver/smoother — exactly `stop` steps are ever attempted, never
  more. This is what guarantees a strategy never "solves to convergence and
  then discards most of the trajectory": the cap *is* the selection
  parameter, not something decided independently of it.
- **`start`** (optional, default -1 = last step only) is the first step
  kept, inclusive. Negative values count back from the last step
  (`-stop..-1`); positive values are absolute (`1..stop`). Step 0 is never
  emitted: generation always starts CG at x0 = 0, so iterate 0 is the base
  pair (r0 = b, e0 = x*), and keeping it would put the right-hand side and
  the true solution into the dataset as trajectory rows. The rule is
  enforced once, in `_StepWindowFields`, by resolving the window against a
  full-length trajectory and rejecting it if the first kept index is 0, so
  every trajectory config (`residuals`, `gaussian_residuals`,
  `smoother_filtered_probes`) inherits it. Zero and
  starts below `-stop` are rejected. Row count per base system is
  `len(window.resolve_indices(stop + 1))`, fixed and independent of x0; no
  value-based trimming or scanning is done.
- **`step`** (default 1) keeps every `step`-th row within the selected
  range.

Trajectory configs expose no CG tolerance. A reachable `rtol`/`atol` would
let the solver stop before `stop`, and the row budget below depends on every
trajectory being full length, so the keys are rejected at config construction
as unknown fields (`extra="forbid"`).

Call sites use `window.select_with_indices(trajectory)` to get the kept
rows and their original step indices from one call over one array — this
is what guarantees the two can never be computed from mismatched arrays by
mistake.

### Trajectory CG tolerance

`residuals.py` passes a fixed tolerance of `1e-20` to the solver
(`_TRAJECTORY_UNREACHABLE_TOLERANCE`). That value is below float64 machine
epsilon, so CG can never satisfy it and always runs exactly `stop`
iterations. It must be passed explicitly: torchalg's defaults (`1e-6` /
`1e-14`) are reachable and would stop the trajectory early.

The row budget assumes every base system yields exactly K_rows rows, which holds only
when its solve runs to `window.stop`. A base system that converges earlier is an error:
`trace_utils.require_full_trajectory` raises `TrajectoryShortfallError` (a `RuntimeError`)
on its raw rows, before trimming, in `residuals` and
`smoother_filtered_probes`. Padding a short trajectory would hide the early stop, so it is
not done. A window the solver cannot reach is therefore a config error: tighten `stop` or
pick a window the solver reaches. `smoother_filtered_probes` always runs
exactly `window.stop` sweeps, so it never raises this error.

## Residual Families

The repo now uses explicit residual strategy names:

- `residual_traces` for residual-to-iterate pairs
- `residuals` for residual-to-error pairs
- `gaussian_residuals` for residual-to-error pairs without archive solutions

These names are the supported user-facing identifiers in dataset configs and
tests.

Residual and trace strategies interpret positive `samples` as the exact final
flattened row budget. Internally they generate ceil(samples / K_rows) complete CG
traces, then trim the final trace block so downstream arrays and row-kind metadata have
exactly `samples` rows. `samples = -1` means all `M*K` (matrix, file) base systems for
finite archive-backed trace sources.

Archive-backed pure-pair strategies can skip an initial slice of the deterministic
archive order with `skip`. When `shuffle = true`, files are shuffled once with
the configured seed. The pool is the remaining `permutation[skip:]`. Explicit
`file_indices` (set per binding by the orchestrator, not by users) index into that
pool, so every binding of a multi-matrix archive draws its own disjoint files.

`FileInputProvider` (in `providers.py`) memoizes its file reads with
`functools.lru_cache`, keyed on `(glob_pattern, count, shuffle, seed, skip, file_indices)`. Every
archive-backed strategy (`solution_archive`, `rhs_archive`, `scaled_solutions`,
`validated_archive`, and `residuals`/`gaussian_residuals` when `solutions_glob` is set)
routes through it, so an archive shared across many matrix bindings — or across several
dataset configs in one `generate-all` batch that point at the same glob — is read from
disk once per distinct selection, not once per binding or per dataset file. `ArchiveData`
(pre-loaded in-memory archives, e.g. from `single_solution`) always takes priority over
`solutions_glob` when both are available for a strategy. With an explicit `solution_path`,
`_prepare_generation_context` loads every row of the file once (file order) into
`solution_rows`, and the file takes no part in binding, so its ids are not matched to the
matrix ids. Only solution-archive strategies receive these rows; a generated strategy never
does, whatever its count. With `samples = -1`, each binding's archive is that full block. With
an explicit positive count `N` on `solution_archive`, binding `b` draws rows `(b + p) mod K` for
`p` in `range(N)`, the same cyclic map as the glob archives, so bindings get different rows. `N`
above the file's `K` rows is capped at `K` per binding with one warning.

## Package Map

- `orchestration.py`: the streamed batch pipeline. `open_batch_stream()` opens the run's
  streams (`_open_streams`), resolves the per-binding allocation (`binding_allocation.py`) and
  plan, then drives `generate_batches()` (see `batch_generator.py`) batch by batch.
  `_make_strategy_runner()` returns the `StrategyRunner` that produces one strategy's rows for
  one binding, loading each binding's inputs once (`_BindingInputs`, `_load_binding_inputs`).
  Dataset-level norm and scale values are folded in by `ScalarAggregator`. Internal state
  (opened streams, per-binding strategy allocation, the run's resolved context) is held in frozen
  dataclasses (`OpenedStreams`, `BindingAllocation`, `_GenerationRunContext`) threaded through the
  pipeline instead of positional tuples. There is no whole-dataset payload builder. The pure
  allocation math, strategy properties, row generation and matrix caching it drives live in the
  four sibling modules below, each with no dependency on `orchestration.py` itself.
- `binding_allocation.py`: pure per-binding strategy-count and archive-file-index allocation,
  given a seed. `_resolve_binding_strategy_counts()` is the entry point: generated counts are
  split across matrices then across each matrix's bindings with the remainder allocation
  (`allocation.py`), so no sample is dropped; archive counts are mapped onto the (matrix, file)
  grid with `archive_units()`, so no pair is emitted twice. `_single_matrix_allocation()` handles
  the one-matrix case, where every binding takes the global counts.
- `strategy_properties.py`: compile-time properties of known generation strategies —
  `_STRATEGY_PROPERTIES` says which strategies are file-backed archives and which override key
  holds their glob (`_archive_glob_for_strategy`), and which are solution archives specifically
  (`_is_solution_archive`, `_explicit_solution_strategies`).
- `strategy_rows.py`: `_generate_strategy_rows()` runs one strategy for one binding through
  `run_generation()` (`runner.py`) and returns its `_StrategyRows` (rhs, solutions, row-kind
  codes), classifying each row's `RowKind` from the strategy's trace iteration indices when
  present (`_row_kind_codes_for`).
- `matrix_cache.py`: `_cached_matrix_loader()` returns a loader that normalizes and measures one
  matrix sample per call, keeping only the most recently requested sample (`_CachedMatrix`)
  cached — bindings are visited in order, so one matrix in memory at a time is enough.
- `batch_plan.py`: the row budget of a run, fixed before generation. `plan_batches()` turns
  the resolved `BindingAllocation` into an immutable `BatchPlan` of per-binding, per-strategy
  row counts. `BindingAllocation` is defined here (re-exported by `orchestration.py`). A plan
  holding the open-ended `ALL_SAMPLES` count is not exact, so its totals raise instead of
  returning a guess. `BatchPlan.require_exact()` raises a ValueError that names the open-ended
  strategies, so the streamed writers refuse such a plan instead of falling back.
  Every archive-style source is sized from its file list before generation, so its plan is
  exact and is never refused. An explicit `solution_path` file is sized by `solution_row_count()`
  in `archive_files.py` (a .npy header read, no values; a .txt file is one sample). A glob
  `solution_path` is sized by its matched file count K, from the stream's sample ids (a listing,
  no content read). Its rows feed each binding by position: `samples = -1` takes all K rows on
  every binding, and an explicit count N draws the cyclic rows `(b + p) mod K`, capped at K with
  one warning. Archive strategies with an archive glob map `samples = -1` to M×K (matrix, file)
  units, or K rows for a single matrix. `require_exact` only refuses a generated strategy with
  `samples = -1` that has no file list to size it.
- `batch.py`: `SampleBatch`, one strategy's rows for one binding (rhs, solutions, row kinds,
  matrix sample ids, the binding's parameter vectors). Its constructor rejects inconsistent
  row counts, and `slice()` cuts a bounded chunk
- `batch_generator.py`: `generate_batches(plan, run_strategy, batch_size=)` yields batches in
  binding order, then plan strategy order, each at most `batch_size` rows. Strategy outputs
  are still produced whole, so the bound applies to what is yielded, not to generation
- `sample_writer.py`: `SampleWriter`, the write-only sink for streamed dense generation.
  `write_batch(batch)` writes rhs, solutions, row kinds, matrix sample ids, parameter
  vectors and the matrix (one broadcast copy per row, or one row when the layout is
  single-matrix) at a running row offset. `finalize()` checks the written rows against
  the plan, closes the store, and returns `DatasetArtifacts` (array names, shapes, dtypes).
  Arrays are created lazily at the first batch, at the plan's full row count. It has no
  read methods.
- `scalar_aggregate.py`: `ScalarAggregator` keeps the first binding's matrix norm, value
  scale and scale payload, plus one disagreement flag per field, so its state is constant
  in the number of bindings. Tolerances and warning text are the same as before
- `specs.py`: frozen input DTOs mirroring the config's own `[source]`/`[generation]`
  sections — `SourceSpec` (where samples come from), `MixtureSpec` (strategy mixing + RNG),
  `DatasetSpec` (assembly: mixture + replacement/normalize/norm-type). `open_batch_stream()`
  and `build_dataset()` accept these instead of the same ~15-20 fields re-declared as loose
  kwargs at every call-chain layer
- `ports.py`: `ArrayStore` (the write-only dense store that `SampleWriter` depends on) and
  `TracingSolverPort` protocol definitions consumed by the composition layer
- `runner.py`: strategy registry and dispatch
- `source_streams.py`: sample discovery and loading. The `_RawSampleSource` protocol
  (defined here, next to its consumer) is composed into a generic `_SampleStream[T]`
  wrapper that decides what a sample *means* — `_MatrixStream` (dense/sparse matrix API)
  or `_VectorStream` (1D vector API). The seven public `{Npy,Txt,Glob,Mtx}{Matrix,Vector}Stream`
  classes (minus `MtxVectorStream`, which doesn't exist) are thin subclasses that only pick a
  source, built from `file_sources.py`; `open_matrix_stream()`/`open_vector_stream()` are the
  entrypoints. Not yet collapsed into a single lookup-by-format dispatch (plan item S4): the
  single-file variants (`Npy`/`Txt`/`Mtx`) are directly tested by name and by `isinstance` in
  `test_source_streams_characterization.py`, so collapsing them means rewriting that test's
  class-identity assertions to behavioral ones — a separate decision, not folded into this split.
- `sample_ids.py`: sample-id derivation and enumeration rules for glob-matched files —
  `EnumerateBy` (assign sequential ids by name/ctime/mtime instead of parsing filenames),
  `_build_glob_index` (resolve a glob to `{sample_id: path}`), `_is_glob_expression`.
- `file_sources.py`: raw per-sample file readers satisfying `_RawSampleSource` structurally —
  `_NpyFileSource` (single file or a stack via mmap), `_TxtFileSource`, `_GlobFileSource`
  (one sample per matched file), `_MtxFileSource` (MatrixMarket, read as CSR).
- `bindings.py`: pure ID-level binding across matrix/rhs/parameters/solution sample streams —
  `SystemBinding` and `bind_sources()` (single-matrix broadcast when only one matrix id exists,
  otherwise bindings are keyed by matrix id).
- `providers.py`: archive or synthetic sample providers
- `allocation.py`: pure remainder-aware splits (no I/O). `split_remainder(total, M, seed=...)`
  gives each matrix `total // M` units and hands the `total % M` leftovers to distinct
  matrices chosen by a seeded `numpy` generator, so nothing is dropped and the split is
  reproducible. `archive_units(M, K, count)` maps unit positions onto (matrix, file) pairs
  with `i = t % M`, `j = (i + t // M) % K`, so archive files never repeat until the M*K grid
  is exhausted; the count is capped there by the function itself
- `matrix_operator.py`: `MatrixOperator`, a frozen wrapper around one system matrix (dense
  ndarray or CSR) that exposes `matvec`, `solve_direct`, and `eigensystem`. The LU/Cholesky
  factor and eigen results are memoized on a private cache, so all samples drawn from one
  matrix share one factorization. Dense uses `scipy.linalg` (`cho_factor` when
  `assume_pos_def`, else `lu_factor`; full `eigh` sliced to the requested end). CSR uses
  `splu` on the CSC form and `eigsh`; the CSR "smallest" eigensolve is shift-invert at
  `sigma=0` with `which="LM"` because ARPACK cannot reuse the operator's LU factor, and
  "largest" uses `which="LA"`. CSR never densifies and cannot return the full spectrum, so
  `random` eigenvector selection is dense-only. `ensure_symmetric` holds the symmetry check
  shared by both formats.
- `transforms.py`: pure transforms such as `A @ x`. `SolveTransform` and
  `EigenvectorCombinationTransform` accept a `SystemMatrix` and wrap it in a `MatrixOperator`
- `trace_utils.py`: trace trimming, offsets, and indexing helpers;
  `TrajectoryShortfallError` and `require_full_trajectory` reject base systems that
  converged before `window.stop`
- `step_window.py`: `StepWindow` — which steps of a bounded trajectory to
  run and keep (see "Step Selection" above)
- `strategies/`: concrete generation implementations
- `helpers.py`: a thin re-export surface over the seven focused modules below, kept so
  existing `from .helpers import X` imports keep working. New code imports directly from
  the owning module instead.
- `seeds.py`: deterministic, independently-labelled seeds (`derive_seed`,
  `derive_strategy_seed`, `rng_from_seed`) — every random stream that must not repeat
  another one takes its own label path under the mixture seed, hashed with CRC32 so string
  labels are stable across processes.
- `counts.py`: strategy sample counts (`rounded_counts`, `resolve_strategy_counts`) and
  trace-row counts for trajectory-harvesting strategies (`trace_rows_per_system`,
  `trace_rows_per_base_system`, `required_trace_systems`, `resolve_trace_generation_counts`,
  `_build_trace_indices`).
- `scaling.py`: matrix normalization for synthetic generation (`normalize_matrix_for_generation`)
  and scale-metadata serialization for the manifest (`serialize_scale_metadata`).
- `linear_solve.py`: `_solve_linear_systems` dispatches through the `_SOLVERS` registry, keyed by
  method alone (`"direct"`, `"cg"`). `"direct"` calls `MatrixOperator.solve_direct` (cached
  factor), and `"cg"` calls `scipy.sparse.linalg.cg` on the wrapped matrix. The format is
  read by the operator, not the dispatch. A missing key raises `ValueError`.
  `_verify_solution_accuracy` computes relative residuals and warns above tolerance.
- `eigen_strategies.py`: `_compute_eigendecomposition` requests only the `count` eigenpairs
  for "smallest"/"largest" and the full dense spectrum for "random"; `_select_eigenvectors`
  and `_generate_eigenvector_combinations` pick and linearly combine the result.
- `archive_files.py`: `select_archive_files` (deterministic, optionally shuffled file
  selection from a glob) and `solution_row_count` (a cheap header-only row count for an
  explicit solution file).
- `krylov.py`: `_lanczos_iteration` builds a Krylov subspace basis with early termination on
  breakdown; `_generate_krylov_combinations` draws random combinations from it.
- `strategy_configs.py`: `require_which_supported_by_format` rejects `which="random"` for CSR.
  `platform/config/models/data_models.py::DataConfigFile` calls it for every
  eigenvector strategy, because `matrix_format` (`[output]`) and `which`
  (`[[generation.strategy]]`) are declared in the same TOML file

Config-driven generation entrypoints now live in
`neuralls.composition.generation.processing`, which wires the generation domain
to default tracing solvers through `neuralls.domain.generation.ports`.

## Dataset Storage

The generation domain is storage-agnostic. It emits one
`GeneratedDatasetPayload` plus a staged matrix artifact path, and composition
selects the concrete storage family through `[output].dataset_format`.

Stored order is generation order: rows are written binding by binding, and strategy by strategy within a binding. The sample-level shuffle is gone, because dlkit shuffles every epoch during training. The mixture `shuffle` setting is accepted for compatibility and no longer changes stored order. A strategy-level `shuffle` on an archive still selects which files are drawn when the count is smaller than the file pool. That is selection randomness, not row order.

The matrix format is chosen by `[output].matrix_format` and threaded to the
streamed writer. A CSR run keeps every sample sparse end to end: the
source loads as `csr_array`, normalization scales the data without densifying,
the operator-based solves and the smoother-filtered probes accept csr matrices,
and the CSR stream stores each sample as-is. Strategies outside that set have
not been checked against csr input, and `random` eigenvector selection is
rejected for csr by config validation. CSR storage is written by the zarr and hdf5 dataset formats (npy is refused at generation), and a dataset cannot mix dense and CSR samples.
Dense storage formats densify CSR input explicitly.

Supported generation values: `zarr` and `hdf5`. `npy` is refused.

Platform storage owns the concrete implementations:

| Component | Location | Role |
| --- | --- | --- |
| `write_dense_streamed` / `write_csr_streamed` | `composition/generation/` | Stream batches into a staged directory and commit it |
| `ArrayStore` (`ZarrArrayStore`, `Hdf5ArrayStore`) | `platform/storage/array_store.py` | Dense write backends for `SampleWriter` |
| `DatasetManifest` | `platform/storage/manifest.py` | Typed manifest contract for persisted datasets |

The manifest is the canonical dataset contract. Read paths do not assume fixed
filenames beyond what the manifest declares.

Generated datasets also persist row-level comparison metadata in the same
storage family as the dataset itself. The persisted metadata artifacts are:

- `rhs_kind`
- `target_kind`
- `matrix_sample_index`

Internal workflow logic uses `StrEnum` semantic types, while storage encodes
those enums as compact integer arrays through shared pure codecs. Safe
comparison selection is derived from the persisted `rhs_kind` metadata rather
than from a separate stored allowlist.

## Normalization Metadata

Generation writes normalized matrix samples, RHS vectors, and solutions as one
consistent system. The dataset manifest stores one dataset-level normalization
block only.

For single-matrix datasets, that manifest block may include reversible scale
metadata such as `spectral_radius_bound` and `dimension_scale`.

For multi-matrix datasets, each matrix is still normalized independently before
storage. If those bindings do not share one exact scale payload, the manifest
intentionally leaves `normalization.scale` empty instead of pretending there is
one dataset-wide reversible scale.

Comparison safety semantics apply to RHS rows, not matrices. A single persisted
matrix may legitimately pair with both safe non-residual RHS rows and unsafe
residual-derived RHS rows.

## Extension Rules

When adding a strategy:

1. add a config model in `strategy_configs.py`
2. implement the strategy under `strategies/`
3. register it through `@register_strategy`
4. document the public strategy name and its required fields in user-facing docs
5. add generation and config tests

## Where It Connects

Generation stays inside the domain layer and depends only on:

- solver tracing for CG-derived strategies
- normalization trace containers
- shared constants and math helpers

`build_dataset()` in `neuralls.composition.generation.dataset_builder` is the only generation path.
Every dense and CSR build, in `zarr` or `hdf5`, streams through `write_dense_streamed()` or
`write_csr_streamed()`: `open_batch_stream()` yields batches, `SampleWriter` writes them, and the
manifest is written last. Memory is bounded by `write_batch_size`. A run whose plan is not exact
(an open-ended `ALL_SAMPLES` count) raises a ValueError naming the strategy; there is no fallback.
The generation domain never imports or instantiates storage objects. All file I/O is confined to
the composition and platform layers.
