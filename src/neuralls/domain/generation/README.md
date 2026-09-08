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

For multi-matrix sources, dataset-level `counts` and `mix/total` budgets are
global across the matrix family rather than applied once per matrix. Set
`replacement = true` under `[generation]` only when you want supported random
strategies to reuse matrix bindings explicitly during that global allocation.
Archive-backed and deterministic strategies remain strict and do not honor
matrix replacement.

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
  matrix — see `_allocate_strategy_counts_across_bindings` in `orchestration.py`.

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
from neuralls.domain.generation import generate_mixture, run_generation
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
| Advanced | `search_directions` | trace pairs for direction learning |

## Residual Families

The repo now uses explicit residual strategy names:

- `residual_traces` for residual-to-iterate pairs
- `residuals` for residual-to-error pairs
- `gaussian_residuals` for residual-to-error pairs without archive solutions

These names are the supported user-facing identifiers in dataset configs and
tests.

Residual and trace strategies interpret positive `samples` as the exact final
flattened row budget. Internally they generate enough complete CG traces to
cover that budget, then trim the final trace block so downstream arrays and
row-kind metadata have exactly `samples` rows. `samples = -1` still means all
available base systems for finite archive-backed trace sources.

Archive-backed pure-pair strategies can skip an initial slice of the deterministic
archive order with `skip`. When `shuffle = true`, files are shuffled once with
the configured seed and then selected as `permutation[skip:skip + samples]`.
This is useful when combining a residual strategy with `solution_archive`: set
`solution_archive.skip` to the number of base systems consumed by the residual
block to avoid reusing the same `(b, x)` pairs.

`FileInputProvider` (in `providers.py`) memoizes its file reads with
`functools.lru_cache`, keyed on `(glob_pattern, count, shuffle, seed, skip)`. Every
archive-backed strategy (`solution_archive`, `rhs_archive`, `scaled_solutions`,
`validated_archive`, and `residuals`/`gaussian_residuals` when `solutions_glob` is set)
routes through it, so an archive shared across many matrix bindings — or across several
dataset configs in one `generate-all` batch that point at the same glob — is read from
disk once per distinct selection, not once per binding or per dataset file. `ArchiveData`
(pre-loaded in-memory archives, e.g. from `single_solution`) always takes priority over
`solutions_glob` when both are available for a strategy.

## Package Map

- `orchestration.py`: mixed-strategy payload assembly; `build_dataset_payload()` requires an
  injected `DatasetAccumulatorPort` — the domain never creates storage objects directly.
  Internal stream/binding/accumulation state (opened streams, per-binding strategy
  allocation, accumulated blocks, the run's resolved context) are each a frozen dataclass
  (`OpenedStreams`, `BindingAllocation`, `AccumulatedBindings`, `_GenerationRunContext`)
  threaded through the pipeline instead of positional tuples, so a step's output can't be
  silently misread by position at its call site
- `specs.py`: frozen input DTOs mirroring the config's own `[source]`/`[generation]`
  sections — `SourceSpec` (where samples come from), `MixtureSpec` (strategy mixing + RNG),
  `DatasetSpec` (assembly: mixture + replacement/normalize/norm-type). `generate_mixture()`
  and `build_dataset()` accept these instead of the same ~15-20 fields re-declared as loose
  kwargs at every call-chain layer
- `payloads.py`: pure DTO — `GeneratedDatasetPayload` only; no accumulation helpers
- `ports.py`: `DatasetAccumulatorPort`, `DatasetWriterPort`, and `TracingSolverPort` protocol
  definitions consumed by the composition layer
- `runner.py`: strategy registry and dispatch
- `source_streams.py`: sample discovery and loading. One `_RawSampleSource` per source
  shape (single stacked `.npy`, single `.txt`, glob of per-sample files) is composed into
  a generic `_SampleStream[T]` wrapper that decides what a sample *means* — `_MatrixStream`
  (dense/sparse matrix API) or `_VectorStream` (1D vector API). The six public
  `{Npy,Txt,Glob}{Matrix,Vector}Stream` classes are thin subclasses that only pick a source;
  `open_matrix_stream()`/`open_vector_stream()` are the entrypoints
- `providers.py`: archive or synthetic sample providers
- `transforms.py`: pure transforms such as `A @ x`
- `trace_utils.py`: trace trimming, offsets, and indexing helpers
- `strategies/`: concrete generation implementations

Config-driven generation entrypoints now live in
`neuralls.composition.generation.processing`, which wires the generation domain
to default tracing solvers through `neuralls.domain.generation.ports`.

## Dataset Storage

The generation domain is storage-agnostic. It emits one
`GeneratedDatasetPayload` plus a staged matrix artifact path, and composition
selects the concrete storage family through `[output].dataset_format`.

Supported values:
- `zarr`
- `npy`

Platform storage owns the concrete implementations:

| Component | Location | Role |
| --- | --- | --- |
| `DatasetAccumulatorPort` | `domain/generation/ports.py` | Domain-facing accumulator protocol |
| `GenerationDatasetStorage` | `platform/storage/generation_formats.py` | Small write seam used by composition |
| `ZarrGenerationStorage` | `platform/storage/generation_formats.py` | Writes `matrix.zarr`, `rhs.zarr`, `solutions.zarr`, `parameters_*.zarr` |
| `NpyGenerationStorage` | `platform/storage/generation_formats.py` | Writes `matrix.npy`, `rhs.npy`, `solutions.npy`, `parameters_*.npy` |
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

`build_dataset()` in `neuralls.composition.generation.dataset_builder` selects a
`GenerationDatasetStorage`, creates its accumulator, and injects that into
`build_dataset_payload()`. The generation domain receives the accumulator via
`DatasetAccumulatorPort` and never
imports or instantiates storage objects directly. All file I/O is confined to the
composition and platform layers.
