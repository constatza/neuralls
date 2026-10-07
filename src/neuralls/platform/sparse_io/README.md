# sparse_io

Storage-agnostic CSR building blocks and the per-format sparse batch backends.
A batch is a sequence of CSR matrices written to and read from one container.
The package owns the conversions, the on-disk member schema, the reader/writer
contracts and the format registry. Storage-level concerns (manifests, dataset
directories, artifact resolution) stay in `platform/storage/`.

## Modules

- `components.py`: `CsrComponents` (indptr, indices, data, shape) is the canonical
  in-memory form. `to_components` and `to_scipy` convert to and from `csr_array`.
  Indices are int64 and values float64 regardless of the container.
- `layout.py`: `choose_layout` picks the layout from the sparsity patterns, the
  pack functions flatten samples into member arrays, and `members_for` maps a
  layout to its member names. Member name constants live here.
- `protocol.py`: `SparseLocation`, `SparseSummary`, the `SparseReader` and
  `SparseWriter` protocols, and the streaming pair `SparseStreamWriter` and
  `CsrMemberSink`.
- `stream.py`: `CsrStreamWriter`, the incremental writer. It owns the running offsets
  and the sample-count check, and writes through a `CsrMemberSink`. The layout
  (`MANY_MATRICES` or `SHARED_PATTERN`) is resolved once in `__init__` to a
  `_LayoutStrategy` (`_RaggedLayout` or `_SharedLayout`); each strategy holds its own
  created-members guard and layout-specific state (the ragged nnz cursor, or the shared
  pattern), so the writer itself never branches on the layout again.
- `pattern.py`: `canonicalise_csr` and `csr_pattern_matches`, the sparsity-pattern
  identity used by the shared-pattern stream (see below).
- `zarr_store.py`: the zarr backend.
- `hdf5_store.py`: the hdf5 backend.
- `registry.py`: `backend_for`, the format-name to backend mapping, and
  `open_stream_writer`, the streamed counterpart.
- `torch_convert.py`: `to_torch_csr`, a float64 `torch.sparse_csr` tensor built from
  components without densifying.

## Protocol

- `SparseWriter.write(location, samples) -> SparseSummary` replaces whatever already
  sits at `location` and returns the sample count and the chosen layout. An empty
  batch is rejected.
- `SparseReader.summary(location) -> SparseSummary` reports the sample count and
  layout without reading any sample data.
- `SparseReader.read_sample(location, sample_index) -> CsrComponents` reads one sample
  and touches only that sample's slices. An out-of-range index raises `IndexError`.
- `SparseLocation(path, key)`: zarr treats `path` as the group directory and ignores
  `key`. hdf5 treats `path` as the file and `key` as the group name, defaulting to
  `HDF5_GROUP_KEY` (`"matrix"`).

Backends implement the protocols structurally, so callers depend on the contract
and never import a storage library.

## Streamed writes

`registry.open_stream_writer(format_name, location, planned_samples=, layout=)`
opens a `SparseStreamWriter` that receives samples batch by batch:

- `write_batch(samples)` appends one batch of `csr_array` samples. It never reads
  back. An empty batch, or a batch that would exceed the planned count, raises.
- `close() -> SparseSummary` raises `ValueError` if the number of samples written
  differs from `planned_samples`, and otherwise returns the summary.
- The layout is an argument, not an inference. `choose_layout` needs every sample,
  which a stream does not have up front. Under `SHARED_PATTERN`, a sample whose
  pattern differs from the first raises on write, naming the stored sample index.
- Pattern check under `SHARED_PATTERN`: each sample is canonicalised first (a copy
  with indices sorted within rows and duplicate entries summed), so the same stored
  positions compare equal whatever their index order. Explicitly stored zeros are kept,
  because the shared layout stores them. The pattern is the shape, `indptr` and
  `indices`; values are never compared. The comparison is one pass over the integer
  arrays and never densifies a matrix. The check runs before a batch's rows are
  written, so a mismatch leaves no partial rows in the staging container.
- Per-sample members (`shape`, `sample_offsets`, shared `data`) are created at full
  shape and filled by row offset. The ragged `indptr`, `indices` and `data` are
  growable and extended per batch. The stored arrays and summary equal those of
  the whole-group writer for the same samples and layout.

The stream is backend-agnostic: `CsrStreamWriter` drives a `CsrMemberSink`, and
each format registers a sink opener in `registry.py`.

## On-disk layouts

Both formats use the same member names. Only the container differs.

- `LayoutType.MANY_MATRICES` (each sample has its own pattern):
  - `indptr` (int64): the per-sample indptr arrays concatenated.
  - `indices` (int64): the per-sample column indices concatenated.
  - `data` (float64): the per-sample values concatenated.
  - `sample_offsets` (int64, N+1): nnz boundaries of each sample in `indices` and `data`.
  - `shape` (int64, N x 2): `(rows, cols)` of each sample.
- `LayoutType.SHARED_PATTERN` (one pattern for all samples):
  - `indptr` and `indices`: stored once.
  - `data` (float64, N x nnz): values of sample i are `data[i]`.
  - `shape` (int64, N x 2).
  - `sample_offsets` is absent. Its absence marks this layout.

Containers:

- zarr: a group directory whose arrays carry the member names above.
- hdf5: a group named `matrix` inside `dataset.h5`, whose datasets carry the member
  names above. Datasets are uncompressed with default chunking, so the stored values
  are bit-identical to the input arrays.

The reader decides the layout from the container's members, not from the manifest,
so a container always reads back the layout it was written with.

## Format dispatch

`registry.backend_for(format_name)` is the only place where a storage format name
selects code for sparse batches. Callers pass the format name recorded in the
dataset manifest (`"zarr"` or `"hdf5"`). Adding a format means registering one
entry in `registry.py`, not adding a branch at each call site. A format without an
entry raises `ValueError` naming the format.

## Content digest

The dataset content digest (`platform/storage/dataset_digest.py`) hashes each CSR
sample from its components: shape, indptr, indices and data. It never hashes the
stored layout or the container. A zarr and an hdf5 CSR dataset holding the same
matrices therefore share one content digest, and a dataset can move between
containers without changing its identity.
