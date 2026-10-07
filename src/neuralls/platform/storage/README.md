# Storage

Filesystem and dataset I/O for generated datasets, comparison inputs, checkpoints and training artifacts. Nothing here decides workflow order; callers choose when to read or write.

## Modules

- `array_store.py`: `ZarrArrayStore` and `Hdf5ArrayStore`, the dense backends of the `ArrayStore` protocol (defined in `domain/generation/ports.py`). Arrays are created at full shape; `close()` raises if any row was left unwritten.
- `dense_stream.py`: opens the dense store a streamed run fills and writes the manifest last, from the shapes the writer reports.
- `staged_commit.py`: `commit_staged_directory` writes a dataset into a sibling `<name>.partial` staging directory and renames it to the final name only once everything, including the manifest, is written, so the final name is never a partial dataset. A forced regeneration moves the existing final directory aside to `<name>.old` first, so a failed rename can restore it. Every rename goes through one retrying helper (`RENAME_ATTEMPTS` tries on `PermissionError`, for transient Windows locks) that reports failure through `_raise_storage_error`.
- `csr_storage.py`, `csr_layout.py`: write and lay out CSR system matrices stored as a zarr or hdf5 group.
- `matrix_readers.py`: suffix-keyed registry `MATRIX_READERS` (`.npy`, `.txt`, `.npz`, `.mtx`, `.mtx.gz`) returning `SystemMatrix`. Unknown suffixes raise. `.npz` must hold a sparse matrix.
- `manifest.py`, `manifest_io.py`: typed manifest models and their serialization.
- `dataset_readers.py`, `datasets.py`: manifest-driven readers and artifact resolution. `datasets.py` is a facade that re-exports the reader helpers.
- `dataset_digest.py`: content digest (format-independent) and stat digest (cheap snapshot) of a generated dataset. A stat digest that still matches is trusted; a copied or touched dataset is re-hashed.
- `generation_formats.py`: write-time storage for generation workflows.
- `comparison.py`, `arrays.py`, `base.py`: loaders for comparison systems and plain array I/O.
- `checkpoints.py`, `training_artifacts.py`, `workspaces.py`: checkpoint, training-artifact and workspace helpers.
- `errors.py`: the `OSError` translation every backend shares.
- `enum_codecs.py`: compatibility re-export of the semantic enum codecs.
