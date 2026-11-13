# Domain Module

The domain package contains the pure computational core of `neuralls`.

## Package Map

- `solver/`: solver/comparison config and reporting DTOs, cost metrics, and comparison orchestration over `torchalg` — CG/PCG/FCG algorithms, monitoring primitives, and preconditioners live in `torchalg`, not here (see `domain/solver/README.md`)
- `generation/`: strategy-driven dataset payload generation
- `presentation/`: typed scientific presentation of datasets, methods, and comparison cohorts: provenance, presentation states and their planner, titles, and their rendering (see `domain/presentation/README.md`)
- `linalg.py`: pure linear algebra utilities (matrix norms, normalization scale)
- `normalization.py`: data scaling helpers and normalization ABC; scales preserve the input format (dense or `csr_array`), and `matrix_norm` dispatches to `np.linalg.norm` or `scipy.sparse.linalg.norm`
- `inference.py`: cross-boundary inference DTOs (`InferencePredictions`, `InferenceOutputs`)
- `inference_ports.py`: framework-agnostic batch inference predictor port used by application and platform
- `masking.py`: vector zero-masks — `random_support_mask` chooses a support, `apply_mask` zeros everything outside it; no knowledge of where the vector comes from
- `graph.py`: pure CSR-to-edge-list view, computed on demand, never stored

- `identity.py` / `identity_ports.py`: `StageIdentity` (derived key + per-input components), the `IdentityTag` names, and the `IdentityStore` port that composition's `gate_reuse` uses to decide reuse (implemented in `platform`)

## Semantic Difference

Domain code explains the mathematics and workflow invariants of the project.
If a module needs only arrays, pure data models, and algorithmic rules, it
belongs here. The moment it starts parsing config files, opening MLflow runs,
or deciding artifact locations, it has crossed into `platform` or
`composition`.

## Boundary

Domain code depends only on `neuralls.shared`. It does not load configs,
resolve filesystem layout, start MLflow runs, or persist artifacts. When
domain output must be written, it is returned as typed payloads and handled by
`neuralls.composition` plus `neuralls.platform`.

No ambient actions either: domain code does no file I/O (archive files go through the injected
`ArchiveSource`), writes no log line, reads no clock outside the shared timing helpers, keeps no
process-wide cache, calls no `torch.cuda` API and picks no device (`resolve_device`,
`release_device_memory`); it works on the device of the tensors it is given. An outcome an
operator must see is returned as data in a closed type (generation notices, `ScaleSummary`
disagreements, the solver's `Reference` and `EnergyError`) and composition logs it.
`tests/architecture/test_io_boundary.py` enforces this for `domain/` and `application/`.
