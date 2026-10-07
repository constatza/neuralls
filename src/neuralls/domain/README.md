# Domain Module

The domain package contains the pure computational core of `neuralls`.

## Package Map

- `solver/`: solver/comparison config and reporting DTOs, cost metrics, and comparison orchestration over `torchalg` — CG/PCG/FCG algorithms, monitoring primitives, and preconditioners live in `torchalg`, not here (see `domain/solver/README.md`)
- `generation/`: strategy-driven dataset payload generation
- `analysis/`: pure numerical diagnostics used by higher layers
- `linalg.py`: pure linear algebra utilities (matrix norms, normalization scale)
- `normalization.py`: data scaling helpers and normalization ABC; scales preserve the input format (dense or `csr_array`), and `matrix_norm` dispatches to `np.linalg.norm` or `scipy.sparse.linalg.norm`
- `inference.py`: cross-boundary inference DTOs (`InferencePredictions`, `InferenceOutputs`)
- `inference_ports.py`: framework-agnostic batch inference predictor port used by application and platform
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
