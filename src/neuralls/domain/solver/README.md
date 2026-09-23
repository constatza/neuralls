# Solver Workflow Boundary

`neuralls.domain.solver` no longer owns solver or preconditioner algorithms.
CG, PCG, FCG, monitoring primitives, solver results, and preconditioner
implementations are delegated to `torchalg`.

## What Stays Here

- Solver and comparison configuration DTOs:
  - `SolverConfig`
  - `SolverParams`
  - `ComparisonData`
  - `ComparisonGeneral`
- Workflow/reporting DTOs:
  - `CGComparisonResult`
  - `ComparisonResult`
  - `PlotPaths`
  - recommendation records
- Comparison orchestration helpers that package `torchalg` solver output for
  neuralls reporting workflows. The reference `x*` is a Jacobi-preconditioned
  `torchalg.pcg` solve (`reference_solution`) driven until its relative
  residual is at most `rtol * reference_precision_margin`, clamped so the
  target never asks below float64 machine precision; `reference_precision_margin`
  defaults to `1e-2` and is configurable via `SolverParams.reference_precision_margin`.
  A dense direct solve (`O(n^3)` per factorization) doesn't scale to the
  systems this module compares preconditioners on; Jacobi-PCG stays `O(n)`
  per iteration regardless of system size, at the cost of no longer being a
  ground truth from a structurally independent algorithm. `compute_reference_solution`
  runs on whichever device its `A`/`b` are already on — `compare_preconditioners`
  computes it once per comparison, on GPU, before the preconditioner loop, and
  passes it as `run_cg_comparison(..., x_exact=...)` to every preconditioner
  (including the `"none"` baseline) instead of each one re-deriving it from
  scratch. The resulting `x*` is passed into `pcg`/`flexible_cg` as
  `x_exact=`, so `torchalg`
  tracks the exact energy-norm error `||e_k||_A / ||e_0||_A` (the norm CG
  minimizes; every curve starts at 1) every iteration from a single dot
  product — no per-iterate vector tracing needed for this metric, so the solve
  always runs at `TraceMode.MINIMAL`. `extract_energy_error`
  (`error_metrics.py`) reads this off the returned `SolverResult` into
  `CGComparisonResult.error_history_a_rel`. When the reference solve fails (or
  `x_exact` otherwise isn't available), `error_history_a_rel` stays unset and
  `error_bound_a_rel` is populated instead from a Golub-Meurant lower bound on
  the same quantity (`golub_meurant_error_bound`, driven by `energy_decrements`
  — always available, no reference solution needed) — the two fields are
  mutually exclusive, an approximate bound never masquerades as the exact
  error. A failing reference solve or energy-error extraction leaves those
  metrics unset instead of failing the preconditioner. Between preconditioners
  it calls `shared.device.release_device_memory()` — cross-layer since
  `composition/assignments` also calls it between sweep children and pipeline
  stages, not solver-comparison-specific, so it lives in `shared/`, not here.
- Validation and artifact export helpers used by platform/composition layers.

## What Lives In Torchalg

- `torchalg.pcg`
- `torchalg.flexible_cg`
- `torchalg.monitoring.TraceMode`
- `torchalg.monitoring.IterationHistory`
- `torchalg.monitoring.analysis.golub_meurant_error_bound`
- `torchalg.models.result.SolverResult`
- `torchalg.preconditioners.*`

Production code must not import local solver factories, solver classes,
strategy classes, monitoring implementations, or preconditioner algorithms
from this package.

## Boundary Rule

Composition-layer adapters map neuralls config and loaded tensor data to
`torchalg` runtime objects. Solver-facing matrix, RHS, initial guess,
residuals, and preconditioner state are `torch.Tensor` values. Conversion to
NumPy is reserved for reporting, storage, and diagnostics that still emit
NumPy-backed artifacts.

Test coverage follows the same split: `torchalg` owns all solver/preconditioner
algorithm tests (exactness benchmarks, paper reproductions, unit tests for CG
variants and preconditioner implementations). `neuralls` tests only its own
composition/platform glue — config-to-object factories, adapters, and export
utilities — never the algorithm behavior itself.
