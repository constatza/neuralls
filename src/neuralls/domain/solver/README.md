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
  neuralls reporting workflows. `run_cg_comparison` traces every iterate and
  reduces it to `CGComparisonResult.error_history_a_rel` — the energy-norm
  error `||e_k||_A / ||e_0||_A` (the norm CG minimizes; every curve starts at
  1). The reference `x*` is a float64 direct solve with iterative refinement
  until its relative residual is at most `rtol * 1e-4`
  (`_reference_solution`), so it is far more precise than the solvers compared.
- Validation and artifact export helpers used by platform/composition layers.

## What Lives In Torchalg

- `torchalg.pcg`
- `torchalg.flexible_cg`
- `torchalg.monitoring.TraceMode`
- `torchalg.monitoring.IterationHistory`
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
