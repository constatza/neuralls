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
  - `CGComparisonResult` — a unified, three-stage cost model instead of bare
    optional floats, so "not applicable," "applicable but unknown," and
    "applicable and known" are three distinct, checkable states that can
    never collapse into each other:
    - `generation_cost: StageCost | None` — dataset-generation cost feeding
      this preconditioner's fit/train step. `None` when generation doesn't
      apply at all (classical/geometric AMG, standard/Jacobi/IC0 — no
      training data). When present, always `HISTORICAL` (read back from the
      dataset's own manifest, `generation_duration_seconds`) or `UNAVAILABLE`
      — a comparison run never generates a dataset itself, that's a separate,
      earlier pipeline stage; see `composition/comparison/_generation_cost.py`.
    - `setup_cost: StageCost | None` — cost to fit/train/build the
      preconditioner itself. `None` only for a placeholder result where no
      build was attempted (`comparison_run.py::_breakdown_result`). For a
      checkpoint-backed preconditioner (a POD-2G basis fit ahead of time, a
      trained neural preconditioner), this is not the freshly-measured
      checkpoint-load/coarse-assembly overhead — it's that checkpoint's real,
      historical fit/train duration, charged back from the origin job's own
      MLflow run (`resolved_run_id` on the config's `NeuralCheckpointRef`) by
      `composition/comparison/comparison_run.py::_resolve_setup_usage`, tagged
      `CostProvenance.HISTORICAL`. When that lookup can't run or finds
      nothing, the fresh (negligible) measurement is used instead but tagged
      `CostProvenance.UNAVAILABLE` — never silently indistinguishable from a
      genuinely fast build (tagged `MEASURED`). See `shared/types.py`'s
      `CostProvenance` docstring for the full three-value contract.
    - `solve_time_seconds`/`solve_peak_memory_bytes` — unchanged bare fields;
      solve is always live-measured in a comparison run, so there's no
      provenance question for it.
    - `StageCost` (`wall_time_seconds`, `peak_memory_bytes`, `provenance`) is
      the one value object both `generation_cost` and `setup_cost` share.
    - Derived properties: `total_time_seconds` (sums generation + setup +
      solve, **excluding** any `UNAVAILABLE`-provenance component — counting
      an unknown cost as `0.0` would silently understate the total) and
      `has_unavailable_cost` (true when generation or setup exists but
      couldn't be resolved to a real number — plots use this to skip a result
      rather than render a misleading position/value).
  - `ComparisonResult`
  - `PlotPaths` — including `generation_time_barplot`/`setup_time_barplot`/
    `solve_time_barplot`/`peak_memory_barplot`/`time_breakdown_barplot`
    (the last now a three-segment generation/setup/solve stacked bar, each
    segment styled by its own `CostProvenance`) and `work_precision`
  - recommendation records
- `cost_metrics.py`: pure size-normalized/throughput functions
  (`iterations_per_second`, `time_per_dof_per_iteration`, `setup_time_per_dof`,
  `generation_time_per_dof`, `peak_memory_per_dof`) — take a `CGComparisonResult`
  plus the comparison's `system_size` (from `ComparisonResult.matrix_shape[0]`,
  since every preconditioner in one comparison shares the same matrix) so cost
  is comparable across comparisons run on different-sized matrices. Both
  `*_time_per_dof` functions are deliberately provenance-agnostic — they
  return a number whenever the corresponding `StageCost` is present,
  regardless of its provenance; provenance-based filtering is a
  presentation-layer concern (plots, MLflow logging), not this pure-math
  layer's job.
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
  passes it as `run_cg_comparison(..., x_exact=...)` to every configured
  preconditioner instead of each one re-deriving it from scratch. The runner
  executes exactly the mapping supplied by composition; it never injects or
  infers a baseline from a result-key string. The resulting `x*` is passed into
  `pcg`/`flexible_cg` as
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
  `run_cg_comparison` wraps each `_solve_one` call in
  `shared.device.track_resource_usage`, attaching `solve_time_seconds`/
  `solve_peak_memory_bytes` to the resulting `CGComparisonResult` — cost is
  measured as an averaged/aggregate value per preconditioner (this module
  makes no attempt at a true per-iteration cost curve, since `torchalg`
  exposes no per-iteration timing/memory hook). Preconditioner *construction*
  time/memory (checkpoint load or inline fit, e.g. POD-2G's SVD) is measured
  the same way one layer up, in
  `composition/comparison/comparison_run.py::_run_preconditioner` around
  `create_preconditioner()` — this module never measures its own
  preconditioner's setup cost, since it only ever receives already-built
  `Preconditioner` instances.
- Validation and artifact export helpers used by platform/composition layers.

Comparison results intentionally contain solver behavior only. Raw and
preconditioned condition numbers are not computed, plotted, tracked, or
serialized by the comparison workflow; optional spectral investigations use
`neuralls.domain.analysis.spectra` explicitly outside the CG hot path.

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
