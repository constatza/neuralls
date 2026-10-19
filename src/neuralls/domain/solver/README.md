# Solver Workflow Boundary

`neuralls.domain.solver` no longer owns solver or preconditioner algorithms.
CG, PCG, FCG, monitoring primitives, solver results, and preconditioner
implementations are delegated to `torchalg`.

## Layout

- `comparison.py` runs the CG comparison; `comparison_summary.py` formats and
  ranks its results; `error_metrics.py` and `cost_metrics.py` derive per-run
  metrics.
- `models/` holds one concept per file: `cg_result` (`CGComparisonResult`),
  `cost` (`StageCost`), `cost_fields`, `sharing`, `projection`, `labels`,
  `group_outcomes`, `plot_style`, `plot_paths`, `presentation`,
  `recommendations`, `result` (`ComparisonResult`) and `config`.
- `reporting/` turns measured results into report groups: `metrics_frame`,
  `solve_time`, `report_levels`, `aggregation`, `classification`.

## What Stays Here

- Solver and comparison configuration DTOs:
  - `SolverConfig`
  - `SolverParams`
  - `ComparisonData`
  - `ComparisonGeneral`
- Workflow/reporting DTOs:
  - `CGComparisonResult` — a three-stage cost model: generation, setup, and
    solve, each measuring a conceptually different thing, so a result never
    has to encode "which stage is this number" in a comment or a naming
    convention:
    - `generation_cost: StageCost | None` — dataset-generation cost feeding
      this preconditioner's fit/train step. `None` when generation doesn't
      apply at all (classical/geometric AMG, standard/Jacobi/IC0 — no
      training data). When present, always `HISTORICAL` (read back from the
      dataset's own manifest, `generation_duration_seconds`) or `UNAVAILABLE`
      — a comparison run never generates a dataset itself, that's a separate,
      earlier pipeline stage; see `composition/comparison/_generation_cost.py`.
      `StageCost` (`wall_time_seconds`, `peak_memory_bytes`, `provenance`)
      stays the value object for this one still-ambiguous stage.
    - `setup_cost: float` — wall-clock seconds to construct and `.setup()`
      this preconditioner. Always measured, every run, for every algorithm —
      `torchalg.Preconditioner`'s two-phase contract (`setup()` must run once
      before `apply()`, taking `matrix` for a `MatrixSetupPreconditioner` or
      no argument at all for a `StaticSetupPreconditioner`) makes this
      deterministic, so there is no provenance question for it any more than
      there is for
      `solve_time_seconds`. Concretely, what gets measured per algorithm:
      IC0/ILU/Jacobi's factorization; classical/POD-2G AMG's
      hierarchy build (strength-of-connection pass, aggregation, transfer
      operators, and — for POD-2G — the SVD fit plus the Galerkin coarse
      triple product); AdaptiveSA/BootstrapAMG's bootstrap cycles. Checkpoint
      loading (a neural preconditioner's, a fitted POD-2G basis's) is deliberately
      *not* included: it is file I/O done once per dataset ahead of the timed
      build and reported separately as `checkpoint_load_s`, so a neural
      preconditioner's setup is near zero. `setup_peak_memory_bytes` is the
      matching peak-memory field (`None` only for a placeholder result where
      no build was ever attempted). Measured in
      `composition/comparison/comparison_run.py::_run_preconditioner`, wrapping
      both the preconditioner's own construction+`.setup()` and any
      scheduling wrapper's `.setup()` in one `shared.device.track_resource_usage`
      block.
    - `solve_time_seconds`/`solve_peak_memory_bytes` — unchanged bare fields;
      solve is always live-measured in a comparison run.
    - Derived properties: `total_time_seconds` (sums generation + setup +
      solve; `setup_cost` is always included since it is always a real
      number, `generation_cost` excludes any `UNAVAILABLE`-provenance
      component — counting an unknown cost as `0.0` would silently understate
      the total) and `has_unavailable_cost` (true only when `generation_cost`
      exists but couldn't be resolved to a real number — `setup_cost` can no
      longer be unavailable — plots use this to skip a result rather than
      render a misleading position/value).
  - `ComparisonResult`
  - `PlotPaths` — including `generation_time_barplot`/`setup_time_barplot`/
    `solve_time_barplot`/`peak_memory_barplot`/`time_breakdown_barplot`
    (the last a three-segment generation/setup/solve stacked bar; only the
    generation segment can still be styled `UNAVAILABLE`) and `work_precision`
  - recommendation records
  - `SetupSharingPolicy`/`ProjectedComparisonResult` — a scaling-law
    projection aggregated across the systems of one report group (see the
    levels below; never the N it extrapolates to). Two independent booleans
    (`share_setup`/`share_generation`) decide, per cost term, whether it
    amortizes (contributes its one-time value once, regardless of N) or
    repeats (contributes `N *` its per-system average);
    `projected_total_seconds(n)`/`projected_avg_per_system_seconds(n)` are
    the resulting closed-form extrapolations, never a real N-system run.
    `domain/solver/reporting/aggregation.py::report_level` builds these
    from systems' `CGComparisonResult`s (via the long-form `build_metrics_frame`
    polars table) plus a pre-computed sharing policy
    per preconditioner — it cannot compute `share_setup` itself (that needs
    the live `torchalg.Preconditioner` object's role interface, which no
    longer exists by aggregation time), so the execution layer
    (`composition/comparison/_preconditioner_setup.py::DatasetPreconditionerCache`)
    computes and passes it in. See `composition/README.md`'s "report levels"
    and "setup cache" sections for the full wiring.
  - `reporting/` — `metrics_frame.py::build_metrics_frame` turns `Mapping[SystemTag,
    Mapping[str, CGComparisonResult]]` into one polars row per (system,
    preconditioner) (`dataset`, `recipe`, `matrix_id`, `rhs_ordinal`,
    `system_id`, `preconditioner`, `setup_s`, `generation_s`, `solve_s`,
    `iterations`, `time_to_solution_s`, `peak_memory_bytes`). `report_level(
    frame, level, sharing_by_name)` groups by the level's keys plus the
    preconditioner, where `level` is the closed `FineLevel` (dataset +
    recipe) | `MediumLevel` (dataset) | `GlobalLevel` (no keys), and returns
    projections per `GroupLabel` (`FineLabel` | `MediumLabel` | `GlobalLabel`;
    the latter two carry `mixture_of`, the pooled recipes, because their
    spread mostly reflects recipes differing). Unmeasured values are nulls:
    unmeasured solve times and unavailable generation costs are skipped by
    the means and spreads rather than read as 0.0, and the generation
    average divides by the available results only. A spread is `None` below
    two values; a one-system group is reported, with its n. The only
    statistical control is the per-system `TimingRepeatPolicy`.
    `time_to_solution_by_system` gives each system's comparable
    time-to-solution per group; `system_counts` gives each group's distinct
    measured systems and matrices; `report_key`/`label_key`/`label_for_tags`
    relate planned systems to labels, including a group nothing was measured in.
  - `ComparisonGroupOutcome` (`GroupAggregated | GroupSkippedAllCacheHit`) —
    the closed outcome of attempting to aggregate one report group: a
    projection to act on, or nothing measured this run. `GroupAggregated`
    carries its `label`, the level-unique `group_id` (`render_group_id`:
    `fine/<dataset>/<recipe slug>`, `medium/<dataset>`, `global`),
    `n_aggregated`/`n_matrices`/`n_in_group` (a partially-cached or
    partly-failed group is reported with a visible gap, handled like a full
    one) and `time_to_solution_by_system`, each system's point behind the
    means. `domain/solver/reporting/classification.py::classify_level` is the pure
    classifier producing one outcome per report group of a level from the
    planned systems' tags and the measured results;
    `composition/comparison/group_reporting.py` matches it exhaustively and
    hands each `GroupAggregated` and its `GroupPresentation` to the
    `GroupReportSink` port (`ports.py`). `PreconditionerPlotStyle`/
    `plot_style_maps` (label, family, colour key, marker key per
    preconditioner, recorded on `ComparisonResult.plot_styles`) are the one
    derivation of plot styling shared by per-run and per-group plots.
  - `COST_FIELD_REGISTRY` — one `(field_name, extractor)` pair per
    reportable scalar cost, read by both the cost bar-charts
    (`composition/comparison/_plots.py`, which owns each cost's title and
    filename in `COST_PLOT_STYLE`, kept in step by a test) and the per-run
    `comparison.toml` writer (`platform/reporting/artifacts.py::
    _save_comparison_toml`). `comparison.toml` also records
    `iterations_to_tolerance`: iterations to first reach each rung of a
    ladder stepping every 3 decades from 1e-3 down to the config's `rtol`
    (`cost_metrics.py::tolerance_ladder`/`iterations_to_tolerances`), read off
    the stored residual history. 1D arrays — per-iteration histories
    (`residual_history_rel/abs`, `error_history_a_rel`, `error_bound_a_rel`)
    and per-repeat timing samples (`setup_time_samples`,
    `solve_time_samples`) — go to one `histories.h5` per run instead of inline
    JSON (`platform/reporting/artifacts.py::write_histories_h5`).
  - `total_cost_per_iteration_seconds` is (generation + setup + solve) /
    iterations: the overhead-inclusive effective cost of an iteration, a
    diagnostic against `avg_iteration_time_seconds`. It depends on the
    stopping tolerance through the iteration count, so it is not a ranking
    criterion; `summarize_best_combinations` ranks converged runs by
    time-to-solution (`total_time_seconds`), then iterations.
  - Timing effort is a `shared.timing.TimingRepeatPolicy` (`SolverParams.
    timing`, from `[comparison_baseline.timing]`): `repeats`, `warmup`,
    `statistic` (median/mean/min), capped by `MAX_TIMING_REPEATS`. The default
    is one timed run. `shared.device.repeat_timed` applies it to both the
    solve (`run_cg_comparison`) and the preconditioner build; peak memory is
    taken from the very first execution (CPU peak RSS has no reset), and a
    repeat whose iteration count differs from the first logs a warning. The
    policy is `Cosmetic()` for comparison identity: changing it does not
    invalidate prior runs; pass `--force` to remeasure.
- `cost_metrics.py`: pure size-normalized/throughput functions
  (`iterations_per_second`, `time_per_dof_per_iteration`, `setup_time_per_dof`,
  `generation_time_per_dof`, `peak_memory_per_dof`) — take a `CGComparisonResult`
  plus the comparison's `system_size` (from `ComparisonResult.matrix_shape[0]`,
  since every preconditioner in one comparison shares the same matrix) so cost
  is comparable across comparisons run on different-sized matrices.
  `generation_time_per_dof` is deliberately provenance-agnostic — it returns a
  number whenever `generation_cost` is present, regardless of its provenance;
  provenance-based filtering is a presentation-layer concern (plots, MLflow
  logging), not this pure-math layer's job. `setup_time_per_dof` has no
  provenance to be agnostic about any more — `setup_cost` is just a float.
- `ports.py`: `CostRecorder` — a narrow Protocol (`record(result, *,
  child_run_tags)`) so composition-layer callers depend on an abstraction
  instead of importing `platform/tracking/comparison_tracking.py`'s MLflow
  logging function directly. `MLflowCostRecorder` (same module) is the
  concrete implementation, wired at the `composition/assignments/` call site
  that currently logs comparison results.
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
  exposes no per-iteration timing/memory hook). Preconditioner *construction
  and setup* time/memory (every algorithm's own `.setup()` call — hierarchy
  build, factorization, whatever that preconditioner's setup concretely means,
  except checkpoint loading, which runs ahead of the timed build) is measured the same way one layer up, in
  `composition/comparison/comparison_run.py::_run_preconditioner`, wrapping
  `PreconditionerService.create_preconditioner()` (which calls `.setup()`
  internally) and any scheduling wrapper's own `.setup()` — this module
  never measures its own preconditioner's setup cost, since it only ever
  receives already-built, already-`setup()` `Preconditioner` instances. On a
  `SystemPreconditionerCache` hit, `_run_preconditioner` skips rebuilding
  entirely and reports the real cost from whichever system built it
  first, rather than a fresh (and misleadingly near-zero) measurement.
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
- `torchalg.preconditioners.*`, including the `setup()` role-interface split:
  `MatrixSetupPreconditioner.setup(matrix, context=None)` for anything whose
  setup genuinely depends on the matrix's values (Jacobi, IC0, ILU, ICholesky,
  every AMG/BootCMatch variant), `StaticSetupPreconditioner.setup(context=None)`
  — no matrix argument at all — for anything whose setup never reads it
  (`NeuralPreconditioner`'s checkpoint binding, `CallablePreconditioner`/
  `LinearOperatorPreconditioner`'s wrapped callables, `Identity`). This is a
  property of each concrete class, not of "neural vs classical" — it is what
  `DatasetPreconditionerCache` checks (`isinstance(precond,
  StaticSetupPreconditioner)`) to decide whether a built preconditioner can
  be reused across sibling systems with a different matrix.

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
