# Solver Workflow Boundary

`neuralls.domain.solver` owns comparison DTOs and pure reporting reductions.
CG, PCG, FCG, monitoring primitives, solver results, and preconditioner
implementations are delegated to `torchalg`.

## Canonical benchmark data

`models/benchmark_events.py` defines the canonical online measurements:

- `MatrixSetupEvent` records one method setup for one matrix.
- `SolveAttempt` records one RHS attempt and requires its `SetupEventId`.
- setup and solve success/failure are closed variants;
- peak memory is either measured or explicitly unknown;
- representative seconds and every repeat sample are retained separately.

Online time is setup plus solve only. Checkpoint and dataset reads, reference
solves, staging, plotting, persistence, MLflow calls, and historical generation
or training remain outside it. Required computation after I/O belongs to a
timed setup or solve event.

`models/workload.py` represents a method summary as either complete or
explicitly incomplete. Both variants retain planned, attempted, and converged
RHS counts, setup and solve totals, total online workload time, attempted-RHS
average when defined, and workload peak memory. Incomplete summaries name the
unattempted systems and cannot be ranked.

## Event reductions

`reporting/event_frames.py` is the Polars serialization boundary. Nullable
columns encode closed event variants only in normalized frames; core domain
objects remain variant-based.

`reporting/aggregation.py` validates planned identities and solve-to-setup
links before deriving costs. Planned RHS membership is counted by
`(dataset, matrix_id)` before attempts are joined:

```text
allocated_setup_s = setup_s / planned_rhs_count
amortized_online_s = solve_s + allocated_setup_s
matrix_online_s = setup_s + sum(solve_s)
workload_online_s = sum(matrix_online_s)
```

One matrix setup is therefore charged once even when sibling RHS values belong
to different recipes. Workload totals use `math.fsum` over representative event
values; warmups and individual repeat samples are never added to totals.

`reporting/classification.py` consumes planned tags, setup events, solve
attempts, and a report level. A `ConvergenceSummary` keeps planned, attempted,
and converged counts distinct and reports iteration statistics over converged
attempts only. `Solved` carries matrix identity, iterations, solve seconds, and
amortized-online seconds. `Failed` carries matrix identity, iterations, and a
reason, but no imputed numeric cost; the measured failed-attempt time remains
only in the raw event/frame and workload totals.

## Benchmarking statistics

`domain/benchmarking/` provides:

- Dolan–Moré performance profiles and absolute ECDFs that retain failures in
  their denominators;
- explicit `solve_seconds` and `amortized_online_seconds` cost scenarios;
- paired reference/method ratios using only positive-time both-solved systems
  while retaining all both-solved, reference-only, and method-only counts;
- percentile intervals that resample whole matrix clusters with every sibling
  RHS observation.

Performance profiles normalize by the fastest method on each problem. A
failed attempt remains censored rather than receiving a penalty time.

## Diagnostic solver results

`CGComparisonResult` remains a solver-diagnostic DTO. Its
`solve_time_seconds` is the explicit PCG/FCG measurement and
`offline_generation_time_seconds` exposes available historical generation
cost separately. It does not aggregate generation, setup, and solve into an
online total; setup-aware aggregate reporting derives totals only from linked
benchmark events.

Per-iteration residual and energy-error histories, timing samples, and memory
diagnostics remain available for artifacts and single-run plots. Comparison
recommendations rank converged diagnostic results by measured solve time, then
iterations; workload ranking uses complete event summaries.

## Boundary rule

Composition maps typed configuration and loaded tensors to `torchalg` runtime
objects. Platform code persists and plots returned domain payloads. Domain code
does not parse TOML, open MLflow runs, resolve filesystem layout, or write
artifacts.
