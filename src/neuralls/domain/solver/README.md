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

`run_cg_comparison` uses the failure-aware repeat timer, so a solver exception
still yields a measured failed attempt with its representative duration,
samples, and peak memory rather than an unmeasured zero-cost row.

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

`validate_setup_solve_frame_links` is the shared frame-level reuse check: each
solve links to exactly one completed setup with matching dataset, matrix, and
method identity. Platform Parquet readers call the same validator before an
artifact can be reused.

The setup and solve Parquet frames are canonical for realized event timing,
repeat samples, outcomes, per-event memory, and attempted/converged counts.
They intentionally do not persist the planned `SystemTag` sequence or the
method-lifetime peak observed by the Task 2 reducer. `summary.toml` is therefore
canonical for planned membership, missing system IDs, and workload-lifetime
memory; those values cannot be reconstructed from only the two event frames.

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
to different recipes. Fine recipe summaries allocate that setup using the
complete matrix membership; medium/global summaries retain the whole setup
once. Workload totals use `math.fsum` over representative event values;
warmups and individual repeat samples are never added to totals.

`reporting/classification.py` consumes planned tags, setup events, solve
attempts, and a report level. A `ConvergenceSummary` keeps planned, attempted,
and converged counts distinct and reports iteration statistics over converged
attempts only. `Solved` carries matrix identity, iterations, solve seconds, and
amortized-online seconds. `Failed` carries matrix identity, iterations, and a
reason, but no imputed numeric cost; the measured failed-attempt time remains
only in the raw event/frame and workload totals.

## Benchmarking statistics

### Comparison vocabulary

The benchmark separates costs by when they are paid. **Generation** is the
offline construction of a dataset or training artifact; it is reported only
when a measured historical value is available. Checkpoint and input-data
loading, validation, staging, plotting, persistence, and MLflow work are
excluded from online timing. **Setup** is method preparation for one matrix;
**solve** is one attempted RHS. **Workload online time** is the sum of setup
and solve events for the workload. **Amortized online time per RHS** is solve
time plus that matrix setup divided by the planned RHS count. Thus no
checkpoint or data-read cost is silently charged to a method.

The **success rate** is converged attempts divided by planned RHS systems;
failures remain in the denominator and are never assigned a penalty time. An
**empirical cumulative distribution function (ECDF)** shows the fraction of
planned systems solved within an absolute time budget. A **Dolan--Moré
performance profile** instead shows the fraction within a factor of the
fastest method on each system. Profiles are emitted only for one homogeneous
recipe; heterogeneous recipes are reported as workload mixtures.

A **paired time** comparison uses the same matrix/RHS system for a configured
reference and method. Its **speed ratio** is reference time divided by method
time, so values above one favor the method. Pair counts retain both-solved,
reference-only, and method-only failures. Uncertainty resamples whole
**(dataset, matrix) clusters**, preserving dependence among sibling RHS
observations.
The reference is always the explicitly configured
`reference_preconditioner`; no observed result may select a fallback. If the
reference is absent or unavailable, paired ratios are omitted.

These are deterministic heterogeneous systems, not repeated COCO-style
restarts of one stochastic problem, so ordinary benchmark sweeps do not have
a meaningful expected running time (ERT). ERT would be appropriate only for
a separately designed restart experiment with a specified success
distribution and stopping rule.

`domain/benchmarking/` provides:

- Dolan–Moré performance profiles and absolute ECDFs that retain failures in
  their denominators;
- explicit `solve_seconds` and `amortized_online_seconds` cost scenarios;
- paired reference/method ratios using only positive-time both-solved systems
  while retaining all both-solved, reference-only, and method-only counts;
- percentile intervals that resample whole `(dataset, matrix_id)` clusters with
  every sibling RHS observation; paired Wilcoxon p-values are not reported.

Performance profiles normalize by the fastest method on each problem. A
failed attempt remains censored rather than receiving a penalty time, and
all-failed systems remain in every profile denominator. Paired
comparisons require a `ConfiguredReference`; `NoReferenceConfigured` and
`ReferenceUnavailable` produce no ratios, and no observed outcome can select a
fallback reference.

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

`BenchmarkMethodSink` is the narrow persistence port for a completed method
workload. Its domain-owned `MethodWorkload` payload carries the validated
method configuration and diagnostics as required data alongside canonical
events; the port
accepts only `(workload, summary)`. Metric calls are forbidden during event
emission. MLflow run ownership stays in composition, so the domain port cannot
create system runs or method grandchildren.

Composition maps typed configuration and loaded tensors to `torchalg` runtime
objects. Platform code persists and plots returned domain payloads. Domain code
does not parse TOML, open MLflow runs, resolve filesystem layout, or write
artifacts.
