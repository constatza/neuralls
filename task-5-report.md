# Task 5 report: versioned benchmark event artifacts

## RED/GREEN evidence

- RED: `uv run pytest tests/platform/test_benchmark_event_artifacts.py -q`
  failed during collection with `ModuleNotFoundError` for the missing
  `neuralls.platform.reporting.benchmark_events` module.
- GREEN: the focused artifact suite passes with 4 tests.
- Adjacent artifact/domain suite passes with 45 tests:
  `tests/platform/test_benchmark_event_artifacts.py`, existing platform
  serialization/artifact tests, event aggregation, and event lifecycle.
- Adjacent workflow/group suite passes with 49 tests.
- Targeted `ty`, Ruff lint, Ruff format, `tach check`, and `git diff --check`
  pass for the changed platform/comparison/reporting scope.

## Files

- Added `platform/reporting/benchmark_events.py`, with schema-versioned
  Parquet writing/reading, setup-link validation, artifact-relative paths, and
  reducer-reconciled `summary.toml` output.
- Re-exported the benchmark artifact API from `platform/reporting/artifacts.py`.
- Added round-trip, outcome/memory/sample/order, schema, link, digest, and
  summary reconciliation tests in
  `tests/platform/test_benchmark_event_artifacts.py`.
- Added the stable method config digest to completed workloads and populate it
  from the resolved method config at execution time.
- Updated platform/composition reporting documentation.

## Self-review

Parquet frames are the canonical raw records and carry schema metadata version
1. The summary is generated only after `summarize_method_workload` reconciles
the supplied summary against the canonical event collections. Setup failures
remain setup rows and do not create solve attempts. Unknown/measured memory,
all repeat samples, event order, and setup links survive the round trip.

The only optional offline fields are explicitly generation-named
`generation_s` and `generation_source`; checkpoint and data I/O costs are not
serialized, derived, measured, or named.

## Concerns

Task 6 tracking-topology wiring is intentionally not included. The serializer
is available as a platform API and does not alter MLflow run ownership.

## Fix round 1

### RED/GREEN evidence

- RED: after adding the five review regressions, the focused suite produced 10
  fixture errors because `CompletedMethodWorkload` did not yet expose the
  required `generation_cost` field.
- GREEN: focused artifact tests pass with 10 tests; the adjacent artifact,
  domain, lifecycle, and composition suites pass with 60 tests.
- Targeted `ty`, Ruff lint, Ruff format, `tach check`, and `git diff --check`
  pass.

### Files and fixes

- `CompletedMethodWorkload` now requires a typed `Digest` and an explicit
  `generation_cost: StageCost | None`; all constructors and execution wiring
  use that field, eliminating generic offline-cost relabeling.
- Unknown workload memory writes only `memory.status = "unknown"`; measured
  workload memory writes `peak_bytes`.
- `validate_setup_solve_frame_links` is shared by domain frame validation and
  Parquet readers. It enforces one setup per link, matching dataset/matrix/
  method identity, and completed setup status.
- The writer rejects empty or malformed config digests (`sha256:` plus 64
  lowercase hexadecimal characters).
- Artifact tests now pin literal Polars schemas and complete ordered rows, and
  recompute summary counts, setup/solve/online totals, attempted average, and
  peak memory from the returned frames.
- Documentation now describes the shared frame-link validator and explicit
  generation cost boundary.

### Self-review and concerns

No checkpoint/data-I/O cost shape exists on the workload or serializer. The
only persisted optional cost is explicitly `generation_s`/`generation_source`.
The reader validates setup links when the sibling or explicit setup artifact is
provided, as required for artifact reuse. Task 6 tracking topology remains out
of scope.
