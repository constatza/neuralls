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
