# Task 12 report: setup-aware comparison acceptance

## Outcome

Added one deterministic real-render workflow acceptance test covering:

- physical matrix IDs `5` and `9`;
- two RHS recipes (`sinusoidal ω=1` and `sinusoidal ω=2`);
- a measured zero-second static Identity setup and a matrix-dependent Jacobi setup;
- a configured Jacobi reference with asymmetric reference-only pairs for the deterministic
  nonconverging Identity method;
- raw setup/solve Parquet schemas, setup links, timing samples, summary totals, and attempted
  RHS denominators;
- fine-only profiles/ECDFs, medium/global mixture bars, concise titles, linear paired axes, and
  exact figure-manifest coverage;
- one MLflow session parent with one direct child per method, no grandchildren, method artifact
  placement, schema tags, and reducer-only method scalars.

The tracking sink now retains the configured reference object in figure metadata, so manifests
carry `configured:jacobi` instead of losing the configured method name behind a generic enum.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/workflows/test_task12_reporting_e2e.py -q -m integration`
  — 1 passed, with the documented MLflow SQLAlchemy deprecation and an empty-diagnostics legend
  warning.
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/platform/test_comparison_sink.py tests/platform/test_figure_metadata.py tests/domain/test_event_aggregation.py tests/domain/test_reference_comparison.py tests/workflows/test_task12_reporting_e2e.py -m 'not integration'`
  — 29 passed, 1 deselected.
- Focused Ruff and `ty` checks pass for the changed source and test files.
- The production/user-facing terminology scan over `README.md src/neuralls` is clean. The broader
  scan also finds the requested historical terms in the permanent implementation-plan checklist;
  those are acceptance-search terms, not user-facing definitions.

## Render check

The real-render rerun used:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/workflows/test_task12_reporting_e2e.py -q -m integration --basetemp=/tmp/task12-e2e-EK34PM
```

The inspected artifact was:

`/tmp/task12-e2e-EK34PM/test_setup_aware_reporting_rec0/mlartifacts/b52b3b43efcd46c79d3388aa3df369bb/artifacts/group_reports/paired_time_scatter_fine_sinusoidal-workload_sinusoidal_omega_1.png`

It visibly has the concise `Online Time` title, recipe scope subtitle, linear equal axes, diagonal,
and `jacobi (reference)` axis annotation.

## Scope boundary

No production SQLite fallback, new test framework, architecture ownership, or dependency edge was
introduced. The full suite and repository-wide type/lint gates remain parent-agent verification work.
