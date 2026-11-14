# Application Module

The application package holds use-case logic only.

## Package Map

- `inference/prediction.py`: batch prediction execution over prepared inference data through a domain-owned inference predictor port; the caller passes the device whose memory is measured
- `inference/models.py`: `InferenceConfig` and `InferenceData` DTOs (`InferenceData.source` is a `DataSource`)
- `models.py`: `AssignmentResult` (the closed union `AssignmentSucceeded | AssignmentFailed`: a success always holds its `mlflow_run_id`, a failure always holds its `error`; both carry the assignment's typed `TrainingPresentation`, and sinks render its text) and `TaskResult` (`TaskSucceeded | TaskFailed`) reporting DTOs

Comparison logic lives directly in `domain/solver/comparison.py` and is called
from `composition/comparison/`; there is no application-layer redirect for it.

## Semantic Difference

Application code owns decisions about what a workflow does, not how runtime
systems are wired or where artifacts live. If a module starts resolving config
paths, creating MLflow clients, or touching filesystem layout, it belongs in
`composition` or `platform` instead.

## Boundary

Application modules do not load configs, resolve workspaces, create MLflow
runs, or touch filesystem layout concerns directly. Cross-layer DTOs that are
shared across multiple layers live in `domain/` or `shared/`, not here.
The same no-ambient-action rule as `domain/` applies (no logging, clock reads, process-wide
caches, CUDA calls or device choice), checked by `tests/architecture/test_io_boundary.py`. The
caller passes the device whose memory is measured; composition passes `resolve_device()`. The
DLKit predictor picks its own device (`device="auto"`), so the two are expected to agree but
nothing checks it.
