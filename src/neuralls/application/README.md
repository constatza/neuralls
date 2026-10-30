# Application Module

The application package holds use-case logic only.

## Package Map

- `inference/prediction.py`: batch prediction execution over prepared inference data through a domain-owned inference predictor port
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
