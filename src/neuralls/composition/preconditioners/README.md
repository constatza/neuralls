# Preconditioners

Turns a preconditioner configuration into a torchalg preconditioner for a matrix in a given format. This is the only place that translates config models into solver objects.

## Modules

- `factory.py`: `create_preconditioner` dispatches on `(type, matrix format)` through a builder registry (`_lookup_builder`). Dense AMG also returns its coarsening strategy (`create_preconditioner_with_coarsening`), because the realized coarse dimension is only known from that object.
- AMG coarsening is built per kind: target-dimension search, POD-2G (reconstructed from a fitted checkpoint, or fit inline from snapshots), neural POD-2G (fit on a model's predictions), and aggregation. A checkpoint is read from `active_checkpoint_path`, which prefers the resolved MLflow artifact.
- `pod_fittable.py`: the dlkit `Fittable` adapter over torchalg's `PODCoarseningStrategy`. It must stay importable at its module path, because `FitJobConfig` jobs reference it by name.
- `_weighting.py`: maps `SnapshotWeightingConfig` to torchalg's POD row-scale functions.

The sparse AMG preset builds its aggregation internally, so for CSR AMG the coarsening is `None`.
