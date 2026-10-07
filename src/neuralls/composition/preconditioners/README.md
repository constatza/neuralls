# Preconditioners

Turns a preconditioner configuration into a torchalg preconditioner for a matrix in a given format. This is the only place that translates config models into solver objects.

## Modules

- `factory.py`: `create_preconditioner`/`create_preconditioner_with_coarsening` are the single construction path. Both just resolve the matrix format (`_resolve_format_input`, `builders.py`), look up the `(type, matrix format)` builder (`_lookup_builder`), and call it — every builder returns `(Preconditioner, CoarseningStrategy | None)`, so the table is the only dispatch mechanism; no type gets a special case before it.
- `builders.py`: the per-`(type, matrix format)` builder registry (`_BUILDERS`) and all 18 `_build_*` functions it dispatches to. Every builder constructs its `torchalg.Preconditioner` with hyperparameters only, then calls `.setup(matrix)` before returning — `torchalg`'s two-phase contract requires an explicit `setup(matrix)` call before `apply()` for every preconditioner kind, so `create_preconditioner`/`create_preconditioner_with_coarsening` always hand back an already-`setup()`, ready-to-apply preconditioner. Dense AMG's builder (`_build_dense_amg`) is the one whose coarsening is a real object rather than `None`, because the realized coarse dimension is only known from that object. `_DENSIFY_FOR_CSR` names the dense-only types (neural, neural AMG) that get their CSR input densified first, with one `O(n^2)` warning.
- `coarsening.py`: AMG coarsening-strategy construction (`_build_amg_coarsening`), dispatched per kind: target-dimension search, POD-2G (reconstructed from a fitted checkpoint, or fit inline from snapshots), neural POD-2G (fit on a model's predictions), and aggregation. A checkpoint is read from `active_checkpoint_path`, which prefers the resolved MLflow artifact.
- `schedule.py`: `create_scheduled_preconditioner` wraps an already-`setup()` primary preconditioner in a `ScheduledPreconditioner` that switches to a fallback after `PreconditionerScheduleConfig.limit_iters`, or does nothing if unscheduled. `ScheduledPreconditioner` itself needs its own `setup(matrix)` call to ready both its primary and fallback branches, so wrapping requires passing `matrix` through.
- `pod_fittable.py`: the dlkit `Fittable` adapter over torchalg's `PODCoarseningStrategy`. It must stay importable at its module path, because `FitJobConfig` jobs reference it by name.
- `_weighting.py`: maps `SnapshotWeightingConfig` to torchalg's POD row-scale functions.

The sparse AMG preset builds its aggregation internally, so for CSR AMG the coarsening is `None`.
