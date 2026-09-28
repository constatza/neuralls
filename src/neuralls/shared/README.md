# Shared Module

The shared package holds cross-layer primitives only.

## Package Map

- `constants.py`: package-wide constants and config keys
- `types.py`: shared numerical/config metadata types (`ScaleMetadata`, `MatrixNormType`, `PreconditionerFamily`, comparison RHS enums) used by ≥2 layers
- `enum_codecs.py`: pure semantic enum encoders/decoders shared by domain, composition, and platform
- `digest.py`: deterministic SHA-256 identity primitives — `canonical_digest` (fail-fast canonical JSON of configs; unknown types and unmarked `Path` fields raise), `content_digest` (file/directory bytes), `toml_digest` (parsed TOML meaning, comments ignored), `array_digest` (logical array content, storage-format independent) — plus the `Cosmetic`/`InputData`/`InputConfig` field markers that say how a config field contributes to identity
- `device.py`: `release_device_memory()` — returns cached CUDA blocks to the driver; called between CUDA-heavy operations that share one long-lived process with no allocator reset of their own (solver comparisons between preconditioners, the training sweep between fit-job children, the case pipeline between its training and comparison stages). `track_resource_usage(device)` — GPU-synchronized wall-time + peak-memory measurement context manager, used by `domain/solver/comparison.py` (CG solve), `composition/comparison/comparison_run.py` (preconditioner construction), and `composition/generation/dataset_builder.py` (dataset generation, stamped onto `DatasetManifest`) so the sync/reset/read dance around a measured region lives in one place; peak memory is always a real number on both CPU (`resource.getrusage` RSS delta) and GPU (exact `torch.cuda` scoped peak), never silently `None` on one device type. `begin_resource_usage(device)`/`end_resource_usage(token)` — the same measurement split into two calls, for callers that can't use a single `with` block because the start and end of the measured region happen in two separate callback invocations (`composition/assignments/training_batch.py`'s `_ChildTimingTracker` bridges `dlkit`'s `on_child_planned`/`on_child_completed`/`on_child_failed` lifecycle hooks this way, since every dlkit training executor hardcodes `duration_seconds=0.0` and never logs a real one itself); `track_resource_usage` is a thin wrapper over this pair.

## Semantic Difference

Shared code exists only to prevent duplication across the other five top-level
packages. A module belongs here when it is pure, dependency-light, and used by
multiple layers. If a helper is specific to one owner package, it should live
with that owner instead of becoming generic by convenience.

## Boundary

Shared code must stay dependency-free with respect to the other architectural
layers. If a type is needed by more than one layer, move it here instead of
letting `application`, `platform`, or `composition` import one another.
