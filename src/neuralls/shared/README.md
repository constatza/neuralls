# Shared Module

The shared package holds cross-layer primitives only.

## Package Map

- `constants.py`: package-wide constants and config keys
- `types.py`: shared numerical/config metadata types (`ScaleMetadata`, `MatrixNormType`, `PreconditionerFamily`, comparison RHS enums) used by ≥2 layers
- `enum_codecs.py`: pure semantic enum encoders/decoders shared by domain, composition, and platform
- `digest.py`: deterministic SHA-256 identity primitives — `canonical_digest` (fail-fast canonical JSON of configs; unknown types and unmarked `Path` fields raise), `content_digest` (file/directory bytes), `toml_digest` (parsed TOML meaning, comments ignored), `array_digest` (logical array content, storage-format independent) — plus the `Cosmetic`/`InputData`/`InputConfig` field markers that say how a config field contributes to identity
- `device.py`: `release_device_memory()` — returns cached CUDA blocks to the driver; called between CUDA-heavy operations that share one long-lived process with no allocator reset of their own (solver comparisons between preconditioners, the training sweep between fit-job children, the case pipeline between its training and comparison stages)

## Semantic Difference

Shared code exists only to prevent duplication across the other five top-level
packages. A module belongs here when it is pure, dependency-light, and used by
multiple layers. If a helper is specific to one owner package, it should live
with that owner instead of becoming generic by convenience.

## Boundary

Shared code must stay dependency-free with respect to the other architectural
layers. If a type is needed by more than one layer, move it here instead of
letting `application`, `platform`, or `composition` import one another.
