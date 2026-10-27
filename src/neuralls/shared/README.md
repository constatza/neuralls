# Shared Module

The shared package holds cross-layer primitives only.

## Package Map

- `constants.py`: package-wide constants and config keys
- `types.py`: shared numerical/config metadata types (`ScaleMetadata`, `MatrixNormType`, `MatrixFormat`, `PreconditionerFamily`, `CostProvenance`, `GenerationRepeatability`, `NormalizeType`, comparison RHS enums) and benchmark identities (`SystemId`, `PreconditionerName`, `SetupEventId`) used by ≥2 layers; `SystemMatrix` is the closed alias `np.ndarray | csr_array` for a system matrix in either format. `GenerationRepeatability` classifies whether a preconditioner's fit/train cost is paid once (checkpoint-backed) or reruns per comparison system (no checkpoint identity set) — a resolved-config fact (`composition/comparison/_generation_cost.py::resolve_generation_repeatability`), not an algorithm fact, so it lives here next to `CostProvenance` rather than in `domain/solver`.
- `enum_codecs.py`: pure semantic enum encoders/decoders shared by domain, composition, and platform
- `scientific_description/`: the vocabulary of scientific facts, free of infrastructure dependencies (no Pydantic, Matplotlib, MLflow, filesystem, or logging). `models.py` holds semantic value types only: `FieldKey`, the `IntegerFact`/`DecimalFact`/`TextFact`/`DisplayTextFact` sum type, `ValidatedLabel`, and `ScientificDescription`; none contains a rendered label or target markup. `catalog.py` is the single controlled registry for fact roles, configured/realized phase, order, precision, plain Unicode/MathText notation, categorical spellings, and visibility. Machine identity remains in explicit ID fields outside scientific facts, and `NEVER`-visible identity facts are excluded by both rendering entry points. `fact_definition()` and rendering reject an unregistered key, controlled categories reject unknown values, and text facts without a catalog formatter or a validated semantic representation are rejected; no path falls back to a Python field name, raw enum value, or arbitrary Python string. `policy.py` selects a render target, profiles, and optional symbol overrides. `render.py` renders one `ScientificDescription`: punctuation, deterministic numeric formatting, escaping, and target notation. `render()` retains every profile-selected fact, including `ALWAYS`-visible algorithmic context, and returns text with preferred wrap boundaries when a profile's length is exceeded; length limits are layout hints, never permission to delete an identifying scientific fact. Nothing is re-exported from `__init__.py`, so each name has one import path. Per-domain describers choose semantic facts and live with their owner package. Presentation states, dataset provenance, titles, and the rendering of methods, datasets, and titles live in `domain/presentation/`, because they know what a dataset or a dataset-trained method is.
- `digest.py`: deterministic SHA-256 identity primitives — `canonical_digest` (fail-fast canonical JSON of configs; unknown types and unmarked `Path` fields raise), `content_digest` (file/directory bytes), `toml_digest` (parsed TOML meaning, comments ignored), `array_digest` (logical array content, storage-format independent) — plus the `Cosmetic`/`InputData`/`InputConfig` field markers that say how a config field contributes to identity
- `device.py`: `release_device_memory()` returns cached CUDA blocks to the driver between CUDA-heavy operations. `track_resource_usage(device)` supplies GPU-synchronized wall time and peak memory; `begin_resource_usage`/`end_resource_usage` expose the same measurement across split lifecycle callbacks. `repeat_timed` returns a representative value plus all samples, while `repeat_timed_outcome` additionally retains duration and memory when the measured call raises. `WorkloadMemoryObserver` combines independently measured operation peaks over a method lifetime without resetting CUDA counters around nested setup/solve timers. CPU observations use process-wide monotonic `ru_maxrss` (`peak_wset` on Windows), so a scoped workload delta is a lower bound when an earlier workload established the process high-water mark.
- `workflow_failures.py`: closed batch/method/matrix locations for blocking comparison failures plus the narrow `WorkflowFailureSink` port shared by composition and tracking adapters.
- `timing.py`: `TimingRepeatPolicy` (validated, frozen: `repeats`, `warmup`, `statistic`, capped by `MAX_TIMING_REPEATS`) and `TimingStatistic` (median/mean/min) — how much measurement effort a timed region gets and how repeats collapse to one number. Default is one timed run.

## Semantic Difference

Shared code exists only to prevent duplication across the other five top-level
packages. A module belongs here when it is pure, dependency-light, and used by
multiple layers. If a helper is specific to one owner package, it should live
with that owner instead of becoming generic by convenience.

## Boundary

Shared code must stay dependency-free with respect to the other architectural
layers. If a type is needed by more than one layer, move it here instead of
letting `application`, `platform`, or `composition` import one another.
