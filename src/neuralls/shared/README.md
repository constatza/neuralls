# Shared Module

The shared package holds cross-layer primitives only.

## Package Map

- `constants.py`: package-wide constants and config keys
- `types.py`: shared numerical/config metadata types (`ScaleMetadata`, `MatrixNormType`, `MatrixFormat`, `PreconditionerFamily`, `CostProvenance`, `GenerationRepeatability`, comparison RHS enums) and benchmark identities (`SystemId`, `PreconditionerName`, `SetupEventId`) used by ≥2 layers; `SystemMatrix` is the closed alias `np.ndarray | csr_array` for a system matrix in either format. `GenerationRepeatability` classifies whether a preconditioner's fit/train cost is paid once (checkpoint-backed) or reruns per comparison system (no checkpoint identity set) — a resolved-config fact (`composition/comparison/_generation_cost.py::resolve_generation_repeatability`), not an algorithm fact, so it lives here next to `CostProvenance` rather than in `domain/solver`.
- `enum_codecs.py`: pure semantic enum encoders/decoders shared by domain, composition, and platform
- `scientific_description/`: standard-library-only core for turning validated configuration and realized runtime facts into publication-ready labels (`docs/plan.md`, Phase 1). `models.py` holds the pure value types — `FieldKey` (validated snake_case identifier), `FieldImportance` (`essential`/`variant`, drives length-budget dropping), the `IntegerFact`/`DecimalFact`/`TextFact` sum type, and `ScientificDescription` (a base name plus ordered facts and ordered context facts, no rendered string, no I/O). `policy.py` holds `RenderTarget` (`plain`/`mathtext`/`latex`), `SymbolSpec`, the default symbol table (`N`, `k`, `m`, `c`, `θ`, ...), `NotationPolicy`, and `RenderProfile` (visibility/order/precision/length budget — `LEGEND_PROFILE`, `LOG_PROFILE`, `TITLE_PROFILE`, `DETAILED_PROFILE`). `render.py::render()` is the only function that turns a description into text; it drops context facts before variant facts before essential facts when `max_length` is exceeded, and never truncates with an ellipsis. Text-valued facts are escaped for mathtext/latex targets (an underscore in a class name would otherwise be a subscript). The default symbol table also carries the preconditioner-label vocabulary (`τ` drop tolerance, `ν1`/`ν2` smoothing steps, `ω_s` smoother damping, `k`, `k_r`, `η`, `n_it`, `θ_ad`, ...), so a presentation override of any of them applies uniformly. Per-domain describers live with their domain (`platform/reporting/preconditioner_labels.py`, `domain/generation/`), not here.
- `digest.py`: deterministic SHA-256 identity primitives — `canonical_digest` (fail-fast canonical JSON of configs; unknown types and unmarked `Path` fields raise), `content_digest` (file/directory bytes), `toml_digest` (parsed TOML meaning, comments ignored), `array_digest` (logical array content, storage-format independent) — plus the `Cosmetic`/`InputData`/`InputConfig` field markers that say how a config field contributes to identity
- `device.py`: `release_device_memory()` returns cached CUDA blocks to the driver between CUDA-heavy operations. `track_resource_usage(device)` supplies GPU-synchronized wall time and peak memory; `begin_resource_usage`/`end_resource_usage` expose the same measurement across split lifecycle callbacks. `repeat_timed` returns a representative value plus all samples, while `repeat_timed_outcome` additionally retains duration and memory when the measured call raises. `WorkloadMemoryObserver` combines independently measured operation peaks over a method lifetime without resetting CUDA counters around nested setup/solve timers. CPU observations use process-wide monotonic `ru_maxrss` (`peak_wset` on Windows), so a scoped workload delta is a lower bound when an earlier workload established the process high-water mark.
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
