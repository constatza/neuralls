# Domain Presentation

What datasets, methods, and comparison cohorts are called, as typed values. The
vocabulary (facts, the fact catalog, notation policy, rendering of one
`ScientificDescription`) lives in `shared/scientific_description/`; this package owns
the parts that know what a dataset, a dataset-trained method, or a title is.

- `provenance.py`: `DatasetSourceProvenance`, `StrategyProvenance` (realized `rows`, algorithmic `parameters`, closed `StrategySeed = ExplicitSeed | DerivedSeed | NoSeed`; its `facts` derive the `N` and `s` facts so a count or seed cannot be stored twice), `DatasetGenerationProvenance` (unique strategy kinds whose rows sum to `generated_rows`, plus the `root_seed` that derived streams come from) and `DatasetPresentation`.
- `strategy_names.py`: `strategy_base_name`, the exhaustive controlled display name of each `GenerationStrategyKind`.
- `qualifier.py`: `dataset_qualifier` describes a dataset by its strategies; a single strategy's `N` is the dataset total, a mixture adds a leading `Mixture (N=…)` description. Seeds stay in the structured provenance only.
- `titles.py`: the validated `EditorialPresentation`, `ComparisonScopePresentation`, and the closed `ComparisonTitle` union.
- `methods.py`: the presentation lifecycle. `StandaloneMethodPresentation` / `DatasetQualifiedMethodPresentation` (closed union `ConfiguredMethodPresentation`; only the qualified variant owns a `DatasetPresentation`), `UnvalidatedComparisonPresentation` -> `PreparedComparisonPresentation` -> `RealizedMethodPresentation`; a finished workload reports an `ExecutedMethodPresentation` (realized when any setup succeeded, otherwise still prepared). Prepared and realized types validate on construction, so they cannot be built around `planner.py`. No state stores rendered text, and assignment identity is never a fact.
- `planner.py`: the pure transitions. `prepare_presentations` turns an unvalidated cohort into the only state execution and tracking accept: it keeps every algorithmic fact, scheduling facts (describers emit them only when non-default), and a reproducibility fact such as a seed only when its value differs between methods of the same family. Collisions are decided on structured compact facts plus the dataset qualifier, never on rendered text, and raise `DuplicateMethodPresentationError` before any run opens. `realize_method` attaches per-system realized facts; `prepared_of` and `realizations_of` read either `ExecutedMethodPresentation` outcome; `reduce_realized_facts` keeps only facts equal on every system of a population, so a group label never claims a matrix-varying value.
- `render.py`: `render_method` (appends the training dataset exactly once), `render_dataset`, and `render_title`, built on the shared `render`.

Import from these submodules directly; `__init__.py` re-exports nothing.
