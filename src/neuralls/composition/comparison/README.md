# Comparison workflow

Composition for comparing preconditioners on one system. The entry point runs every configured preconditioner on the same input and collects the results. Aggregating results *across* several sibling systems (a `[[comparison_sweeps]]` group's projected scaling-law metric) is a level up, in `composition/assignments/comparison_batch.py` — this package only ever sees one system at a time.

## Modules

- `comparison_run.py`: one comparison run across all configured preconditioners. Pure orchestration; I/O and algorithms are delegated. `compare_preconditioners` takes an optional `cache: SweepPreconditionerCache | None` — when the caller is running one sibling of a `comparison_sweeps` group, this lets a preconditioner built for an earlier sibling be reused instead of rebuilt (see `_preconditioner_setup.py`).
- `config_assembler.py`: builds `ComparisonConfig` from the case registry, defaults and per-method overrides.
- `_input_resolution.py`, `source_handlers.py`: resolve the input system from a canonical right-hand-side source, one handler per source kind.
- `_linear_system.py`: loads and normalizes the linear system.
- `rhs_generation.py`: direct right-hand-side generators.
- `_preconditioner_setup.py`: creates preconditioners, their schedules, input bindings and the stopping criterion. Also owns `SweepPreconditionerCache`, scoped to one sweep group: it reuses a built preconditioner across sibling systems only when doing so is provably valid — always for a `torchalg.StaticSetupPreconditioner` (setup never reads the matrix), or for a `torchalg.MatrixSetupPreconditioner` only when every sibling resolves to the same matrix. A cache hit copies forward the real one-time `setup_cost`/peak-memory measurement rather than reporting a misleadingly-free 0.0.
- `_generation_cost.py`: the dataset-generation cost attributed to a comparison (`resolve_generation_cost`), and whether that cost is a one-time artifact or reruns per system (`resolve_generation_repeatability` — a `GenerationRepeatability`, driven by checkpoint presence on the resolved config, not by preconditioner type).
- `result_keys.py`: stable result-key names. Changing a key breaks stored results.
- `_presentation.py`, `_plots.py`: labels and diagnostic plots for one run.
- `models.py`: the composition-layer DTOs of the workflow.
