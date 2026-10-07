# Comparison workflow

Composition for comparing preconditioners on one system. The entry point runs every configured preconditioner on the same input and collects the results.

## Modules

- `comparison_run.py`: one comparison run across all configured preconditioners. Pure orchestration; I/O and algorithms are delegated.
- `config_assembler.py`: builds `ComparisonConfig` from the case registry, defaults and per-method overrides.
- `_input_resolution.py`, `source_handlers.py`: resolve the input system from a canonical right-hand-side source, one handler per source kind.
- `_linear_system.py`: loads and normalizes the linear system.
- `rhs_generation.py`: direct right-hand-side generators.
- `_preconditioner_setup.py`: creates preconditioners, their schedules, input bindings and the stopping criterion.
- `_generation_cost.py`: the dataset-generation cost attributed to a comparison.
- `result_keys.py`: stable result-key names. Changing a key breaks stored results.
- `_presentation.py`, `_plots.py`: labels and diagnostic plots for one run.
- `models.py`: the composition-layer DTOs of the workflow.
