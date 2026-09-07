# CLI Module

The CLI package defines one public executable: `neuralls`.

## Public Surface

- `neuralls config ...`: manage machine-specific profiles
- `neuralls generate <case.toml> [--force]`: build every dataset declared in one case (skips a dataset that already exists unless `--force`)
- `neuralls generate-single <dataset.toml> --case-config <case.toml> [--force]`: build one dataset config
- `neuralls train <case.toml> [--force]`: train every assignment declared in one case (assumes datasets were already generated — run `generate` first; skips an assignment whose dataset/checkpoint already matches unless `--force`), and log the aggregate metric plot/label map
- `neuralls eval <case.toml>`: evaluate completed assignment checkpoints on their logged test splits
- `neuralls run <case.toml> [--force] [--force-generate] [--force-compare]`: the full per-case pipeline — generate every dataset, train every assignment, then run every comparison, each stage independently skippable/forceable
- `neuralls compare <case.toml> [--force]`: run every comparison profile declared in one case after their benchmark datasets exist (skips a comparison whose resolved checkpoints already produced a result unless `--force`)

## Package Map

- `main.py`: root Typer assembler for the public command surface
- `config.py`: profile management subcommands
- `generate.py`: case-wide dataset generation
- `generate_single.py`: explicit single-dataset generation
- `train.py`: case-wide training and aggregate reporting
- `eval.py`: case-wide checkpoint evaluation and aggregate reporting
- `run.py`: end-to-end case execution (generate → train → compare via `composition.assignments.case_pipeline.run_case_pipeline`)
- `compare.py`: case-wide solver benchmarking
- `options.py`: shared option aliases for profile and env-file resolution

## Boundary

CLI modules parse user input, print progress, resolve runtime settings at the
boundary, and delegate immediately to `neuralls.composition`. They must not
instantiate platform adapters directly or contain workflow business logic.

CLI owns argument parsing, top-level option semantics, and user-facing error
messages. It may resolve the active settings/profile context, but it should not
contain workflow assembly, filesystem policy, or service-integration logic.
Generation commands therefore render failures at the CLI boundary, while
lower layers supply the detailed operation/path context needed for diagnosis.
