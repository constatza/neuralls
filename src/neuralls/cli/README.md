# CLI Module

The CLI package defines one public executable: `neuralls`.

## Public Surface

- `neuralls config ...`: manage machine-specific profiles
- `neuralls generate <case.toml> [--force]`: build every dataset declared in one case (skips a dataset that already exists unless `--force`; `--force` also recreates sweep-generated dataset configs, which are otherwise created only when missing)
- `neuralls generate-single <dataset.toml> --case-config <case.toml> [--force]`: build one dataset config
- `neuralls train <case.toml> [--force]`: train every assignment declared in one case (assumes datasets were already generated — run `generate` first; skips an assignment whose dataset/checkpoint already matches unless `--force`), and log the aggregate metric plot/label map
- `neuralls eval <case.toml>`: evaluate completed assignment checkpoints on their logged test splits
- `neuralls run <case.toml> [--force] [--force-expand] [--force-generate] [--force-train] [--force-compare]`: the full per-case pipeline — generate every dataset, train every assignment, then run every comparison and its report, each stage independently skippable/forceable
- `neuralls compare <case.toml> [--force]`: run every comparison profile declared in one case after their benchmark datasets exist and store each method's raw tables (skips a comparison whose resolved checkpoints already produced a result unless `--force`). Compute only: it draws no figures; run `neuralls report` for them
- `neuralls report <case.toml> [--force] [--session-run-id ID]`: read the newest finished compare session back and write the nested report run tree (metrics, figures with cards, `figure_index.toml`) into the `Reports` experiment; a finished report for the same session, notation and schema is skipped unless `--force`. Never solves or times

## Package Map

- `main.py`: root Typer assembler for the public command surface
- `config.py`: profile management subcommands
- `generate.py`: case-wide dataset generation
- `generate_single.py`: explicit single-dataset generation
- `train.py`: case-wide training and aggregate reporting
- `eval.py`: case-wide checkpoint evaluation and aggregate reporting
- `run.py`: end-to-end case execution (generate → train → compare → report via `composition.assignments.case_pipeline.run_case_pipeline`)
- `compare.py`: case-wide solver benchmarking (compute only)
- `report.py`: report generation from a stored compare session
- `errors.py`: the shared top-level failure renderer; blocking workflow exceptions retain
  their concrete type, message, contextual exception notes, and logged traceback
- `options.py`: shared option aliases (profile, env file, case config, `--force`) and `require_existing_config`

## Printed text

Commands print the plain text of typed presentations through `composition/display.py`
(`comparison_label`, `assignment_label`, `generation_label`): a comparison line is the same
text as its MLflow run name, a training line names the model and the dataset it trained on,
and a dataset line is its stamped scientific description. A command never prints a registry id
as a name; MLflow run ids and file paths appear only as such. Profile subcommands print the
user's profile names, which are settings keys rather than scientific entities.

## Boundary

CLI modules parse user input, print progress, resolve runtime settings at the
boundary, and delegate immediately to `neuralls.composition`. They must not
instantiate platform adapters directly or contain workflow business logic.

CLI owns argument parsing, top-level option semantics, and user-facing error
messages. It may resolve the active settings/profile context, but it should not
contain workflow assembly, filesystem policy, or service-integration logic.
Commands render blocking failures at the CLI boundary, while lower layers
supply the detailed operation/path or workflow-phase context needed for
diagnosis. A command never catches `BaseException`, but every ordinary
`Exception` escaping a workflow is printed with its concrete type and exact
message before the command exits nonzero.
