# neuralls

`neuralls` is the experiment codebase behind an ongoing investigation into
learned (neural-network) preconditioners for Conjugate Gradient on sparse
linear systems, benchmarked against classical preconditioners (Jacobi,
IC(0), AMG, POD-2G).

Each research question is a **case**: one TOML config binding the datasets,
training/search jobs, and solver comparisons that make up that experiment.
Machine-specific data roots are kept out of the repo (`~/.config/neuralls/`)
so the same case configs reproduce results on any machine.

This checkout is pinned to CUDA 13.0. To change backends, edit
`pyproject.toml`, then re-run `uv lock` and `uv sync`.

## Setup

- Python managed with [`uv`](https://docs.astral.sh/uv/)
- Access to your own raw matrix data, processed dataset root, and output root

Install the project environment:

```bash
uv sync
```

The project is packaged with `uv_build`, so editable installs and local builds
use the native uv backend instead of the legacy setuptools fallback.

## Quickstart

1. pick a case config
2. configure machine-specific data roots once
3. generate that case's datasets
4. train that case's assignments
5. run or compare the full case
6. evaluate trained checkpoints

### 1. Choose a case config

Cases live under `configs/cases/<family>/<variant>.toml`, one directory per
matrix family:

- `configs/cases/45x15/` — fixed 45x15 stiffness matrix
- `configs/cases/45x15randomE/` — parametric family (~100 matrices, randomized
  Young's moduli)
- `configs/cases/93x31/`
- `configs/cases/rectangular-high-condition/` — also has a `sample-sweep.toml`
  varying 0-CG/CG-10/CG-50 training-set size (1000/2000/5000/10000 samples)
- `configs/cases/spheres-1000x/`, `spheres-50x/`, `spheres-1x/` — sphere-RVE
  matrices at decreasing sphere/matrix stiffness contrast

Each family typically has a `default.toml` (classical + POD-2G comparisons)
and `*-search.toml` variants for neural-network training/search jobs. See
[Configuration Guide](configs/README.md) for how a case config is assembled.

### 2. Set machine-specific roots

Profiles are defined in the user config file:

```text
~/.config/neuralls/config.toml
```

That file is the persistent machine-specific config store. `neuralls config`
is the CLI for creating, updating, listing, and selecting profiles inside it.
You can also edit `~/.config/neuralls/config.toml` manually as long as it
follows the expected TOML structure.

Set up the active machine profile once:

```bash
uv run neuralls config init
uv run neuralls config create default --raw-dir /data/raw --processed-dir /data/processed --output-dir /data/output
```

Example profile file:

```toml
[default]
raw_dir = "/data/raw"
processed_dir = "/data/processed"
output_dir = "/data/output"

[profiles.laptop]
raw_dir = "/mnt/external/raw"
processed_dir = "/mnt/external/processed"
output_dir = "/home/archer/laptop-output"
```

Profile format rules:

- `[default]` defines the fallback profile used when no named profile is selected
- `[profiles.<name>]` defines a named profile such as `laptop` or `windows`
- every profile must define `raw_dir`, `processed_dir`, and `output_dir`
- paths are expanded with `~` and normalized to absolute paths at load time

You can manage the same file either way:

```bash
uv run neuralls config path
uv run neuralls config list
uv run neuralls config show
uv run neuralls config create laptop --raw-dir /mnt/external/raw --processed-dir /mnt/external/processed --output-dir /home/archer/laptop-output
uv run neuralls config set output-dir /new/output laptop
uv run neuralls config delete laptop
```

or by editing `~/.config/neuralls/config.toml` directly.

If you want a starter file instead of answering prompts immediately, use:

```bash
uv run neuralls config init
```

That writes a commented template to `~/.config/neuralls/config.toml`. `config create`
is non-interactive and requires explicit `--raw-dir`, `--processed-dir`, and
`--output-dir` flags.

Profiles provide:

- `raw_dir`
- `processed_dir`
- `output_dir`

Profile selection works like this:

1. `--profile <name>` picks a named profile for one command
2. `NEURALLS_PROFILE=<name>` picks a named profile from the environment
3. otherwise `default` is used

Root overrides work after profile selection:

1. process env vars such as `NEURALLS_OUTPUT_DIR`
2. `--env-file <path>`
3. `NEURALLS_ENV_FILE=<path>`
4. the selected profile in `~/.config/neuralls/config.toml`

`config set` overwrites the existing field value in place for an existing
profile. `config delete NAME` removes a named profile; `default` cannot be
deleted.

`neuralls` does **not** auto-discover `.env`, `.env.local`, or a repo root.
There is no hidden cwd-based configuration search.

Example env file override:

```dotenv
NEURALLS_RAW_DIR=D:/neuralls/raw
NEURALLS_PROCESSED_DIR=D:/neuralls/processed
NEURALLS_OUTPUT_DIR=D:/neuralls/output
```

On Windows, prefer forward slashes in env-file paths.

### 3. Generate datasets

Use the batch form for a full case, or `generate-single` when you want to
inspect one dataset config directly.

```bash
uv run neuralls generate configs/cases/45x15/default.toml --env-file .env.windows
uv run neuralls generate-single configs/datasets/train/45x15/gaussian-cg50.toml \
  --case-config configs/cases/45x15/default.toml \
  --env-file .env.windows
```

The batch form materializes every dataset referenced by the case config under
the resolved processed root. The `generate-single` form restores the one-dataset path
when you want to validate one dataset config in isolation.

### 4. Train one case batch

```bash
uv run neuralls train configs/cases/45x15/default-search.toml --env-file .env.windows
```

This trains every assignment declared in the case config and writes aggregate
training outputs under the resolved output root.

### 5. Run or compare a full case

```bash
uv run neuralls run configs/cases/45x15/default.toml --env-file .env.windows
uv run neuralls compare configs/cases/45x15/default.toml --env-file .env.windows
```

`neuralls run` generates datasets as needed and trains the full assignment
matrix. `neuralls compare` benchmarks the configured solver setups for the same
case.

### 6. Evaluate trained checkpoints

```bash
uv run neuralls eval configs/cases/45x15/default-search.toml --env-file .env.windows --metric mae
```

Evaluates trained assignment checkpoints on their logged test splits and logs
a batch metric plot to MLflow. Restrict to specific assignments with repeated
`--assignment <id>` flags.

## Case Configs

A case config is the authoritative persisted source for one experiment. It
binds:

- `[[datasets]]` — processed-dataset generation configs
- `[[jobs]]` — thin runnable entrypoints (`run.type = "train" | "search" | "fit"`)
  referencing a model profile, data profile, and training profile
- `[[comparisons]]` — inline solver-comparison scenarios (matrix, RHS source,
  seed), evaluated against `[comparison_defaults]` preconditioners
- `[[assignments]]` — pairs one dataset with one job
- optional `[mlflow]` topology and `[names]` for training/comparison experiment
  buckets

Relative paths inside a case config resolve against the case file's own
directory, and `${NEURALLS_*}` placeholders expand from resolved settings. If
`[mlflow]` is omitted, local SQLite tracking and artifact paths derive from
the active settings `output_dir`.

See [Configuration Guide](configs/README.md) for the full schema (dataset,
model-profile, training-profile, and job anatomy) and the reasoning behind
each checked-in case.

## Command Reference

| Goal | Command |
| --- | --- |
| Manage machine profiles | `uv run neuralls config ...` |
| Generate all datasets in one case | `uv run neuralls generate <case.toml>` |
| Generate one dataset config | `uv run neuralls generate-single <dataset.toml> --case-config <case.toml>` |
| Train all assignments in one case | `uv run neuralls train <case.toml>` |
| Generate datasets and train the full case | `uv run neuralls run <case.toml>` |
| Compare solver setups for one case | `uv run neuralls compare <case.toml>` |
| Evaluate trained checkpoints in one case | `uv run neuralls eval <case.toml>` |

Root-resolution rules are also explicit:

1. process env vars
2. `--env-file`
3. `NEURALLS_ENV_FILE`
4. the selected profile from `~/.config/neuralls/config.toml`
5. otherwise fail

There are no other fallbacks.

## Configuration Layout

- `configs/datasets/{train,test}/**/*.toml`: dataset generation and
  input-source definitions
- `configs/profiles/model/**/*.toml`: reusable DLKit model architecture
  fragments
- `configs/profiles/data/**/*.toml`: shared data-shape profiles referenced by
  model profiles via `run.data`
- `configs/profiles/training/**/*.toml`: shared optimization-policy profiles
- `configs/jobs/**/*.toml`: thin runnable entrypoints binding one model,
  data, and training profile
- `configs/cases/**/*.toml`: case configs tying datasets, jobs, and
  comparisons together into one experiment

Use the narrowest config that matches the task:

- debugging data generation: start with a dataset config
- validating one architecture: add one model profile and job
- running a repeatable experiment batch: move to a case config

Additional guidance:

- [Configuration Guide](configs/README.md)
- [Architecture Docs](docs/README.md)

## Outputs

Two roots matter operationally:

- `processed_dir`: generated datasets used by training and comparison
- `output_dir`: MLflow tracking, model checkpoints, figures, reports, and
  artifacts

`neuralls` does not migrate external data for you. Moving to a new machine
means you are responsible for copying any required raw datasets, processed
datasets, checkpoints, and MLflow state into the new roots.

## Repository Guide

If you need to work below the CLI layer:

- `src/neuralls/cli/`: the `neuralls` root CLI plus config and case-batch commands
- `src/neuralls/composition/`: workflow assembly and orchestration
- `src/neuralls/application/`: use-case logic
- `src/neuralls/platform/`: config, storage, tracking, DLKit integration
- `src/neuralls/domain/`: generation, solver logic, analysis, normalization
- `src/neuralls/shared/`: constants, shared types, functional helpers

Module-level architecture notes live alongside the code under the corresponding
package directories.

## Development

Install the development toolchain:

```bash
uv tool install prek
uv sync --dev
prek install -t pre-commit -t pre-push
```

Useful verification commands:

```bash
uv run ruff check src tests
uv run ty check src/ tests/
uv run pytest
```

The repository uses local hooks plus CI. Ruff is the linting baseline, `ty` is
the type-checking baseline, and `uv run ...` is the expected entrypoint for
Python tooling.
