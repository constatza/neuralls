"""Generate one dataset from one dataset config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neuralls.cli.error_messages import format_cli_error
from neuralls.cli.options import CaseConfigOption, EnvFileOption, ProfileOption
from neuralls.composition.config import load_case_settings, resolve_case_config_path
from neuralls.composition.generation.process_data import process_data_from_config
from neuralls.shared.constants import EXIT_FAILURE, SYMBOL_ERROR, SYMBOL_SUCCESS


def _resolve_case_config(case_config: Path | None) -> Path | None:
    """Resolve explicit case config or NEURALLS_CASE_CONFIG override."""
    return resolve_case_config_path(case_config)


def generate_single(
    config: Annotated[Path, typer.Argument(help="Path to a dataset config TOML.")],
    case_config: CaseConfigOption = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Regenerate even if a matching dataset already exists."),
    ] = False,
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Generate or collect one dataset from one dataset config."""
    try:
        resolved_case_config = _resolve_case_config(case_config)
        if resolved_case_config is None:
            raise ValueError(
                "This command requires a case config. Pass --case-config or set "
                "NEURALLS_CASE_CONFIG."
            )

        settings = load_case_settings(resolved_case_config, env_file, profile=profile)
        typer.echo(f"Loading data config: {config}")
        output_path = process_data_from_config(config, settings, force=force)

        typer.echo(f"\n{SYMBOL_SUCCESS} Data processing complete!")
        typer.echo(f"  Output: {output_path}")
        typer.echo("  Files:")
        for file in sorted(output_path.glob("*")):
            typer.echo(f"    - {file.name}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(
            f"\n{SYMBOL_ERROR} {format_cli_error('Error during data generation', exc)}", err=True
        )
        raise typer.Exit(code=EXIT_FAILURE) from exc
