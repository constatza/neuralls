"""Generate every dataset referenced by one case config."""

from __future__ import annotations

from typing import Any

import typer

from neuralls.cli.error_messages import format_cli_error
from neuralls.cli.options import CaseConfigArgument, EnvFileOption, ProfileOption
from neuralls.composition.assignments.assembler import load_validated_case_config
from neuralls.composition.config import load_case_settings, load_raw_toml
from neuralls.composition.generation.multi_generation import generate_batch
from neuralls.shared.constants import EXIT_FAILURE


def _looks_like_dataset_config(raw_config: dict[str, Any]) -> bool:
    """Return True when the TOML has the top-level shape of a dataset config."""
    dataset_keys = {"source", "generation", "output"}
    return any(key in raw_config for key in dataset_keys)


def generate_case(
    config: CaseConfigArgument,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Regenerate every dataset even if a matching one already exists.",
    ),
    env_file: EnvFileOption = None,
    profile: ProfileOption = None,
) -> None:
    """Materialize every dataset declared in one case config."""
    try:
        raw_config = load_raw_toml(config)
        if _looks_like_dataset_config(raw_config):
            raise ValueError(
                "Bare 'neuralls generate' expects a case config. "
                "Use 'neuralls generate-single <dataset.toml> --case-config <case.toml>' "
                "for one dataset config."
            )

        settings = load_case_settings(config, env_file, profile=profile)
        cfg, _ = load_validated_case_config(config, settings)
        if not cfg.datasets:
            raise ValueError(
                "Case config does not define any [[datasets]] entries for batch generation."
            )
        results = generate_batch(
            cfg=cfg, configs_dir=config.resolve().parent, settings=settings, force=force
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(format_cli_error("Error during batch generation", exc), err=True)
        raise typer.Exit(code=EXIT_FAILURE) from exc

    for result in results:
        typer.echo(f"[{result.dataset_id}] -> {result.output_dir}")
