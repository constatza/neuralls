"""CLI tool for managing neuralls machine-specific profiles."""

from __future__ import annotations

import os

import typer
from pydantic import ValidationError

from neuralls.composition.config import (
    PROFILE_ENV_VAR,
    USER_CONFIG_FILE,
    ProfileConfig,
    delete_profile,
    list_profiles,
    load_profile,
    resolve_typed_profile,
    save_profile,
)

app = typer.Typer(
    name="config",
    help="Manage neuralls machine-specific profile configurations.",
    no_args_is_help=True,
)

_VALID_FIELDS = {"raw-dir", "processed-dir", "output-dir"}
_FIELD_MAP = {
    "raw-dir": "raw_dir",
    "processed-dir": "processed_dir",
    "output-dir": "output_dir",
}
_TEMPLATE = """# neuralls machine profile configuration
# Edit this file directly or use `uv run neuralls config create ...`.
#
# One active default profile:
# [default]
# raw_dir = "/path/to/raw"
# processed_dir = "/path/to/processed"
# output_dir = "/path/to/output"
#
# Optional named profiles:
# [profiles.laptop]
# raw_dir = "/mnt/external/raw"
# processed_dir = "/mnt/external/processed"
# output_dir = "/home/user/laptop-output"
"""


def _active_profile() -> str:
    """Return the active profile name from the environment or ``'default'``."""
    return os.getenv(PROFILE_ENV_VAR) or "default"


def _write_template_config(*, force: bool = False) -> None:
    """Write a commented starter config to the user config path."""
    if USER_CONFIG_FILE.exists() and not force:
        raise FileExistsError(
            f"Config file already exists at {USER_CONFIG_FILE}. Use --force to replace it."
        )
    USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_FILE.write_text(_TEMPLATE, encoding="utf-8")


@app.command()
def path() -> None:
    """Print the absolute path to the user config file.

    Example:
        $ uv run neuralls config path
        /home/archer/.config/neuralls/config.toml
    """
    typer.echo(USER_CONFIG_FILE)


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file."),
) -> None:
    """Write a commented starter config file to the user config path.

    Example:
        $ uv run neuralls config init
        Wrote starter config to /home/archer/.config/neuralls/config.toml
    """
    try:
        _write_template_config(force=force)
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote starter config to {USER_CONFIG_FILE}")


@app.command(name="list")
def list_cmd() -> None:
    """List all profiles, marking the active one with ``*``.

    The active profile is determined by the ``NEURALLS_PROFILE`` environment
    variable, falling back to ``"default"``.

    Example:
        $ uv run neuralls config list
          name         raw_dir                      processed_dir                output_dir
        * default      /home/archer/raw            /home/archer/processed       /home/archer/outputs
    """
    names = list_profiles()
    if not names:
        typer.echo(
            "No profiles found. Run: uv run neuralls config init or "
            "uv run neuralls config create default --raw-dir ... --processed-dir ... --output-dir ..."
        )
        return

    active = _active_profile()
    typer.echo(
        "  name         raw_dir                      processed_dir                output_dir"
    )
    for name in names:
        try:
            profile = load_profile(name)
        except ValueError:
            profile = None
        marker = "*" if name == active else " "
        if profile is not None:
            typer.echo(
                f"{marker} {name:<12} {profile.raw_dir!s:<28} "
                f"{profile.processed_dir!s:<28} {profile.output_dir!s}"
            )
        else:
            typer.echo(f"{marker} {name:<12} (unreadable)")


@app.command()
def show(
    name: str | None = typer.Argument(None, help="Profile name. Defaults to the active profile."),
) -> None:
    """Pretty-print a single profile's settings.

    Args:
        name: Profile name to display. Defaults to the active profile.

    Example:
        $ uv run neuralls config show laptop
        Profile: laptop
          raw_dir:       /mnt/external/raw
          processed_dir: /mnt/external/processed
          output_dir:    /home/archer/laptop-outputs
    """
    resolved = name or _active_profile()
    try:
        profile = load_profile(resolved)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if profile is None:
        typer.echo(f"Profile '{resolved}' not found.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Profile: {resolved}")
    typer.echo(f"  raw_dir:       {profile.raw_dir}")
    typer.echo(f"  processed_dir: {profile.processed_dir}")
    typer.echo(f"  output_dir:    {profile.output_dir}")


@app.command()
def create(
    name: str = typer.Argument("default", help="Profile name to create."),
    raw_dir: str = typer.Option(..., "--raw-dir", help="Raw matrices and archives root."),
    processed_dir: str = typer.Option(..., "--processed-dir", help="Processed datasets root."),
    output_dir: str = typer.Option(..., "--output-dir", help="Outputs root."),
    force: bool = typer.Option(False, "--force", help="Overwrite if the profile already exists."),
) -> None:
    """Create a new named profile.

    Args:
        name: Profile name. Defaults to ``"default"``.
        raw_dir: Raw matrices and archives root directory.
        processed_dir: Processed datasets root directory.
        output_dir: Outputs (MLflow, checkpoints) root directory.
        force: When ``True``, overwrite an existing profile with the same name.

    Example:
        $ uv run neuralls config create laptop --raw-dir /mnt/raw --processed-dir /mnt/data --output-dir /mnt/out
        Profile 'laptop' written to /home/archer/.config/neuralls/config.toml
    """
    try:
        profile = resolve_typed_profile(
            ProfileConfig.model_validate(
                {
                    "raw_dir": raw_dir,
                    "processed_dir": processed_dir,
                    "output_dir": output_dir,
                }
            )
        )
        save_profile(name, profile, overwrite=force)
    except (ValidationError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Profile '{name}' written to {USER_CONFIG_FILE}")


@app.command(name="set")
def set_cmd(
    field: str = typer.Argument(
        ...,
        help="Field to update: 'raw-dir', 'processed-dir', or 'output-dir'.",
    ),
    value: str = typer.Argument(..., help="New path value."),
    name: str | None = typer.Argument(None, help="Profile name. Defaults to the active profile."),
) -> None:
    """Overwrite a single field in an existing profile.

    Args:
        field: Which field to update. Must be ``"raw-dir"``, ``"processed-dir"``, or ``"output-dir"``.
        value: The new path to assign, replacing the existing field value.
        name: Profile name. Defaults to the active profile.

    Example:
        $ uv run neuralls config set output-dir /new/outputs laptop
    """
    if field not in _VALID_FIELDS:
        typer.echo(
            f"Invalid field '{field}'. Choose from: {', '.join(sorted(_VALID_FIELDS))}", err=True
        )
        raise typer.Exit(code=1)

    resolved_name = name or _active_profile()
    try:
        existing = load_profile(resolved_name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if existing is None:
        typer.echo(f"Profile '{resolved_name}' not found.", err=True)
        raise typer.Exit(code=1)

    attr = _FIELD_MAP[field]
    updated_data = existing.model_dump()
    updated_data[attr] = value
    try:
        updated = resolve_typed_profile(ProfileConfig.model_validate(updated_data))
    except (ValidationError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    save_profile(resolved_name, updated, overwrite=True)
    typer.echo(f"Profile '{resolved_name}' updated: {attr} = {getattr(updated, attr)}")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Profile name to delete."),
) -> None:
    """Delete a named profile.

    The ``"default"`` profile cannot be deleted. Named profiles are removed
    from ``[profiles.<name>]`` in the user config file.

    Args:
        name: Profile name to remove.

    Example:
        $ uv run neuralls config delete old-laptop
        Profile 'old-laptop' deleted.
    """
    if name == "default":
        typer.echo("Cannot delete the 'default' profile.", err=True)
        raise typer.Exit(code=1)

    try:
        delete_profile(name)
    except (KeyError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Profile '{name}' deleted.")
