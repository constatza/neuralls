"""File and directory operations.

This module provides simple file system operations like directory creation
and path manipulation utilities.

Functions in this module perform I/O operations and are not pure.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from neuralls.platform.config.job_metadata import read_job_metadata
from neuralls.platform.config.models.workspace import AssignmentWorkspace


def sanitize_identifier(value: str, default: str = "run") -> str:
    """Convert arbitrary identifier text into a filesystem-friendly slug.

    Pure function - transforms string without I/O.

    Args:
        value: Identifier to sanitize
        default: Default value if sanitized result is empty

    Returns:
        Filesystem-safe identifier string
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return default

    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("_-.")

    return cleaned or default


def extract_model_name(job_config_path: Path | str) -> str:
    """Extract a stable job/model name from a DLKit job config."""
    job_config_path = Path(job_config_path)

    metadata = read_job_metadata(job_config_path)
    if metadata.experiment_name is not None:
        return metadata.experiment_name
    if metadata.model_name is not None:
        return metadata.model_name
    return job_config_path.stem


def derive_model_identifier(
    settings: Any,
    context: AssignmentWorkspace,
    config_path: str | Path,
) -> str:
    """Derive a stable model identifier for artifact naming.

    Pure function - extracts identifier from settings and context.

    Preference order:
        1. ``model.name`` from the config settings
        2. ``experiment.name`` if present
        3. Context ``run_id``
        4. Config filename stem

    Args:
        settings: DLKit workflow settings
        context: Assignment workspace
        config_path: Path to config file

    Returns:
        Sanitized model identifier string
    """
    config_path = Path(config_path)

    candidates: list[str | None] = []
    model = getattr(settings, "model", None)
    if model is not None:
        model_name = getattr(model, "name", None)
        if model_name is not None:
            candidates.append(str(model_name))

    experiment = getattr(settings, "experiment", None)
    if experiment is not None:
        candidates.append(getattr(experiment, "name", None))

    candidates.append(getattr(context, "run_id", None))
    candidates.append(config_path.stem)

    for candidate in candidates:
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized and not normalized.lower().startswith("dlkit-session"):
                return sanitize_identifier(normalized, default="model")

    return sanitize_identifier(config_path.stem, default="model")


def parse_data_dir_name(dir_name: str) -> dict[str, Any]:
    """Parse data directory name to extract parameters.

    Pure function - extracts metadata from string.

    Args:
        dir_name: Directory name (e.g., "collect-504-norm", "generate-90-krylov50-norm")

    Returns:
        Dictionary with parsed parameters

    Examples:
        >>> parse_data_dir_name("collect-504-norm")
        {'source': 'collect', 'dimension': 504, 'normalized': True, 'krylov_percent': None}
        >>> parse_data_dir_name("generate-90-krylov50-norm")
        {'source': 'generate', 'dimension': 90, 'normalized': True, 'krylov_percent': 50}
    """
    parts = dir_name.split("-")

    if len(parts) < 3:
        return {
            "source": "unknown",
            "dimension": None,
            "normalized": False,
            "krylov_percent": None,
        }

    result: dict[str, Any] = {
        "source": parts[0],
        "dimension": None,
        "normalized": parts[-1] == "norm",
        "krylov_percent": None,
    }

    # Extract dimension (second part)
    try:
        result["dimension"] = int(parts[1])
    except ValueError, IndexError:
        pass

    # Check for krylov tag (if present, it's the second-to-last or third part)
    for part in parts[2:-1]:
        if part.startswith("krylov"):
            try:
                result["krylov_percent"] = int(part.replace("krylov", ""))
            except ValueError:
                pass

    return result


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    I/O action - creates directory on filesystem.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Copy file from source to destination.

    I/O action - copies file on filesystem.

    Args:
        src: Source path
        dst: Destination path
    """
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def get_file_size(path: str | Path) -> int:
    """Get file size in bytes.

    I/O action - reads file metadata.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def file_exists(path: str | Path) -> bool:
    """Check if file exists.

    I/O action - checks filesystem.

    Args:
        path: File path

    Returns:
        True if file exists
    """
    return Path(path).exists()


def get_relative_path(path: str | Path, base: str | Path) -> Path:
    """Get relative path from base directory.

    Pure function - computes relative path.

    Args:
        path: Target path
        base: Base directory

    Returns:
        Relative path
    """
    return Path(path).relative_to(Path(base))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.

    Pure function - formats number as string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def save_text_file(content: str, path: str | Path) -> None:
    """Save text content to file.

    I/O action - writes file to disk.

    Args:
        content: Text content
        path: Output path
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def load_text_file(path: str | Path) -> str:
    """Load text content from file.

    I/O action - reads file from disk.

    Args:
        path: Input path

    Returns:
        Text content
    """
    return Path(path).read_text(encoding="utf-8")
