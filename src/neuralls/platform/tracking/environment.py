"""Runtime environment helpers for MLflow execution."""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager


@contextmanager
def scoped_mlflow_environment(overrides: Mapping[str, str | None]) -> Generator[None]:
    """Temporarily apply MLflow environment variable overrides."""
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
                continue
            os.environ[key] = value
