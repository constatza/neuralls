"""Fixtures for tests of the standalone dataset-sweep expansion script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SWEEP_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "expand_dataset_sweep.py"


@pytest.fixture(scope="session")
def dataset_sweep_module() -> ModuleType:
    """Load ``scripts/expand_dataset_sweep.py`` for unit-testing its pure helpers."""
    spec = importlib.util.spec_from_file_location("expand_dataset_sweep", _SWEEP_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
