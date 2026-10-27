"""Fixtures for testing standalone scripts under ``scripts/``.

``scripts/`` is not part of the installed package, so its modules are loaded
directly from their file path rather than imported by dotted name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pod2g_amg_convergence_study.py"


@pytest.fixture(scope="session")
def convergence_study_module() -> ModuleType:
    """Load ``scripts/pod2g_amg_convergence_study.py`` for unit-testing its pure helpers."""
    spec = importlib.util.spec_from_file_location("pod2g_amg_convergence_study", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass field resolution needs this module registered
    spec.loader.exec_module(module)
    return module
