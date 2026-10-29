"""Production wiring of the preconditioner builders' ports."""

from __future__ import annotations

from neuralls.composition.preconditioners.ports import FitDataLoader
from neuralls.platform.storage.fit_data import FsFitDataLoader

FS_FIT_DATA: FitDataLoader = FsFitDataLoader()
"""Reads fit arrays from the dataset directory through its manifest."""
