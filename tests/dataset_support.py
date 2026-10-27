"""Shared helper for tests that build real datasets without caring about their problem name."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

from neuralls.composition.generation.dataset_builder import build_dataset
from neuralls.domain.presentation.provenance import DatasetSourceProvenance
from neuralls.platform.storage.manifest_io import (
    load_dataset_manifest,
    read_dataset_manifest,
    save_dataset_manifest,
)
from neuralls.shared.scientific_description.models import ScientificDescription

DEFAULT_TEST_PROBLEM = DatasetSourceProvenance(ScientificDescription("Test problem"))
"""The source problem stamped on datasets whose tests do not choose one."""

build_test_dataset = partial(build_dataset, source_provenance=DEFAULT_TEST_PROBLEM)
"""``build_dataset`` with the default problem; pass ``source_provenance=`` to override."""


def strip_provenance(dataset_dir: str | Path) -> None:
    """Rewrite a dataset's manifest without provenance, as a dataset generated earlier has none."""
    manifest = read_dataset_manifest(dataset_dir)
    save_dataset_manifest(dataset_dir, replace(manifest, provenance=None))
    load_dataset_manifest.cache_clear()
