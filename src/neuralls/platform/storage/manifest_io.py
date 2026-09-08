"""Serialization and deserialization helpers for dataset manifests."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from neuralls.platform.storage.manifest import (
    _DATASET_SCHEMA,
    DatasetArtifact,
    DatasetManifest,
    DatasetNormalization,
    manifest_path_for,
)
from neuralls.shared.types import LayoutType


def manifest_to_dict(manifest: DatasetManifest) -> dict[str, Any]:
    """Serialize a typed manifest into the on-disk JSON shape.

    Args:
        manifest: The typed dataset manifest to serialize.

    Returns:
        JSON-compatible dictionary representation.
    """
    payload = asdict(manifest)
    payload["matrix"]["shape"] = list(manifest.matrix.shape)
    payload["rhs"]["shape"] = list(manifest.rhs.shape)
    payload["solutions"]["shape"] = list(manifest.solutions.shape)
    if manifest.row_kind is not None:
        payload["row_kind"] = {**asdict(manifest.row_kind), "shape": list(manifest.row_kind.shape)}
    if manifest.matrix_sample_index is not None:
        payload["matrix_sample_index"] = {
            **asdict(manifest.matrix_sample_index),
            "shape": list(manifest.matrix_sample_index.shape),
        }
    payload["params"] = [
        {**entry, "shape": list(param.shape)}
        for entry, param in zip(payload["params"], manifest.params, strict=True)
    ] or None
    return payload


def save_dataset_manifest(dataset_dir: str | Path, manifest: DatasetManifest) -> None:
    """Write a typed dataset manifest to disk.

    Invalidates `load_dataset_manifest`'s cache so a read after this write sees
    the manifest just written rather than the one it replaced.

    Args:
        dataset_dir: Directory in which to write the manifest file.
        manifest: The typed dataset manifest to persist.
    """
    path = manifest_path_for(dataset_dir)
    path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    load_dataset_manifest.cache_clear()


@lru_cache(maxsize=128)
def load_dataset_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load the raw manifest dictionary and validate the schema marker.

    Args:
        dataset_dir: Directory containing the manifest file.

    Returns:
        Raw manifest as a dictionary.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest does not have the expected schema marker.
    """
    path = manifest_path_for(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not str(manifest.get("schema", "")).startswith(
        "neuralls.dataset"
    ):
        raise ValueError(f"Invalid dataset manifest schema in {path}")
    return manifest


def read_dataset_manifest(dataset_dir: str | Path) -> DatasetManifest:
    """Load the typed dataset manifest.

    Args:
        dataset_dir: Directory containing the manifest file.

    Returns:
        Typed DatasetManifest object.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest schema is invalid.
    """
    raw = load_dataset_manifest(dataset_dir)

    def _artifact(payload: dict[str, Any]) -> DatasetArtifact:
        return DatasetArtifact(
            path=str(payload["path"]),
            format=str(payload["format"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            index=int(payload["index"]) if payload.get("index") is not None else None,
            n_matrix_samples=(
                int(payload["n_matrix_samples"])
                if payload.get("n_matrix_samples") is not None
                else None
            ),
            broadcast=bool(payload["broadcast"]) if payload.get("broadcast") is not None else None,
            key=str(payload["key"]) if payload.get("key") is not None else None,
            layout=LayoutType(payload["layout"]) if payload.get("layout") is not None else None,
            logical_sample_count=(
                int(payload["logical_sample_count"])
                if payload.get("logical_sample_count") is not None
                else None
            ),
        )

    params_raw = raw.get("params") or []
    normalization_raw = raw["normalization"]
    return DatasetManifest(
        schema=str(raw.get("schema", _DATASET_SCHEMA)),
        matrix=_artifact(raw["matrix"]),
        rhs=_artifact(raw["rhs"]),
        solutions=_artifact(raw["solutions"]),
        normalization=DatasetNormalization(
            type=str(normalization_raw["type"]),
            matrix_norm=float(normalization_raw["matrix_norm"]),
            matrix_norm_type=str(normalization_raw["matrix_norm_type"]),
            scale=dict(normalization_raw.get("scale") or {}),
        ),
        params=tuple(_artifact(payload) for payload in params_raw),
        row_kind=_artifact(raw["row_kind"]) if raw.get("row_kind") is not None else None,
        matrix_sample_index=(
            _artifact(raw["matrix_sample_index"])
            if raw.get("matrix_sample_index") is not None
            else None
        ),
        dataset_fingerprint=(
            str(raw["dataset_fingerprint"]) if raw.get("dataset_fingerprint") is not None else None
        ),
    )


def make_dataset_manifest(
    *,
    matrix: DatasetArtifact,
    rhs: DatasetArtifact,
    solutions: DatasetArtifact,
    normalization: DatasetNormalization,
    params: tuple[DatasetArtifact, ...] = (),
    row_kind: DatasetArtifact | None = None,
    matrix_sample_index: DatasetArtifact | None = None,
    dataset_fingerprint: str | None = None,
) -> DatasetManifest:
    """Construct a typed dataset manifest with the repo schema marker.

    Args:
        matrix: Matrix artifact descriptor.
        rhs: RHS vector artifact descriptor.
        solutions: Solution vector artifact descriptor.
        normalization: Normalization metadata.
        params: Optional parameter artifact descriptors.
        row_kind: Optional row-kind artifact descriptor.
        matrix_sample_index: Optional per-row matrix binding artifact descriptor.
        dataset_fingerprint: Optional artifact fingerprint stamped after the
            artifacts are on disk; storage writers leave this unset.

    Returns:
        Immutable DatasetManifest.
    """
    return DatasetManifest(
        schema=_DATASET_SCHEMA,
        matrix=matrix,
        rhs=rhs,
        solutions=solutions,
        normalization=normalization,
        params=params,
        row_kind=row_kind,
        matrix_sample_index=matrix_sample_index,
        dataset_fingerprint=dataset_fingerprint,
    )
