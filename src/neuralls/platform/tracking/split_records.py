"""The ``splits.toml`` session artifact: what each dataset's pinned split held out.

Written once on the comparison session run (``mlflow.log_text`` of ``encode_split_records``), because a split is a property of the dataset and
shared by every method child. It lets train/test separation be checked without the dataset
directory.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import tomli_w
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from neuralls.domain.solver.models.split_record import (
    DatasetSplitRecord,
    MatrixDisjoint,
    MatrixSeparation,
    SingleMatrixShared,
)
from neuralls.shared.types import DatasetId, MatrixId, SessionRunId

SPLITS_ARTIFACT: Final = "splits.toml"
_DATASETS_TABLE: Final = "datasets"
_SINGLE_MATRIX_SHARED: Final = "single_matrix_shared"
_MATRIX_DISJOINT: Final = "matrix_disjoint"


def encode_split_records(records: Iterable[DatasetSplitRecord]) -> bytes:
    """Serialize split records as one TOML table per dataset id."""
    tables = {str(record.dataset): _encode_record(record) for record in records}
    return tomli_w.dumps({_DATASETS_TABLE: tables}).encode("utf-8")


def decode_split_records(payload: bytes) -> tuple[DatasetSplitRecord, ...]:
    """Rebuild the records of a ``splits.toml`` payload.

    Raises:
        TypeError: A table is missing or is not a table.
        ValueError: A table contradicts its own separation claim.
    """
    tables = tomllib.loads(payload.decode("utf-8")).get(_DATASETS_TABLE)
    if not isinstance(tables, dict):
        raise TypeError(f"{SPLITS_ARTIFACT} has no [{_DATASETS_TABLE}] table")
    return tuple(_decode_record(DatasetId(name), table) for name, table in tables.items())


def download_session_splits(
    client: MlflowClient, session_run_id: SessionRunId, destination: Path
) -> tuple[DatasetSplitRecord, ...]:
    """Read back the split records a session logged.

    Raises:
        ValueError: The session logged none (it predates split records), or the file is malformed.
    """
    try:
        local = client.download_artifacts(session_run_id, SPLITS_ARTIFACT, str(destination))
    except MlflowException:
        raise ValueError(f"session {session_run_id} has no {SPLITS_ARTIFACT}") from None
    return decode_split_records(Path(local).read_bytes())


def _encode_record(record: DatasetSplitRecord) -> dict[str, Any]:
    table: dict[str, Any] = {
        "split_digest": record.split_digest,
        "train_rows": record.train_rows,
        "validation_rows": record.validation_rows,
        "test_rows": record.test_rows,
        "fit_matrix_ids": [int(matrix_id) for matrix_id in record.fit_matrix_ids],
        "test_matrix_ids": [int(matrix_id) for matrix_id in record.test_matrix_ids],
    }
    match record.separation:
        case SingleMatrixShared(matrix_id=matrix_id):
            table["separation"] = _SINGLE_MATRIX_SHARED
            table["shared_matrix_id"] = int(matrix_id)
        case MatrixDisjoint():
            table["separation"] = _MATRIX_DISJOINT
    return table


def _decode_record(dataset: DatasetId, table: object) -> DatasetSplitRecord:
    if not isinstance(table, dict):
        raise TypeError(f"split record of dataset {dataset!r} is not a table")
    return DatasetSplitRecord(
        dataset=dataset,
        split_digest=str(table["split_digest"]),
        train_rows=int(table["train_rows"]),
        validation_rows=int(table["validation_rows"]),
        test_rows=int(table["test_rows"]),
        fit_matrix_ids=tuple(MatrixId(int(value)) for value in table["fit_matrix_ids"]),
        test_matrix_ids=tuple(MatrixId(int(value)) for value in table["test_matrix_ids"]),
        separation=_decode_separation(dataset, table),
    )


def _decode_separation(dataset: DatasetId, table: dict[str, Any]) -> MatrixSeparation:
    match table["separation"]:
        case "single_matrix_shared":
            return SingleMatrixShared(MatrixId(int(table["shared_matrix_id"])))
        case "matrix_disjoint":
            return MatrixDisjoint()
        case other:
            raise ValueError(f"unknown separation {other!r} for dataset {dataset!r}")
