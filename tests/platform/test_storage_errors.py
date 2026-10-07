"""Characterization of the shared storage error translation."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.platform.storage.errors import _raise_storage_error

_OPERATION = "Writing rhs"
_TARGET = Path("/data/dataset.zarr/rhs")
_MISSING_SOURCE = "/data/source.npy"


@pytest.fixture
def missing_file_error() -> OSError:
    """A real OSError with errno and filename, as the filesystem raises it."""
    return FileNotFoundError(2, "No such file or directory", _MISSING_SOURCE)


def test_raise_storage_error_wraps_as_oserror_with_fixed_message(
    missing_file_error: OSError,
) -> None:
    with pytest.raises(OSError) as info:
        _raise_storage_error(_OPERATION, _TARGET, missing_file_error)

    assert type(info.value) is OSError
    assert info.value.__cause__ is missing_file_error
    assert str(info.value) == (
        "Writing rhs at /data/dataset.zarr/rhs failed. "
        "FileNotFoundError: [Errno 2] No such file or directory: '/data/source.npy', "
        "errno=2, src=/data/source.npy"
    )
