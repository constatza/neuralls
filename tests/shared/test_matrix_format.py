"""Tests for ``MatrixFormat`` and the ``[output].matrix_format`` config key."""

from __future__ import annotations

import pytest

from neuralls.platform.config.models.data_models import OutputConfig
from neuralls.shared.types import MatrixFormat


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("csr", MatrixFormat.CSR),
        ("dense", MatrixFormat.DENSE),
    ],
)
def test_matrix_format_accepts_known_values(raw: str, expected: MatrixFormat) -> None:
    """Each persisted string value maps to its enum member."""
    assert MatrixFormat(raw) is expected


def test_output_config_defaults_to_csr() -> None:
    """The default matrix format lives only in ``OutputConfig``."""
    assert OutputConfig().matrix_format is MatrixFormat.CSR
