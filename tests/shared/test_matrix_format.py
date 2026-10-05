"""Tests for ``MatrixFormat`` and the ``[output].matrix_format`` config key."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize("raw", ["coo", "CSR", "", "sparse"])
def test_matrix_format_rejects_unknown_values(raw: str) -> None:
    """Values outside the closed set raise ``ValueError``."""
    with pytest.raises(ValueError):
        MatrixFormat(raw)


def test_output_config_defaults_to_csr() -> None:
    """The default matrix format lives only in ``OutputConfig``."""
    assert OutputConfig().matrix_format is MatrixFormat.CSR


def test_output_config_accepts_explicit_dense() -> None:
    """An explicit ``dense`` value validates to the dense member."""
    cfg = OutputConfig.model_validate({"matrix_format": "dense"})
    assert cfg.matrix_format is MatrixFormat.DENSE


def test_output_config_rejects_invalid_matrix_format() -> None:
    """An unknown format string fails model validation."""
    with pytest.raises(ValidationError):
        OutputConfig.model_validate({"matrix_format": "coo"})
