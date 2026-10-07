"""Cross-binding agreement on dataset-level scalars, kept in constant memory.

Each binding contributes one matrix norm, one matrix value scale and one scale payload.
The manifest exposes a single value per field, so the first binding's value is kept and
a flag records whether any later binding disagreed. Only the flags and the first
observation are stored, so the state does not grow with the number of bindings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import numpy as np
from loguru import logger

from neuralls.shared.types import ScaleMetadata

NORM_AGREEMENT_RTOL: Final[float] = 1e-10
NORM_AGREEMENT_ATOL: Final[float] = 1e-12

_NORM_DISAGREEMENT_WARNING: Final[str] = (
    "Bindings produced different normalized matrix norms; "
    "the dataset manifest keeps a representative matrix_norm only."
)
_SCALE_DISAGREEMENT_WARNING: Final[str] = (
    "Bindings produced different matrix value scales; "
    "the dataset manifest omits a shared reversible matrix scale."
)
_METADATA_DISAGREEMENT_WARNING: Final[str] = (
    "Bindings produced different scale metadata payloads; "
    "the dataset manifest intentionally stores no shared scale metadata."
)
_NULL_PAYLOAD_KEY: Final[str] = "null"


@dataclass(frozen=True)
class BindingScale:
    """Scalars one binding reports after normalizing its matrix.

    Attributes:
        matrix_norm_value: Norm of the normalized matrix.
        matrix_value_scale: Factor applied to the matrix values.
        scale_params: Serialized scale metadata, or ``None`` when the matrix was not scaled.
    """

    matrix_norm_value: float
    matrix_value_scale: float
    scale_params: ScaleMetadata | None


@dataclass(frozen=True)
class ScaleSummary:
    """Dataset-level scalars resolved from every binding.

    Attributes:
        matrix_norm: Representative matrix norm (the first binding's value).
        matrix_value_scale: Shared value scale, or 1.0 when bindings disagree.
        scale_metadata: Shared scale payload, or ``None`` when bindings disagree.
    """

    matrix_norm: float
    matrix_value_scale: float
    scale_metadata: ScaleMetadata | None


def _agrees(value: float, reference: float) -> bool:
    return bool(np.isclose(value, reference, rtol=NORM_AGREEMENT_RTOL, atol=NORM_AGREEMENT_ATOL))


def _payload_key(payload: ScaleMetadata | None) -> str:
    """Canonical text of a payload, so payloads compare by content and key order is ignored."""
    return json.dumps(payload, sort_keys=True) if payload is not None else _NULL_PAYLOAD_KEY


class ScalarAggregator:
    """Accumulate per-binding scalars in O(1) state and resolve one summary.

    Warnings are emitted once each by ``result``, in the order norm, scale, metadata.
    """

    def __init__(self) -> None:
        self._first: BindingScale | None = None
        self._first_key: str = _NULL_PAYLOAD_KEY
        self._norm_disagrees = False
        self._scale_disagrees = False
        self._metadata_disagrees = False

    def observe(self, binding: BindingScale) -> None:
        """Fold one binding's scalars into the agreement state."""
        first = self._first
        if first is None:
            self._first = binding
            self._first_key = _payload_key(binding.scale_params)
            return
        if not _agrees(binding.matrix_norm_value, first.matrix_norm_value):
            self._norm_disagrees = True
        if not _agrees(binding.matrix_value_scale, first.matrix_value_scale):
            self._scale_disagrees = True
        if _payload_key(binding.scale_params) != self._first_key:
            self._metadata_disagrees = True

    def result(self) -> ScaleSummary:
        """Resolve the summary and log one warning per disagreeing field.

        Raises:
            ValueError: If no binding was observed.
        """
        first = self._first
        if first is None:
            raise ValueError("No norm or scale values to resolve")

        if self._norm_disagrees:
            logger.warning(_NORM_DISAGREEMENT_WARNING)
        matrix_value_scale = float(first.matrix_value_scale)
        if self._scale_disagrees:
            logger.warning(_SCALE_DISAGREEMENT_WARNING)
            matrix_value_scale = 1.0
        scale_metadata = first.scale_params
        if self._metadata_disagrees:
            logger.warning(_METADATA_DISAGREEMENT_WARNING)
            scale_metadata = None

        return ScaleSummary(
            matrix_norm=float(first.matrix_norm_value),
            matrix_value_scale=matrix_value_scale,
            scale_metadata=scale_metadata,
        )


__all__ = [
    "BindingScale",
    "ScalarAggregator",
    "ScaleSummary",
]
