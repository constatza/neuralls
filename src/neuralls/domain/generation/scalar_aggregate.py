"""Cross-binding agreement on dataset-level scalars, kept in constant memory.

Each binding contributes one matrix norm, one matrix value scale and one scale payload.
The manifest exposes a single value per field, so the first binding's value is kept and
the set of disagreeing fields records whether any later binding disagreed. Only that set and the first
observation are stored, so the state does not grow with the number of bindings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from neuralls.shared.types import ScaleMetadata

NORM_AGREEMENT_RTOL: Final[float] = 1e-10
NORM_AGREEMENT_ATOL: Final[float] = 1e-12


class ScaleField(StrEnum):
    """Dataset-level scalar on which bindings can disagree."""

    MATRIX_NORM = "matrix_norm"
    MATRIX_VALUE_SCALE = "matrix_value_scale"
    SCALE_METADATA = "scale_metadata"


_NULL_PAYLOAD_KEY: Final[str] = "null"


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ScaleSummary:
    """Dataset-level scalars resolved from every binding.

    Attributes:
        matrix_norm: Representative matrix norm (the first binding's value).
        matrix_value_scale: Shared value scale, or 1.0 when bindings disagree.
        scale_metadata: Shared scale payload, or ``None`` when bindings disagree.
        disagreements: Fields on which some binding disagreed with the first.
    """

    matrix_norm: float
    matrix_value_scale: float
    scale_metadata: ScaleMetadata | None
    disagreements: frozenset[ScaleField] = frozenset()


def _agrees(value: float, reference: float) -> bool:
    return bool(np.isclose(value, reference, rtol=NORM_AGREEMENT_RTOL, atol=NORM_AGREEMENT_ATOL))


def _payload_key(payload: ScaleMetadata | None) -> str:
    """Canonical text of a payload, so payloads compare by content and key order is ignored."""
    return json.dumps(payload, sort_keys=True) if payload is not None else _NULL_PAYLOAD_KEY


class ScalarAggregator:
    """Accumulate per-binding scalars in O(1) state and resolve one summary.

    ``result`` reports every disagreeing field in the summary; it logs nothing.
    """

    def __init__(self) -> None:
        self._first: BindingScale | None = None
        self._first_key: str = _NULL_PAYLOAD_KEY
        self._disagreements: set[ScaleField] = set()

    def observe(self, binding: BindingScale) -> None:
        """Fold one binding's scalars into the agreement state."""
        first = self._first
        if first is None:
            self._first = binding
            self._first_key = _payload_key(binding.scale_params)
            return
        if not _agrees(binding.matrix_norm_value, first.matrix_norm_value):
            self._disagreements.add(ScaleField.MATRIX_NORM)
        if not _agrees(binding.matrix_value_scale, first.matrix_value_scale):
            self._disagreements.add(ScaleField.MATRIX_VALUE_SCALE)
        if _payload_key(binding.scale_params) != self._first_key:
            self._disagreements.add(ScaleField.SCALE_METADATA)

    def result(self) -> ScaleSummary:
        """Resolve the summary; a disagreeing field falls back to its neutral value.

        Raises:
            ValueError: If no binding was observed.
        """
        first = self._first
        if first is None:
            raise ValueError("No norm or scale values to resolve")
        disagreements = frozenset(self._disagreements)
        return ScaleSummary(
            matrix_norm=float(first.matrix_norm_value),
            matrix_value_scale=(
                1.0
                if ScaleField.MATRIX_VALUE_SCALE in disagreements
                else float(first.matrix_value_scale)
            ),
            scale_metadata=(
                None if ScaleField.SCALE_METADATA in disagreements else first.scale_params
            ),
            disagreements=disagreements,
        )


__all__ = [
    "BindingScale",
    "ScalarAggregator",
    "ScaleField",
    "ScaleSummary",
]
