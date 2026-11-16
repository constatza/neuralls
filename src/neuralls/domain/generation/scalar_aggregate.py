"""Cross-binding scalars of a generation run: one matrix norm and one scale factor per matrix.

Each binding contributes the norm of its scaled matrix and the factor ``alpha`` its matrix
was divided by. The manifest exposes one representative matrix norm, so the first binding's
norm is kept and later disagreements are recorded as a field set. ``alpha`` is recorded per
matrix sample id, because a comparison must divide each raw RHS by its own matrix's factor;
the state grows with the number of distinct matrices, never with the number of bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from neuralls.shared.types import MatrixSampleId

NORM_AGREEMENT_RTOL: Final[float] = 1e-10
NORM_AGREEMENT_ATOL: Final[float] = 1e-12


class ScaleField(StrEnum):
    """Dataset-level scalar on which bindings can disagree."""

    MATRIX_NORM = "matrix_norm"


@dataclass(frozen=True, slots=True)
class BindingScale:
    """Scalars one binding reports after scaling its matrix.

    Attributes:
        matrix_sample_id: The matrix sample the binding is bound to.
        matrix_norm_value: Norm of the scaled matrix.
        alpha: Factor the raw matrix was divided by.
    """

    matrix_sample_id: MatrixSampleId
    matrix_norm_value: float
    alpha: float


@dataclass(frozen=True, slots=True)
class ScaleSummary:
    """Dataset-level scalars resolved from every binding.

    Attributes:
        matrix_norm: Representative matrix norm (the first binding's value).
        matrix_scales: Scale factor ``alpha`` of every matrix sample id the run used.
        disagreements: Fields on which some binding disagreed with the first.
    """

    matrix_norm: float
    matrix_scales: Mapping[MatrixSampleId, float]
    disagreements: frozenset[ScaleField] = frozenset()


class ConflictingMatrixScaleError(ValueError):
    """Two bindings of one matrix sample reported different scale factors."""

    def __init__(self, matrix_sample_id: MatrixSampleId, first: float, second: float) -> None:
        self.matrix_sample_id = matrix_sample_id
        self.first = first
        self.second = second
        super().__init__(
            f"Matrix sample {matrix_sample_id} was scaled by {first!r} and by {second!r}; "
            "one matrix must have one scale factor."
        )


def _agrees(value: float, reference: float) -> bool:
    return bool(np.isclose(value, reference, rtol=NORM_AGREEMENT_RTOL, atol=NORM_AGREEMENT_ATOL))


class ScalarAggregator:
    """Accumulate per-binding scalars and resolve one summary.

    ``result`` reports every disagreeing field in the summary; it logs nothing.
    """

    def __init__(self) -> None:
        self._first_norm: float | None = None
        self._scales: dict[MatrixSampleId, float] = {}
        self._disagreements: set[ScaleField] = set()

    def observe(self, binding: BindingScale) -> None:
        """Fold one binding's scalars into the summary state.

        Raises:
            ConflictingMatrixScaleError: If the binding's matrix was already seen with a
                different scale factor.
        """
        known = self._scales.setdefault(binding.matrix_sample_id, binding.alpha)
        if known != binding.alpha:
            raise ConflictingMatrixScaleError(binding.matrix_sample_id, known, binding.alpha)
        if self._first_norm is None:
            self._first_norm = binding.matrix_norm_value
            return
        if not _agrees(binding.matrix_norm_value, self._first_norm):
            self._disagreements.add(ScaleField.MATRIX_NORM)

    def result(self) -> ScaleSummary:
        """Resolve the summary.

        Raises:
            ValueError: If no binding was observed.
        """
        if self._first_norm is None:
            raise ValueError("No norm or scale values to resolve")
        return ScaleSummary(
            matrix_norm=float(self._first_norm),
            matrix_scales=dict(self._scales),
            disagreements=frozenset(self._disagreements),
        )


__all__ = [
    "BindingScale",
    "ConflictingMatrixScaleError",
    "ScalarAggregator",
    "ScaleField",
    "ScaleSummary",
]
