"""Doc lint: every catalog metric's label and formula appears verbatim in the solver README."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.domain.solver.reporting.metrics import METRICS, MetricSpec

README = Path(__file__).parents[2] / "src/neuralls/domain/solver/README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("spec", list(METRICS.values()), ids=lambda spec: spec.id.value)
def test_label_and_formula_are_documented_verbatim(spec: MetricSpec, readme_text: str) -> None:
    assert spec.label in readme_text
    assert spec.formula in readme_text
