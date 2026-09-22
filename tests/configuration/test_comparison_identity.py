"""`comparison_identity`: what invalidates a comparison, and what must not."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralls.composition.identity.comparison import comparison_identity
from neuralls.platform.config.models.comparison import ComparisonConfig
from neuralls.shared.digest import canonical_digest
from tests.configuration.conftest import ComparisonFactory


def _key(cfg: ComparisonConfig, checkpoints: dict[str, str] | None = None) -> str:
    return comparison_identity(cfg, cfg.preconditioners, checkpoints or {}).key


@pytest.fixture
def baseline(make_comparison: ComparisonFactory) -> ComparisonConfig:
    """The reference comparison."""
    return make_comparison()


def test_identical_config_gives_the_same_key(
    baseline: ComparisonConfig, make_comparison: ComparisonFactory
) -> None:
    assert _key(baseline) == _key(make_comparison())


@pytest.mark.parametrize(
    "override",
    [
        {"rtol": 1e-6},
        {"max_iterations": 50},
        {"ic0_threshold": 0.5},
        {"pod_rank": 8},
        {"rhs_std": 2.0},
    ],
    ids=["rtol", "max_iterations", "static-preconditioner-param", "pod-rank", "rhs-source"],
)
def test_result_determining_settings_change_the_key(
    baseline: ComparisonConfig, make_comparison: ComparisonFactory, override: dict[str, object]
) -> None:
    assert _key(make_comparison(**override)) != _key(baseline)


def test_a_preconditioner_label_does_not_change_the_key(
    baseline: ComparisonConfig, make_comparison: ComparisonFactory
) -> None:
    """Renaming a preconditioner is cosmetic."""
    assert _key(make_comparison(jacobi_name="renamed")) == _key(baseline)


def test_matrix_content_changes_the_key_but_its_location_does_not(
    baseline: ComparisonConfig,
    make_comparison: ComparisonFactory,
    matrix_file: Path,
    tmp_path: Path,
) -> None:
    """The matrix is identified by content: a copy elsewhere is identical, an edit is not."""
    copy = tmp_path / "elsewhere" / "m.bin"
    copy.parent.mkdir()
    copy.write_bytes(matrix_file.read_bytes())
    assert _key(make_comparison(matrix_path=copy)) == _key(baseline)

    matrix_file.write_bytes(b"different matrix")
    assert _key(baseline) != _key(make_comparison(matrix_path=copy))


def test_checkpoint_content_changes_the_key(baseline: ComparisonConfig) -> None:
    """A retrained model changes the key; an identical one does not."""
    old = {"0:coarsening": canonical_digest("weights-v1")}
    new = {"0:coarsening": canonical_digest("weights-v2")}
    assert _key(baseline, old) == _key(baseline, dict(old))
    assert _key(baseline, old) != _key(baseline, new)


def test_static_only_comparison_is_keyed_too(make_comparison: ComparisonFactory) -> None:
    """The old dependency hash was a constant for static-only comparisons; this one is not."""
    a = make_comparison(rtol=1e-8)
    b = make_comparison(rtol=1e-7)
    assert comparison_identity(a, (), {}).key != comparison_identity(b, (), {}).key
