"""Tests for ``neuralls.shared.digest``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralls.shared import digest as digest_module
from neuralls.shared.digest import (
    array_digest,
    canonical_digest,
    content_digest,
    toml_digest,
)
from tests.shared.conftest import (
    CosmeticDataclass,
    CosmeticModel,
    DefaultModel,
    InputModel,
    Unknown,
    UnmarkedPathModel,
)


def test_digest_format() -> None:
    assert canonical_digest(1).startswith("sha256:")
    assert len(canonical_digest(1)) == len("sha256:") + 64


def test_dict_key_order_independent() -> None:
    assert canonical_digest({"a": 1, "b": [1, 2]}) == canonical_digest({"b": [1, 2], "a": 1})


def test_set_order_independent() -> None:
    assert canonical_digest({3, 1, 2}) == canonical_digest(frozenset({2, 3, 1}))


def test_value_change_changes_digest() -> None:
    assert canonical_digest({"a": 1}) != canonical_digest({"a": 2})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        canonical_digest({"x": bad})


def test_unknown_type_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        canonical_digest(Unknown())


def test_unmarked_path_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unmarked Path"):
        canonical_digest(UnmarkedPathModel(path=tmp_path))
    with pytest.raises(TypeError, match="unmarked Path"):
        canonical_digest({"p": tmp_path})


def test_content_digest_file_path_independent(tmp_path: Path) -> None:
    first = tmp_path / "x" / "f.bin"
    second = tmp_path / "y" / "renamed.bin"
    for target in (first, second):
        target.parent.mkdir()
        target.write_bytes(b"same")
    assert content_digest(first) == content_digest(second)


def test_content_digest_directory_path_independent(dir_a: Path, dir_b: Path) -> None:
    assert content_digest(dir_a) == content_digest(dir_b)


def test_content_digest_directory_changes_on_nested_edit(dir_a: Path) -> None:
    before = content_digest(dir_a)
    (dir_a / "nested" / "deep" / "c.bin").write_bytes(b"changed")
    assert content_digest(dir_a) != before


def test_content_digest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        content_digest(tmp_path / "missing")


def test_toml_digest_ignores_comments_whitespace_and_order(
    toml_pair: tuple[Path, Path],
) -> None:
    first, second = toml_pair
    assert toml_digest(first) == toml_digest(second)


def test_toml_digest_changes_on_value_change(simple_toml: Path) -> None:
    before = toml_digest(simple_toml)
    simple_toml.write_text("x = 2\n")
    assert toml_digest(simple_toml) != before


def test_toml_digest_follows_referenced_toml(toml_main: Path) -> None:
    before = toml_digest(toml_main)
    (toml_main.parent / "child.toml").write_text('lr = 0.2\nname = "child"\n')
    assert toml_digest(toml_main) != before


def test_toml_digest_cycle_raises(toml_cycle: Path) -> None:
    with pytest.raises(ValueError, match="Cyclic"):
        toml_digest(toml_cycle)


def test_cosmetic_field_excluded_pydantic() -> None:
    assert canonical_digest(CosmeticModel(label="a")) == canonical_digest(CosmeticModel(label="b"))
    assert canonical_digest(CosmeticModel(value=1)) != canonical_digest(CosmeticModel(value=2))


def test_cosmetic_field_excluded_dataclass() -> None:
    assert canonical_digest(CosmeticDataclass(label="a")) == canonical_digest(
        CosmeticDataclass(label="b")
    )
    assert canonical_digest(CosmeticDataclass(value=1)) != canonical_digest(
        CosmeticDataclass(value=2)
    )


def test_input_data_hashed_by_content(dir_a: Path, dir_b: Path) -> None:
    assert canonical_digest(InputModel(data=dir_a)) == canonical_digest(InputModel(data=dir_b))
    before = canonical_digest(InputModel(data=dir_a))
    (dir_a / "a.bin").write_bytes(b"edited")
    assert canonical_digest(InputModel(data=dir_a)) != before


def test_input_none_stays_none() -> None:
    assert canonical_digest(InputModel()) == canonical_digest(InputModel(data=None, config=None))


def test_input_config_hashed_semantically(toml_main: Path) -> None:
    before = canonical_digest(InputModel(config=toml_main))
    toml_main.write_text('epochs = 3\n\nprofile = "child.toml"  # moved\n')
    assert canonical_digest(InputModel(config=toml_main)) == before
    toml_main.write_text('profile = "child.toml"\nepochs = 4\n')
    assert canonical_digest(InputModel(config=toml_main)) != before


def test_default_values_are_included() -> None:
    assert canonical_digest(DefaultModel()) == canonical_digest(DefaultModel(threshold=0.0))
    assert canonical_digest(DefaultModel()) != canonical_digest(DefaultModel(threshold=0.5))


def test_array_digest_is_storage_format_independent(
    sample_array: np.ndarray, stored_arrays: dict[str, object]
) -> None:
    """npy, HDF5 and zarr copies of one array share a logical digest."""
    digests = {array_digest(array) for array in (sample_array, *stored_arrays.values())}
    assert len(digests) == 1


def test_array_digest_is_chunk_size_independent(
    sample_array: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming in tiny chunks yields the same digest as one pass."""
    whole = array_digest(sample_array)
    monkeypatch.setattr(digest_module, "_ARRAY_CHUNK_BYTES", 100)
    assert array_digest(sample_array) == whole


def test_array_digest_detects_value_dtype_and_shape_changes(sample_array: np.ndarray) -> None:
    """Any change to values, dtype or shape changes the digest."""
    base = array_digest(sample_array)
    edited = sample_array.copy()
    edited[3, 2] += 1e-12
    assert array_digest(edited) != base
    assert array_digest(sample_array.astype(np.float32)) != base
    assert array_digest(sample_array.reshape(20, 12)) != base


def test_array_digest_handles_scalars_and_big_endian(sample_array: np.ndarray) -> None:
    """0-d arrays hash, and byte order does not matter."""
    assert array_digest(np.asarray(3.5)) == array_digest(np.asarray(3.5))
    assert array_digest(sample_array.astype(">f8")) == array_digest(sample_array)
