"""Pure tests of `anchor` and the path-expansion helpers: no disk, no ambient cwd or HOME."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuralls.platform.config.context import (
    ConfigContext,
    anchor,
    expand_config_glob,
    expand_config_path,
)
from neuralls.platform.config.models.data_models import SourceConfig

CONFIG_DIR = Path("cfg")
HOME_DIR = Path("home")
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="posix path semantics")
WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")


def make_context(*, raw_dir: str | Path | None = Path("raw")) -> ConfigContext:
    """A context whose directories are all relative, so every result is a pure value."""
    settings = SimpleNamespace(
        raw_dir=raw_dir,
        processed_dir=Path("processed"),
        output_dir=Path("output"),
    )
    return ConfigContext(
        config_path=CONFIG_DIR / "config.toml",
        settings=settings,
        home_dir=HOME_DIR,
    )


@pytest.fixture
def ctx() -> ConfigContext:
    return make_context()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("matrix.mtx", Path("cfg/matrix.mtx")),
        ("relative/path.mtx", Path("cfg/relative/path.mtx")),
        ("a/../b", Path("cfg/b")),
        ("./a//b/", Path("cfg/a/b")),
        ("../up/x", Path("up/x")),
        ("~", Path("home")),
        ("~/matrix.mtx", Path("home/matrix.mtx")),
        ("~/a/../b", Path("home/b")),
        ("${NEURALLS_RAW_DIR}/matrix.mtx", Path("cfg/raw/matrix.mtx")),
        ("${NEURALLS_OUTPUT_DIR}/reports", Path("cfg/output/reports")),
        ("sqlite:///rel.db", Path("cfg/rel.db")),
        ("file:///a%20b/x", Path("/a b/x")),
    ],
)
@POSIX_ONLY
def test_anchor_gives_the_pure_path(ctx: ConfigContext, value: str, expected: Path) -> None:
    assert anchor(value, ctx) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/abs/x", Path("/abs/x")),
        ("/abs/a/../x", Path("/abs/x")),
        ("file:///abs/x", Path("/abs/x")),
        ("file://localhost/abs/x", Path("/abs/x")),
        ("file:////abs/x", Path("/abs/x")),
        ("sqlite:////abs/mlflow.db", Path("/abs/mlflow.db")),
    ],
)
@POSIX_ONLY
def test_anchor_keeps_an_absolute_path(ctx: ConfigContext, value: str, expected: Path) -> None:
    assert anchor(value, ctx) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("~user/x", "Cannot expand '~user/x': only '~' and '~/...' are supported, not '~user'."),
        (
            "file://host/x",
            (
                "File URL has non-empty host 'host': 'file://host/x'. "
                "Use file:///path (3 slashes) for a local file."
            ),
        ),
        ("ftp://host/x", "Unsupported scheme for local resolution: ftp"),
        (
            "${GRAPH_CG_RAW_DIR}/matrix.mtx",
            (
                "Stale placeholder '${GRAPH_CG_RAW_DIR}' in config. "
                "Rename GRAPH_CG_* → NEURALLS_* (see docs/env-reference.md)."
            ),
        ),
        (
            "${NEURALLS_UNKNOWN}/matrix.mtx",
            "Unknown placeholder '${NEURALLS_UNKNOWN}'. Available: RAW_DIR, PROCESSED_DIR, OUTPUT_DIR.",
        ),
    ],
)
def test_anchor_rejects(ctx: ConfigContext, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        anchor(value, ctx)


def test_an_unset_raw_dir_is_an_unknown_placeholder() -> None:
    message = (
        "Unknown placeholder '${NEURALLS_RAW_DIR}'. Available: RAW_DIR, PROCESSED_DIR, OUTPUT_DIR."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        anchor("${NEURALLS_RAW_DIR}/matrix.mtx", make_context(raw_dir=None))


@pytest.mark.parametrize(
    "value",
    [r"C:\x", r"\\server\share\x", "file:///C:/x"],
)
@POSIX_ONLY
def test_anchor_rejects_a_windows_absolute_path_on_posix(ctx: ConfigContext, value: str) -> None:
    with pytest.raises(ValueError, match=r"^Windows absolute path '.*' cannot be used on posix"):
        anchor(value, ctx)


@pytest.mark.parametrize(
    ("raw_dir", "value", "expected"),
    [
        (
            r"M:\shared\raw",
            "${NEURALLS_RAW_DIR}/SpectralData/matrix.txt",
            r"M:\shared\raw\SpectralData\matrix.txt",
        ),
        (Path("raw"), r"\\192.168.1.51\MGroup\matrix.txt", r"\\192.168.1.51\MGroup\matrix.txt"),
        (Path("raw"), "C:/data/../x", r"C:\x"),
    ],
)
@WINDOWS_ONLY
def test_anchor_keeps_a_windows_absolute_path_on_windows(
    raw_dir: str | Path, value: str, expected: str
) -> None:
    assert str(anchor(value, make_context(raw_dir=raw_dir))) == expected


def test_expand_config_path_is_the_anchor_as_a_string(ctx: ConfigContext) -> None:
    assert expand_config_path("a/../b", ctx) == str(Path("cfg/b"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("*.mtx", Path("cfg/*.mtx")),
        ("data/*.mtx", Path("cfg/data/*.mtx")),
        ("data/part*", Path("cfg/data/part*")),
        ("a/../data/**/x?.mtx", Path("cfg/data/**/x?.mtx")),
        ("${NEURALLS_OUTPUT_DIR}/*.mtx", Path("cfg/output/*.mtx")),
        ("~/runs/*/K[0-9].mtx", Path("home/runs/*/K[0-9].mtx")),
        ("plain.mtx", Path("cfg/plain.mtx")),
    ],
)
@POSIX_ONLY
def test_glob_anchors_only_the_prefix(ctx: ConfigContext, value: str, expected: Path) -> None:
    assert Path(expand_config_glob(value, ctx)) == expected


def test_a_validator_anchors_when_a_context_is_given(ctx: ConfigContext) -> None:
    config = SourceConfig.model_validate(
        {"matrix_path": "workspace/../A.txt"}, context=ctx.as_pydantic_context()
    )

    assert config.matrix_path == str(Path("cfg/A.txt"))


def test_a_model_built_without_a_context_keeps_its_paths() -> None:
    config = SourceConfig.model_validate({"matrix_path": "workspace/../A.txt"})

    assert config.matrix_path == "workspace/../A.txt"


def test_the_pydantic_context_round_trips(ctx: ConfigContext) -> None:
    assert ConfigContext.from_pydantic_context(ctx.as_pydantic_context()) == ctx
