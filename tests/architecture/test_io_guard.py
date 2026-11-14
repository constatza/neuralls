"""The runtime IO guard classifies audit events as a guarded test's own IO or as ambient."""

from __future__ import annotations

import os

import pytest
from tests.io_guard import TESTS_DIR, GuardRoots, IoViolation, classify

# Pure path arithmetic under an absolute anchor that exists on every OS; nothing is touched.
_ANCHOR = TESTS_DIR.parent / "virtual"
_BASETEMP = _ANCHOR / "basetemp"
_PREFIX = _ANCHOR / "prefix"
_ELSEWHERE = _ANCHOR / "elsewhere"
_ROOTS = GuardRoots(basetemp=_BASETEMP, tests=TESTS_DIR, ignored=(_PREFIX,))
_READ = os.O_RDONLY
_CREATE = os.O_WRONLY | os.O_CREAT


@pytest.mark.parametrize(
    ("event", "args", "expected"),
    [
        (
            "open",
            (_BASETEMP / "a.npy", "wb", _CREATE),
            IoViolation("open", str(_BASETEMP / "a.npy")),
        ),
        ("open", (_BASETEMP / "a.npy", "rb", _READ), IoViolation("open", str(_BASETEMP / "a.npy"))),
        (
            "open",
            (_ELSEWHERE / "cache", None, _CREATE),
            IoViolation("open", str(_ELSEWHERE / "cache")),
        ),
        (
            "open",
            (TESTS_DIR / "data.toml", "r", _READ),
            IoViolation("open", str(TESTS_DIR / "data.toml")),
        ),
        ("open", (TESTS_DIR / "conftest.py", "r", _READ), None),
        ("open", (_ELSEWHERE / "library.json", "r", _READ), None),
        ("open", (_PREFIX / "site.json", "w", _CREATE), None),
        ("open", (_ELSEWHERE / "__pycache__" / "m.pyc", "wb", _CREATE), None),
        ("open", (3, "rb", _READ), None),
        ("os.mkdir", (_BASETEMP / "tmp_path0", 0o777, -1), None),
        (
            "os.mkdir",
            (_ELSEWHERE / "out", 0o777, -1),
            IoViolation("os.mkdir", str(_ELSEWHERE / "out")),
        ),
        ("os.remove", (_BASETEMP / "current", -1), None),
        ("os.remove", (_ELSEWHERE / "f", -1), IoViolation("os.remove", str(_ELSEWHERE / "f"))),
        (
            "shutil.copyfile",
            (_PREFIX / "template.db", _ELSEWHERE / "copy.db"),
            IoViolation("shutil.copyfile", str(_ELSEWHERE / "copy.db")),
        ),
        (
            "os.rename",
            (_BASETEMP / "out.npy", _ELSEWHERE / "out.npy", -1, -1),
            IoViolation("os.rename", str(_ELSEWHERE / "out.npy")),
        ),
        ("os.rename", (_BASETEMP / "a", _BASETEMP / "b", -1, -1), None),
        (
            "shutil.rmtree",
            (_ELSEWHERE / "tree", -1),
            IoViolation("shutil.rmtree", str(_ELSEWHERE / "tree")),
        ),
        (
            "open",
            (b"relative.bin", "wb", _CREATE),
            IoViolation("open", str(_ELSEWHERE / "relative.bin")),
        ),
        ("open", (_BASETEMP / "a.h5", "a", None), IoViolation("open", str(_BASETEMP / "a.h5"))),
        ("sqlite3.connect", ("file::memory:?cache=shared",), None),
        ("sqlite3.connect", (":memory:",), None),
        (
            "sqlite3.connect",
            ("mlflow.db",),
            IoViolation("sqlite3.connect", str(_ELSEWHERE / "mlflow.db")),
        ),
        ("sys._getframe", (None,), None),
    ],
)
def test_audit_events_are_classified_as_own_or_ambient_io(
    event: str, args: tuple[object, ...], expected: IoViolation | None
) -> None:
    assert classify(event, args, _ROOTS, cwd=_ELSEWHERE) == expected
