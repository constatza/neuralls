"""Runtime guard: tests outside the IO-boundary directories do no file, database or store IO.

The static marker check in ``tests/architecture/test_io_boundary.py`` reads test source, so it
misses IO done through fixtures and libraries (a conftest fixture writing arrays, MLflow's
fluent API opening ``sqlite:///mlflow.db`` when no run is active). This plugin watches the
interpreter's audit events while a guarded test sets up, runs and tears down, and fails the
test that touched the filesystem or a database.

What counts: any write or sqlite connection; a mkdir, remove, rename, copy or truncate outside
pytest's basetemp; and any read of a file under pytest's basetemp or of a non-Python file under
``tests/``. What does not: imports and bytecode caches, reads of the installed environment and
of the package sources, empty directories pytest creates for ``tmp_path``.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
BOUNDARY_DIRS = ("io", "integration", "platform", "architecture", "manual")
"""Top-level test directories whose subject is IO; only they may touch files or stores."""

PENDING_FILES = frozenset(
    {
        "generation/test_archive_no_repeat.py",
        "generation/test_zero_row_binding_scale.py",
    }
)
"""Test files outside the boundary that still read fixture-written files; the set only shrinks."""

AMBIENT_STORES = ("mlflow.db", "mlruns")
"""What MLflow creates in the working directory when it falls back to its default store."""

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_SYSTEM_ROOTS = (Path("/dev"), Path("/proc"), Path("/sys"))
_MUTATING_EVENTS = frozenset(
    {"os.remove", "os.rmdir", "os.rename", "os.truncate", "shutil.rmtree", "shutil.copyfile"}
)
_TWO_PATH_EVENTS = frozenset({"shutil.copyfile", "os.rename"})
_IN_MEMORY_DATABASES = frozenset({":memory:", ""})


@dataclass(frozen=True, slots=True)
class GuardRoots:
    """Where IO is a test's own (basetemp, tests) and where reads are ambient (ignored)."""

    basetemp: Path
    tests: Path
    ignored: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class IoViolation:
    """One audited filesystem or database action a guarded test performed."""

    event: str
    path: str


def classify(
    event: str, args: tuple[object, ...], roots: GuardRoots, cwd: Path
) -> IoViolation | None:
    """The violation an audit event represents for a guarded test, if any."""
    match event, args:
        case "open", (str() | bytes() | os.PathLike() as target, mode, flags, *_):
            path = _absolute(target, cwd)
            if _ignored(path, roots):
                return None
            if _writes(mode, flags) or _reads_test_data(path, roots):
                return IoViolation(event, str(path))
            return None
        case "os.mkdir", (str() | bytes() | os.PathLike() as target, *_):
            path = _absolute(target, cwd)
            if path.is_relative_to(roots.basetemp) or _ignored(path, roots):
                return None
            return IoViolation(event, str(path))
        case "sqlite3.connect", (str() | bytes() | os.PathLike() as database, *_):
            if isinstance(database, str) and _in_memory(database):
                return None
            return IoViolation(event, str(_absolute(database, cwd)))
        case mutating, (str() | bytes() | os.PathLike() as target, *rest) if (
            mutating in _MUTATING_EVENTS
        ):
            # copyfile and rename report (src, dst, ...); the destination is the write.
            written = rest[0] if mutating in _TWO_PATH_EVENTS and rest else target
            if not isinstance(written, str | bytes | os.PathLike):
                return None
            path = _absolute(written, cwd)
            # Under basetemp pytest itself rotates numbered dirs and their `current` links;
            # a test's own IO there always starts with an `open`, which is classified above.
            if path.is_relative_to(roots.basetemp) or _ignored(path, roots):
                return None
            return IoViolation(event, str(path))
        case _:
            return None


def _absolute(target: str | bytes | os.PathLike[str] | os.PathLike[bytes], cwd: Path) -> Path:
    path = Path(os.fsdecode(target))
    return path if path.is_absolute() else cwd / path


def _ignored(path: Path, roots: GuardRoots) -> bool:
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        return True
    return any(path.is_relative_to(root) for root in (*roots.ignored, *_SYSTEM_ROOTS))


def _writes(mode: object, flags: object) -> bool:
    if isinstance(flags, int):
        return bool(flags & _WRITE_FLAGS)
    return isinstance(mode, str) and any(char in mode for char in "wax+")


def _reads_test_data(path: Path, roots: GuardRoots) -> bool:
    if path.is_relative_to(roots.basetemp):
        return True
    return path.is_relative_to(roots.tests) and path.suffix != ".py"


def _in_memory(database: str) -> bool:
    return database in _IN_MEMORY_DATABASES or database.startswith("file::memory:")


def is_boundary_test(path: Path) -> bool:
    """Whether a test file lives in a directory whose subject is IO."""
    return path.relative_to(TESTS_DIR).parts[0] in BOUNDARY_DIRS


def _relative_test_path(path: Path) -> str | None:
    return path.relative_to(TESTS_DIR).as_posix() if path.is_relative_to(TESTS_DIR) else None


# --- pytest shell -------------------------------------------------------------------------


@dataclass(slots=True)
class _Watch:
    """Process-wide audit state: the hook cannot be removed, so it reads this to stay inert."""

    roots: GuardRoots | None = None
    nodeid: str | None = None
    found: list[IoViolation] = field(default_factory=list)


_WATCH = _Watch()
_WATCHED_EVENTS = frozenset({"open", "os.mkdir", "sqlite3.connect", *_MUTATING_EVENTS})
_STORES_BEFORE = pytest.StashKey[frozenset[str]]()


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event not in _WATCHED_EVENTS or _WATCH.nodeid is None or _WATCH.roots is None:
        return
    violation = classify(event, args, _WATCH.roots, _cwd())
    if violation is not None:
        _WATCH.found.append(violation)


# Audit hooks cannot be removed; the plugin module is imported once per process.
sys.addaudithook(_audit)


def _cwd() -> Path:
    try:
        return Path(os.getcwd())
    except FileNotFoundError:  # a test removed its working directory
        return TESTS_DIR


def _audited_hdf5_open(original: Callable[..., None]) -> Callable[..., None]:
    """libhdf5 opens files in C, out of sight of Python's `open` audit event; report it."""

    @functools.wraps(original)
    def init(
        self: h5py.File, name: object, mode: str | None = None, *args: object, **kwargs: object
    ) -> None:
        if isinstance(name, str | bytes | os.PathLike):
            sys.audit("open", name, mode or "r", None)
        original(self, name, mode, *args, **kwargs)

    return init


h5py.File.__init__ = _audited_hdf5_open(h5py.File.__init__)


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_STORES_BEFORE] = _ambient_stores()


@pytest.fixture(scope="session", autouse=True)
def _io_guard_roots(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Resolve basetemp once, before any guarded phase can see pytest create it."""
    nodeid, _WATCH.nodeid = _WATCH.nodeid, None
    _WATCH.roots = GuardRoots(
        basetemp=tmp_path_factory.getbasetemp(),
        tests=TESTS_DIR,
        ignored=_ambient_read_roots(),
    )
    _WATCH.nodeid = nodeid


def _ambient_read_roots() -> tuple[Path, ...]:
    prefixes = {Path(sys.prefix), Path(sys.base_prefix), Path(sys.exec_prefix)}
    return (*sorted(prefixes), REPO_ROOT / "src", REPO_ROOT / ".venv")


def _guarded(item: pytest.Item) -> bool:
    match _relative_test_path(item.path):
        case None:
            return False
        case relative:
            return not (
                is_boundary_test(item.path)
                or relative in PENDING_FILES
                or item.get_closest_marker("deferred_io") is not None
            )


def _watch_phase(item: pytest.Item) -> Generator[None, object, object]:
    if not _guarded(item):
        return (yield)
    _WATCH.nodeid = item.nodeid
    _WATCH.found.clear()
    try:
        result = yield
    finally:
        _WATCH.nodeid = None
    if _WATCH.found:
        raise IoOutsideBoundaryError(item.nodeid, tuple(_WATCH.found))
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _watch_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _watch_phase(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item) -> Generator[None, object, object]:
    return (yield from _watch_phase(item))


class IoOutsideBoundaryError(AssertionError):
    """A test outside the IO-boundary directories touched the filesystem or a database."""

    def __init__(self, nodeid: str, violations: tuple[IoViolation, ...]) -> None:
        self.nodeid = nodeid
        self.violations = violations
        shown = "\n".join(f"  {v.event}: {v.path}" for v in _unique(violations))
        super().__init__(
            f"{nodeid} does IO outside the IO-boundary test directories "
            f"({', '.join(BOUNDARY_DIRS)}). Move it to tests/io/, or give it in-memory "
            f"values instead of files:\n{shown}"
        )


def _unique(violations: Iterable[IoViolation]) -> list[IoViolation]:
    return list(dict.fromkeys(violations))


def _ambient_stores() -> frozenset[str]:
    return frozenset(name for name in AMBIENT_STORES if (REPO_ROOT / name).exists())


def pytest_sessionfinish(session: pytest.Session) -> None:
    if hasattr(session.config, "workerinput"):
        return
    leaked = _ambient_stores() - session.config.stash[_STORES_BEFORE]
    if not leaked:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"Tests created {sorted(leaked)} in {REPO_ROOT}: an MLflow call fell back to its "
            "ambient default store.",
            red=True,
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
