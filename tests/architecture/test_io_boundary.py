"""Real IO belongs to the IO-boundary test directories; elsewhere it is declared and temporary.

A test file outside the boundary that creates files, a dataset, or an MLflow store must carry
``pytestmark = pytest.mark.deferred_io("<reason>")`` naming the future refactor that will
remove that IO: ``mlflow-run-tracking`` (MLflow runs behind ports), ``dataset-fixture-io`` (a test
that needs a dataset directory on disk as its input fixture) or ``digest-hashing`` (file hashing behind a port). A file
that no longer does IO must drop the marker, so the set only shrinks.

``domain/`` and ``application/`` also take no ambient actions: no file I/O, no log output, no
clock read, no process-wide cache, no CUDA call and no ambient device choice. Their outcomes are
returned as data and composition logs them.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from tests.architecture_support import parsed
from tests.io_guard import BOUNDARY_DIRS

_TESTS = Path(__file__).resolve().parents[1]
_REASONS = frozenset({"mlflow-run-tracking", "dataset-fixture-io", "digest-hashing"})
_PURE_LAYERS = ("domain", "application")
_SRC = _TESTS.parent / "src" / "neuralls"
_FILESYSTEM_METHODS = frozenset(
    {
        "exists", "is_file", "is_dir", "glob", "rglob", "iterdir", "stat", "mkdir", "open",
        "read_text", "write_text", "read_bytes", "write_bytes", "unlink",
    }
)  # fmt: skip
_ARCHIVE_SOURCE_NAMES = frozenset({"source", "archive_source"})
_IO_USE = re.compile(
    r"(?<![.\w])(tmp_path|tmp_path_factory|mlflow_tracking_uri|mlflow_experiment|"
    r"log_run_with_checkpoint|log_run_with_named_checkpoints|build_test_dataset)\b"
    r"|\.write_text\(|\.write_bytes\(|np\.save|mmwrite|tomli_w"
)


def _test_files_outside_the_boundary() -> list[Path]:
    return [
        path
        for path in sorted(_TESTS.rglob("test_*.py"))
        if path.relative_to(_TESTS).parts[0] not in BOUNDARY_DIRS
    ]


def _deferred_plan(source: str) -> str | None:
    """The plan named by a module-level ``pytestmark = pytest.mark.deferred_io("Dn")``."""
    for node in ast.parse(source).body:
        match node:
            case ast.Assign(
                targets=[ast.Name(id="pytestmark")],
                value=ast.Call(
                    func=ast.Attribute(attr="deferred_io"),
                    args=[ast.Constant(value=str(plan))],
                ),
            ):
                return plan
    return None


def test_io_outside_the_boundary_is_declared_with_its_retiring_plan() -> None:
    undeclared, unknown_plan, stale = [], [], []
    for path in _test_files_outside_the_boundary():
        source = path.read_text(encoding="utf-8")
        plan = _deferred_plan(source)
        name = path.relative_to(_TESTS).as_posix()
        uses_io = _IO_USE.search(source) is not None
        if uses_io and plan is None:
            undeclared.append(name)
        elif plan is not None and plan not in _REASONS:
            unknown_plan.append(name)
        elif plan is not None and not uses_io:
            stale.append(name)

    assert (undeclared, unknown_plan, stale) == ([], [], [])


def _filesystem_calls(path: Path) -> list[int]:
    """Lines of file I/O calls; the injected archive source port is the only allowed receiver."""
    lines = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Call(func=ast.Name(id="open")):
                lines.append(node.lineno)
            case ast.Call(func=ast.Attribute(attr=attr, value=ast.Name(id=receiver))) if (
                attr in _FILESYSTEM_METHODS and receiver not in _ARCHIVE_SOURCE_NAMES
            ) or (receiver in {"np", "numpy"} and attr.startswith(("load", "save"))):
                lines.append(node.lineno)
            case ast.Call(func=ast.Attribute(attr=attr, value=receiver)) if (
                attr in _FILESYSTEM_METHODS and not isinstance(receiver, ast.Name)
            ):
                lines.append(node.lineno)
    return lines


def test_pure_layers_do_no_file_io() -> None:
    offenders = [
        f"{path.relative_to(_SRC).as_posix()}:{line}"
        for layer in _PURE_LAYERS
        for path in sorted((_SRC / layer).rglob("*.py"))
        for line in _filesystem_calls(path)
    ]
    assert offenders == []


_BANNED_MODULES = frozenset({"loguru", "logging", "warnings", "time", "timeit"})
_CLOCK_READS = frozenset({"now", "utcnow", "today"})
_CACHE_DECORATORS = frozenset({"lru_cache", "cache"})
_DEVICE_ACTIONS = frozenset({"resolve_device", "release_device_memory"})


def _root(module: str | None) -> str:
    return (module or "").split(".")[0]


def _is_cuda_module(module: str | None) -> bool:
    return (module or "").startswith("torch.cuda")


def _unseeded(args: list[ast.expr], keywords: list[ast.keyword]) -> bool:
    """No seed argument, or a literal ``None`` one: the generator draws OS entropy."""
    seeds = [*args, *(keyword.value for keyword in keywords)]
    return all(isinstance(seed, ast.Constant) and seed.value is None for seed in seeds)


def _ambient_actions(path: Path) -> list[str]:
    """Logging, clock reads, process-wide caches, CUDA use and ambient device picks.

    ``import time`` is banned outright, not only its clock calls: the one timing the
    domain does goes through ``shared/device.py``.
    """
    found = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Import(names=names) if any(
                _root(alias.name) in _BANNED_MODULES or _is_cuda_module(alias.name)
                for alias in names
            ):
                found.append(f"{node.lineno} import")
            case ast.ImportFrom(module=module) if _root(module) in _BANNED_MODULES or (
                _is_cuda_module(module)
            ):
                found.append(f"{node.lineno} import")
            case ast.ImportFrom(module="torch", names=names) if any(
                alias.name == "cuda" for alias in names
            ):
                found.append(f"{node.lineno} cuda")
            case ast.ImportFrom(module="functools", names=names) if any(
                alias.name in _CACHE_DECORATORS for alias in names
            ):
                found.append(f"{node.lineno} cache")
            case ast.ImportFrom(names=names) if any(
                alias.name in _DEVICE_ACTIONS for alias in names
            ):
                found.append(f"{node.lineno} device")
            case (
                ast.Attribute(attr="lru_cache")
                | ast.Attribute(attr="cache", value=ast.Name(id="functools"))
            ):
                found.append(f"{node.lineno} cache")
            case ast.Attribute(attr="cuda"):
                found.append(f"{node.lineno} cuda")
            case ast.Attribute(attr=attr) if attr in _DEVICE_ACTIONS:
                found.append(f"{node.lineno} device")
            case ast.Call(func=ast.Attribute(attr=attr)) if attr in _CLOCK_READS:
                found.append(f"{node.lineno} clock")
            case ast.Call(
                func=ast.Attribute(attr="default_rng") | ast.Name(id="default_rng"),
                args=args,
                keywords=keywords,
            ) if _unseeded(args, keywords):
                found.append(f"{node.lineno} entropy")
    return found


def test_pure_layers_take_no_ambient_actions() -> None:
    offenders = [
        f"{path.relative_to(_SRC).as_posix()}:{finding}"
        for layer in _PURE_LAYERS
        for path in sorted((_SRC / layer).rglob("*.py"))
        for finding in _ambient_actions(path)
    ]
    assert offenders == []


_PATH_READS = frozenset(
    {
        "resolve",
        "absolute",
        "cwd",
        "home",
        "expanduser",
        "expandvars",
        "getcwd",
        "abspath",
        "realpath",
    }
)
_PATH_RESOLVERS = frozenset(
    {
        "url_resolver",
        "resolve_local_path",
        "resolve_user_path",
        "expand_user_path",
        "require_absolute_user_path",
        "resolve_case_config_path",
        "resolve_env_file_path",
        "resolve_registry_path",
        "normalize_tracking_uri",
        "normalize_artifacts_destination",
        "build_sqlite_uri",
        "build_sqlite_tracking_uri",
        "build_mlflow_environment",
        "derive_output_root_from_tracking_uri",
        "derive_sqlite_artifacts_destination",
    }
)
_PATH_PURE_SCOPES = (*_PURE_LAYERS, "platform/config/models")


def _ambient_path_reads(path: Path) -> list[str]:
    """Making a path absolute reads the working directory, ``$HOME`` or symlinks.

    Any reference counts, called or not (``default_factory=Path.home``), whatever its
    receiver, and so does a bare name imported from ``os`` / ``os.path``: the scopes have
    no other function of those names, and a new one is better renamed than allowed.
    """
    found = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Attribute(attr=attr) if attr in _PATH_READS:
                found.append(f"{node.lineno} {attr}")
            case ast.ImportFrom(module="os" | "os.path", names=names) if any(
                alias.name in _PATH_READS for alias in names
            ):
                found.append(f"{node.lineno} import")
            case ast.Import(names=names) if any(
                alias.name.endswith("url_resolver") for alias in names
            ):
                found.append(f"{node.lineno} resolver")
            case ast.ImportFrom(module=module, names=names) if (module or "").endswith(
                "url_resolver"
            ) or any(alias.name in _PATH_RESOLVERS for alias in names):
                found.append(f"{node.lineno} resolver")
            case ast.Attribute(attr=attr) if attr in _PATH_RESOLVERS:
                found.append(f"{node.lineno} resolver")
    return found


def test_pure_layers_and_config_models_read_no_ambient_paths() -> None:
    offenders = [
        f"{path.relative_to(_SRC).as_posix()}:{finding}"
        for scope in _PATH_PURE_SCOPES
        for path in sorted((_SRC / scope).rglob("*.py"))
        for finding in _ambient_path_reads(path)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "p.resolve()",
        "Path.cwd()",
        "Field(default_factory=Path.home)",
        "os.path.abspath(p)",
        "from os import getcwd",
        "from os.path import realpath",
        "import dlkit.infrastructure.io.url_resolver as ur",
        "from neuralls.platform.config.resolution import resolve_user_path",
        "resolution.normalize_tracking_uri(u, base_dir=d)",
    ],
)
def test_the_ambient_path_lint_catches(tmp_path: Path, source: str) -> None:
    module = tmp_path / "module.py"
    module.write_text(source + "\n", encoding="utf-8")

    assert _ambient_path_reads(module) != []


_ALLOWED_MLFLOW_IMPORTS: dict[str, frozenset[str] | None] = {
    "mlflow.tracking": frozenset({"MlflowClient"}),
    "mlflow.tracking.context.registry": frozenset({"resolve_tags"}),
    "mlflow.entities": None,
    "mlflow.exceptions": None,
    "mlflow.protos.databricks_pb2": None,
}
"""The MLflow imports that cannot reach process-wide state; ``None`` allows every name."""


def _allowed_mlflow_import(module: str, names: list[ast.alias]) -> bool:
    if module not in _ALLOWED_MLFLOW_IMPORTS:
        return False
    allowed = _ALLOWED_MLFLOW_IMPORTS[module]
    return allowed is None or all(alias.name in allowed for alias in names)


def _ambient_tracking(path: Path) -> list[int]:
    """An MLflow import that can reach its fluent API.

    ``mlflow.log_*``, ``mlflow.start_run`` and their kin write into the process-wide active run
    on the process-wide tracking URI; with neither set they start a run on ``./mlflow.db``.
    Tracking writes go through ``platform/tracking/run_writer.py`` on an explicit client and
    run id, so only the client, entities, exceptions, error codes and tag resolver may be imported.
    """
    found = []
    for node in ast.walk(parsed(path)):
        match node:
            case ast.Import(names=names) if any(
                alias.name.split(".")[0] == "mlflow" for alias in names
            ):
                found.append(node.lineno)
            case ast.ImportFrom(module=str(module), names=names) if module.split(".")[
                0
            ] == "mlflow" and not _allowed_mlflow_import(module, names):
                found.append(node.lineno)
    return found


def test_code_reaches_mlflow_only_through_explicit_clients() -> None:
    offenders = [
        f"{path.relative_to(root.parent).as_posix()}:{line}"
        for root in (_SRC, _TESTS)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
        for line in _ambient_tracking(path)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "import mlflow",
        "import mlflow.tracking",
        "from mlflow import log_metric",
        "from mlflow.tracking.fluent import start_run",
        "from mlflow.tracking import set_tracking_uri",
    ],
)
def test_the_ambient_tracking_lint_catches(tmp_path: Path, source: str) -> None:
    module = tmp_path / "module.py"
    module.write_text(source + "\n", encoding="utf-8")

    assert _ambient_tracking(module) != []


def test_the_ambient_tracking_lint_allows_client_imports(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text("from mlflow.tracking import MlflowClient\n", encoding="utf-8")

    assert _ambient_tracking(module) == []
