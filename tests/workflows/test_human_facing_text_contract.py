"""Human-facing text is rendered from typed presentations, never from ids, paths or class names.

One guard for every sink. Each sink registers a scope: the source it covers and the
legitimate machine-only matches inside it, each with the reason it is allowed. The checks
parse the code and look at call and attribute structure, so formatting and renamed
variables do not hide or fake a violation.

Rules:
- ``mapping.get(key, key)`` and ``mapping.get(key, str(key))``: an identifier standing in for a
  missing label.
- ``Path(...).name`` / ``.stem`` and ``str(<path or directory>)``: a path used as a name.
- ``type(x).__name__`` / ``x.__class__.__name__``: a class name used as a name.
- an f-string that interpolates a method, dataset, assignment or system identifier into a
  title, label or tick-label parameter.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "neuralls"


@dataclass(frozen=True, slots=True)
class Violation:
    """One rule match at a source location."""

    rule: str
    path: str
    function: str
    line: int


@dataclass(frozen=True, slots=True)
class Allowed:
    """A legitimate machine-only match: the rule, the function it sits in, and why it is fine."""

    rule: str
    path: str
    function: str
    reason: str


@dataclass(frozen=True, slots=True)
class SinkScope:
    """The source one sink owns and the machine-only matches allowed inside it."""

    roots: tuple[str, ...]
    allowed: tuple[Allowed, ...]


LABEL_FALLBACK = "label-fallback"
PATH_AS_NAME = "path-as-name"
CLASS_NAME = "class-name"
ID_INTERPOLATION = "id-interpolation"

_HUMAN_CALLS = frozenset(
    {
        "set_title",
        "set_xlabel",
        "set_ylabel",
        "suptitle",
        "text",
        "set_yticklabels",
        "set_xticklabels",
    }
)
_HUMAN_KEYWORDS = frozenset({"label", "title", "subtitle", "xlabel", "ylabel"})
_PATHLIKE = re.compile(r"(^|_)(path|dir|file|root)$")
_IDENTIFIER = re.compile(r"^(?!n_)(.*_)?(method|dataset|assignment|system|name)s?$|_id$")

SCOPES: dict[str, SinkScope] = {
    "reporting-and-plots": SinkScope(
        roots=(
            "platform/reporting",
            "composition/comparison",
            "domain/solver/reporting",
        ),
        allowed=(
            Allowed(
                PATH_AS_NAME,
                "composition/comparison/comparison_run.py",
                "_pod2g_style_keys",
                "the fit dataset directory is a colour-grouping key, compared and never drawn",
            ),
            Allowed(
                PATH_AS_NAME,
                "composition/comparison/comparison_run.py",
                "_resolve_comparison_paths",
                "the matrix file stem names the output directory on disk",
            ),
            Allowed(
                PATH_AS_NAME,
                "composition/comparison/_presentation.py",
                "_path_label",
                "the diagnostic log line names its source files; no figure or table uses it",
            ),
            Allowed(
                PATH_AS_NAME,
                "platform/reporting/figure_metadata.py",
                "_collision_key",
                "file-name collision detection compares machine file names",
            ),
            Allowed(
                PATH_AS_NAME,
                "platform/reporting/figure_metadata.py",
                "add",
                "validates that a figure file name is a bare basename",
            ),
            Allowed(
                PATH_AS_NAME,
                "platform/reporting/predictions.py",
                "derive_run_identifier",
                "builds a machine run identifier from checkpoint locations; it is an id, not a label",
            ),
            Allowed(
                CLASS_NAME,
                "platform/reporting/serialization.py",
                "to_json_primitive",
                "names the unsupported type in a programmer-facing TypeError, not in any figure",
            ),
            Allowed(
                PATH_AS_NAME,
                "composition/comparison/source_handlers.py",
                "resolve",
                "records the RHS source location as a machine provenance field, never drawn",
            ),
            Allowed(
                CLASS_NAME,
                "composition/comparison/comparison_run.py",
                "_evaluate_preconditioner",
                "exception class names in a diagnostic log line and a failure record",
            ),
            Allowed(
                CLASS_NAME,
                "composition/comparison/comparison_run.py",
                "_execute_matrix_bundle",
                "exception class names in diagnostic log lines and failure records",
            ),
        ),
    ),
}


def _functions(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Yield ``(enclosing function name, node)`` for every node of the module."""

    def walk(node: ast.AST, function: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else function
            )
            yield inner, child
            yield from walk(child, inner)

    yield from walk(tree, "<module>")


def _identifier_of(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(attr=name):
            return name
        case _:
            return None


def _is_label_fallback(call: ast.Call) -> bool:
    if not (isinstance(call.func, ast.Attribute) and call.func.attr == "get"):
        return False
    if len(call.args) != 2:
        return False
    key, default = call.args
    wrapped = (
        default.args[0]
        if isinstance(default, ast.Call)
        and isinstance(default.func, ast.Name)
        and default.func.id == "str"
        and default.args
        else default
    )
    return ast.dump(key) == ast.dump(wrapped)


def _is_path_as_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr in {"name", "stem"}:
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return value.func.id in {"Path", "PurePath"}
        identifier = _identifier_of(value)
        return identifier is not None and bool(_PATHLIKE.search(identifier))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
    ):
        identifier = _identifier_of(node.args[0])
        return identifier is not None and bool(_PATHLIKE.search(identifier))
    return False


def _is_class_name(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Attribute) and node.attr == "__name__"):
        return False
    value = node.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "type":
        return True
    return isinstance(value, ast.Attribute) and value.attr == "__class__"


def _interpolates_identifier(fstring: ast.JoinedStr) -> bool:
    for part in fstring.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        value = part.value
        if isinstance(value, ast.Call) and _identifier_of(value.func) == "label":
            continue  # a legend lookup: the text comes from a presentation
        for inner in ast.walk(value):
            identifier = _identifier_of(inner)
            if identifier is not None and _IDENTIFIER.search(identifier):
                return True
    return False


def _is_id_interpolation(call: ast.Call) -> bool:
    human_call = isinstance(call.func, ast.Attribute) and call.func.attr in _HUMAN_CALLS
    candidates = list(call.args) if human_call else []
    candidates += [kw.value for kw in call.keywords if kw.arg in _HUMAN_KEYWORDS]
    return any(isinstance(c, ast.JoinedStr) and _interpolates_identifier(c) for c in candidates)


def violations_in(source: str, path: str) -> list[Violation]:
    """Every rule match in ``source``, located by the function that contains it."""
    found: list[Violation] = []
    for function, node in _functions(ast.parse(source)):
        rules: list[str] = []
        if isinstance(node, ast.Call) and _is_label_fallback(node):
            rules.append(LABEL_FALLBACK)
        if _is_path_as_name(node):
            rules.append(PATH_AS_NAME)
        if _is_class_name(node):
            rules.append(CLASS_NAME)
        if isinstance(node, ast.Call) and _is_id_interpolation(node):
            rules.append(ID_INTERPOLATION)
        found += [Violation(rule, path, function, getattr(node, "lineno", 0)) for rule in rules]
    return found


def scope_violations(scope: SinkScope) -> list[Violation]:
    """The matches of every file under the scope's roots that its allow-list does not explain."""
    allowed = {(a.rule, a.path, a.function) for a in scope.allowed}
    found: list[Violation] = []
    for root in scope.roots:
        for file in sorted((SOURCE_ROOT / root).rglob("*.py")):
            relative = file.relative_to(SOURCE_ROOT).as_posix()
            found += [
                violation
                for violation in violations_in(file.read_text(encoding="utf-8"), relative)
                if (violation.rule, violation.path, violation.function) not in allowed
            ]
    return found


def stale_allowances(scope: SinkScope) -> list[Allowed]:
    """Allow-list entries that no longer match any code, so exceptions cannot outlive their reason."""
    present: set[tuple[str, str, str]] = set()
    for root in scope.roots:
        for file in sorted((SOURCE_ROOT / root).rglob("*.py")):
            relative = file.relative_to(SOURCE_ROOT).as_posix()
            present |= {
                (v.rule, v.path, v.function)
                for v in violations_in(file.read_text(encoding="utf-8"), relative)
            }
    return [a for a in scope.allowed if (a.rule, a.path, a.function) not in present]


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_sink_scope_has_no_unexplained_human_text_violations(name: str) -> None:
    assert scope_violations(SCOPES[name]) == []


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_every_allowance_still_matches_code(name: str) -> None:
    assert stale_allowances(SCOPES[name]) == []


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_every_allowance_states_its_reason(name: str) -> None:
    assert all(len(a.reason) > 20 for a in SCOPES[name].allowed)


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("def f(labels, name):\n    return labels.get(name, name)\n", LABEL_FALLBACK),
        ("def f(labels, key):\n    return labels.get(key, str(key))\n", LABEL_FALLBACK),
        ("def f(p):\n    return Path(p).name\n", PATH_AS_NAME),
        ("def f(dataset_dir):\n    return str(dataset_dir)\n", PATH_AS_NAME),
        ("def f(matrix_path):\n    return matrix_path.stem\n", PATH_AS_NAME),
        ("def f(x):\n    return type(x).__name__\n", CLASS_NAME),
        ("def f(x):\n    return x.__class__.__name__\n", CLASS_NAME),
        ('def f(ax, method_id):\n    ax.set_title(f"{method_id} results")\n', ID_INTERPOLATION),
        ('def f(ax, dataset):\n    ax.plot(label=f"{dataset} run")\n', ID_INTERPOLATION),
    ],
)
def test_the_checker_detects_each_rule(source: str, rule: str) -> None:
    assert [v.rule for v in violations_in(source, "x.py")] == [rule]


@pytest.mark.parametrize(
    "source",
    [
        "def f(labels, name):\n    return labels[name]\n",
        "def f(labels, name):\n    return labels.get(name, 'unknown')\n",
        "def f(legend, name):\n    return f'{legend.label(name)} (reference)'\n",
        "def f(ax, legend, name):\n    ax.set_title(f'{legend.label(name)} runs')\n",
        "def f(ax, n):\n    ax.set_title(f'N={n}')\n",
        "def f(d):\n    return d.name\n",
    ],
)
def test_the_checker_accepts_typed_lookups_and_numeric_text(source: str) -> None:
    assert violations_in(source, "x.py") == []
