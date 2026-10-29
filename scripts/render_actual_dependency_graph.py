from __future__ import annotations

import json
import sys
from pathlib import Path


def _module_from_path(path: str) -> str | None:
    candidate = Path(path)
    parts = candidate.parts
    if "src" not in parts or candidate.suffix != ".py":
        return None
    src_index = parts.index("src")
    module_parts = list(parts[src_index + 1 :])
    if not module_parts:
        return None
    if module_parts[-1] == "__init__.py":
        module_parts = module_parts[:-1]
    else:
        module_parts[-1] = candidate.stem
    return ".".join(module_parts) if module_parts else None


def _all_modules(dep_map: dict[str, list[str]]) -> set[str]:
    modules: set[str] = set()
    for source_path, dep_paths in dep_map.items():
        source_module = _module_from_path(source_path)
        if source_module is not None:
            modules.add(source_module)
            parts = source_module.split(".")
            for index in range(1, len(parts)):
                modules.add(".".join(parts[:index]))
        for dep_path in dep_paths:
            dep_module = _module_from_path(dep_path)
            if dep_module is not None:
                modules.add(dep_module)
                parts = dep_module.split(".")
                for index in range(1, len(parts)):
                    modules.add(".".join(parts[:index]))
    return modules


def _is_direct_child(module_name: str, root_module: str) -> bool:
    if not module_name.startswith(f"{root_module}."):
        return False
    suffix = module_name[len(root_module) + 1 :]
    return "." not in suffix


def _direct_children(modules: set[str], root_module: str) -> set[str]:
    return {module for module in modules if _is_direct_child(module, root_module)}


def _display_label(module_name: str, root_module: str) -> str:
    if module_name == root_module:
        return root_module.rsplit(".", maxsplit=1)[-1]
    prefix = f"{root_module}."
    if module_name.startswith(prefix):
        return module_name[len(prefix) :]
    if module_name.startswith("neuralls."):
        return module_name[len("neuralls.") :]
    return module_name


def _resolve_node(
    module_name: str,
    *,
    root_module: str,
    direct_children: set[str],
    context_modules: list[str],
) -> str | None:
    if module_name == root_module:
        return root_module
    for child in direct_children:
        if module_name == child or module_name.startswith(f"{child}."):
            return child
    for context in context_modules:
        if module_name == context or module_name.startswith(f"{context}."):
            return context
    return None


def _build_edges(
    dep_map: dict[str, list[str]],
    *,
    root_module: str,
    direct_children: set[str],
    context_modules: list[str],
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source_path, dep_paths in dep_map.items():
        source_module = _module_from_path(source_path)
        if source_module is None:
            continue
        source_node = _resolve_node(
            source_module,
            root_module=root_module,
            direct_children=direct_children,
            context_modules=context_modules,
        )
        if source_node is None:
            continue
        for dep_path in dep_paths:
            dep_module = _module_from_path(dep_path)
            if dep_module is None:
                continue
            dep_node = _resolve_node(
                dep_module,
                root_module=root_module,
                direct_children=direct_children,
                context_modules=context_modules,
            )
            if dep_node is None or dep_node == source_node:
                continue
            if source_node == root_module and dep_node in direct_children:
                continue
            edges.add((source_node, dep_node))
    return edges


def _render_dot(
    *,
    root_module: str,
    direct_children: set[str],
    context_modules: list[str],
    edges: set[tuple[str, str]],
) -> str:
    is_overview = root_module == "neuralls" and not context_modules
    nodes = {root_module, *direct_children, *context_modules}
    if is_overview:
        nodes.discard(root_module)
    lines = ["digraph {"]
    for node in sorted(nodes):
        label = _display_label(node, root_module)
        if node == root_module:
            lines.append(f'"{node}" [label="{label}" style="bold"];')
        else:
            lines.append(f'"{node}" [label="{label}"];')
    if not is_overview:
        for child in sorted(direct_children):
            lines.append(
                f'"{root_module}" -> "{child}" [style="dashed" color="gray50" arrowhead="none"];'
            )
    for source, target in sorted(edges):
        if is_overview and (source == root_module or target == root_module):
            continue
        lines.append(f'"{source}" -> "{target}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        raise SystemExit(
            "usage: python scripts/render_actual_dependency_graph.py "
            "<map.json> <output.dot> <root_module> [context_modules_csv]"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    root_module = sys.argv[3]
    context_modules = []
    if len(sys.argv) == 5 and sys.argv[4]:
        context_modules = [item.strip() for item in sys.argv[4].split(",") if item.strip()]

    dep_map = json.loads(input_path.read_text(encoding="utf-8"))
    modules = _all_modules(dep_map)
    direct_children = _direct_children(modules, root_module)
    edges = _build_edges(
        dep_map,
        root_module=root_module,
        direct_children=direct_children,
        context_modules=context_modules,
    )
    output_path.write_text(
        _render_dot(
            root_module=root_module,
            direct_children=direct_children,
            context_modules=context_modules,
            edges=edges,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
