#!/usr/bin/env python3
"""CI container-hierarchy invariant checker.

Verifies the 4-tier container hierarchy invariant against the ``.pyi`` stubs via AST
parsing:

* every ``ScalarArrayIndex*`` alias must not reference ``Series``;
* every ``ScalarArrayIndexSeries*`` alias must not reference ``DataFrame``;
* the forward arithmetic dunders of ``Index`` must not reference ``Series`` in their
  ``other`` operand;
* the forward arithmetic dunders of ``Series`` must not reference ``DataFrame`` in
  their ``other`` operand.

Reverse dunders (``__r*__``) and ``__matmul__`` are deliberately out of scope (see
``docs/type-architecture.md``).
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

FORWARD_DUNDERS = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
]


def collect_aliases(paths: list[Path]) -> dict[str, ast.AST]:
    """Collect every ``TypeAlias`` RHS from ``paths``, keyed by alias name."""
    aliases: dict[str, ast.AST] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.annotation, ast.Name)
                and node.annotation.id == "TypeAlias"
                and node.value is not None
            ):
                aliases[node.target.id] = node.value
    return aliases


def references_name(
    node: ast.AST | None, target: str, aliases: dict[str, ast.AST]
) -> bool:
    """Return whether ``node`` references ``target`` directly or via a ``TypeAlias``."""
    if node is None:
        return False
    to_expand = [node]
    expanded: set[str] = set()
    while to_expand:
        current = to_expand.pop()
        for child in ast.walk(current):
            if not isinstance(child, ast.Name):
                continue
            if child.id == target:
                return True
            if child.id in aliases and child.id not in expanded:
                expanded.add(child.id)
                to_expand.append(aliases[child.id])
    return False


def find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    """Return the top-level class ``name`` from ``tree``, if present."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _other_annotation(func: ast.FunctionDef) -> ast.AST | None:
    all_args = getattr(func.args, "posonlyargs", []) + func.args.args
    for arg in all_args:
        if arg.arg == "other":
            return arg.annotation
    return None


def check_alias_level(base_tree: ast.AST, aliases: dict[str, ast.AST]) -> bool:
    ok = True
    for node in ast.walk(base_tree):
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Name)
            and node.annotation.id == "TypeAlias"
            and node.value is not None
        ):
            continue
        name = node.target.id
        if name.startswith("ScalarArrayIndexSeries"):
            forbidden = "DataFrame"
        elif name.startswith("ScalarArrayIndex"):
            forbidden = "Series"
        else:
            continue
        if references_name(node.value, forbidden, aliases):
            sys.stdout.write(
                f"❌ ERROR: alias {name} references {forbidden} — "
                "violates the container hierarchy.\n"
            )
            ok = False
    return ok


def check_forward_dunders(
    class_node: ast.ClassDef | None,
    class_name: str,
    forbidden: str,
    aliases: dict[str, ast.AST],
) -> bool:
    if class_node is None:
        sys.stdout.write(f"Error: Could not find class '{class_name}' in stubs.\n")
        return False
    ok = True
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in FORWARD_DUNDERS:
            continue
        if references_name(_other_annotation(node), forbidden, aliases):
            sys.stdout.write(
                f"❌ ERROR: {class_name}.{node.name} `other` operand references "
                f"{forbidden} — violates the container hierarchy.\n"
            )
            ok = False
    return ok


def check_container_hierarchy(repo_root: Path) -> bool:
    base_file = repo_root / "pandas-stubs" / "core" / "base.pyi"
    typing_file = repo_root / "pandas-stubs" / "_typing.pyi"
    index_file = repo_root / "pandas-stubs" / "core" / "indexes" / "base.pyi"
    series_file = repo_root / "pandas-stubs" / "core" / "series.pyi"

    for path in (base_file, typing_file, index_file, series_file):
        if not path.exists():
            sys.stdout.write(f"Error: Missing required file {path}.\n")
            return False

    aliases = collect_aliases([base_file, typing_file])
    base_tree = ast.parse(base_file.read_text(encoding="utf-8"))
    index_tree = ast.parse(index_file.read_text(encoding="utf-8"))
    series_tree = ast.parse(series_file.read_text(encoding="utf-8"))

    ok = True
    if not check_alias_level(base_tree, aliases):
        ok = False
    if not check_forward_dunders(
        find_class(index_tree, "Index"), "Index", "Series", aliases
    ):
        ok = False
    if not check_forward_dunders(
        find_class(series_tree, "Series"), "Series", "DataFrame", aliases
    ):
        ok = False

    if ok:
        sys.stdout.write("✅ Container hierarchy invariant holds.\n")
    return ok


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    if not check_container_hierarchy(repo_root):
        sys.exit(1)
    sys.exit(0)
