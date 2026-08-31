#!/usr/bin/env python3
"""Check the container-hierarchy constraints in a pandas-stubs tree.

The checker reads the stubs as syntax trees. It verifies that:

* ``ScalarArrayIndex*`` aliases do not reference ``Series`` or ``DataFrame``;
* ``ScalarArrayIndexSeries*`` aliases do not reference ``DataFrame``; and
* forward binary dunders declared directly on ``Index`` and ``Series`` do not name a
  higher-tier container in their ``other`` annotation, unless an explicit exception
  permits it.

The checks include direct and transitive references through ``TypeAlias`` definitions.
Reflected dunders are deliberately outside this structural check.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import (
    TYPE_CHECKING,
    Final,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

ExceptionKey = tuple[str, str, str]


@dataclass(frozen=True)
class HierarchyException:
    """A deliberate higher-tier reference in a forward binary dunder."""

    rationale: str
    documentation: str


# Adding an exception is a compatibility decision: update the linked documentation
# and the exact-registry test in tests/test_check_container_hierarchy.py as well.
FORWARD_DUNDER_EXCEPTIONS: Final[dict[ExceptionKey, HierarchyException]] = {
    ("Series", "__matmul__", "DataFrame"): HierarchyException(
        rationale="Series matrix multiplication with a DataFrame returns a Series.",
        documentation=(
            "docs/type-architecture/container-hierarchy.md#matrix-multiplication"
        ),
    ),
}

# A forward dunder can itself start with ``__r`` (for example, ``__rshift__``), so
# reverse operations are enumerated instead of being filtered by a name prefix.
REVERSE_BINARY_DUNDERS: Final[frozenset[str]] = frozenset(
    {
        "__radd__",
        "__rand__",
        "__rdivmod__",
        "__rfloordiv__",
        "__rlshift__",
        "__rmatmul__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rrshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
    }
)


def _type_aliases(tree: ast.AST) -> dict[str, ast.AST]:
    """Return ``TypeAlias`` right-hand sides from ``tree``, keyed by alias name."""
    aliases: dict[str, ast.AST] = {}
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


def collect_aliases(stub_root: Path) -> dict[str, ast.AST]:
    """Collect aliases from every stub file below ``stub_root``."""
    aliases: dict[str, ast.AST] = {}
    for path in sorted(stub_root.rglob("*.pyi")):
        aliases.update(_type_aliases(ast.parse(path.read_text(encoding="utf-8"))))
    return aliases


def references_name(
    node: ast.AST | None, target: str, aliases: Mapping[str, ast.AST]
) -> bool:
    """Return whether ``node`` references ``target`` directly or through aliases."""
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
            alias = aliases.get(child.id)
            if alias is not None and child.id not in expanded:
                expanded.add(child.id)
                to_expand.append(alias)
    return False


def find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    """Return the top-level class ``name`` from ``tree``, if present."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _other_annotation(function: ast.FunctionDef) -> ast.AST | None:
    all_args = function.args.posonlyargs + function.args.args
    for arg in all_args:
        if arg.arg == "other":
            return arg.annotation
    return None


def is_forward_binary_dunder(function: ast.FunctionDef) -> bool:
    """Return whether ``function`` is an in-scope forward binary dunder."""
    return (
        function.name.startswith("__")
        and function.name.endswith("__")
        and function.name not in REVERSE_BINARY_DUNDERS
        and _other_annotation(function) is not None
    )


def check_alias_level(aliases: Mapping[str, ast.AST]) -> bool:
    """Check hierarchy-bearing aliases for direct or transitive higher tiers."""
    ok = True
    for name, value in sorted(aliases.items()):
        forbidden_names: tuple[str, ...]
        if name.startswith("ScalarArrayIndexSeries"):
            forbidden_names = ("DataFrame",)
        elif name.startswith("ScalarArrayIndex"):
            forbidden_names = ("Series", "DataFrame")
        else:
            continue

        for forbidden in forbidden_names:
            if references_name(value, forbidden, aliases):
                message = "ERROR: alias {} references {} — violates the container hierarchy.\n"
                sys.stdout.write(
                    message.format(
                        name,
                        forbidden,
                    )
                )
                ok = False
    return ok


def check_forward_binary_dunders(
    class_node: ast.ClassDef | None,
    class_name: str,
    forbidden: str,
    aliases: Mapping[str, ast.AST],
    exceptions: Mapping[ExceptionKey, HierarchyException],
) -> bool:
    """Check every direct forward binary dunder with an ``other`` operand."""
    if class_node is None:
        sys.stdout.write(f"ERROR: could not find class {class_name!r} in stubs.\n")
        return False

    ok = True
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or not is_forward_binary_dunder(node):
            continue

        if references_name(_other_annotation(node), forbidden, aliases):
            key = (class_name, node.name, forbidden)
            if key in exceptions:
                continue
            message = "ERROR: {}.{} `other` operand references {} — violates the container hierarchy.\n"
            sys.stdout.write(
                message.format(
                    class_name,
                    node.name,
                    forbidden,
                )
            )
            ok = False
    return ok


def _read_required_trees(stub_root: Path) -> dict[Path, ast.Module] | None:
    required_files = (
        Path("core/base.pyi"),
        Path("core/indexes/base.pyi"),
        Path("core/series.pyi"),
    )
    paths = [stub_root / path for path in required_files]
    for path in paths:
        if not path.exists():
            sys.stdout.write(f"ERROR: missing required stub file {path}.\n")
            return None
    return {
        path.relative_to(stub_root): ast.parse(path.read_text(encoding="utf-8"))
        for path in paths
    }


def check_container_hierarchy(
    stub_root: Path,
    *,
    exceptions: Mapping[ExceptionKey, HierarchyException] = FORWARD_DUNDER_EXCEPTIONS,
) -> bool:
    """Check the hierarchy constraints in the ``pandas-stubs`` directory ``stub_root``."""
    trees = _read_required_trees(stub_root)
    if trees is None:
        return False

    aliases = collect_aliases(stub_root)
    ok = check_alias_level(aliases)
    index_class = find_class(trees[Path("core/indexes/base.pyi")], "Index")
    for forbidden in ("Series", "DataFrame"):
        if not check_forward_binary_dunders(
            index_class,
            "Index",
            forbidden,
            aliases,
            exceptions,
        ):
            ok = False
    if not check_forward_binary_dunders(
        find_class(trees[Path("core/series.pyi")], "Series"),
        "Series",
        "DataFrame",
        aliases,
        exceptions,
    ):
        ok = False

    if ok:
        sys.stdout.write("Container hierarchy invariant holds.\n")
    return ok


if __name__ == "__main__":
    STUB_ROOT = Path(__file__).parent.parent / "pandas-stubs"
    sys.exit(not check_container_hierarchy(STUB_ROOT))
