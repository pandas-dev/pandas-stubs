#!/usr/bin/env python3
# ruff: noqa: T201
"""Check the operand-hierarchy constraints in a pandas-stubs tree.

The checker reads the stubs as syntax trees. It verifies that:

* ``ScalarArrayIndex*`` aliases do not reference ``Series`` or ``DataFrame``;
* ``ScalarArrayIndexSeries*`` aliases do not reference ``DataFrame``; and
* forward binary dunders declared directly on ``Index``, ``MultiIndex``, and
  ``Series`` do not name a
  higher-tier operand type in their ``other`` annotation, unless an explicit exception
  permits it.

The checks include direct and transitive references through ``TypeAlias`` definitions.
Reflected dunders are deliberately outside this structural check.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

ExceptionKey = tuple[str, str, str]


@dataclass(frozen=True)
class HierarchyException:
    """A deliberate higher-tier reference in a forward binary dunder."""

    rationale: str
    documentation: str


# Adding an exception is a compatibility decision: update the linked documentation
# and the exact-registry test in tests/test_check_operand_hierarchy.py as well.
FORWARD_DUNDER_EXCEPTIONS: Final[dict[ExceptionKey, HierarchyException]] = {
    ("Series", "__matmul__", "DataFrame"): HierarchyException(
        rationale="Series matrix multiplication with a DataFrame returns a Series.",
        documentation=(
            "docs/type-architecture/operand-hierarchy.md#matrix-multiplication"
        ),
    ),
}

# The scanned set is enumerated explicitly. Name heuristics cannot separate forward
# from reflected operations (a forward dunder can itself start with ``__r``, as
# ``__rshift__`` does), and a dunder outside this set — every reflected dunder, like
# ``__radd__``, plus non-operator methods such as ``__init__`` — is not scanned.
FORWARD_BINARY_DUNDERS: Final[frozenset[str]] = frozenset(
    {
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__lshift__",
        "__rshift__",
        "__and__",
        "__or__",
        "__xor__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
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


def _other_parameter(function: ast.FunctionDef) -> ast.arg | None:
    arguments = function.args
    all_args = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    if arguments.vararg is not None:
        all_args.append(arguments.vararg)
    if arguments.kwarg is not None:
        all_args.append(arguments.kwarg)
    for arg in all_args:
        if arg.arg == "other":
            return arg
    return None


def _other_annotation(function: ast.FunctionDef) -> ast.AST | None:
    parameter = _other_parameter(function)
    return None if parameter is None else parameter.annotation


def is_forward_binary_dunder(function: ast.FunctionDef) -> bool:
    """Return whether ``function`` is an in-scope forward binary dunder."""
    return function.name in FORWARD_BINARY_DUNDERS


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
                print(
                    f"ERROR: alias {name} references {forbidden} — "
                    "violates the operand hierarchy.",
                    file=sys.stderr,
                )
                ok = False
    return ok


def check_forward_binary_dunders(
    class_node: ast.ClassDef | None,
    class_name: str,
    forbidden_names: tuple[str, ...],
    aliases: Mapping[str, ast.AST],
    exceptions: Mapping[ExceptionKey, HierarchyException],
) -> bool:
    """Check every direct forward binary dunder for a higher-tier ``other`` operand."""
    if class_node is None:
        print(
            f"ERROR: could not find class {class_name!r} in stubs.",
            file=sys.stderr,
        )
        return False

    ok = True
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or not is_forward_binary_dunder(node):
            continue

        # A scanned dunder that renames its operand would otherwise slip through the
        # reference check below, so demand the ``other`` operand by name.
        annotation = _other_annotation(node)
        if annotation is None:
            print(
                f"ERROR: {class_name}.{node.name} declares no 'other' operand — "
                "violates the operand hierarchy.",
                file=sys.stderr,
            )
            ok = False
            continue

        for forbidden in forbidden_names:
            if not references_name(annotation, forbidden, aliases):
                continue
            if (class_name, node.name, forbidden) in exceptions:
                continue
            print(
                f"ERROR: {class_name}.{node.name} `other` operand references "
                f"{forbidden} — violates the operand hierarchy.",
                file=sys.stderr,
            )
            ok = False
    return ok


def _read_required_trees(stub_root: Path) -> dict[Path, ast.Module] | None:
    required_files = (
        Path("core/base.pyi"),
        Path("core/indexes/base.pyi"),
        Path("core/indexes/multi.pyi"),
        Path("core/series.pyi"),
    )
    paths = [stub_root / path for path in required_files]
    for path in paths:
        if not path.exists():
            print(f"ERROR: missing required stub file {path}.", file=sys.stderr)
            return None
    return {
        path.relative_to(stub_root): ast.parse(path.read_text(encoding="utf-8"))
        for path in paths
    }


def check_operand_hierarchy(
    stub_root: Path,
    *,
    exceptions: Mapping[ExceptionKey, HierarchyException] = FORWARD_DUNDER_EXCEPTIONS,
) -> bool:
    """Check the operand-hierarchy constraints in the ``pandas-stubs`` root directory."""
    trees = _read_required_trees(stub_root)
    if trees is None:
        return False

    aliases = collect_aliases(stub_root)
    ok = check_alias_level(aliases)
    if not check_forward_binary_dunders(
        find_class(trees[Path("core/indexes/base.pyi")], "Index"),
        "Index",
        ("Series", "DataFrame"),
        aliases,
        exceptions,
    ):
        ok = False
    if not check_forward_binary_dunders(
        find_class(trees[Path("core/indexes/multi.pyi")], "MultiIndex"),
        "MultiIndex",
        ("Series", "DataFrame"),
        aliases,
        exceptions,
    ):
        ok = False
    if not check_forward_binary_dunders(
        find_class(trees[Path("core/series.pyi")], "Series"),
        "Series",
        ("DataFrame",),
        aliases,
        exceptions,
    ):
        ok = False

    if ok:
        print("Operand hierarchy invariant holds.")
    return ok


if __name__ == "__main__":
    STUB_ROOT = Path(__file__).parents[1] / "pandas-stubs"
    sys.exit(not check_operand_hierarchy(STUB_ROOT))
