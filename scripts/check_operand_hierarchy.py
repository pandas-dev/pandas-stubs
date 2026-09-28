#!/usr/bin/env python3
# ruff: noqa: T201
"""Check the operand-hierarchy constraints in a pandas-stubs tree.

The checker reads the stubs as syntax trees. Every class it scans is given a tier from
the operand hierarchy:

======  ===================================================================
Tier    Scanned operands
======  ===================================================================
0       Scalars: ``Timedelta``, ``Timestamp``, ``Period``, ``Interval``, ``NAType``
2       ``Index``, ``MultiIndex``, and every ``Index`` subclass
3       ``Series``
4       ``DataFrame``
======  ===================================================================

Tier 1 holds array-like values such as ``Categorical``; it owns no scanned class, so the
checker cannot constrain it structurally.

It verifies that:

* ``ScalarArrayIndex*`` aliases do not reference ``Series`` or ``DataFrame``;
* ``ScalarArrayIndexSeries*`` aliases do not reference ``DataFrame``; and
* a forward binary dunder declared directly on a scanned class does not name an operand
  of a strictly higher tier in its ``other`` annotation, unless an explicit exception
  permits it. Any spelling the tier model knows counts, so ``TimedeltaIndex`` is a
  tier-2 operand just as ``Index`` is, and the exception list is keyed by the
  tier's operand name rather than by the spelling.

The checks include direct and transitive references through ``TypeAlias`` definitions;
every definition of an alias name is considered, and a qualified terminal name such as
``pd.DataFrame`` counts as a reference. Reflected dunders and tier-1 array-likes are
deliberately outside this structural check.

Run as a script, the checker also requires every exception key to be exercised by the
tree it scans.
"""

from __future__ import annotations

import ast
from collections.abc import (  # noqa: TC003
    Mapping,
    Set as AbstractSet,
)
from pathlib import Path
import sys
from typing import Final

# The checker runs both as a package module (imported by the tests) and as a script:
# `.github/workflows/test.yml` runs `python scripts/check_operand_hierarchy.py`. Script
# mode fails the relative import because ``__package__`` is empty, so the bare branch is
# the one a script takes; the bare name resolves through ``sys.path[0]``, which is
# ``scripts/``. Going through ``scripts.operand_hierarchy_exceptions`` instead would
# execute ``scripts/__init__.py`` and drag in loguru, which the stdlib-only script path
# must not need.
#
# mypy reports the bare branch of this idiom as `no-redef`. `if __package__:` is reported
# the same way, and binding the module rather than its names does not help either. Both
# branches import the same names by design, so the redefinition is the intent.
try:
    from .operand_hierarchy_exceptions import (
        FORWARD_DUNDER_EXCEPTIONS,
        TEMPORARY_EXCEPTION_NOTE,
        ExceptionKey,
    )
except ImportError:
    from operand_hierarchy_exceptions import (  # type: ignore[no-redef]
        FORWARD_DUNDER_EXCEPTIONS,
        TEMPORARY_EXCEPTION_NOTE,
        ExceptionKey,
    )

# The file the pass summary and the dead-key message name, spelled as a reader would.
_EXCEPTIONS_FILE: Final[str] = "scripts/operand_hierarchy_exceptions.py"

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

# Operand names that can appear in ``other``, mapped to their tier.
TIER_OPERANDS: Final[dict[str, int]] = {
    "Index": 2,
    "MultiIndex": 2,
    "Series": 3,
    "DataFrame": 4,
}

# The operand name a tier is registered under, so a site that spells its operand
# ``TimedeltaIndex`` or ``MultiIndex`` is looked up as ``Index``. Keyed by tier, not by
# spelling, so a spelling cannot drift out of the exception list.
CANONICAL_OPERAND_BY_TIER: Final[dict[int, str]] = {
    TIER_OPERANDS["Index"]: "Index",
    TIER_OPERANDS["Series"]: "Series",
    TIER_OPERANDS["DataFrame"]: "DataFrame",
}

# Tier 0 is the scalars. Each is named together with the stub file that declares it, so
# the class scan can require that file rather than silently skipping the class.
TIER_0_CLASSES: Final[dict[str, Path]] = {
    "Timedelta": Path("_libs/tslibs/timedeltas.pyi"),
    "Timestamp": Path("_libs/tslibs/timestamps.pyi"),
    "Period": Path("_libs/tslibs/period.pyi"),
    "Interval": Path("_libs/interval.pyi"),
    "NAType": Path("_libs/missing.pyi"),
}

# Every file that declares a scanned class. The list is explicit so that moving a class
# to a new file fails the scan loudly instead of silently dropping it.
REQUIRED_STUB_FILES: Final[tuple[Path, ...]] = tuple(
    sorted(
        {
            Path("core/base.pyi"),
            Path("core/frame.pyi"),
            Path("core/indexes/base.pyi"),
            Path("core/indexes/category.pyi"),
            Path("core/indexes/datetimelike.pyi"),
            Path("core/indexes/datetimes.pyi"),
            Path("core/indexes/extension.pyi"),
            Path("core/indexes/interval.pyi"),
            Path("core/indexes/multi.pyi"),
            Path("core/indexes/period.pyi"),
            Path("core/indexes/range.pyi"),
            Path("core/indexes/timedeltas.pyi"),
            Path("core/series.pyi"),
            Path("_stubs_only/__init__.pyi"),
            *TIER_0_CLASSES.values(),
        }
    )
)


def _type_aliases(tree: ast.AST) -> dict[str, tuple[ast.AST, ...]]:
    """Return ``TypeAlias`` right-hand sides from ``tree``, keyed by alias name.

    Every definition of a name is kept: a redefinition inside a version branch must not
    hide the definition it shadows.
    """
    aliases: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Name)
            and node.annotation.id == "TypeAlias"
            and node.value is not None
        ):
            aliases.setdefault(node.target.id, []).append(node.value)
    return {name: tuple(values) for name, values in aliases.items()}


def collect_aliases(stub_root: Path) -> dict[str, tuple[ast.AST, ...]]:
    """Collect every alias definition from every stub file below ``stub_root``.

    Every definition of a name is kept rather than letting the last parsed module win, so
    a name that collides across modules cannot hide a violation behind the collision.
    """
    aliases: dict[str, list[ast.AST]] = {}
    for path in sorted(stub_root.rglob("*.pyi")):
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        for name, values in _type_aliases(parsed).items():
            aliases.setdefault(name, []).extend(values)
    return {name: tuple(values) for name, values in aliases.items()}


def references_name(
    node: ast.AST | tuple[ast.AST, ...] | None,
    target: str,
    aliases: Mapping[str, tuple[ast.AST, ...]],
) -> bool:
    """Return whether ``node`` references ``target`` directly or through aliases.

    ``node`` may be one expression or every definition of an alias name. A qualified
    terminal name counts, so ``pd.DataFrame`` references ``DataFrame``.
    """
    if node is None:
        return False

    to_expand = list(node) if isinstance(node, tuple) else [node]
    expanded: set[str] = set()
    while to_expand:
        current = to_expand.pop()
        for child in ast.walk(current):
            if isinstance(child, ast.Attribute) and child.attr == target:
                return True
            if not isinstance(child, ast.Name):
                continue
            if child.id == target:
                return True
            alias = aliases.get(child.id)
            if alias is not None and child.id not in expanded:
                expanded.add(child.id)
                to_expand.extend(alias)
    return False


def _terminal_name(node: ast.AST) -> str | None:
    """Return the terminal name of a possibly subscripted, possibly qualified node."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def collect_class_bases(stub_root: Path) -> dict[str, tuple[str, ...]]:
    """Map every class name below ``stub_root`` to its terminal base-class names."""
    bases: dict[str, list[str]] = {}
    for path in sorted(stub_root.rglob("*.pyi")):
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(parsed):
            if not isinstance(node, ast.ClassDef):
                continue
            names = bases.setdefault(node.name, [])
            for base in node.bases:
                terminal = _terminal_name(base)
                if terminal is not None:
                    names.append(terminal)
    return {name: tuple(names) for name, names in bases.items()}


def _descends_from(
    name: str, target: str, bases: Mapping[str, tuple[str, ...]]
) -> bool:
    """Return whether ``name`` is ``target`` or inherits from it, by name."""
    seen: set[str] = set()
    to_visit = [name]
    while to_visit:
        current = to_visit.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == target:
            return True
        to_visit.extend(bases.get(current, ()))
    return False


def class_tiers(bases: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Assign a tier to every scanned class.

    Explicit operand names win, then the tier-0 scalars, then every ``Index`` subclass
    inherits tier 2. Base classes resolve by name, matching the rest of the checker.
    """
    tiers: dict[str, int] = {**TIER_OPERANDS, **dict.fromkeys(TIER_0_CLASSES, 0)}
    index_tier = TIER_OPERANDS["Index"]
    for name in bases:
        if name not in tiers and _descends_from(name, "Index", bases):
            tiers[name] = index_tier
    return tiers


def _higher_tier_operands(tier: int, tiers: Mapping[str, int]) -> tuple[str, ...]:
    """Return every spelling whose tier is strictly higher than ``tier``.

    The spellings come from ``tiers`` rather than from ``TIER_OPERANDS``, because an
    ``Index`` subclass that ``class_tiers`` discovered is a tier-2 operand whether or
    not its own name is an operand name.
    """
    return tuple(
        sorted(name for name, operand_tier in tiers.items() if operand_tier > tier)
    )


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


def check_alias_level(aliases: Mapping[str, tuple[ast.AST, ...]]) -> bool:
    """Check hierarchy-bearing aliases for direct or transitive higher tiers."""
    ok = True
    for name, definitions in sorted(aliases.items()):
        forbidden_names: tuple[str, ...]
        if name.startswith("ScalarArrayIndexSeries"):
            forbidden_names = ("DataFrame",)
        elif name.startswith("ScalarArrayIndex"):
            forbidden_names = ("Series", "DataFrame")
        else:
            continue

        for forbidden in forbidden_names:
            if references_name(definitions, forbidden, aliases):
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
    tiers: Mapping[str, int],
    aliases: Mapping[str, tuple[ast.AST, ...]],
    exceptions: AbstractSet[ExceptionKey],
) -> tuple[bool, set[ExceptionKey], set[tuple[str, str]]]:
    """Check every direct forward binary dunder for a higher-tier ``other`` operand.

    Also returns the exception key of every exception that permitted a site and the
    ``(class, dunder)`` slot each one permitted, so the caller can tell which keys the
    tree still justifies and how many distinct slots they cover.
    """
    if class_node is None:
        print(
            f"ERROR: could not find class {class_name!r} in the required stub files; "
            "add the file that declares it to REQUIRED_STUB_FILES.",
            file=sys.stderr,
        )
        return False, set(), set()

    tier = tiers[class_name]
    higher_tiers = _higher_tier_operands(tier, tiers)
    consulted: set[ExceptionKey] = set()
    sites: set[tuple[str, str]] = set()
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

        for spelling in higher_tiers:
            if not references_name(annotation, spelling, aliases):
                continue
            # The exception list is keyed by the tier's operand name, so a subclass
            # spelling is looked up as the operand name it stands for.
            forbidden = CANONICAL_OPERAND_BY_TIER[tiers[spelling]]
            if (class_name, node.name, forbidden) in exceptions:
                consulted.add((class_name, node.name, forbidden))
                sites.add((class_name, node.name))
                continue
            if (class_name, "*", forbidden) in exceptions:
                consulted.add((class_name, "*", forbidden))
                sites.add((class_name, node.name))
                continue
            described = (
                spelling
                if spelling == forbidden
                else f"{spelling} (the tier-{tiers[spelling]} operand {forbidden})"
            )
            print(
                f"ERROR: {class_name}.{node.name} `other` operand references "
                f"{described}, a higher tier than {class_name} (tier {tier}) — "
                "violates the operand hierarchy.",
                file=sys.stderr,
            )
            ok = False
    return ok, consulted, sites


def _read_required_trees(stub_root: Path) -> dict[Path, ast.Module] | None:
    paths = [stub_root / path for path in REQUIRED_STUB_FILES]
    for path in paths:
        if not path.exists():
            print(f"ERROR: missing required stub file {path}.", file=sys.stderr)
            return None
    return {
        path.relative_to(stub_root): ast.parse(path.read_text(encoding="utf-8"))
        for path in paths
    }


def _class_index(trees: Mapping[Path, ast.Module]) -> dict[str, ast.ClassDef]:
    """Map every class name in the required trees to its class definition."""
    index: dict[str, ast.ClassDef] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                index[node.name] = node
    return index


def check_operand_hierarchy(
    stub_root: Path,
    *,
    exceptions: AbstractSet[ExceptionKey] = FORWARD_DUNDER_EXCEPTIONS,
    require_all_exercised: bool = False,
) -> bool:
    """Check the operand-hierarchy constraints in the ``pandas-stubs`` root directory.

    ``require_all_exercised`` also fails when the scanned tree never needs one of the
    ``exceptions``, so an exception key cannot outlive the overload that justified it.
    It is off by default because a synthetic fixture legitimately exercises almost none
    of the global exception list.
    """
    trees = _read_required_trees(stub_root)
    if trees is None:
        return False

    aliases = collect_aliases(stub_root)
    ok = check_alias_level(aliases)

    tiers = class_tiers(collect_class_bases(stub_root))
    classes = _class_index(trees)
    consulted: set[ExceptionKey] = set()
    sites: set[tuple[str, str]] = set()
    for class_name in sorted(tiers):
        class_ok, class_consulted, class_sites = check_forward_binary_dunders(
            classes.get(class_name),
            class_name,
            tiers,
            aliases,
            exceptions,
        )
        consulted |= class_consulted
        sites |= class_sites
        if not class_ok:
            ok = False

    if require_all_exercised:
        for key in sorted(exceptions):
            if key in consulted:
                continue
            print(
                f"ERROR: temporary exception {key} is never exercised by the scanned "
                "tree — remove it from scripts/operand_hierarchy_exceptions.py, or add "
                "the stub overload that justifies it.",
                file=sys.stderr,
            )
            ok = False

    if not ok:
        return False
    print("Operand hierarchy invariant holds.")
    if exceptions:
        print(
            f"{_EXCEPTIONS_FILE} -- {len(consulted)} of {len(exceptions)} applied at "
            f"{len(sites)} sites. {TEMPORARY_EXCEPTION_NOTE}."
        )
    return True


if __name__ == "__main__":
    STUB_ROOT = Path(__file__).parents[1] / "pandas-stubs"
    sys.exit(not check_operand_hierarchy(STUB_ROOT, require_all_exercised=True))
