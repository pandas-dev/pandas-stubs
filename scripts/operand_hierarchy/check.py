# ruff: noqa: T201
"""Validate signature structure against docs/type-architecture/operand-hierarchy.md."""

from __future__ import annotations

import ast
from collections import (
    Counter,
    defaultdict,
)
from collections.abc import (  # noqa: TC003
    Iterator,
    Mapping,
)
from dataclasses import (
    dataclass,
)
from pathlib import Path
import sys

from .exceptions import (
    EXCEPTIONS_FILE,
    FORWARD_DUNDER_EXCEPTIONS,
    ExceptionKey,
)
from .model import (
    CANONICAL_OPERAND_BY_TIER,
    CLASS_TIERS,
    FORWARD_BINARY_DUNDERS,
    REQUIRED_STUB_FILES,
)


@dataclass(frozen=True)
class StubTree:
    """Aliases and bases are discovered globally; checked classes require registered files."""

    aliases: Mapping[str, tuple[ast.AST, ...]]
    tiers: Mapping[str, int]
    classes: Mapping[str, tuple[ast.ClassDef, ...]]

    @classmethod
    def load(cls, root: Path) -> StubTree | None:
        """Parse each stub once; report missing required files before scanning."""
        for relative in REQUIRED_STUB_FILES:
            if not (root / relative).is_file():
                print(
                    f"ERROR: missing required stub file {root / relative}.",
                    file=sys.stderr,
                )
                return None

        aliases: dict[str, list[ast.AST]] = defaultdict(list)
        bases: dict[str, list[str]] = defaultdict(list)
        classes: dict[str, list[ast.ClassDef]] = defaultdict(list)
        for path in sorted(root.rglob("*.pyi")):
            parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            _collect_types(parsed, aliases, bases)
            if path.relative_to(root) in REQUIRED_STUB_FILES:
                for node in _conditional_definitions(parsed.body):
                    if isinstance(node, ast.ClassDef):
                        classes[node.name].append(node)

        tiers = dict(CLASS_TIERS)
        tiers.update(
            (name, CLASS_TIERS["Index"])
            for name in bases
            if name not in tiers and _descends_from(name, "Index", bases)
        )
        return cls(
            {name: tuple(values) for name, values in aliases.items()},
            tiers,
            {name: tuple(nodes) for name, nodes in classes.items()},
        )

    def references(
        self, node: ast.AST | tuple[ast.AST, ...] | None, target: str
    ) -> bool:
        """Return whether ``node`` references ``target`` directly or through aliases.

        ``node`` may be one expression or every definition of an alias name. Both node
        kinds are reduced to a terminal name before the lookup, so ``pd.DataFrame``
        references ``DataFrame``, and ``types.Higher`` references whatever ``Higher``
        stands for. Alias resolution is by name, not by module scope, like the rest of the
        checker.
        """
        if node is None:
            return False

        to_expand = list(node) if isinstance(node, tuple) else [node]
        expanded: set[str] = set()
        while to_expand:
            current = to_expand.pop()
            for child in ast.walk(current):
                terminal = _terminal_name(child)
                if terminal is None:
                    continue
                if terminal == target:
                    return True
                alias = self.aliases.get(terminal)
                if alias is not None and terminal not in expanded:
                    expanded.add(terminal)
                    to_expand.extend(alias)
        return False

    def check_aliases(self) -> bool:
        """Check hierarchy-bearing aliases for direct or transitive higher tiers."""
        ok = True
        for name, definitions in sorted(self.aliases.items()):
            forbidden_names: tuple[str, ...]
            if name.startswith("ScalarArrayIndexSeries"):
                forbidden_names = ("DataFrame",)
            elif name.startswith("ScalarArrayIndex"):
                forbidden_names = ("Series", "DataFrame")
            else:
                continue

            for forbidden in forbidden_names:
                if self.references(definitions, forbidden):
                    print(
                        f"ERROR: alias {name} references {forbidden} — violates the operand hierarchy.",
                        file=sys.stderr,
                    )
                    ok = False
        return ok

    def check_class(self, class_name: str, observed: Counter[ExceptionKey]) -> bool:
        """Collect one occurrence per declaration and distinct canonical higher tier."""
        class_nodes = self.classes.get(class_name)
        if not class_nodes:
            print(
                f"ERROR: could not find class {class_name!r} in the required stub files; add the file that declares it to REQUIRED_STUB_FILES.",
                file=sys.stderr,
            )
            return False

        higher = {
            name: CANONICAL_OPERAND_BY_TIER[tier]
            for name, tier in self.tiers.items()
            if tier > self.tiers[class_name]
        }
        ok = True
        for class_node in class_nodes:
            for node in _conditional_definitions(class_node.body):
                if (
                    not isinstance(node, ast.FunctionDef)
                    or node.name not in FORWARD_BINARY_DUNDERS
                ):
                    continue
                parameter = _other_parameter(node)
                if parameter is None or parameter.annotation is None:
                    print(
                        f"ERROR: {class_name}.{node.name} requires an annotated 'other' operand.",
                        file=sys.stderr,
                    )
                    ok = False
                    continue
                operands = {
                    canonical
                    for spelling, canonical in higher.items()
                    if self.references(parameter.annotation, spelling)
                }
                observed.update(
                    (class_name, node.name, operand) for operand in operands
                )
        return ok


def check_operand_hierarchy(
    stub_root: Path,
    *,
    exceptions: Mapping[ExceptionKey, int] = FORWARD_DUNDER_EXCEPTIONS,
) -> bool:
    """Require exact agreement between observed higher-tier operands and allowances."""
    tree = StubTree.load(stub_root)
    if tree is None:
        return False
    ok = tree.check_aliases()
    observed: Counter[ExceptionKey] = Counter()
    for class_name in sorted(tree.tiers):
        if not tree.check_class(class_name, observed):
            ok = False
    for key in sorted(exceptions.keys() | observed.keys()):
        expected = exceptions.get(key, 0)
        actual = observed[key]
        if key in exceptions and expected <= 0:
            print(
                f"ERROR: allowance {key} must be a positive integer; got {expected!r}.",
                file=sys.stderr,
            )
            ok = False
        if expected != actual:
            print(
                f"ERROR: {key[0]}.{key[1]} higher-tier operand {key[2]}: expected {expected} overload(s), observed {actual}; review the signature or update {EXCEPTIONS_FILE}.",
                file=sys.stderr,
            )
            ok = False
    if ok:
        print("Operand hierarchy invariant holds.")
        print(
            f"{EXCEPTIONS_FILE}: {len(observed)} combinations, {sum(observed.values())} occurrences."
        )
    return ok


def main() -> int:
    """Run the checker over this repository's stubs."""
    stub_root = Path(__file__).resolve().parents[2] / "pandas-stubs"
    return 0 if check_operand_hierarchy(stub_root) else 1


def _collect_types(
    tree: ast.Module,
    aliases: dict[str, list[ast.AST]],
    bases: dict[str, list[str]],
) -> None:
    """Retain every alias definition and base spelling, including duplicate names."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _terminal_name(node.annotation) == "TypeAlias"
            and node.value is not None
        ):
            aliases[node.target.id].append(node.value)
        elif isinstance(node, ast.ClassDef):
            names = bases[node.name]
            for base in node.bases:
                terminal = _terminal_name(base)
                if terminal is not None:
                    names.append(terminal)


def _conditional_definitions(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Visit both conditional branches without entering functions or classes."""
    for node in body:
        if isinstance(node, ast.If):
            yield from _conditional_definitions(node.body)
            yield from _conditional_definitions(node.orelse)
        else:
            yield node


def _terminal_name(node: ast.AST) -> str | None:
    """Return the terminal name of a possibly subscripted, possibly qualified node."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _descends_from(name: str, target: str, bases: Mapping[str, list[str]]) -> bool:
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


def _other_parameter(function: ast.FunctionDef) -> ast.arg | None:
    """Return the argument named ``other``, positional, keyword or variadic."""
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
