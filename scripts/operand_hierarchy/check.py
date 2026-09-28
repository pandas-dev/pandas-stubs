# ruff: noqa: T201
"""Check the operand-hierarchy constraints in a pandas-stubs tree.

The checker reads the stubs as syntax trees. Every class it scans is given a tier from
the operand hierarchy:

======  ===================================================================
Tier    Scanned operands
======  ===================================================================
0       Scalars: ``Timedelta``, ``Timestamp``, ``Period``, ``Interval``, ``NAType``
1       Array-likes: ``ExtensionArray``
2       ``Index``, ``MultiIndex``, and every ``Index`` subclass
3       ``Series``
4       ``DataFrame``
======  ===================================================================

It verifies that:

* ``ScalarArrayIndex*`` aliases that are not ``ScalarArrayIndexSeries*`` aliases do not
  reference ``Series`` or ``DataFrame``;
* ``ScalarArrayIndexSeries*`` aliases do not reference ``DataFrame``; and
* a forward binary dunder declared directly on a scanned class does not name an operand
  of a strictly higher tier in its ``other`` annotation, unless an explicit exception
  permits it. Any spelling the tier model knows counts, so ``TimedeltaIndex`` is a
  tier-2 operand just as ``Index`` is, and the exception list is keyed by the
  tier's operand name rather than by the spelling.

The checks include direct and transitive references through ``TypeAlias`` definitions;
every definition of an alias name is considered, and a qualified name counts like a bare
one, whether it names the target directly (``pd.DataFrame``) or names an alias of it
(``types.Higher`` where ``Higher: TypeAlias = DataFrame``). Reflected dunders are
deliberately outside this structural check.

Run as a script, the checker also requires every exception key to be exercised by the
tree it scans.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import (  # noqa: TC003
    Iterator,
    Mapping,
    Set as AbstractSet,
)
from pathlib import Path
import sys

from .exceptions import (
    EXCEPTIONS_FILE,
    FORWARD_DUNDER_EXCEPTIONS,
    TEMPORARY_EXCEPTION_NOTE,
    ExceptionKey,
)
from .model import (
    CANONICAL_OPERAND_BY_TIER,
    CLASS_STUB_FILES,
    FORWARD_BINARY_DUNDERS,
    REQUIRED_STUB_FILES,
    TIER_OPERANDS,
)


class StubTree:
    """One parsed stub tree: what the checks query, and the parse products behind it.

    ``load`` is the only caller of the parsing helpers at the foot of this module, so it is
    also the only place the missing-file error can come from. A plain class, not a
    dataclass: the state is containers, so ``frozen`` would only make the immutability
    cosmetic while the generated ``__hash__`` raised, and neither equality nor hashing is a
    question anyone asks of a parsed tree.
    """

    def __init__(
        self,
        root: Path,
        aliases: Mapping[str, tuple[ast.AST, ...]],
        bases: Mapping[str, tuple[str, ...]],
        tiers: Mapping[str, int],
        classes: Mapping[str, tuple[ast.ClassDef, ...]],
    ) -> None:
        self.root = root
        self.aliases = aliases
        self.bases = bases
        self.tiers = tiers
        self.classes = classes

    @classmethod
    def load(cls, root: Path) -> StubTree | None:
        """Parse ``root``, or return ``None`` after reporting a missing required file."""
        trees = _read_required_trees(root)
        if trees is None:
            return None
        bases = _collect_class_bases(root)
        return cls(
            root=root,
            aliases=_collect_aliases(root),
            bases=bases,
            tiers=_class_tiers(bases),
            classes=_class_index(trees),
        )

    def references(
        self, node: ast.AST | tuple[ast.AST, ...] | None, target: str
    ) -> bool:
        """Return whether ``node`` references ``target`` directly or through aliases.

        ``node`` may be one expression or every definition of an alias name. Both node
        kinds are reduced to a terminal name before the lookup, so ``pd.DataFrame``
        references ``DataFrame``, and ``types.Higher`` references whatever ``Higher``
        stands for. A quoted annotation reads like the spelling it quotes, so
        ``other: "DataFrame"`` references ``DataFrame``. Alias resolution is by name, not
        by module scope, like the rest of the checker.
        """
        if node is None:
            return False

        to_expand = list(node) if isinstance(node, tuple) else [node]
        expanded: set[str] = set()
        while to_expand:
            current = to_expand.pop()
            for child in _reference_walk(current):
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

    def higher_tier_operands(self, tier: int) -> tuple[str, ...]:
        """Return every spelling whose tier is strictly higher than ``tier``.

        The spellings come from this tree's tiers rather than from ``TIER_OPERANDS``,
        because an ``Index`` subclass that ``_class_tiers`` discovered is a tier-2 operand
        whether or not its own name is an operand name.
        """
        return tuple(
            sorted(
                name for name, operand_tier in self.tiers.items() if operand_tier > tier
            )
        )

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
                        f"ERROR: alias {name} references {forbidden} — "
                        "violates the operand hierarchy.",
                        file=sys.stderr,
                    )
                    ok = False
        return ok

    def check_class(
        self, class_name: str, exceptions: AbstractSet[ExceptionKey]
    ) -> tuple[bool, set[ExceptionKey], set[tuple[str, str]]]:
        """Check every direct forward binary dunder for a higher-tier ``other`` operand.

        Also returns the exception key of every exception that permitted a site and the
        ``(class, dunder)`` slot each one permitted, so the caller can tell which keys the
        tree still justifies and how many distinct slots they cover.
        """
        class_nodes = self.classes.get(class_name)
        if class_nodes is None:
            print(
                f"ERROR: could not find class {class_name!r} in the required stub files; "
                "add the file that declares it to REQUIRED_STUB_FILES.",
                file=sys.stderr,
            )
            return False, set(), set()

        tier = self.tiers[class_name]
        higher_tiers = self.higher_tier_operands(tier)
        consulted: set[ExceptionKey] = set()
        sites: set[tuple[str, str]] = set()
        ok = True
        for class_node in class_nodes:
            for node in class_node.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                method_ok, method_consulted, method_sites = self._check_dunder(
                    node, class_name, tier, higher_tiers, exceptions
                )
                consulted |= method_consulted
                sites |= method_sites
                if not method_ok:
                    ok = False
        return ok, consulted, sites

    def _check_dunder(
        self,
        node: ast.FunctionDef,
        class_name: str,
        tier: int,
        higher_tiers: tuple[str, ...],
        exceptions: AbstractSet[ExceptionKey],
    ) -> tuple[bool, set[ExceptionKey], set[tuple[str, str]]]:
        """Check one declared method, with the same contract as ``check_class``.

        A method outside the scanned dunder set is permitted by definition and consults
        nothing, so the caller needs no branch of its own.
        """
        if not _is_forward_binary_dunder(node):
            return True, set(), set()

        # A scanned dunder that renames its operand would otherwise slip through the
        # reference check below, so demand the ``other`` operand by name.
        annotation = _other_annotation(node)
        if annotation is None:
            print(
                f"ERROR: {class_name}.{node.name} declares no 'other' operand — "
                "violates the operand hierarchy.",
                file=sys.stderr,
            )
            return False, set(), set()

        consulted: set[ExceptionKey] = set()
        sites: set[tuple[str, str]] = set()
        ok = True
        for spelling in higher_tiers:
            if not self.references(annotation, spelling):
                continue
            # The exception list is keyed by the tier's operand name, so a subclass
            # spelling is looked up as the operand name it stands for.
            forbidden = CANONICAL_OPERAND_BY_TIER[self.tiers[spelling]]
            key = _permitting_key(class_name, node.name, forbidden, exceptions)
            if key is not None:
                consulted.add(key)
                sites.add((class_name, node.name))
                continue
            described = (
                spelling
                if spelling == forbidden
                else f"{spelling} (the tier-{self.tiers[spelling]} operand {forbidden})"
            )
            print(
                f"ERROR: {class_name}.{node.name} `other` operand references "
                f"{described}, a higher tier than {class_name} (tier {tier}) — "
                "violates the operand hierarchy.",
                file=sys.stderr,
            )
            ok = False
        return ok, consulted, sites


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
    tree = StubTree.load(stub_root)
    if tree is None:
        return False

    ok = tree.check_aliases()

    consulted: set[ExceptionKey] = set()
    sites: set[tuple[str, str]] = set()
    for class_name in sorted(tree.tiers):
        class_ok, class_consulted, class_sites = tree.check_class(
            class_name, exceptions
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
                f"tree — remove it from {EXCEPTIONS_FILE}, or add "
                "the stub overload that justifies it.",
                file=sys.stderr,
            )
            ok = False

    if not ok:
        return False
    print("Operand hierarchy invariant holds.")
    if exceptions:
        print(
            f"{EXCEPTIONS_FILE} -- {len(consulted)} of {len(exceptions)} applied at "
            f"{len(sites)} sites. {TEMPORARY_EXCEPTION_NOTE}."
        )
    return True


def main() -> int:
    """Run the checker over this repository's ``pandas-stubs`` tree."""
    stub_root = Path(__file__).resolve().parents[2] / "pandas-stubs"
    ok = check_operand_hierarchy(stub_root, require_all_exercised=True)
    return 0 if ok else 1


def _type_aliases(tree: ast.AST) -> dict[str, tuple[ast.AST, ...]]:
    """Return ``TypeAlias`` right-hand sides from ``tree``, keyed by alias name.

    Every definition of a name is kept: a redefinition inside a version branch must not
    hide the definition it shadows. The annotation is read by its terminal name, so a
    qualified ``typing.TypeAlias`` is collected like the bare spelling.
    """
    aliases: dict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _terminal_name(node.annotation) == "TypeAlias"
            and node.value is not None
        ):
            aliases[node.target.id].append(node.value)
    return {name: tuple(values) for name, values in aliases.items()}


def _collect_aliases(stub_root: Path) -> dict[str, tuple[ast.AST, ...]]:
    """Collect every alias definition from every stub file below ``stub_root``.

    Every definition of a name is kept rather than letting the last parsed module win, so
    a name that collides across modules cannot hide a violation behind the collision.
    """
    aliases: dict[str, list[ast.AST]] = defaultdict(list)
    for path in sorted(stub_root.rglob("*.pyi")):
        parsed = _parse(path)
        for name, values in _type_aliases(parsed).items():
            aliases[name].extend(values)
    return {name: tuple(values) for name, values in aliases.items()}


def _parse(path: Path) -> ast.Module:
    """Parse one stub file, naming it in any ``SyntaxError`` the parse raises.

    ``ast.parse`` leaves ``SyntaxError.filename`` unset otherwise, so a syntax error
    anywhere below the stub root would report a file the caller cannot identify.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _terminal_name(node: ast.AST) -> str | None:
    """Return the terminal name of a possibly subscripted, possibly qualified node."""
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _quoted_expression(node: ast.AST) -> ast.AST | None:
    """Return the expression a quoted annotation spells, or ``None`` for any other node.

    A stub may quote an annotation — ``other: "DataFrame"``, a legal forward reference —
    which parses as a string constant rather than as the expression it spells. Reading the
    constant alone would leave the reference scan blind to the operand actually named.
    """
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return None
    try:
        return ast.parse(node.value, mode="eval").body
    except SyntaxError:
        return None


def _reference_walk(node: ast.AST) -> Iterator[ast.AST]:
    """Walk ``node`` as ``ast.walk`` does, descending into any quoted annotation."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        quoted = _quoted_expression(current)
        if quoted is not None:
            stack.append(quoted)
        stack.extend(ast.iter_child_nodes(current))


def _collect_class_bases(stub_root: Path) -> dict[str, tuple[str, ...]]:
    """Map every class name below ``stub_root`` to its terminal base-class names."""
    bases: dict[str, list[str]] = defaultdict(list)
    for path in sorted(stub_root.rglob("*.pyi")):
        parsed = _parse(path)
        for node in ast.walk(parsed):
            if not isinstance(node, ast.ClassDef):
                continue
            names = bases[node.name]
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


def _class_tiers(bases: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Assign a tier to every scanned class.

    ``TIER_OPERANDS`` is spread over ``CLASS_STUB_FILES`` last, so it, and not the tier-0
    seed, decides the tier of a name both hold.

    ``bases`` is needed only because ``Index`` has subclasses: the registry names
    ``Index``, so a spelling such as ``TimedeltaIndex`` is a tier-2 operand only because
    this walk resolves its base by name. It runs for ``Index`` alone, so an array-like
    such as ``Categorical`` is deliberately not registered and is not read at tier 1; a
    second walk is what registering an array-like subclass would take.
    """
    tiers: dict[str, int] = {**dict.fromkeys(CLASS_STUB_FILES, 0), **TIER_OPERANDS}
    index_tier = TIER_OPERANDS["Index"]
    tiers.update(
        {
            name: index_tier
            for name in bases
            if name not in tiers and _descends_from(name, "Index", bases)
        }
    )
    return tiers


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


def _other_annotation(function: ast.FunctionDef) -> ast.AST | None:
    """Return the annotation of that argument, or ``None`` if it has none."""
    parameter = _other_parameter(function)
    return None if parameter is None else parameter.annotation


def _is_forward_binary_dunder(function: ast.FunctionDef) -> bool:
    """Return whether ``function`` is an in-scope forward binary dunder."""
    return function.name in FORWARD_BINARY_DUNDERS


def _permitting_key(
    class_name: str,
    dunder: str,
    forbidden: str,
    exceptions: AbstractSet[ExceptionKey],
) -> ExceptionKey | None:
    """Return the key of the exception permitting ``forbidden`` here, if any.

    The exact ``(class, dunder, forbidden)`` key wins; the ``(class, "*", forbidden)``
    key covers every forward binary dunder of the class.
    """
    exact = (class_name, dunder, forbidden)
    if exact in exceptions:
        return exact
    wildcard = (class_name, "*", forbidden)
    return wildcard if wildcard in exceptions else None


def _read_required_trees(stub_root: Path) -> dict[Path, ast.Module] | None:
    """Parse every required stub file, or report the first missing one and give up."""
    paths = [stub_root / path for path in REQUIRED_STUB_FILES]
    for path in paths:
        if not path.exists():
            print(f"ERROR: missing required stub file {path}.", file=sys.stderr)
            return None
    return {path.relative_to(stub_root): _parse(path) for path in paths}


def _class_index(
    trees: Mapping[Path, ast.Module],
) -> dict[str, tuple[ast.ClassDef, ...]]:
    """Map every class name in the required trees to every definition of it.

    Every definition is kept, like the alias collector above: these stubs do declare
    duplicate class names, and letting the last one parsed win would hide a violation
    declared in the definition it shadowed.
    """
    classes: dict[str, list[ast.ClassDef]] = defaultdict(list)
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name].append(node)
    return {name: tuple(nodes) for name, nodes in classes.items()}
