from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import pytest

from scripts.operand_hierarchy.check import (
    StubTree,
    check_operand_hierarchy,
    main,
)
from scripts.operand_hierarchy.exceptions import (
    EXCEPTIONS_DOCUMENTATION,
    FORWARD_DUNDER_EXCEPTIONS,
    ExceptionKey,
)
from scripts.operand_hierarchy.model import (
    CANONICAL_OPERAND_BY_TIER,
    FORWARD_BINARY_DUNDERS,
    TIER_OPERANDS,
)

_REPO_ROOT = Path(__file__).parents[1]

# A class as the renderer needs it: its name, its base or ``None``, and its dunders.
_Class = tuple[str, str | None, tuple[tuple[str, str], ...]]

# The stub files the cases override, named once so a case's fixture fits on one line.
_INDEX = "core/indexes/base.pyi"
_INTERVAL_INDEX = "core/indexes/interval.pyi"
_MULTI = "core/indexes/multi.pyi"
_RANGE = "core/indexes/range.pyi"
_SERIES = "core/series.pyi"
_INTERVAL = "_libs/interval.pyi"
_NATYPE = "_libs/missing.pyi"
_TIMEDELTAS = "_libs/tslibs/timedeltas.pyi"
_TIMESTAMPS = "_libs/tslibs/timestamps.pyi"

# The smallest tree the checker requires, as data: every required stub file and the
# ``(class, base)`` pairs it declares. ``IndexSubclassBase`` resolves ``Index`` by name, so
# no test tree has to declare ``Index`` for the derived subclasses to register.
_DEFAULT_FILES: Mapping[str, tuple[tuple[str, str | None], ...]] = {
    "core/arrays/base.pyi": (("ExtensionArray", None),),
    "core/base.pyi": (),
    "core/frame.pyi": (("DataFrame", None),),
    "core/indexes/base.pyi": (("Index", None),),
    "core/indexes/category.pyi": (("CategoricalIndex", "ExtensionIndex"),),
    "core/indexes/datetimelike.pyi": (
        ("DatetimeIndexOpsMixin", "ExtensionIndex"),
        ("DatetimeTimedeltaMixin", "DatetimeIndexOpsMixin"),
    ),
    "core/indexes/datetimes.pyi": (("DatetimeIndex", "DatetimeTimedeltaMixin"),),
    "core/indexes/extension.pyi": (("ExtensionIndex", "IndexSubclassBase"),),
    "core/indexes/interval.pyi": (("IntervalIndex", "ExtensionIndex"),),
    "core/indexes/multi.pyi": (("MultiIndex", None),),
    "core/indexes/period.pyi": (("PeriodIndex", "DatetimeIndexOpsMixin"),),
    "core/indexes/range.pyi": (("RangeIndex", "IndexSubclassBase"),),
    "core/indexes/timedeltas.pyi": (("TimedeltaIndex", "DatetimeTimedeltaMixin"),),
    "core/series.pyi": (("Series", None),),
    "_libs/interval.pyi": (("Interval", None),),
    "_libs/missing.pyi": (("NAType", None),),
    "_libs/tslibs/period.pyi": (("Period", None),),
    "_libs/tslibs/timedeltas.pyi": (("Timedelta", None),),
    "_libs/tslibs/timestamps.pyi": (("Timestamp", None),),
    "_stubs_only/__init__.pyi": (("IndexSubclassBase", "Index"),),
}


def _render(classes: tuple[_Class, ...]) -> str:
    """Render stub text: one class per entry, one ``def`` per dunder."""
    blocks: list[str] = []
    for name, base, members in classes:
        head = f"class {name}({base}):" if base else f"class {name}:"
        body = "".join(
            f"\n    def {dunder}(self, other: {operand}, /) -> None: ..."
            for dunder, operand in members
        )
        blocks.append(head + (body or "\n    pass"))
    return "\n\n".join(blocks) + "\n"


def _dunders(
    path: str,
    cls: str,
    *members: tuple[str, str],
    base: str | None = None,
) -> dict[str, str]:
    """Render ``path``, with ``cls`` carrying one ``def`` per ``(dunder, operand)``.

    The file's other classes keep the defaults, so a case names one class and one
    diagnostic; ``base`` overrides the class's default base, for the cases about how a base
    is spelled.
    """
    classes = dict(_DEFAULT_FILES[path])
    classes[cls] = base if base is not None else classes.get(cls)
    text = _render(
        tuple(
            (name, default_base, members if name == cls else ())
            for name, default_base in classes.items()
        )
    )
    # Two cases render a class their file does not declare, and adding it here is what
    # gives those rows their content, so both it and every dunder are asserted to have
    # reached the text: a helper that stopped adding it would delete them without failing.
    assert f"class {cls}(" in text or f"class {cls}:" in text, (path, cls)
    assert all(f"def {dunder}(" in text for dunder, _ in members), (path, cls)
    return {path: text}


def _aliases(
    *definitions: str,
    path: str = "core/base.pyi",
    header: str = "from typing import TypeAlias",
) -> dict[str, str]:
    """Render ``path``, where the alias cases declare their ``TypeAlias``\\ s."""
    return {path: "\n".join((header, "", *definitions, ""))}


def _write_tree(tmp_path: Path, *overrides: Mapping[str, str]) -> Path:
    """Write the default stub tree, each ``overrides`` mapping replacing a whole file."""
    contents = {
        path: _render(tuple((name, base, ()) for name, base in classes))
        for path, classes in _DEFAULT_FILES.items()
    }
    for override in overrides:
        contents |= override

    stub_root = tmp_path / "pandas-stubs"
    for relative, text in contents.items():
        path = stub_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub_root


# Shared by the ``interval-comparison`` row and the sibling-key test: one scanned site.
_INTERVAL_COMPARISON = _dunders(_INTERVAL, "Interval", ("__gt__", "IntervalIndex"))

# One higher-tier site per row: the stub files it lives in, the message substrings the
# verdict must name, and the key that permits exactly that site. The driver checks both
# halves, so a row pins the rejection and its key together.
_SINGLE_SITE_REJECTIONS: Mapping[
    str, tuple[tuple[Mapping[str, str], ...], tuple[str, ...], ExceptionKey]
] = {
    "qualified-terminal-name": (
        (_dunders(_SERIES, "Series", ("__add__", "pd.DataFrame")),),
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "qualified-alias": (
        (
            _aliases("Higher: TypeAlias = DataFrame"),
            _dunders(_SERIES, "Series", ("__add__", "types.Higher")),
        ),
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "qualified-type-alias": (
        (
            _aliases("Higher: typing.TypeAlias = DataFrame", header="import typing"),
            _dunders(_SERIES, "Series", ("__add__", "Higher")),
        ),
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "comparison-dunder": (
        (_dunders(_SERIES, "Series", ("__lt__", "DataFrame")),),
        ("Series.__lt__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "undeclared-matmul": (
        (_dunders(_SERIES, "Series", ("__matmul__", "DataFrame")),),
        ("Series.__matmul__ `other` operand references DataFrame",),
        ("Series", "__matmul__", "DataFrame"),
    ),
    "index-subclass-operand": (
        (_dunders(_INTERVAL_INDEX, "IntervalIndex", ("__eq__", "Series")),),
        (
            "IntervalIndex.__eq__ `other` operand references Series",
            "higher tier than IntervalIndex (tier 2)",
        ),
        ("IntervalIndex", "*", "Series"),
    ),
    "interval-comparison": (
        (_INTERVAL_COMPARISON,),
        ("references IntervalIndex (the tier-2 operand Index)",),
        ("Interval", "*", "Index"),
    ),
    # ``Interval`` is declared in both of these files: the violating one sorts first, and
    # the clean definition must not hide it, since these stubs do reuse class names.
    "duplicate-class-definition": (
        (
            _dunders(_INTERVAL, "Interval", ("__gt__", "Series")),
            _dunders(_TIMESTAMPS, "Interval"),
        ),
        ("Interval.__gt__ `other` operand references Series",),
        ("Interval", "*", "Series"),
    ),
    # Every registered scalar is enforced, not just the one a case spells out.
    "registered-scalar-operand": (
        (_dunders(_TIMEDELTAS, "Timedelta", ("__eq__", "Index")),),
        (
            "Timedelta.__eq__ `other` operand references Index",
            "higher tier than Timedelta (tier 0)",
        ),
        ("Timedelta", "*", "Index"),
    ),
    # A subclass spelling is reported as the operand its tier registers.
    "subclass-spelling": (
        (_dunders(_TIMEDELTAS, "Timedelta", ("__sub__", "TimedeltaIndex")),),
        (
            "Timedelta.__sub__ `other` operand references TimedeltaIndex",
            "(the tier-2 operand Index)",
            "higher tier than Timedelta (tier 0)",
        ),
        ("Timedelta", "*", "Index"),
    ),
    # A sibling spelling of a recognized tier resolves to the tier's operand name.
    "sibling-spelling": (
        (_dunders(_TIMEDELTAS, "Timedelta", ("__eq__", "MultiIndex")),),
        ("references MultiIndex (the tier-2 operand Index)",),
        ("Timedelta", "*", "Index"),
    ),
    # Tier 1 is scanned too: a tier-0 scalar may not claim an array-like as its operand.
    "scalar-naming-an-array-like": (
        (_dunders(_NATYPE, "NAType", ("__eq__", "ExtensionArray")),),
        (
            "NAType.__eq__ `other` operand references ExtensionArray",
            "higher tier than NAType (tier 0)",
        ),
        ("NAType", "*", "ExtensionArray"),
    ),
    "registered-natype-operand": (
        (_dunders(_NATYPE, "NAType", ("__add__", "Index")),),
        (
            "NAType.__add__ `other` operand references Index",
            "higher tier than NAType (tier 0)",
        ),
        ("NAType", "*", "Index"),
    ),
    # The scanned set is the operator set, not one spelling of it.
    "bitwise-dunder": (
        (_dunders(_INDEX, "Index", ("__or__", "Series")),),
        ("Index.__or__ `other` operand references Series",),
        ("Index", "*", "Series"),
    ),
    # ``MultiIndex`` is a tier-2 operand name in its own right, so it is scanned like
    # ``Index`` rather than reached through the base walk.
    "multiindex-series-operand": (
        (_dunders(_MULTI, "MultiIndex", ("__add__", "Series")),),
        ("MultiIndex.__add__ `other` operand references Series",),
        ("MultiIndex", "*", "Series"),
    ),
    "multiindex-dataframe-operand": (
        (_dunders(_MULTI, "MultiIndex", ("__add__", "DataFrame")),),
        ("MultiIndex.__add__ `other` operand references DataFrame",),
        ("MultiIndex", "*", "DataFrame"),
    ),
    # Every ``Index`` subclass the base walk discovers spells its base with arguments --
    # ``MultiIndex`` is an operand name in its own right, so the walk never has to find it
    # -- and only the unwrapping of a subscripted base reads those bases at all.
    "subscripted-base": (
        (
            _dunders(
                _RANGE,
                "RangeIndex",
                ("__add__", "Series"),
                base="IndexSubclassBase[int, np.int64]",
            ),
        ),
        (
            "RangeIndex.__add__ `other` operand references Series",
            "higher tier than RangeIndex (tier 2)",
        ),
        ("RangeIndex", "*", "Series"),
    ),
}

# One passing rule per row, as the override mappings alone: every row runs against the empty
# exception set, so nothing may be reported at all. That is strictly stronger than the
# shipped registry, which the one case needing a key asserts separately below.
_ACCEPTED_OPERANDS: Mapping[str, tuple[Mapping[str, str], ...]] = {
    # A lower-tier operand passes, and a reflected dunder is not scanned at all.
    "lower-tier-operands": (
        _aliases(
            "ScalarArrayIndexOperand: TypeAlias = int",
            "ScalarArrayIndexSeriesOperand: TypeAlias = int | Series",
        ),
        _dunders(_INDEX, "Index", ("__add__", "int"), ("__radd__", "Series")),
        _dunders(
            _SERIES, "Series", ("__add__", "int | Series"), ("__radd__", "DataFrame")
        ),
    ),
    "qualified-alias-of-a-lower-tier-operand": (
        _aliases("Lower: TypeAlias = Index"),
        _dunders(_SERIES, "Series", ("__add__", "types.Lower")),
    ),
    "same-tier-scalar-operand": (
        _dunders(_TIMEDELTAS, "Timedelta", ("__add__", "Timedelta")),
    ),
    # A scalar class the model does not name, such as ``IntervalLike``, is not scanned.
    "unregistered-scalar-class": (
        _dunders(_INTERVAL, "IntervalLike", ("__add__", "Series")),
    ),
    "non-binary-dunder": (
        _dunders(_INDEX, "Index", ("__foo__", "DataFrame")),
        _dunders(_SERIES, "Series", ("__add__", "int")),
    ),
    "multiindex-lower-tier-operand": (
        _dunders(_INDEX, "Index", ("__add__", "int")),
        _dunders(_MULTI, "MultiIndex", ("__add__", "int")),
        _dunders(_SERIES, "Series", ("__add__", "int | Series")),
    ),
}


@pytest.mark.parametrize("case", _SINGLE_SITE_REJECTIONS)
def test_rejects_one_higher_tier_site_and_a_key_permits_it(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One rule per row: the site is reported, and the key covering it permits it."""
    overrides, substrings, permitting_key = _SINGLE_SITE_REJECTIONS[case]
    stub_root = _write_tree(tmp_path, *overrides)

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    output = capsys.readouterr().err
    for substring in substrings:
        assert substring in output

    assert check_operand_hierarchy(stub_root, exceptions={permitting_key})
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("case", _ACCEPTED_OPERANDS)
def test_accepts_a_permitted_or_invisible_operand(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One rule per row: the operand is permitted or invisible, so nothing is reported."""
    stub_root = _write_tree(tmp_path, *_ACCEPTED_OPERANDS[case])

    assert check_operand_hierarchy(stub_root, exceptions=frozenset())
    assert capsys.readouterr().err == ""


def test_accepts_a_matmul_site_the_shipped_registry_permits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one case that needs the shipped registry: ``Series.__matmul__(DataFrame)``."""
    stub_root = _write_tree(
        tmp_path, _dunders(_SERIES, "Series", ("__matmul__", "DataFrame"))
    )

    assert check_operand_hierarchy(stub_root)
    assert capsys.readouterr().err == ""


# The alias-violating trees, which report through ``check_aliases`` and so have a message
# shape of their own: the aliases to declare and the operands to spell through them.
_ALIAS_VIOLATIONS: Mapping[str, tuple[tuple[str, ...], str, str]] = {
    "direct": (
        (
            "ScalarArrayIndexOperand: TypeAlias = Series | DataFrame",
            "ScalarArrayIndexSeriesOperand: TypeAlias = DataFrame",
        ),
        "Series | DataFrame",
        "DataFrame",
    ),
    "transitive": (
        (
            "SeriesAlias: TypeAlias = Series",
            "DataFrameAlias: TypeAlias = DataFrame",
            "ScalarArrayIndexOperand: TypeAlias = SeriesAlias | DataFrameAlias",
            "ScalarArrayIndexSeriesOperand: TypeAlias = DataFrameAlias",
            "IndexOperand: TypeAlias = SeriesAlias | DataFrameAlias",
            "SeriesOperand: TypeAlias = DataFrameAlias",
        ),
        "IndexOperand",
        "SeriesOperand",
    ),
}

_ALIAS_VIOLATION_MESSAGES = (
    "alias ScalarArrayIndexOperand references Series",
    "alias ScalarArrayIndexOperand references DataFrame",
    "alias ScalarArrayIndexSeriesOperand references DataFrame",
    "Index.__add__ `other` operand references Series",
    "Index.__add__ `other` operand references DataFrame",
    "Series.__add__ `other` operand references DataFrame",
)


@pytest.mark.parametrize("case", _ALIAS_VIOLATIONS)
def test_reports_alias_and_operand_violations(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An alias naming a higher tier fails, directly and through another alias."""
    definitions, index_operand, series_operand = _ALIAS_VIOLATIONS[case]
    stub_root = _write_tree(
        tmp_path,
        _aliases(*definitions),
        _dunders(_INDEX, "Index", ("__add__", index_operand)),
        _dunders(_SERIES, "Series", ("__add__", series_operand)),
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    for message in _ALIAS_VIOLATION_MESSAGES:
        assert message in output


def test_detects_an_alias_collision_across_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second, clean definition of a name must not hide a violating first one."""
    stub_root = _write_tree(
        tmp_path,
        # The violating definition sorts first, so a last-wins collector reports nothing.
        _aliases(
            "ScalarArrayIndexOperand: TypeAlias = Series",
            path="_stubs_only/aliases.pyi",
        ),
        _aliases("ScalarArrayIndexOperand: TypeAlias = int"),
    )

    assert not check_operand_hierarchy(stub_root)
    assert "alias ScalarArrayIndexOperand references Series" in capsys.readouterr().err


def test_rejects_a_sibling_key_for_an_interval_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sibling key covers the sibling operand only; the Index key covers this one."""
    stub_root = _write_tree(tmp_path, _INTERVAL_COMPARISON)
    series_only = {("Interval", "*", "Series")}

    assert not check_operand_hierarchy(stub_root, exceptions=series_only)
    assert "references IntervalIndex" in capsys.readouterr().err

    assert check_operand_hierarchy(
        stub_root, exceptions={*series_only, ("Interval", "*", "Index")}
    )


def test_rejects_forward_binary_dunder_without_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scanned dunder whose operand is spelled differently is reported, not skipped."""
    stub_root = _write_tree(
        tmp_path,
        {_INDEX: """\
class Index:
    def __add__(self, right: int, /) -> None: ...
"""},
    )

    assert not check_operand_hierarchy(stub_root)
    assert "Index.__add__ declares no 'other' operand" in capsys.readouterr().err


def test_class_level_exception_covers_every_dunder(tmp_path: Path) -> None:
    stub_root = _write_tree(
        tmp_path,
        _dunders(
            _SERIES,
            "Series",
            ("__matmul__", "DataFrame"),
            ("__add__", "DataFrame"),
        ),
    )
    exact = {("Series", "__matmul__", "DataFrame")}

    # An exact key covers its own dunder only; the class-level key covers both.
    assert not check_operand_hierarchy(stub_root, exceptions=exact)
    assert check_operand_hierarchy(
        stub_root,
        exceptions={*exact, ("Series", "*", "DataFrame")},
    )


@pytest.mark.parametrize(
    "relative_path",
    # One file the scan must read, and one only the tree walk reaches.
    ["core/indexes/range.pyi", "core/arrays/masked.pyi"],
)
def test_a_syntax_error_names_the_stub_file(relative_path: str, tmp_path: Path) -> None:
    """A malformed stub reports its own path, required or not, on any platform."""
    stub_root = _write_tree(tmp_path, {relative_path: "class Broken(:\n"})

    with pytest.raises(SyntaxError) as failure:
        check_operand_hierarchy(stub_root)

    assert failure.value.filename == str(stub_root / relative_path)


def test_rejects_a_temporary_exception_the_tree_never_exercises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_tree(
        tmp_path, _dunders(_TIMEDELTAS, "Timedelta", ("__add__", "Timedelta"))
    )
    dead: set[ExceptionKey] = {("Timedelta", "*", "Index")}

    assert check_operand_hierarchy(stub_root, exceptions=dead)
    assert not check_operand_hierarchy(
        stub_root, exceptions=dead, require_all_exercised=True
    )
    output = capsys.readouterr().err
    assert (
        "temporary exception ('Timedelta', '*', 'Index') is never exercised" in output
    )
    assert "remove it from scripts/operand_hierarchy/exceptions.py" in output
    assert "add the stub overload that justifies it" in output


def test_two_registry_keys_each_cover_a_whole_class() -> None:
    """The Interval and NAType keys are not formalities: each covers its class's dunders.

    Measured against the shipped registry: ``("Interval", "*", "Index")`` alone permits all
    6 Interval comparisons and ``("NAType", "*", "Series")`` alone the 15 NAType overloads
    that name ``Series``, 21 of the 52 ``(class, dunder)`` slots the scan reports. These are
    measurements, so update them when the stubs legitimately change, and read a collapse as
    a reason to delete a key rather than to edit a number.
    """
    tree = StubTree.load(_REPO_ROOT / "pandas-stubs")
    assert tree is not None

    for class_name, key, expected in (
        ("Interval", ("Interval", "*", "Index"), 6),
        ("NAType", ("NAType", "*", "Series"), 15),
    ):
        _, consulted, sites = tree.check_class(class_name, {key})
        assert consulted == {key}
        assert len(sites) == expected


def _document_anchors(document: Path) -> set[str]:
    """Return the GitHub heading anchors a markdown document exposes."""
    anchors: set[str] = set()
    headings: list[str] = re.findall(
        r"^#{1,6}\s+(.+?)\s*$", document.read_text(encoding="utf-8"), flags=re.MULTILINE
    )
    for heading in headings:
        slug = re.sub(r"[^\w\- ]", "", heading.lower())
        anchors.add(slug.replace(" ", "-"))
    return anchors


def test_temporary_exception_keys_are_well_formed() -> None:
    """Every key names a class, dunder and operand, and resolves to the guide.

    A key whose class is never scanned, or that can never be consulted, is already failed as
    dead by ``require_all_exercised`` in CI; this test names the faults it can name without
    building a stub tree, and keeps the mechanical code-to-guide link.
    """
    # Every tier has exactly one canonical operand name, because a site that spells the
    # operand indexes this table by tier, and a missing entry is a ``KeyError`` out of a
    # scan rather than a diagnostic. Each name must also name its own tier, so a
    # transposed pair of entries cannot pass by covering the same set of tiers.
    assert sorted(CANONICAL_OPERAND_BY_TIER) == sorted(set(TIER_OPERANDS.values()))
    for tier, operand in CANONICAL_OPERAND_BY_TIER.items():
        assert TIER_OPERANDS[operand] == tier

    for key in FORWARD_DUNDER_EXCEPTIONS:
        class_name, dunder, forbidden = key
        assert class_name
        assert dunder == "*" or dunder in FORWARD_BINARY_DUNDERS, key
        # The lookup canonicalizes the spelling first, so a key naming a bare spelling
        # such as ``MultiIndex`` could never be consulted.
        assert forbidden in CANONICAL_OPERAND_BY_TIER.values(), key

    relative_path, _, anchor = EXCEPTIONS_DOCUMENTATION.partition("#")
    assert anchor, "the exception list documents no anchor"
    document = _REPO_ROOT / relative_path
    if not document.is_file():
        # The guide is documentation, so it can land after the checker it documents.
        # Once it is in the tree, this assertion keeps the anchor resolving.
        pytest.skip(f"{relative_path} is not in the tree")
    assert anchor in _document_anchors(document), "the anchor does not resolve"


# The two lines CI and every review round quote, spelled out rather than assembled from the
# constants that produce them: the summary is the only report of the debt still on the
# books, so its path, its wording and its counts are all pinned.
_EXPECTED_SUMMARY = (
    "Operand hierarchy invariant holds.\n"
    "scripts/operand_hierarchy/exceptions.py -- 12 of 12 applied at 52 sites. "
    "Temporary exceptions; the target is an empty set.\n"
)


def test_this_repository_passes_and_prints_the_expected_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shipped stub tree is clean and its two-line summary is byte-identical.

    Apart from the well-formedness test, which scans nothing, and the key-coverage test
    above, the other tests drive synthetic trees under ``tmp_path``. So this is the one
    place the registry in ``exceptions.py`` is run with ``require_all_exercised`` over
    the real tree: a key dropped from it, or gone dead, would otherwise leave this suite
    green while only the ``architecture`` CI job noticed. ``main`` is the process CI
    runs, so its exit status is asserted here too.
    """
    assert main() == 0
    captured = capsys.readouterr()
    assert captured.out == _EXPECTED_SUMMARY
    assert captured.err == ""
