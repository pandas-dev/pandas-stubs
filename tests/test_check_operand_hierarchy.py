from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import pytest

from scripts.operand_hierarchy.check import (
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

# The minimal content each required stub file needs for the checker to find the class it
# declares. ``IndexSubclassBase`` resolves ``Index`` by name, so a test tree does not have
# to define ``Index`` for the derived subclasses to register.
_DEFAULT_STUB_FILES: Mapping[str, str] = {
    "core/arrays/base.pyi": "class ExtensionArray:\n    pass\n",
    "core/base.pyi": "",
    "core/frame.pyi": "class DataFrame:\n    pass\n",
    "core/indexes/base.pyi": "class Index:\n    pass\n",
    "core/indexes/category.pyi": "class CategoricalIndex(ExtensionIndex):\n    pass\n",
    "core/indexes/datetimelike.pyi": (
        "class DatetimeIndexOpsMixin(ExtensionIndex):\n"
        "    pass\n"
        "\n"
        "\n"
        "class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin):\n"
        "    pass\n"
    ),
    "core/indexes/datetimes.pyi": (
        "class DatetimeIndex(DatetimeTimedeltaMixin):\n    pass\n"
    ),
    "core/indexes/extension.pyi": (
        "class ExtensionIndex(IndexSubclassBase):\n    pass\n"
    ),
    "core/indexes/interval.pyi": "class IntervalIndex(ExtensionIndex):\n    pass\n",
    "core/indexes/multi.pyi": "class MultiIndex:\n    pass\n",
    "core/indexes/period.pyi": (
        "class PeriodIndex(DatetimeIndexOpsMixin):\n    pass\n"
    ),
    "core/indexes/range.pyi": "class RangeIndex(IndexSubclassBase):\n    pass\n",
    "core/indexes/timedeltas.pyi": (
        "class TimedeltaIndex(DatetimeTimedeltaMixin):\n    pass\n"
    ),
    "core/series.pyi": "class Series:\n    pass\n",
    "_libs/interval.pyi": "class Interval:\n    pass\n",
    "_libs/missing.pyi": "class NAType:\n    pass\n",
    "_libs/tslibs/period.pyi": "class Period:\n    pass\n",
    "_libs/tslibs/timedeltas.pyi": "class Timedelta:\n    pass\n",
    "_libs/tslibs/timestamps.pyi": "class Timestamp:\n    pass\n",
    "_stubs_only/__init__.pyi": "class IndexSubclassBase(Index):\n    pass\n",
}


def _write_stub_tree(tmp_path: Path, *, files: Mapping[str, str]) -> Path:
    """Create the smallest stub tree the hierarchy checker requires.

    ``files`` overrides any path relative to the stub root, so a case names only the
    files it differs in.
    """
    contents = {**_DEFAULT_STUB_FILES, **files}

    stub_root = tmp_path / "pandas-stubs"
    for relative_path, text in contents.items():
        path = stub_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub_root


# Shared by the ``interval-comparison`` row below and the sibling-key test: the one
# scanned site is ``Interval.__gt__`` against an ``IntervalIndex``, a tier-2 operand.
_INTERVAL_COMPARISON: Mapping[str, str] = {
    "_libs/interval.pyi": """\
class Interval:
    def __gt__(self, other: IntervalIndex, /) -> None: ...
""",
}

# One higher-tier site per rule: the stub files it lives in, the message substrings the
# verdict must name, and the key that permits exactly that site. The driver checks both
# halves, so a row pins the rejection and its exception together.
_SINGLE_SITE_REJECTIONS: Mapping[
    str, tuple[Mapping[str, str], tuple[str, ...], ExceptionKey]
] = {
    "qualified-terminal-name": (
        {
            "core/series.pyi": """\
class Series:
    def __add__(self, other: pd.DataFrame, /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "qualified-alias": (
        {
            "core/base.pyi": """\
from typing import TypeAlias

Higher: TypeAlias = DataFrame
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: types.Higher, /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "index-subclass-operand": (
        {
            "core/indexes/interval.pyi": """\
class IntervalIndex(ExtensionIndex):
    def __eq__(self, other: Series, /) -> None: ...
""",
        },
        (
            "IntervalIndex.__eq__ `other` operand references Series",
            "higher tier than IntervalIndex (tier 2)",
        ),
        ("IntervalIndex", "*", "Series"),
    ),
    "registered-scalar-operand": (
        {
            "_libs/tslibs/timedeltas.pyi": """\
class Timedelta:
    def __eq__(self, other: Index, /) -> None: ...
""",
        },
        (
            "Timedelta.__eq__ `other` operand references Index",
            "higher tier than Timedelta (tier 0)",
        ),
        ("Timedelta", "*", "Index"),
    ),
    # Every registered scalar is enforced, not just the one a case spells out.
    "registered-natype-operand": (
        {
            "_libs/missing.pyi": """\
class NAType:
    def __add__(self, other: Index, /) -> None: ...
""",
        },
        (
            "NAType.__add__ `other` operand references Index",
            "higher tier than NAType (tier 0)",
        ),
        ("NAType", "*", "Index"),
    ),
    # Tier 1 is scanned too: a tier-0 scalar may not claim an array-like as its operand.
    "scalar-naming-an-array-like": (
        {
            "_libs/missing.pyi": """\
class NAType:
    def __eq__(self, other: ExtensionArray, /) -> None: ...
""",
        },
        (
            "NAType.__eq__ `other` operand references ExtensionArray",
            "higher tier than NAType (tier 0)",
        ),
        ("NAType", "*", "ExtensionArray"),
    ),
    # A subclass spelling is reported as the operand its tier registers.
    "subclass-spelling": (
        {
            "_libs/tslibs/timedeltas.pyi": """\
class Timedelta:
    def __sub__(self, other: TimedeltaIndex, /) -> None: ...
""",
        },
        (
            "Timedelta.__sub__ `other` operand references TimedeltaIndex",
            "(the tier-2 operand Index)",
            "higher tier than Timedelta (tier 0)",
        ),
        ("Timedelta", "*", "Index"),
    ),
    # A sibling spelling of a recognized tier resolves to the tier's operand name.
    "sibling-spelling": (
        {
            "_libs/tslibs/timedeltas.pyi": """\
class Timedelta:
    def __eq__(self, other: MultiIndex, /) -> None: ...
""",
        },
        ("references MultiIndex (the tier-2 operand Index)",),
        ("Timedelta", "*", "Index"),
    ),
    "interval-comparison": (
        _INTERVAL_COMPARISON,
        ("references IntervalIndex (the tier-2 operand Index)",),
        ("Interval", "*", "Index"),
    ),
    "undeclared-matmul": (
        {
            "core/series.pyi": """\
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
""",
        },
        ("Series.__matmul__ `other` operand references DataFrame",),
        ("Series", "__matmul__", "DataFrame"),
    ),
    # A quoted annotation is a legal forward reference, so it must read like the spelling
    # it quotes rather than as a constant that names nothing -- and a quoted spelling is a
    # spelling wherever a type can stand, not only when it is the whole annotation.
    "quoted-annotation": (
        {
            "core/series.pyi": """\
class Series:
    def __add__(self, other: "DataFrame", /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "quoted-annotation-in-union": (
        {
            "core/series.pyi": """\
class Series:
    def __add__(self, other: "DataFrame" | None, /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    "quoted-annotation-under-generic": (
        {
            "core/series.pyi": """\
class Series:
    def __add__(self, other: Optional["DataFrame"], /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    # The first argument of ``Annotated`` is the type it annotates, unlike the metadata
    # after it, which is a value.
    "quoted-annotation-as-annotated-type": (
        {
            "core/series.pyi": """\
class Series:
    def __add__(self, other: Annotated["DataFrame", "meta"], /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    # ``typing.TypeAlias`` is the same declaration spelled through a module.
    "qualified-type-alias": (
        {
            "core/base.pyi": """\
import typing

Higher: typing.TypeAlias = DataFrame
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: Higher, /) -> None: ...
""",
        },
        ("Series.__add__ `other` operand references DataFrame",),
        ("Series", "*", "DataFrame"),
    ),
    # ``Interval`` is declared in both of these files, and the violating one sorts first;
    # the clean definition must not hide it, since these stubs do reuse class names.
    "duplicate-class-definition": (
        {
            "_libs/interval.pyi": """\
class Interval:
    def __gt__(self, other: Series, /) -> None: ...
""",
            "_libs/tslibs/timestamps.pyi": """\
class Timestamp:
    pass


class Interval:
    pass
""",
        },
        ("Interval.__gt__ `other` operand references Series",),
        ("Interval", "*", "Series"),
    ),
    # Every ``Index`` subclass the base walk discovers spells its base with arguments --
    # ``MultiIndex`` is an operand name in its own right, so the walk never has to find it
    # -- and only the unwrapping of a subscripted base reads those bases at all.
    "subscripted-base": (
        {
            "core/indexes/range.pyi": """\
class RangeIndex(IndexSubclassBase[int, np.int64]):
    def __add__(self, other: Series, /) -> None: ...
""",
        },
        (
            "RangeIndex.__add__ `other` operand references Series",
            "higher tier than RangeIndex (tier 2)",
        ),
        ("RangeIndex", "*", "Series"),
    ),
}

# One passing rule per row: the stub files, and the exception set to run with. ``None`` is
# the shipped registry, so such a row also asserts that the registry covers the case.
_ACCEPTED_OPERANDS: Mapping[
    str, tuple[Mapping[str, str], frozenset[ExceptionKey] | None]
] = {
    # A lower-tier operand passes, and a reflected dunder is not scanned at all.
    "lower-tier-operands": (
        {
            "core/base.pyi": """\
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = int
ScalarArrayIndexSeriesOperand: TypeAlias = int | Series
""",
            "core/indexes/base.pyi": """\
class Index:
    def __add__(self, other: int, /) -> None: ...
    def __radd__(self, other: Series, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: int | Series, /) -> None: ...
    def __radd__(self, other: DataFrame, /) -> None: ...
""",
        },
        None,
    ),
    "qualified-alias-of-a-lower-tier-operand": (
        {
            "core/base.pyi": """\
from typing import TypeAlias

Lower: TypeAlias = Index
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: types.Lower, /) -> None: ...
""",
        },
        None,
    ),
    "same-tier-scalar-operand": (
        {
            "_libs/tslibs/timedeltas.pyi": """\
class Timedelta:
    def __add__(self, other: Timedelta, /) -> None: ...
""",
        },
        frozenset(),
    ),
    # A scalar class the model does not name, such as ``IntervalLike``, is not scanned.
    "unregistered-scalar-class": (
        {
            "_libs/interval.pyi": """\
class Interval:
    pass


class IntervalLike:
    def __add__(self, other: Series, /) -> None: ...
""",
        },
        None,
    ),
    "non-binary-dunder": (
        {
            "core/indexes/base.pyi": """\
class Index:
    def __foo__(self, other: DataFrame, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: int, /) -> None: ...
""",
        },
        None,
    ),
    "multiindex-lower-tier-operand": (
        {
            "core/indexes/base.pyi": """\
class Index:
    def __add__(self, other: int, /) -> None: ...
""",
            "core/indexes/multi.pyi": """\
class MultiIndex:
    def __add__(self, other: int, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: int | Series, /) -> None: ...
""",
        },
        None,
    ),
    "default-registry-matmul-key": (
        {
            "core/series.pyi": """\
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
""",
        },
        None,
    ),
    # A string inside a subscript is an argument, not a quoted annotation: the value of
    # ``Literal`` names a value, so it is not the operand it happens to spell.
    "literal-string-operand": (
        {
            "_libs/missing.pyi": """\
class NAType:
    def __eq__(self, other: Literal["DataFrame"], /) -> None: ...
""",
        },
        frozenset(),
    ),
}


@pytest.mark.parametrize("case", _SINGLE_SITE_REJECTIONS)
def test_rejects_one_higher_tier_site_and_a_key_permits_it(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One rule per row: the site is reported, and the key covering it permits it."""
    files, substrings, permitting_key = _SINGLE_SITE_REJECTIONS[case]
    stub_root = _write_stub_tree(tmp_path, files=files)

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
    files, exceptions = _ACCEPTED_OPERANDS[case]
    stub_root = _write_stub_tree(tmp_path, files=files)

    if exceptions is None:
        exceptions = FORWARD_DUNDER_EXCEPTIONS
    assert check_operand_hierarchy(stub_root, exceptions=exceptions)
    assert capsys.readouterr().err == ""


def test_rejects_direct_alias_and_operand_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/base.pyi": """\
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = Series | DataFrame
ScalarArrayIndexSeriesOperand: TypeAlias = DataFrame
""",
            "core/indexes/base.pyi": """\
class Index:
    def __add__(self, other: Series | DataFrame, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: DataFrame, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "alias ScalarArrayIndexOperand references Series" in output
    assert "alias ScalarArrayIndexOperand references DataFrame" in output
    assert "alias ScalarArrayIndexSeriesOperand references DataFrame" in output
    assert "Index.__add__ `other` operand references Series" in output
    assert "Index.__add__ `other` operand references DataFrame" in output
    assert "Series.__add__ `other` operand references DataFrame" in output


def test_rejects_transitive_alias_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/base.pyi": """\
from typing import TypeAlias

SeriesAlias: TypeAlias = Series
DataFrameAlias: TypeAlias = DataFrame
ScalarArrayIndexOperand: TypeAlias = SeriesAlias | DataFrameAlias
ScalarArrayIndexSeriesOperand: TypeAlias = DataFrameAlias
IndexOperand: TypeAlias = SeriesAlias | DataFrameAlias
SeriesOperand: TypeAlias = DataFrameAlias
""",
            "core/indexes/base.pyi": """\
class Index:
    def __add__(self, other: IndexOperand, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __add__(self, other: SeriesOperand, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "alias ScalarArrayIndexOperand references Series" in output
    assert "alias ScalarArrayIndexOperand references DataFrame" in output
    assert "alias ScalarArrayIndexSeriesOperand references DataFrame" in output
    assert "Index.__add__ `other` operand references Series" in output
    assert "Index.__add__ `other` operand references DataFrame" in output
    assert "Series.__add__ `other` operand references DataFrame" in output


def test_detects_an_alias_collision_across_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second, clean definition of a name must not hide a violating first one."""
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/base.pyi": """\
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = int
""",
            "core/indexes/base.pyi": """\
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = Series

class Index:
    pass
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    assert "alias ScalarArrayIndexOperand references Series" in capsys.readouterr().err


def test_checks_bitwise_and_comparison_dunders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/indexes/base.pyi": """\
class Index:
    def __or__(self, other: Series, /) -> None: ...
""",
            "core/series.pyi": """\
class Series:
    def __lt__(self, other: DataFrame, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "Index.__or__ `other` operand references Series" in output
    assert "Series.__lt__ `other` operand references DataFrame" in output


def test_rejects_a_sibling_key_for_an_interval_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sibling key covers the sibling operand only; the Index key covers this one."""
    stub_root = _write_stub_tree(tmp_path, files=_INTERVAL_COMPARISON)
    series_only = {("Interval", "*", "Series")}

    assert not check_operand_hierarchy(stub_root, exceptions=series_only)
    assert "references IntervalIndex" in capsys.readouterr().err

    assert check_operand_hierarchy(
        stub_root, exceptions={*series_only, ("Interval", "*", "Index")}
    )


def test_rejects_forward_binary_dunder_without_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/indexes/base.pyi": """\
class Index:
    def __add__(self, right: int, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    assert "Index.__add__ declares no 'other' operand" in capsys.readouterr().err


def test_class_level_exception_covers_every_dunder(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/series.pyi": """\
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
    def __add__(self, other: DataFrame, /) -> Series: ...
""",
        },
    )
    exact = {("Series", "__matmul__", "DataFrame")}

    # An exact key covers its own dunder only; the class-level key covers both.
    assert not check_operand_hierarchy(stub_root, exceptions=exact)
    assert check_operand_hierarchy(
        stub_root,
        exceptions={*exact, ("Series", "*", "DataFrame")},
    )


def test_rejects_multiindex_higher_tier_operands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/indexes/multi.pyi": """\
class MultiIndex:
    def __add__(self, other: Series | DataFrame, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "MultiIndex.__add__ `other` operand references Series" in output
    assert "MultiIndex.__add__ `other` operand references DataFrame" in output


@pytest.mark.parametrize(
    "relative_path",
    # One file the scan must read, and one only the tree walk reaches.
    ["core/indexes/range.pyi", "core/arrays/masked.pyi"],
)
def test_a_syntax_error_names_the_stub_file(relative_path: str, tmp_path: Path) -> None:
    """A malformed stub reports its own path, required or not, on any platform."""
    stub_root = _write_stub_tree(tmp_path, files={relative_path: "class Broken(:\n"})

    with pytest.raises(SyntaxError) as failure:
        check_operand_hierarchy(stub_root)

    assert failure.value.filename == str(stub_root / relative_path)


def test_rejects_a_temporary_exception_the_tree_never_exercises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """\
class Timedelta:
    def __add__(self, other: Timedelta, /) -> None: ...
""",
        },
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
    """Every key names a scanned class, dunder and operand, and resolves to the guide.

    A key that can never be consulted is already failed as dead by ``require_all_exercised``
    in CI; this test names the same faults locally, without building a stub tree, and keeps
    the mechanical code-to-guide link.
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

    document = Path(__file__).parents[1] / relative_path
    if not document.is_file():
        # The guide is documentation, so it can land after the checker it documents.
        # Once it is in the tree, this assertion keeps the anchor resolving.
        pytest.skip(f"{relative_path} is not in the tree")
    assert anchor in _document_anchors(document), "the anchor does not resolve"


# The two lines CI, the guide and every review round quote, spelled out rather than
# assembled from the constants that produce them: the summary is the only report of the
# debt still on the books, so its path, its wording and its counts are all pinned.
_EXPECTED_SUMMARY = (
    "Operand hierarchy invariant holds.\n"
    "scripts/operand_hierarchy/exceptions.py -- 12 of 12 applied at 52 sites. "
    "Temporary exceptions; the target is an empty set.\n"
)


def test_this_repository_passes_and_prints_the_expected_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The shipped stub tree is clean and its two-line summary is byte-identical.

    Apart from the well-formedness test, which scans nothing, every other test drives a
    synthetic tree under ``tmp_path``. So this is the one place the registry in
    ``exceptions.py`` is run with ``require_all_exercised`` over the real tree: a key
    dropped from it, or a key that has gone dead, would otherwise leave this suite green
    while only the ``architecture`` CI job noticed. The counts pin the aggregate the
    non-vacuity checks are measured against -- 52 ``(class, dunder)`` slots in all, of
    which six are Interval comparisons and fifteen are NAType overloads -- and ``main`` is
    the process CI runs, so its exit status is asserted here too.
    """
    assert main() == 0
    captured = capsys.readouterr()
    assert captured.out == _EXPECTED_SUMMARY
    assert captured.err == ""
