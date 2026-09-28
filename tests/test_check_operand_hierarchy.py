from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import pytest

from scripts.operand_hierarchy.check import check_operand_hierarchy
from scripts.operand_hierarchy.exceptions import (
    EXCEPTIONS_DOCUMENTATION,
    FORWARD_DUNDER_EXCEPTIONS,
    ExceptionKey,
)
from scripts.operand_hierarchy.model import (
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

    A mis-shaped key can never be consulted, so ``require_all_exercised`` already fails
    on it as a dead key in CI; this test keeps the mechanical code-to-guide link and
    catches a key that would be consulted but names an unscanned dunder.
    """
    for key in FORWARD_DUNDER_EXCEPTIONS:
        class_name, dunder, forbidden = key
        assert class_name
        assert dunder == "*" or dunder in FORWARD_BINARY_DUNDERS, key
        assert forbidden in TIER_OPERANDS, key

    relative_path, _, anchor = EXCEPTIONS_DOCUMENTATION.partition("#")
    assert anchor, "the exception list documents no anchor"

    document = Path(__file__).parents[1] / relative_path
    if not document.is_file():
        # The guide is documentation, so it can land after the checker it documents.
        # Once it is in the tree, this assertion keeps the anchor resolving.
        pytest.skip(f"{relative_path} is not in the tree")
    assert anchor in _document_anchors(document), "the anchor does not resolve"
