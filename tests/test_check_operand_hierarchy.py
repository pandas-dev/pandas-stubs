from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import pytest

from scripts.check_operand_hierarchy import (
    FORWARD_BINARY_DUNDERS,
    TIER_OPERANDS,
    check_operand_hierarchy,
)
from scripts.operand_hierarchy_exceptions import (
    EXCEPTIONS_DOCUMENTATION,
    FORWARD_DUNDER_EXCEPTIONS,
    ExceptionKey,
)

# The minimal content each required stub file needs for the checker to find the class it
# declares. ``IndexSubclassBase`` resolves ``Index`` by name, so a test tree does not have
# to define ``Index`` for the derived subclasses to register.
_DEFAULT_STUB_FILES: Mapping[str, str] = {
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


def _write_stub_tree(
    tmp_path: Path,
    *,
    base: str | None = None,
    index: str | None = None,
    multi: str | None = None,
    series: str | None = None,
    files: Mapping[str, str] | None = None,
) -> Path:
    """Create the smallest stub tree the hierarchy checker requires.

    ``base``, ``index``, ``multi`` and ``series`` override the four originally scanned
    stub files; ``files`` overrides any other path relative to the stub root.
    """
    contents = dict(_DEFAULT_STUB_FILES)
    named = {
        "core/base.pyi": base,
        "core/indexes/base.pyi": index,
        "core/indexes/multi.pyi": multi,
        "core/series.pyi": series,
    }
    contents.update({path: text for path, text in named.items() if text is not None})
    if files is not None:
        contents.update(files)

    stub_root = tmp_path / "pandas-stubs"
    for relative_path, text in contents.items():
        path = stub_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return stub_root


def test_accepts_lower_tier_operands(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        base="""
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = int
ScalarArrayIndexSeriesOperand: TypeAlias = int | Series
""",
        index="""
class Index:
    def __add__(self, other: int, /) -> None: ...
    def __radd__(self, other: Series, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: int | Series, /) -> None: ...
    def __radd__(self, other: DataFrame, /) -> None: ...
""",
    )

    assert check_operand_hierarchy(stub_root)


def test_rejects_direct_alias_and_operand_violations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        base="""
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = Series | DataFrame
ScalarArrayIndexSeriesOperand: TypeAlias = DataFrame
""",
        index="""
class Index:
    def __add__(self, other: Series | DataFrame, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: DataFrame, /) -> None: ...
""",
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
        base="""
from typing import TypeAlias

SeriesAlias: TypeAlias = Series
DataFrameAlias: TypeAlias = DataFrame
ScalarArrayIndexOperand: TypeAlias = SeriesAlias | DataFrameAlias
ScalarArrayIndexSeriesOperand: TypeAlias = DataFrameAlias
IndexOperand: TypeAlias = SeriesAlias | DataFrameAlias
SeriesOperand: TypeAlias = DataFrameAlias
""",
        index="""
class Index:
    def __add__(self, other: IndexOperand, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: SeriesOperand, /) -> None: ...
""",
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
        base="""
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = int
""",
        index="""
from typing import TypeAlias

ScalarArrayIndexOperand: TypeAlias = Series

class Index:
    pass
""",
        series="""
class Series:
    pass
""",
    )

    assert not check_operand_hierarchy(stub_root)
    assert "alias ScalarArrayIndexOperand references Series" in capsys.readouterr().err


def test_checks_bitwise_and_comparison_dunders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __or__(self, other: Series, /) -> None: ...
""",
        series="""
class Series:
    def __lt__(self, other: DataFrame, /) -> None: ...
""",
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "Index.__or__ `other` operand references Series" in output
    assert "Series.__lt__ `other` operand references DataFrame" in output


def test_rejects_a_qualified_terminal_operand_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        series="""
class Series:
    def __add__(self, other: pd.DataFrame, /) -> None: ...
""",
    )

    assert not check_operand_hierarchy(stub_root)
    assert (
        "Series.__add__ `other` operand references DataFrame" in capsys.readouterr().err
    )


def test_rejects_an_index_subclass_higher_tier_operand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "core/indexes/interval.pyi": """
class IntervalIndex(ExtensionIndex):
    def __eq__(self, other: Series, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    output = capsys.readouterr().err
    assert "IntervalIndex.__eq__ `other` operand references Series" in output
    assert "higher tier than IntervalIndex (tier 2)" in output


def test_rejects_a_registered_scalar_higher_tier_operand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """
class Timedelta:
    def __eq__(self, other: Index, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    output = capsys.readouterr().err
    assert "Timedelta.__eq__ `other` operand references Index" in output
    assert "higher tier than Timedelta (tier 0)" in output

    assert check_operand_hierarchy(
        stub_root,
        exceptions={("Timedelta", "*", "Index")},
    )


def test_accepts_a_registered_scalar_lower_tier_operand(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """
class Timedelta:
    def __add__(self, other: Timedelta, /) -> None: ...
""",
        },
    )

    assert check_operand_hierarchy(stub_root, exceptions=set())


def test_rejects_a_registered_natype_higher_tier_operand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every registered scalar is enforced, not just the one a fixture spells out."""
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/missing.pyi": """
class NAType:
    def __add__(self, other: Index, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    output = capsys.readouterr().err
    assert "NAType.__add__ `other` operand references Index" in output
    assert "higher tier than NAType (tier 0)" in output

    assert check_operand_hierarchy(
        stub_root,
        exceptions={("NAType", "*", "Index")},
    )


def test_rejects_a_temporary_exception_the_tree_never_exercises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """
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
    assert "remove it from scripts/operand_hierarchy_exceptions.py" in output
    assert "add the stub overload that justifies it" in output


def test_rejects_a_scalar_index_subclass_operand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A subclass spelling is a higher-tier operand, reported as the operand it is."""
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """
class Timedelta:
    def __sub__(self, other: TimedeltaIndex, /) -> None: ...
""",
        },
    )

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    output = capsys.readouterr().err
    assert "Timedelta.__sub__ `other` operand references TimedeltaIndex" in output
    assert "(the tier-2 operand Index)" in output
    assert "higher tier than Timedelta (tier 0)" in output

    assert check_operand_hierarchy(
        stub_root,
        exceptions={("Timedelta", "*", "Index")},
    )


def test_rejects_an_interval_index_subclass_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/interval.pyi": """
class Interval:
    def __gt__(self, other: IntervalIndex, /) -> None: ...
""",
        },
    )
    series_only = {("Interval", "*", "Series")}

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    assert "references IntervalIndex (the tier-2 operand Index)" in (
        capsys.readouterr().err
    )

    # The Series key covers the sibling operand only; the Index key covers this one.
    assert not check_operand_hierarchy(stub_root, exceptions=series_only)
    assert "references IntervalIndex" in capsys.readouterr().err
    assert check_operand_hierarchy(
        stub_root,
        exceptions={*series_only, ("Interval", "*", "Index")},
    )


def test_canonicalizes_multiindex_to_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sibling spelling of a recognized tier resolves to the tier's operand name."""
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/tslibs/timedeltas.pyi": """
class Timedelta:
    def __eq__(self, other: MultiIndex, /) -> None: ...
""",
        },
    )
    index_entry = {("Timedelta", "*", "Index")}

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    assert "references MultiIndex (the tier-2 operand Index)" in (
        capsys.readouterr().err
    )
    assert check_operand_hierarchy(stub_root, exceptions=index_entry)


def test_ignores_an_unregistered_scalar_operand(tmp_path: Path) -> None:
    """A scalar outside ``TIER_0_CLASSES`` is not scanned."""
    stub_root = _write_stub_tree(
        tmp_path,
        files={
            "_libs/interval.pyi": """
class Interval:
    pass


class IntervalLike:
    def __add__(self, other: Series, /) -> None: ...
""",
        },
    )

    assert check_operand_hierarchy(stub_root)


def test_rejects_undeclared_matrix_multiplication_exception(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __add__(self, other: int, /) -> None: ...
""",
        series="""
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
""",
    )

    assert not check_operand_hierarchy(stub_root, exceptions=set())
    assert (
        "Series.__matmul__ `other` operand references DataFrame"
        in capsys.readouterr().err
    )


def test_accepts_declared_matrix_multiplication_exception(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __add__(self, other: int, /) -> None: ...
""",
        series="""
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
""",
    )

    assert check_operand_hierarchy(stub_root)


def test_class_level_exception_covers_every_dunder(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    pass
""",
        series="""
class Series:
    def __matmul__(self, other: DataFrame, /) -> Series: ...
    def __add__(self, other: DataFrame, /) -> Series: ...
""",
    )
    exact = {("Series", "__matmul__", "DataFrame")}

    # An exact key covers its own dunder only; the class-level key covers both.
    assert not check_operand_hierarchy(stub_root, exceptions=exact)
    assert check_operand_hierarchy(
        stub_root,
        exceptions={*exact, ("Series", "*", "DataFrame")},
    )


def test_accepts_multiindex_with_lower_tier_operand(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __add__(self, other: int, /) -> None: ...
""",
        multi="""
class MultiIndex:
    def __add__(self, other: int, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: int | Series, /) -> None: ...
""",
    )

    assert check_operand_hierarchy(stub_root)


def test_rejects_multiindex_higher_tier_operands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    pass
""",
        multi="""
class MultiIndex:
    def __add__(self, other: Series | DataFrame, /) -> None: ...
""",
        series="""
class Series:
    pass
""",
    )

    assert not check_operand_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "MultiIndex.__add__ `other` operand references Series" in output
    assert "MultiIndex.__add__ `other` operand references DataFrame" in output


def test_ignores_non_binary_dunder_with_other(tmp_path: Path) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __foo__(self, other: DataFrame, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: int, /) -> None: ...
""",
    )

    assert check_operand_hierarchy(stub_root)


def test_rejects_forward_binary_dunder_without_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __add__(self, right: int, /) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: int, /) -> None: ...
""",
    )

    assert not check_operand_hierarchy(stub_root)
    assert "Index.__add__ declares no 'other' operand" in capsys.readouterr().err


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
