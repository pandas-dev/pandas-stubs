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
    EXCEPTIONS_FILE,
    FORWARD_DUNDER_EXCEPTIONS,
)
from scripts.operand_hierarchy.model import (
    CANONICAL_OPERAND_BY_TIER,
    CLASS_TIERS,
    FORWARD_BINARY_DUNDERS,
)

_REPO_ROOT = Path(__file__).parents[1]
_MINIMAL_STUBS = {
    "core/arrays/base.pyi": "class ExtensionArray: ...\n",
    "core/base.pyi": "\n",
    "core/frame.pyi": "class DataFrame: ...\n",
    "core/indexes/base.pyi": "class Index: ...\n",
    "core/indexes/category.pyi": "class CategoricalIndex(ExtensionIndex): ...\n",
    "core/indexes/datetimelike.pyi": "class DatetimeIndexOpsMixin(ExtensionIndex): ...\nclass DatetimeTimedeltaMixin(DatetimeIndexOpsMixin): ...\n",
    "core/indexes/datetimes.pyi": "class DatetimeIndex(DatetimeTimedeltaMixin): ...\n",
    "core/indexes/extension.pyi": "class ExtensionIndex(IndexSubclassBase): ...\n",
    "core/indexes/interval.pyi": "class IntervalIndex(ExtensionIndex): ...\n",
    "core/indexes/multi.pyi": "class MultiIndex: ...\n",
    "core/indexes/period.pyi": "class PeriodIndex(DatetimeIndexOpsMixin): ...\n",
    "core/indexes/range.pyi": "class RangeIndex(IndexSubclassBase): ...\n",
    "core/indexes/timedeltas.pyi": "class TimedeltaIndex(DatetimeTimedeltaMixin): ...\n",
    "core/series.pyi": "class Series: ...\n",
    "_libs/interval.pyi": "class Interval: ...\n",
    "_libs/missing.pyi": "class NAType: ...\n",
    "_libs/tslibs/period.pyi": "class Period: ...\n",
    "_libs/tslibs/timedeltas.pyi": "class Timedelta: ...\n",
    "_libs/tslibs/timestamps.pyi": "class Timestamp: ...\n",
    "_stubs_only/__init__.pyi": "class IndexSubclassBase(Index): ...\n",
}


def _write_tree(tmp_path: Path, overrides: Mapping[str, str]) -> Path:
    stub_root = tmp_path / "pandas-stubs"
    for relative, source in {**_MINIMAL_STUBS, **overrides}.items():
        path = stub_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return stub_root


@pytest.mark.parametrize(
    "operand", ["DataFrame", "pd.DataFrame", "Higher", "types.Higher"]
)
def test_alias_spellings(
    tmp_path: Path, operand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(
        tmp_path,
        {
            "aliases.pyi": "Higher: typing.TypeAlias = DataFrame",
            "core/series.pyi": f"class Series:\n    def __add__(self, other: {operand}, /) -> None: ...",
        },
    )
    assert not check_operand_hierarchy(root, exceptions={})
    assert (
        "Series.__add__ higher-tier operand DataFrame: expected 0 overload(s), observed 1"
        in capsys.readouterr().err
    )
    assert check_operand_hierarchy(
        root, exceptions={("Series", "__add__", "DataFrame"): 1}
    )


@pytest.mark.parametrize(
    "operator",
    [
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
    ],
)
def test_scanned_operators(tmp_path: Path, operator: str) -> None:
    root = _write_tree(
        tmp_path,
        {
            "core/indexes/base.pyi": f"class Index:\n    def {operator}(self, other: Series, /) -> None: ...",
        },
    )
    assert not check_operand_hierarchy(root, exceptions={})
    assert check_operand_hierarchy(root, exceptions={("Index", operator, "Series"): 1})


@pytest.mark.parametrize("expected", [0, 1, 2, 3])
def test_exact_counts(
    tmp_path: Path, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(
        tmp_path,
        {"core/series.pyi": """
class Series:
    @overload
    def __matmul__(self, other: DataFrame, /) -> Series: ...
    @overload
    def __matmul__(self, other: DataFrame, /) -> DataFrame: ...
"""},
    )
    allowances: dict[tuple[str, str, str], int] = (
        {("Series", "__matmul__", "DataFrame"): expected} if expected else {}
    )
    assert check_operand_hierarchy(root, exceptions=allowances) == (expected == 2)
    error = capsys.readouterr().err
    if expected != 2:
        assert f"expected {expected} overload(s), observed 2" in error


def test_unused_allowance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _write_tree(tmp_path, {})
    assert not check_operand_hierarchy(
        root, exceptions={("Series", "__matmul__", "DataFrame"): 1}
    )
    assert "expected 1 overload(s), observed 0" in capsys.readouterr().err


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_allowance_count(
    tmp_path: Path, count: int, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(tmp_path, {})
    assert not check_operand_hierarchy(
        root, exceptions={("Series", "__add__", "DataFrame"): count}
    )
    assert "must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize(
    "operand",
    [
        "Index | MultiIndex | TimedeltaIndex | Alias | Alias",
        "Alias",
        "pd.Index",
        "TimedeltaIndex",
    ],
)
def test_canonical_counts_ignore_spelling(tmp_path: Path, operand: str) -> None:
    root = _write_tree(
        tmp_path,
        {
            "aliases.pyi": "Alias: TypeAlias = Index\nUnused: TypeAlias = int",
            "_libs/tslibs/timedeltas.pyi": f"class Timedelta:\n    def __add__(self, other: {operand} | Series, /) -> None: ...",
        },
    )
    assert check_operand_hierarchy(
        root,
        exceptions={
            ("Timedelta", "__add__", "Index"): 1,
            ("Timedelta", "__add__", "Series"): 1,
        },
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_overload_reordering(tmp_path: Path, reverse: bool) -> None:
    declarations = [
        "    def __add__(self, other: DataFrame, /) -> Series: ...",
        "    def __add__(self, other: int, /) -> Series: ...",
        "    def __sub__(self, other: DataFrame, /) -> Series: ...",
    ]
    if reverse:
        declarations.reverse()
    root = _write_tree(
        tmp_path,
        {
            "core/series.pyi": "class Series:\n" + "\n".join(declarations),
        },
    )
    assert check_operand_hierarchy(
        root,
        exceptions={
            ("Series", "__add__", "DataFrame"): 1,
            ("Series", "__sub__", "DataFrame"): 1,
        },
    )


def test_conditional_methods_and_nested_scope_exclusion(tmp_path: Path) -> None:
    root = _write_tree(
        tmp_path,
        {"core/series.pyi": """
class Series:
    if VERSION:
        def __add__(self, other: DataFrame, /) -> None: ...
    else:
        if OTHER_VERSION:
            def __add__(self, other: DataFrame, /) -> None: ...
        else:
            def __sub__(self, other: DataFrame, /) -> None: ...
    def helper(self):
        def __add__(self, other: DataFrame, /) -> None: ...
    class Nested:
        def __add__(self, other: DataFrame, /) -> None: ...
    def __mul__(self, other: int, /) -> None:
        def __add__(self, other: DataFrame, /) -> None: ...
"""},
    )
    assert check_operand_hierarchy(
        root,
        exceptions={
            ("Series", "__add__", "DataFrame"): 2,
            ("Series", "__sub__", "DataFrame"): 1,
        },
    )


def test_permitted_operands_and_unscanned_classes(tmp_path: Path) -> None:
    root = _write_tree(
        tmp_path,
        {
            "core/series.pyi": """
class Series:
    def __add__(self, other: int | Index | Series, /) -> DataFrame: ...
    def __radd__(self, other: DataFrame, /) -> None: ...
    def __foo__(self, other: DataFrame, /) -> None: ...
class Unregistered:
    def __add__(self, other: DataFrame, /) -> None: ...
""",
            "aliases.pyi": """
ScalarArrayIndexOperand: TypeAlias = int | Index
ScalarArrayIndexSeriesOperand: TypeAlias = int | Series
""",
        },
    )
    assert check_operand_hierarchy(root, exceptions={})


@pytest.mark.parametrize("definition", ["Series | DataFrame", "Cycle"])
def test_alias_constraints_and_cycles(
    tmp_path: Path, definition: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(
        tmp_path,
        {"aliases.pyi": f"""
Cycle: TypeAlias = Other
Other: TypeAlias = Cycle | Series | DataFrame
ScalarArrayIndexOperand: TypeAlias = {definition}
ScalarArrayIndexSeriesOperand: TypeAlias = {definition}
"""},
    )
    assert not check_operand_hierarchy(root, exceptions={})
    error = capsys.readouterr().err
    assert "alias ScalarArrayIndexOperand references Series" in error
    assert "alias ScalarArrayIndexOperand references DataFrame" in error
    assert "alias ScalarArrayIndexSeriesOperand references DataFrame" in error


def test_benign_alias_cycle(tmp_path: Path) -> None:
    root = _write_tree(
        tmp_path,
        {"aliases.pyi": """
A: TypeAlias = B
B: TypeAlias = A | int
ScalarArrayIndexOperand: TypeAlias = A
"""},
    )
    assert check_operand_hierarchy(root, exceptions={})


@pytest.mark.parametrize("branch", ["if", "else"])
def test_duplicate_aliases_and_classes_are_all_scanned(
    tmp_path: Path, branch: str
) -> None:
    method = "class Series:\n        def __add__(self, other: Higher, /) -> None: ..."
    empty = "class Series: ..."
    if_body, else_body = (method, empty) if branch == "if" else (empty, method)
    root = _write_tree(
        tmp_path,
        {
            "a.pyi": "Higher: TypeAlias = DataFrame",
            "z.pyi": "Higher: TypeAlias = int",
            "core/series.pyi": f"""
if VERSION:
    {if_body}
else:
    {else_body}
""",
        },
    )
    assert not check_operand_hierarchy(root, exceptions={})
    assert check_operand_hierarchy(
        root, exceptions={("Series", "__add__", "DataFrame"): 1}
    )


@pytest.mark.parametrize("parameter", ["right: int", "other"])
def test_requires_annotated_other(
    tmp_path: Path, parameter: str, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(
        tmp_path,
        {
            "core/indexes/base.pyi": f"class Index:\n    def __add__(self, {parameter}, /) -> None: ..."
        },
    )
    assert not check_operand_hierarchy(root, exceptions={})
    assert (
        "Index.__add__ requires an annotated 'other' operand" in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "parameter",
    ["other: Series, /", "*, other: Series", "*other: Series", "**other: Series"],
)
def test_other_parameter_kinds(tmp_path: Path, parameter: str) -> None:
    root = _write_tree(
        tmp_path,
        {
            "core/indexes/base.pyi": f"class Index:\n    def __add__(self, {parameter}) -> None: ..."
        },
    )
    assert check_operand_hierarchy(root, exceptions={("Index", "__add__", "Series"): 1})


def test_index_subclass_discovery_requires_registered_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = """
class NewIndex(pd.Index[int]):
    def __add__(self, other: Series, /) -> None: ...
"""
    root = _write_tree(tmp_path, {"new.pyi": source})
    assert not check_operand_hierarchy(root, exceptions={})
    assert "could not find class 'NewIndex'" in capsys.readouterr().err
    (root / "new.pyi").unlink()
    (root / "core/base.pyi").write_text(source, encoding="utf-8")
    assert check_operand_hierarchy(
        root, exceptions={("NewIndex", "__add__", "Series"): 1}
    )


@pytest.mark.parametrize(
    "path, source, key",
    [
        (
            "_libs/missing.pyi",
            "class NAType:\n    def __eq__(self, other: ExtensionArray, /) -> None: ...",
            ("NAType", "__eq__", "ExtensionArray"),
        ),
        (
            "core/indexes/multi.pyi",
            "class MultiIndex:\n    def __add__(self, other: DataFrame, /) -> None: ...",
            ("MultiIndex", "__add__", "DataFrame"),
        ),
        (
            "core/arrays/base.pyi",
            "class ExtensionArray:\n    def __add__(self, other: Index, /) -> None: ...",
            ("ExtensionArray", "__add__", "Index"),
        ),
    ],
)
def test_registered_tiers(
    tmp_path: Path, path: str, source: str, key: tuple[str, str, str]
) -> None:
    root = _write_tree(tmp_path, {path: source})
    assert not check_operand_hierarchy(root, exceptions={})
    assert check_operand_hierarchy(root, exceptions={key: 1})


@pytest.mark.parametrize("missing_file", [False, True])
def test_missing_required_file_or_class(
    tmp_path: Path, missing_file: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_tree(tmp_path, {"core/series.pyi": ""})
    if missing_file:
        (root / "core/series.pyi").unlink()
    assert not check_operand_hierarchy(root, exceptions={})
    assert (
        "missing required stub file"
        if missing_file
        else "could not find class 'Series'"
    ) in capsys.readouterr().err


@pytest.mark.parametrize("relative", ["core/indexes/range.pyi", "other.pyi"])
def test_syntax_errors_name_file(tmp_path: Path, relative: str) -> None:
    root = _write_tree(tmp_path, {relative: "class Broken(:"})
    with pytest.raises(SyntaxError) as failure:
        check_operand_hierarchy(root, exceptions={})
    assert failure.value.filename == str(root / relative)


def test_registry_and_documentation() -> None:
    assert (_REPO_ROOT / EXCEPTIONS_FILE).is_file()
    assert set(CANONICAL_OPERAND_BY_TIER) == set(CLASS_TIERS.values()) - {0}
    for tier, operand in CANONICAL_OPERAND_BY_TIER.items():
        assert CLASS_TIERS[operand] == tier
    for class_name, operator, operand in FORWARD_DUNDER_EXCEPTIONS:
        assert class_name
        assert operator in FORWARD_BINARY_DUNDERS
        assert operand in CANONICAL_OPERAND_BY_TIER.values()
    relative, _, anchor = EXCEPTIONS_DOCUMENTATION.partition("#")
    document = _REPO_ROOT / relative
    assert document.is_file(), f"missing contract: {relative}"
    headings = re.findall(
        r"^#{1,6} (.+)$", document.read_text(encoding="utf-8"), flags=re.MULTILINE
    )
    anchors = {
        re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
        for heading in headings
    }
    assert anchor and anchor in anchors
    assert "type-architecture/README.md" in (_REPO_ROOT / "docs/README.md").read_text(
        encoding="utf-8"
    )
    assert "operand-hierarchy.md" in (
        _REPO_ROOT / "docs/type-architecture/README.md"
    ).read_text(encoding="utf-8")


def test_guide_traced_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    guide = (_REPO_ROOT / EXCEPTIONS_DOCUMENTATION.partition("#")[0]).read_text(
        encoding="utf-8"
    )
    section = guide.split("## A rejected signature, traced", 1)[1].split("\n## ", 1)[0]
    source, diagnostic = re.findall(
        r"```(?:python|text)\n(.*?)```", section, flags=re.DOTALL
    )
    root = _write_tree(tmp_path, {"core/series.pyi": source})
    assert not check_operand_hierarchy(root, exceptions={})
    assert capsys.readouterr().err == diagnostic


def test_repository_matches_allowances(capsys: pytest.CaptureFixture[str]) -> None:
    assert main() == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert output.out == (
        "Operand hierarchy invariant holds.\n"
        f"{EXCEPTIONS_FILE}: {len(FORWARD_DUNDER_EXCEPTIONS)} combinations, "
        f"{sum(FORWARD_DUNDER_EXCEPTIONS.values())} occurrences.\n"
    )


def test_cli_failure_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(stub_root: Path) -> bool:
        return False

    monkeypatch.setattr(
        "scripts.operand_hierarchy.check.check_operand_hierarchy", reject
    )
    assert main() == 1
