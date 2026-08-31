from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_container_hierarchy import (
    FORWARD_DUNDER_EXCEPTIONS,
    HierarchyException,
    check_container_hierarchy,
)


def _write_stub_tree(
    tmp_path: Path,
    *,
    base: str = "",
    index: str = "",
    series: str = "",
) -> Path:
    """Create the smallest stub tree the hierarchy checker requires."""
    stub_root = tmp_path / "pandas-stubs"
    index_file = stub_root / "core" / "indexes" / "base.pyi"
    index_file.parent.mkdir(parents=True)
    index_file.write_text(index, encoding="utf-8")
    (stub_root / "core" / "base.pyi").write_text(base, encoding="utf-8")
    (stub_root / "core" / "series.pyi").write_text(series, encoding="utf-8")
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

    assert check_container_hierarchy(stub_root)


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

    assert not check_container_hierarchy(stub_root)
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

    assert not check_container_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "alias ScalarArrayIndexOperand references Series" in output
    assert "alias ScalarArrayIndexOperand references DataFrame" in output
    assert "alias ScalarArrayIndexSeriesOperand references DataFrame" in output
    assert "Index.__add__ `other` operand references Series" in output
    assert "Index.__add__ `other` operand references DataFrame" in output
    assert "Series.__add__ `other` operand references DataFrame" in output


def test_rejects_non_positional_only_other_parameter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_root = _write_stub_tree(
        tmp_path,
        index="""
class Index:
    def __add__(self, other: int) -> None: ...
""",
        series="""
class Series:
    def __add__(self, other: int, /) -> None: ...
""",
    )

    assert not check_container_hierarchy(stub_root)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Index.__add__ `other` parameter must be positional-only" in captured.err


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

    assert not check_container_hierarchy(stub_root)
    output = capsys.readouterr().err
    assert "Index.__or__ `other` operand references Series" in output
    assert "Series.__lt__ `other` operand references DataFrame" in output


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

    assert not check_container_hierarchy(stub_root, exceptions={})
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

    assert check_container_hierarchy(stub_root)


def test_exception_registry_documents_the_matrix_multiplication_case() -> None:
    expected = {
        ("Series", "__matmul__", "DataFrame"): HierarchyException(
            rationale="Series matrix multiplication with a DataFrame returns a Series.",
            documentation=(
                "docs/type-architecture/container-hierarchy.md#matrix-multiplication"
            ),
        ),
    }

    assert FORWARD_DUNDER_EXCEPTIONS == expected
    document = Path("docs/type-architecture/container-hierarchy.md").read_text(
        encoding="utf-8"
    )
    assert "Series.__matmul__(DataFrame)" in document
