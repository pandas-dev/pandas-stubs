from __future__ import annotations

from collections import (
    UserDict,
    UserList,
    deque,
)
from collections.abc import (
    Hashable,
    Iterator,
)
import datetime
from enum import Enum
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
import pytest
from typing_extensions import assert_type

from pandas._typing import Scalar

from tests import check


def test_types_getitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    a = np.array(["col1", "col2"])
    check(assert_type(df["col1"], pd.Series), pd.Series)
    check(assert_type(df[5], pd.Series), pd.Series)
    check(assert_type(df[["col1", "col2"]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[1:], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[s], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[a], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[select_df], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[i], pd.DataFrame), pd.DataFrame)


def test_types_getitem_with_hashable() -> None:
    # Testing getitem support for hashable types that are not scalar
    # Due to the bug in https://github.com/pandas-dev/pandas-stubs/issues/592
    class MyEnum(Enum):
        FIRST = "tayyar"
        SECOND = "haydar"

    df = pd.DataFrame(
        data=[[12.2, 10], [8.8, 15]], columns=[MyEnum.FIRST, MyEnum.SECOND]
    )
    check(assert_type(df[MyEnum.FIRST], pd.Series), pd.Series)
    check(assert_type(df[1:], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[:2], pd.DataFrame), pd.DataFrame)

    df2 = pd.DataFrame(data=[[12.2, 10], [8.8, 15]], columns=[3, 4])
    check(assert_type(df2[3], pd.Series), pd.Series)
    check(assert_type(df2[[3]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df2[[3, 4]], pd.DataFrame), pd.DataFrame)


def test_slice_setitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    df[1:] = [10, 11, 12]


def test_types_setitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    h = cast(Hashable, "col1")
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    a = np.array(["col1", "col2"])
    df["col1"] = [1, 2]
    df[5] = [5, 6]
    df[h] = [5, 6]
    df.loc[:, h] = [5, 6]
    df.loc[:, UserList([h])] = [[5], [6]]
    df.loc[:, iter([h])] = [[5], [6]]
    df[["col1", "col2"]] = [[1, 2], [3, 4]]
    df[s] = [5, 6]
    df.loc[:, s] = [5, 6]
    df["col1"] = [5, 6]
    df[df["col1"] > 1] = [5, 6, 7]
    df[a] = [[1, 2], [3, 4]]
    df[i] = [8, 9]

    df["col1"] = [None, pd.NaT]
    # TODO: mypy bug, remove after python/mypy#20420 has been resolved
    df[["col1"]] = [[None], [pd.NA]]  # type: ignore[assignment,list-item]
    df[iter(["col1"])] = [[None], [pd.NA]]  # type: ignore[assignment]


def test_types_setitem_mask() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    df[select_df] = [1, 2, 3]


def test_types_iloc_iat() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df.iloc[1, 1], Scalar), np.integer)
    check(assert_type(df.iloc[[1], [1]], pd.DataFrame), pd.DataFrame)

    check(assert_type(df.iat[0, 0], Scalar), np.integer)

    # https://github.com/microsoft/python-type-stubs/issues/31
    check(assert_type(df.iloc[:, [0]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.iloc[:, 0], pd.Series), pd.Series)


def test_types_loc_at() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df.loc[[0], "col1"], pd.Series), pd.Series)
    check(assert_type(df.loc[0, "col1"], Scalar), np.integer)

    check(assert_type(df.at[0, "col1"], Scalar), np.integer)


def test_types_boolean_indexing() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df[df > 1], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[~(df > 1.0)], pd.DataFrame), pd.DataFrame)

    row_mask = df["col1"] >= 2
    col_mask = df.columns.isin(["col2"])
    check(assert_type(df.loc[row_mask], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[~row_mask], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[row_mask, :], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[:, col_mask], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[row_mask, col_mask], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[~row_mask, ~col_mask], pd.DataFrame), pd.DataFrame)


def test_indexslice_setitem() -> None:
    df = pd.DataFrame(
        {"x": [1, 2, 2, 3], "y": [1, 2, 3, 4], "z": [10, 20, 30, 40]}
    ).set_index(["x", "y"])
    s = pd.Series([-1, -2])
    df.loc[pd.IndexSlice[2, :]] = s.values
    df.loc[pd.IndexSlice[2, :], "z"] = [200, 300]
    # GH 314
    df.loc[pd.IndexSlice[pd.Index([2, 3]), :], "z"] = 99


def test_indexslice_getitem() -> None:
    # GH 300
    df = (
        pd.DataFrame({"x": [1, 2, 2, 3, 4], "y": [10, 20, 30, 40, 10]})
        .assign(z=lambda df: df.x * df.y)
        .set_index(["x", "y"])
    )
    ind = pd.Index([2, 3])
    check(
        assert_type(
            pd.IndexSlice[ind, :], tuple["pd.Index[int]", "slice[None, None, None]"]
        ),
        tuple,
    )
    check(assert_type(df.loc[pd.IndexSlice[ind, :]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[pd.IndexSlice[1:2]], pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.loc[pd.IndexSlice[:, df["z"] > 40], :], pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.loc[pd.IndexSlice[2, 30], "z"], Scalar), np.integer)
    check(
        assert_type(df.loc[pd.IndexSlice[[2, 4], [20, 40]], :], pd.DataFrame),
        pd.DataFrame,
    )
    # GH 314
    check(
        assert_type(df.loc[pd.IndexSlice[pd.Index([2, 4]), :], "z"], pd.Series),
        pd.Series,
    )


def test_getset_untyped() -> None:
    """Test that Dataframe.__getitem__ needs to return untyped series."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    check(assert_type(df["x"].max(), Any), np.integer)


def test_getmultiindex_columns() -> None:
    mi = pd.MultiIndex.from_product([[1, 2], ["a", "b"]])
    df = pd.DataFrame([[1, 2, 3, 4], [10, 20, 30, 40]], columns=mi)
    li: list[tuple[int, str]] = [(1, "a"), (2, "b")]
    check(assert_type(df[[(1, "a"), (2, "b")]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[li], pd.DataFrame), pd.DataFrame)
    check(
        assert_type(
            df[[(i, s) for i in [1] for s in df.columns.get_level_values(1)]],
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(df[[df.columns[0]]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[df.columns[0]], pd.Series), pd.Series)
    check(assert_type(df[li[0]], pd.Series), pd.Series)


def test_frame_isin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    check(assert_type(df.isin([1, 3, 5]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.isin({1, 3, 5}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.isin(pd.Series([1, 3, 5])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.isin(pd.Index([1, 3, 5])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.isin(df), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.isin({"x": [1, 2]}), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.isin(UserDict({"x": iter([1, "2"])})), pd.DataFrame),
        pd.DataFrame,
    )


def test_frame_getitem_isin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    check(assert_type(df[df.index.isin([1, 3, 5])], pd.DataFrame), pd.DataFrame)


def test_columns_mixlist() -> None:
    # GH 97
    df = pd.DataFrame({"a": [1, 2, 3], 1: [3, 4, 5]})
    key: list[int | str]
    key = [1]
    check(assert_type(df[key], pd.DataFrame), pd.DataFrame)


def test_frame_scalars_slice() -> None:
    # GH 133
    # scalars:
    # str, bytes, datetime.date, datetime.datetime, datetime.timedelta, bool, int,
    # float, complex, Timestamp, Timedelta

    str_ = "a"
    bytes_ = b"7"
    date = datetime.date(1999, 12, 31)
    datetime_ = datetime.datetime(1999, 12, 31)
    timedelta = datetime.datetime(2000, 1, 1) - datetime.datetime(1999, 12, 31)
    bool_ = True
    int_ = 2
    float_ = 3.14
    complex_ = 1.0 + 3.0j
    timestamp = pd.Timestamp(0)
    pd_timedelta = pd.Timedelta(0, unit="D")
    none = None
    idx = [
        str_,
        bytes_,
        date,
        datetime_,
        timedelta,
        bool_,
        int_,
        float_,
        complex_,
        timestamp,
        pd_timedelta,
        none,
    ]
    values = np.arange(len(idx))[:, None] + np.arange(len(idx))
    df = pd.DataFrame(values, columns=idx, index=idx)

    # Note: bool_ cannot be tested since the index is object and pandas does not
    # support boolean access using loc except when the index is boolean
    check(assert_type(df.loc[str_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[bytes_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[date], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[datetime_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[timedelta], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[int_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[float_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[complex_], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[timestamp], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[pd_timedelta], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[none], pd.Series), pd.Series)

    check(assert_type(df.loc[:, str_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, bytes_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, date], pd.Series), pd.Series)
    check(assert_type(df.loc[:, datetime_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[:, int_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, float_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, complex_], pd.Series), pd.Series)
    check(assert_type(df.loc[:, timestamp], pd.Series), pd.Series)
    check(assert_type(df.loc[:, pd_timedelta], pd.Series), pd.Series)
    check(assert_type(df.loc[:, none], pd.Series), pd.Series)

    # GH749

    multi_idx = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["alpha", "num"])
    df2 = pd.DataFrame({"col1": range(4)}, index=multi_idx)
    check(assert_type(df2.loc[str_], pd.Series | pd.DataFrame), pd.DataFrame)

    df3 = pd.DataFrame({"x": range(2)}, index=pd.Index(["a", "b"]))
    check(assert_type(df3.loc[str_], pd.Series | pd.DataFrame), pd.Series)

    # https://github.com/microsoft/python-type-stubs/issues/62
    df7 = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))
    index = pd.Index(["b"])
    check(assert_type(df7.loc[index], pd.DataFrame), pd.DataFrame)


def test_boolean_loc() -> None:
    # Booleans can only be used in loc when the index is boolean
    df = pd.DataFrame([[0, 1], [1, 0]], columns=[True, False], index=[True, False])
    check(assert_type(df.loc[True], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[:, False], pd.Series), pd.Series)


def test_setitem_list() -> None:
    # GH 153
    lst1: list[str] = ["a", "b", "c"]
    lst2: list[int] = [1, 2, 3]
    lst3: list[float] = [4.0, 5.0, 6.0]
    lst4: list[tuple[str, int]] = [("a", 1), ("b", 2), ("c", 3)]
    lst5: list[complex] = [0 + 1j, 0 + 2j, 0 + 3j]

    columns: list[Hashable] = [
        "a",
        "b",
        "c",
        1,
        2,
        3,
        4.0,
        5.0,
        6.0,
        ("a", 1),
        ("b", 2),
        ("c", 3),
        0 + 1j,
        0 + 2j,
        0 + 3j,
    ]

    df = pd.DataFrame(np.empty((3, 15)), columns=columns)

    check(assert_type(df.set_index(lst1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst3), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst4), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(lst5), pd.DataFrame), pd.DataFrame)

    iter1: Iterator[str] = (v for v in lst1)
    iter2: Iterator[tuple[str, int]] = (v for v in lst4)
    check(assert_type(df.set_index(iter1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index(iter2), pd.DataFrame), pd.DataFrame)


def test_setitem_loc() -> None:
    # GH 254
    df = pd.DataFrame.from_dict(
        dict.fromkeys(["A", "B", "C"], (True, True, True)), orient="index"
    )
    df.loc[["A", "C"]] = False
    my_arr = ["A", "C"]
    df.loc[my_arr] = False


def test_isetframe() -> None:
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    check(assert_type(frame.isetitem(0, 10), None), type(None))
    check(assert_type(frame.isetitem([0], [10, 12]), None), type(None))


def test_getsetitem_multiindex() -> None:
    # GH 466
    rows = pd.Index(["project A", "project B", "project C"])
    years: tuple[str, ...] = ("Year 1", "Year 2", "Year 3")
    quarters: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
    index_tuples: list[tuple[str, ...]] = list(itertools.product(years, quarters))
    cols = pd.MultiIndex.from_tuples(index_tuples)
    budget = pd.DataFrame(index=rows, columns=cols)
    multi_index: tuple[str, str] = ("Year 1", "Q1")
    budget.loc["project A", multi_index] = 4700
    check(assert_type(budget.loc["project A", multi_index], Scalar), int)


def test_getitem_generator() -> None:
    # GH 685
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(
        assert_type(df[(f"col{i + 1}" for i in range(2))], pd.DataFrame), pd.DataFrame
    )


def test_getitem_dict_keys() -> None:
    # GH 770
    some_columns = {"a": [1], "b": [2]}
    df = pd.DataFrame.from_dict(some_columns)
    check(assert_type(df[some_columns.keys()], pd.DataFrame), pd.DataFrame)


def test_frame_setitem_na() -> None:
    # GH 743
    df = pd.DataFrame(
        {"x": [1, 2, 3], "y": pd.date_range("3/1/2023", "3/3/2023")},
        index=pd.Index(["a", "b", "c"]),
    ).convert_dtypes()

    ind = pd.Index(["a", "c"])

    df.loc[ind, :] = pd.NA
    df.iloc[[0, 2], :] = pd.NA
    df.at["a", "x"] = pd.NA
    df.iat[0, 0] = pd.NA

    # reveal_type(df["y"]) gives Series[Any], so we have to cast to tell the
    # type checker what kind of type it is when adding to a Timedelta
    df["x"] = cast("pd.Series[pd.Timestamp]", df["y"]) + pd.Timedelta(days=3)
    df.loc[ind, :] = pd.NaT
    df.iloc[[0, 2], :] = pd.NaT
    df.at["a", "y"] = pd.NaT
    df.iat[0, 0] = pd.NaT

    df.loc["a", "x"] = None
    df.iloc[2, 0] = None
    df.at["a", "y"] = None
    df.iat[0, 0] = None

    df.loc[:, "x"] = [None, pd.NA, pd.NaT]
    df.iloc[:, 0] = [None, pd.NA, pd.NaT]

    # TODO: mypy bug, remove after python/mypy#20420 has been resolved
    df.loc[:, ["x"]] = [[None], [pd.NA], [pd.NaT]]  # type: ignore[assignment,index]
    df.iloc[:, [0]] = [[None], [pd.NA], [pd.NaT]]  # type: ignore[assignment,index]

    # TODO: mypy bug, remove after python/mypy#20420 has been resolved
    df.loc[:, iter(["x"])] = [[None], [pd.NA], [pd.NaT]]  # type: ignore[assignment,index]
    df.iloc[:, iter([0])] = [[None], [pd.NA], [pd.NaT]]  # type: ignore[assignment,index]


def test_loc_set() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.loc["a"] = [3, 4]


def test_loc_int_set() -> None:
    df = pd.DataFrame({1: [1, 2], 2: [3, 4]})
    df.loc[1] = [3, 4]
    df.loc[np.int_(1)] = pd.Series([1, 2])
    df.loc[np.uint(1)] = pd.Series([1, 2])
    df.loc[np.int8(1)] = pd.Series([1, 2])
    df.loc[np.int32(1)] = [2, 3]
    df.loc[np.uint64(1)] = [2, 3]


@pytest.mark.parametrize("col", [1, None])
@pytest.mark.parametrize("typ", [list, tuple, deque, UserList, iter])
def test_loc_iterable(col: Hashable, typ: type) -> None:
    # GH 189, GH 1410
    df = pd.DataFrame({1: [1, 2], None: 5}, columns=pd.Index([1, None], dtype=object))
    check(df.loc[:, typ([col])], pd.DataFrame)

    if TYPE_CHECKING:
        assert_type(df.loc[:, [None]], pd.DataFrame)
        assert_type(df.loc[:, [1]], pd.DataFrame)

        assert_type(df.loc[:, (None,)], pd.DataFrame)
        assert_type(df.loc[:, (1,)], pd.DataFrame)

        assert_type(df.loc[:, deque([None])], pd.DataFrame)
        assert_type(df.loc[:, deque([1])], pd.DataFrame)

        assert_type(df.loc[:, UserList([None])], pd.DataFrame)
        assert_type(df.loc[:, UserList([1])], pd.DataFrame)

        assert_type(df.loc[:, (None for _ in [0])], pd.DataFrame)
        assert_type(df.loc[:, (1 for _ in [0])], pd.DataFrame)


def test_loc_slice() -> None:
    """Test DataFrame.loc with a slice, Index, Series."""
    # GH277
    df1 = pd.DataFrame(
        {"x": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=["num", "let"]),
    )
    check(assert_type(df1.loc[1, :], pd.Series | pd.DataFrame), pd.DataFrame)
    check(assert_type(df1[::-1], pd.DataFrame), pd.DataFrame)

    # GH1299
    ind = pd.Index(["a", "b"])
    mask = pd.Series([True, False])
    mask_col = pd.Series([True, False], index=pd.Index(["a", "b"]))
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # loc with index for columns
    check(assert_type(df.loc[mask, ind], pd.DataFrame), pd.DataFrame)
    # loc with index for columns
    check(assert_type(df.loc[mask, mask_col], pd.DataFrame), pd.DataFrame)


def test_loc_callable() -> None:
    # GH 256
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def select1(df: pd.DataFrame) -> pd.Series:
        return df["x"] > 2.0

    check(assert_type(df.loc[select1], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[select1, :], pd.DataFrame), pd.DataFrame)

    def select2(df: pd.DataFrame) -> list[Hashable]:
        return [i for i in df.index if cast(int, i) % 2 == 1]

    check(assert_type(df.loc[select2, "x"], pd.Series), pd.Series)

    def select3(_: pd.DataFrame) -> int:
        return 1

    check(assert_type(df.loc[select3, "x"], Scalar), np.integer)

    check(
        assert_type(df.loc[:, lambda df: df.columns.str.startswith("x")], pd.DataFrame),
        pd.DataFrame,
    )


def test_npint_loc_indexer() -> None:
    # GH 508

    df = pd.DataFrame({"x": [1, 2, 3]}, index=np.array([10, 20, 30], dtype="uint64"))

    def get_NDArray(df: pd.DataFrame, key: npt.NDArray[np.uint64]) -> pd.DataFrame:
        return df.loc[key]

    a: npt.NDArray[np.uint64] = np.array([10, 30], dtype="uint64")
    check(assert_type(get_NDArray(df, a), pd.DataFrame), pd.DataFrame)


def test_loc_list_str() -> None:
    # GH 1162 (PR)
    df = pd.DataFrame(
        [[1, 2], [4, 5], [7, 8]],
        index=["cobra", "viper", "sidewinder"],
        columns=["max_speed", "shield"],
    )

    result = df.loc[["viper", "sidewinder"]]
    check(assert_type(result, pd.DataFrame), pd.DataFrame)


def test_loc_returns_series() -> None:
    df1 = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[10, 20, 30, 40])
    df2 = df1.loc[10, :]
    check(assert_type(df2, pd.Series | pd.DataFrame), pd.Series)


def test_frame_single_slice() -> None:
    # GH 572
    df = pd.DataFrame([1, 2, 3])
    check(assert_type(df.loc[:], pd.DataFrame), pd.DataFrame)

    df.loc[:] = 1 + df


def test_frame_index_timestamp() -> None:
    # GH 620
    dt1 = pd.to_datetime("2023-05-01")
    dt2 = pd.to_datetime("2023-05-02")
    s = pd.Series([1, 2], index=[dt1, dt2])
    df = pd.DataFrame(s)
    # Next result is Series or DataFrame because the index could be a MultiIndex
    check(assert_type(df.loc[dt1, :], pd.Series | pd.DataFrame), pd.Series)
    check(assert_type(df.loc[[dt1], :], pd.DataFrame), pd.DataFrame)
    df2 = pd.DataFrame({"x": s})
    check(assert_type(df2.loc[dt1, "x"], Scalar), np.integer)
    check(assert_type(df2.loc[[dt1], "x"], pd.Series), pd.Series, np.integer)


def test_df_loc_dict() -> None:
    """Test that we can set a dict to a df.loc result GH1203."""
    df = pd.DataFrame(columns=["X"])
    df.loc[0] = {"X": 0}
    check(assert_type(df, pd.DataFrame), pd.DataFrame)

    df.iloc[0] = {"X": 0}
    check(assert_type(df, pd.DataFrame), pd.DataFrame)

    df.loc[0] = {None: None, pd.NA: pd.NA, pd.NaT: pd.NaT}
    df.iloc[0] = {None: None, pd.NA: pd.NA, pd.NaT: pd.NaT}


def test_iloc_npint() -> None:
    # GH 69
    df = pd.DataFrame({"a": [10, 20, 30], "b": [20, 40, 60], "c": [30, 60, 90]})
    iloc = np.argmin(np.random.standard_normal(3))
    df.iloc[iloc]


# https://github.com/pandas-dev/pandas-stubs/issues/143
def test_iloc_tuple() -> None:
    df = pd.DataFrame({"Char": ["A", "B", "C"], "Number": [1, 2, 3]})
    df = df.iloc[0:2,]


def test_frame_ndarray_assignmment() -> None:
    # GH 100
    df_a = pd.DataFrame({"a": [0.0] * 10})
    df_a.iloc[:, :] = np.array([[-1.0]] * 10)

    df_b = pd.DataFrame({"a": [0.0] * 10, "b": [1.0] * 10})
    df_b.iloc[:, :] = np.array([[-1.0, np.inf]] * 10)
