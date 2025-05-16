from __future__ import annotations

from collections import defaultdict
from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
import csv
import datetime
from enum import Enum
import io
import itertools
from pathlib import Path
import re
import string
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import Timestamp
from pandas.api.typing import NAType
from pandas.core.resample import (
    DatetimeIndexResampler,
    Resampler,
)
import pytest
from typing_extensions import (
    TypeAlias,
    assert_never,
    assert_type,
)
import xarray as xr

from pandas._typing import Scalar

from tests import (
    PD_LTE_22,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    ensure_clean,
    pytest_warns_bounded,
)

from pandas.io.formats.format import EngFormatter
from pandas.io.formats.style import Styler
from pandas.io.parsers import TextFileReader

if TYPE_CHECKING:
    from pandas.core.frame import _PandasNamedTuple
    from pandas.core.series import TimestampSeries
else:
    _PandasNamedTuple: TypeAlias = tuple
    TimestampSeries: TypeAlias = pd.Series

DF = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})


def getCols(k) -> str:
    return string.ascii_uppercase[:k]


def makeStringIndex(k: int = 10) -> pd.Index:
    return pd.Index(rands_array(nchars=10, size=k), name=None)


def rands_array(nchars, size: int) -> np.ndarray:
    chars = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))
    retval = (
        np.random.default_rng(2)
        .choice(chars, size=nchars * np.prod(size), replace=True)
        .view((np.str_, nchars))
        .reshape(size)
    )
    return retval.astype("O")


def getSeriesData() -> dict[str, pd.Series]:
    _N = 30
    _K = 4

    index = makeStringIndex(_N)
    return {
        c: pd.Series(np.random.default_rng(i).standard_normal(_N), index=index)
        for i, c in enumerate(getCols(_K))
    }


def test_types_init() -> None:
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}, index=[2, 1])
    pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]])
    pd.DataFrame(data=itertools.repeat([1, 2, 3], 3))
    pd.DataFrame(data=(range(i) for i in range(5)))
    pd.DataFrame(data=[1, 2, 3, 4], dtype=np.int8)
    pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        columns=["a", "b", "c"],
        dtype=np.int8,
        copy=True,
    )
    check(
        assert_type(pd.DataFrame(0, index=[0, 1], columns=[0, 1]), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_all() -> None:
    df = pd.DataFrame([[False, True], [False, False]], columns=["col1", "col2"])
    check(assert_type(df.all(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(df.all(axis=None), np.bool_), np.bool_)


def test_types_any() -> None:
    df = pd.DataFrame([[False, True], [False, False]], columns=["col1", "col2"])
    check(assert_type(df.any(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(df.any(axis=None), np.bool_), np.bool_)


def test_types_append() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df2 = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
    if TYPE_CHECKING_INVALID_USAGE:
        res1: pd.DataFrame = df.append(df2)  # type: ignore[operator] # pyright: ignore[reportCallIssue]
        res2: pd.DataFrame = df.append([1, 2, 3])  # type: ignore[operator] # pyright: ignore[reportCallIssue]
        res3: pd.DataFrame = df.append([[1, 2, 3]])  # type: ignore[operator] # pyright: ignore[reportCallIssue]
        res4: pd.DataFrame = df.append(  # type: ignore[operator] # pyright: ignore[reportCallIssue]
            {("a", 1): [1, 2, 3], "b": df2}, ignore_index=True
        )
        res5: pd.DataFrame = df.append(  # type: ignore[operator] # pyright: ignore[reportCallIssue]
            {1: [1, 2, 3]}, ignore_index=True
        )
        res6: pd.DataFrame = df.append(  # type: ignore[operator] # pyright: ignore[reportCallIssue]
            {1: [1, 2, 3], "col2": [1, 2, 3]}, ignore_index=True
        )
        res7: pd.DataFrame = df.append(  # type: ignore[operator] # pyright: ignore[reportCallIssue]
            pd.Series([5, 6]), ignore_index=True
        )
        res8: pd.DataFrame = df.append(  # type: ignore[operator] # pyright: ignore[reportCallIssue]
            pd.Series([5, 6], index=["col1", "col2"]), ignore_index=True
        )


def test_types_to_csv() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df.to_csv(), str), str)

    with ensure_clean() as path:
        df.to_csv(path)
        check(assert_type(pd.read_csv(path), pd.DataFrame), pd.DataFrame)

    with ensure_clean() as path:
        df.to_csv(Path(path))
        check(assert_type(pd.read_csv(Path(path)), pd.DataFrame), pd.DataFrame)

    # This keyword was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    with ensure_clean() as path:
        df.to_csv(path, errors="replace")
        check(assert_type(pd.read_csv(path), pd.DataFrame), pd.DataFrame)

    # Testing support for binary file handles, added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.to_csv(io.BytesIO(), encoding="utf-8", compression="gzip")

    # Testing support for binary file handles, added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.to_csv(io.BytesIO(), quoting=csv.QUOTE_ALL, encoding="utf-8", compression="gzip")

    if sys.version_info >= (3, 12):
        with ensure_clean() as path:
            df.to_csv(path, quoting=csv.QUOTE_STRINGS)

        with ensure_clean() as path:
            df.to_csv(path, quoting=csv.QUOTE_NOTNULL)


def test_types_to_csv_when_path_passed() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    with ensure_clean() as file:
        path = Path(file)
        df.to_csv(path)
        check(assert_type(pd.read_csv(path), pd.DataFrame), pd.DataFrame)


def test_types_copy() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    check(assert_type(df.copy(), pd.DataFrame), pd.DataFrame)


def test_types_getitem() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    a = np.array(["col1", "col2"])
    df["col1"]
    df[5]
    df[["col1", "col2"]]
    df[1:]
    df[s]
    df[a]
    df[select_df]
    df[i]


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
    i = pd.Index(["col1", "col2"])
    s = pd.Series(["col1", "col2"])
    a = np.array(["col1", "col2"])
    df["col1"] = [1, 2]
    df[5] = [5, 6]
    df[["col1", "col2"]] = [[1, 2], [3, 4]]
    df[s] = [5, 6]
    df[a] = [[1, 2], [3, 4]]
    df[i] = [8, 9]


def test_types_setitem_mask() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], 5: [6, 7]})
    select_df = pd.DataFrame({"col1": [True, True], "col2": [False, True]})
    df[select_df] = [1, 2, 3]


def test_types_iloc_iat() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.iloc[1, 1]
    df.iloc[[1], [1]]
    df.iat[0, 0]


def test_types_loc_at() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.loc[[0], "col1"]
    df.at[0, "col1"]
    df.loc[0, "col1"]


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


def test_types_df_to_df_comparison() -> None:
    df = pd.DataFrame(data={"col1": [1, 2]})
    df2 = pd.DataFrame(data={"col1": [3, 2]})
    check(assert_type(df > df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df >= df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df < df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df <= df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df == df2, pd.DataFrame), pd.DataFrame)


def test_types_head_tail() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.head(1)
    df.tail(1)


def test_types_assign() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.assign(col3=lambda frame: frame.sum(axis=1))
    df["col3"] = df.sum(axis=1)


def test_assign() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], 1: [4, 5, 6]})

    my_unnamed_func = lambda df: df["a"] * 2

    def my_named_func_1(df: pd.DataFrame) -> pd.Series[str]:
        return df["a"]

    def my_named_func_2(df: pd.DataFrame) -> pd.Series[Any]:
        return df["a"]

    check(assert_type(df.assign(c=lambda df: df["a"] * 2), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.assign(c=lambda df: df["a"].index), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.assign(c=lambda df: df["a"].to_numpy()), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.assign(c=lambda df: df["a"].max()), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.assign(c=df["a"] * 2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=df["a"].index), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=df["a"].to_numpy()), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=my_unnamed_func), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=my_named_func_1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.assign(c=my_named_func_2), pd.DataFrame), pd.DataFrame)


def test_types_sample() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    # GH 67
    check(assert_type(df.sample(frac=0.5), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sample(n=1), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.sample(n=1, random_state=np.random.default_rng()), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_nlargest_nsmallest() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.nlargest(1, "col1")
    df.nsmallest(1, "col2")


def test_types_filter() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.filter(items=["col1"])
    df.filter(regex="co.*")
    df.filter(like="1")
    # [PR 964] Docs state `items` is `list-like`
    df.filter(items=("col2", "col2", 1, tuple([4])))


def test_types_setting() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df["col1"] = 1
    df[df == 1] = 7


def test_types_drop() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(df.drop("col1", axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(columns=["col1"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(columns=pd.Index(["col1"])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(columns={"col1"}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(columns=iter(["col1"])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop([0]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(index=[0]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(index=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(labels=0), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.drop([0, 0], inplace=True), None) is None
    to_drop: list[str] = ["col1"]
    check(assert_type(df.drop(columns=to_drop), pd.DataFrame), pd.DataFrame)
    # GH 302
    check(assert_type(df.drop(pd.Index([1])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(index=pd.Index([1])), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop(columns=pd.Index(["col1"])), pd.DataFrame), pd.DataFrame)


def test_arguments_drop() -> None:
    # GH 950
    if TYPE_CHECKING_INVALID_USAGE:
        df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
        res1 = df.drop()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
        res2 = df.drop([0], columns=["col1"])  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        res3 = df.drop([0], index=[0])  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        # These should also fail, but `None` is Hasheable and i do not know how
        # to type hint a non-None hashable.
        # res4 = df.drop(columns=None)
        # res5 = df.drop(index=None)
        # res6 = df.drop(None)


def test_types_dropna() -> None:
    df = pd.DataFrame(data={"col1": [np.nan, np.nan], "col2": [3, np.nan]})
    check(assert_type(df.dropna(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.dropna(ignore_index=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.dropna(axis=1, thresh=1), pd.DataFrame), pd.DataFrame)
    assert (
        assert_type(df.dropna(axis=0, how="all", subset=["col1"], inplace=True), None)
        is None
    )
    assert (
        assert_type(
            df.dropna(
                axis=0, how="all", subset=["col1"], inplace=True, ignore_index=False
            ),
            None,
        )
        is None
    )


def test_types_drop_duplicates() -> None:
    # GH#59237
    df = pd.DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )

    check(assert_type(df.drop_duplicates(["AAA"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop_duplicates(("AAA",)), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.drop_duplicates("AAA"), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.drop_duplicates("AAA", inplace=True), None) is None
    check(
        assert_type(
            df.drop_duplicates("AAA", inplace=False, ignore_index=True), pd.DataFrame
        ),
        pd.DataFrame,
    )

    if not PD_LTE_22:
        check(assert_type(df.drop_duplicates({"AAA"}), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(df.drop_duplicates({"AAA": None}), pd.DataFrame), pd.DataFrame
        )


def test_types_fillna() -> None:
    df = pd.DataFrame(data={"col1": [np.nan, np.nan], "col2": [3, np.nan]})
    check(assert_type(df.fillna(0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.fillna(0, axis=1, inplace=True), None), type(None))


def test_types_sort_index() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=[5, 1, 3, 2])
    df2 = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=["a", "b", "c", "d"])
    check(assert_type(df.sort_index(), pd.DataFrame), pd.DataFrame)
    level1 = (1, 2)
    check(
        assert_type(df.sort_index(ascending=False, level=level1), pd.DataFrame),
        pd.DataFrame,
    )
    level2: list[str] = ["a", "b", "c"]
    check(assert_type(df2.sort_index(level=level2), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.sort_index(ascending=False, level=3), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(df.sort_index(kind="mergesort", inplace=True), None), type(None))


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_index_with_key() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4]}, index=["a", "b", "C", "d"])
    check(
        assert_type(df.sort_index(key=lambda k: k.str.lower()), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_set_index() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]}, index=[5, 1, 3, 2]
    )
    check(assert_type(df.set_index("col1"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index("col1", drop=False), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index("col1", append=True), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.set_index("col1", verify_integrity=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.set_index(["col1", "col2"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.set_index("col1", inplace=True), None), type(None))
    # GH 140
    check(
        assert_type(df.set_index(pd.Index(["w", "x", "y", "z"])), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_query() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4], "col2": [3, 0, 1, 7]})
    check(assert_type(df.query("col1 > col2"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.query("col1 % col2 == 0", inplace=True), None), type(None))


def test_types_eval() -> None:
    df = pd.DataFrame(data={"col1": [1, 2, 3, 4], "col2": [3, 0, 1, 7]})
    check(assert_type(df.eval("E = col1 > col2", inplace=True), None), type(None))
    check(assert_type(df.eval("C = col1 % col2 == 0", inplace=True), None), type(None))
    check(
        assert_type(
            df.eval("E = col1 > col2"), Scalar | np.ndarray | pd.DataFrame | pd.Series
        ),
        pd.DataFrame,
    )


def test_types_sort_values() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(assert_type(df.sort_values("col1"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.sort_values("col1", ascending=False, inplace=True), None),
        type(None),
    )
    check(
        assert_type(
            df.sort_values(by=["col1", "col2"], ascending=[True, False]), pd.DataFrame
        ),
        pd.DataFrame,
    )


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_values_with_key() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(
        assert_type(df.sort_values(by="col1", key=lambda k: -k), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_shift() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1], "col2": [3, 4]}, index=pd.date_range("2020", periods=2)
    )
    check(assert_type(df.shift(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.shift(1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.shift(-1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.shift(freq="1D"), pd.DataFrame), pd.DataFrame)


def test_types_rank() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.rank(axis=0, na_option="bottom")
    df.rank(method="min", pct=True)
    df.rank(method="dense", ascending=True)
    df.rank(method="first", numeric_only=True)


def test_types_mean() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(assert_type(df.mean(), pd.Series), pd.Series)
    check(assert_type(df.mean(axis=0), pd.Series), pd.Series)
    check(assert_type(df.groupby(level=0).mean(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.mean(axis=1, skipna=True, numeric_only=False), pd.Series),
        pd.Series,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        df3: pd.DataFrame = df.groupby(axis=1, level=0).mean()  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
        df4: pd.DataFrame = df.groupby(axis=1, level=0, dropna=True).mean()  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]


def test_types_median() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(assert_type(df.median(), pd.Series), pd.Series)
    check(assert_type(df.median(axis=0), pd.Series), pd.Series)
    check(assert_type(df.groupby(level=0).median(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.median(axis=1, skipna=True, numeric_only=False), pd.Series),
        pd.Series,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        df3: pd.DataFrame = df.groupby(axis=1, level=0).median()  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]
        df4: pd.DataFrame = df.groupby(axis=1, level=0, dropna=True).median()  # type: ignore[call-overload] # pyright: ignore[reportArgumentType, reportCallIssue]


def test_types_iterrows() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(
        assert_type(df.iterrows(), "Iterable[tuple[Hashable, pd.Series]]"),
        Iterable,
        tuple,
    )


def test_types_itertuples() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    check(
        assert_type(df.itertuples(), Iterable[_PandasNamedTuple]),
        Iterable,
        _PandasNamedTuple,
    )
    check(
        assert_type(
            df.itertuples(index=False, name="Foobar"), Iterable[_PandasNamedTuple]
        ),
        Iterable,
        _PandasNamedTuple,
    )
    check(
        assert_type(df.itertuples(index=False, name=None), Iterable[tuple[Any, ...]]),
        Iterable,
        object,
    )


def test_types_sum() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.sum()
    df.sum(axis=1)


def test_types_cumsum() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.cumsum()
    df.sum(axis=0)


def test_types_min() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.min()
    df.min(axis=0)


def test_types_max() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.max()
    df.max(axis=0)


def test_types_quantile() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.quantile([0.25, 0.5])
    df.quantile(0.75)
    df.quantile()
    # GH 81
    df.quantile(np.array([0.25, 0.75]))


def test_dataframe_clip() -> None:
    """Test different clipping combinations for dataframe."""
    df = pd.DataFrame(data={"col1": [20, 12], "col2": [3, 14]})
    if TYPE_CHECKING_INVALID_USAGE:
        df.clip(lower=pd.Series([4, 5]), upper=None, axis=None)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        df.clip(lower=None, upper=pd.Series([4, 5]), axis=None)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        df.clip(lower=pd.Series([1, 2]), upper=pd.Series([4, 5]), axis=None)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        df.copy().clip(lower=pd.Series([1, 2]), upper=None, axis=None, inplace=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        df.copy().clip(lower=None, upper=pd.Series([1, 2]), axis=None, inplace=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        df.copy().clip(lower=pd.Series([4, 5]), upper=pd.Series([1, 2]), axis=None, inplace=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]

    check(
        assert_type(df.clip(lower=None, upper=None, axis=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, upper=None, axis=None), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.clip(lower=None, upper=15, axis=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.clip(lower=None, upper=None, axis=None, inplace=True), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, upper=None, axis=None, inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, upper=15, axis=None, inplace=True), None),
        type(None),
    )

    check(
        assert_type(df.clip(lower=None, upper=None, axis=0), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(df.clip(lower=5, upper=None, axis=0), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(lower=None, upper=15, axis=0), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.clip(lower=pd.Series([1, 2]), upper=None, axis=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=None, upper=pd.Series([1, 2]), axis=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.clip(lower=None, upper=None, axis="index", inplace=True), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, upper=None, axis="index", inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, upper=15, axis="index", inplace=True), None),
        type(None),
    )
    check(
        assert_type(
            df.clip(lower=pd.Series([1, 2]), upper=None, axis="index", inplace=True),
            None,
        ),
        type(None),
    )
    check(
        assert_type(
            df.clip(lower=None, upper=pd.Series([1, 2]), axis="index", inplace=True),
            None,
        ),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, upper=None, axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, upper=None, axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=None, upper=15, axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.clip(lower=pd.Series([1, 2]), upper=None, axis="index"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.clip(lower=None, upper=pd.Series([1, 2]), axis="index"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.clip(lower=None, upper=None, axis=0, inplace=True), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, upper=None, axis=0, inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, upper=15, axis=0, inplace=True), None),
        type(None),
    )

    # without lower
    check(assert_type(df.clip(upper=None, axis=None), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.clip(upper=15, axis=None), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(upper=None, axis=None, inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.clip(upper=15, axis=None, inplace=True), None), type(None))

    check(assert_type(df.clip(upper=None, axis=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.clip(upper=15, axis=0), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(upper=pd.Series([1, 2]), axis=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(upper=None, axis="index", inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(upper=15, axis="index", inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(upper=pd.Series([1, 2]), axis="index", inplace=True), None),
        type(None),
    )
    check(assert_type(df.clip(upper=None, axis="index"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.clip(upper=15, axis="index"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(upper=pd.Series([1, 2]), axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(upper=None, axis=0, inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.clip(upper=15, axis=0, inplace=True), None), type(None))

    # without upper
    check(
        assert_type(df.clip(lower=None, axis=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.clip(lower=5, axis=None), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(lower=None, axis=None, inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, axis=None, inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, axis=None, inplace=True), pd.DataFrame),
        pd.DataFrame,
    )

    check(assert_type(df.clip(lower=None, axis=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.clip(lower=5, axis=0), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.clip(lower=pd.Series([1, 2]), axis=0), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=None, axis="index", inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, axis="index", inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=pd.Series([1, 2]), axis="index", inplace=True), None),
        type(None),
    )
    check(
        assert_type(df.clip(lower=None, axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=5, axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=pd.Series([1, 2]), axis="index"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.clip(lower=None, axis=0, inplace=True), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.clip(lower=5, axis=0, inplace=True), None), type(None))


def test_types_abs() -> None:
    df = pd.DataFrame(data={"col1": [-5, 1], "col2": [3, -14]})
    df.abs()


def test_types_var() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [1, 4]})
    df.var()
    df.var(axis=1, ddof=1)
    df.var(skipna=True, numeric_only=False)


def test_types_std() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [1, 4]})
    df.std()
    df.std(axis=1, ddof=1)
    df.std(skipna=True, numeric_only=False)


def test_types_idxmin() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.idxmin()
    df.idxmin(axis=0)


def test_types_idxmax() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.idxmax()
    df.idxmax(axis=0)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_value_counts() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 4]})
    check(assert_type(df.value_counts(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(df.value_counts(normalize=True), "pd.Series[float]"),
        pd.Series,
        float,
    )


def test_types_unique() -> None:
    # This is really more for of a Series test
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [1, 4]})
    df["col1"].unique()


def test_types_apply() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})

    def returns_scalar(_: pd.Series) -> int:
        return 2

    def returns_scalar_na(x: pd.Series) -> int | NAType:
        return 2 if (x < 5).all() else pd.NA

    def returns_series(x: pd.Series) -> pd.Series:
        return x**2

    def returns_listlike_of_2(_: pd.Series) -> tuple[int, int]:
        return (7, 8)

    def returns_listlike_of_3(_: pd.Series) -> tuple[int, int, int]:
        return (7, 8, 9)

    def returns_dict(_: pd.Series) -> dict[str, int]:
        return {"col4": 7, "col5": 8}

    # Misc checks
    check(assert_type(df.apply(np.exp), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.apply(str), "pd.Series[str]"), pd.Series, str)

    # GH 393
    def gethead(s: pd.Series, y: int) -> pd.Series:
        return s.head(y)

    check(assert_type(df.apply(gethead, args=(4,)), pd.DataFrame), pd.DataFrame)

    # Check various return types for default result_type (None) with default axis (0)
    check(
        assert_type(df.apply(returns_scalar), "pd.Series[int]"), pd.Series, np.integer
    )
    check(
        assert_type(df.apply(returns_scalar_na), "pd.Series[int]"),
        pd.Series,
        int,
    )
    check(assert_type(df.apply(returns_series), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.apply(returns_listlike_of_3), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.apply(returns_dict), pd.Series), pd.Series)

    # Check various return types for result_type="expand" with default axis (0)
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "expand" to a scalar return
        assert_type(df.apply(returns_scalar, result_type="expand"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(df.apply(returns_series, result_type="expand"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.apply(returns_listlike_of_3, result_type="expand"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.apply(returns_dict, result_type="expand"), pd.DataFrame),
        pd.DataFrame,
    )

    # Check various return types for result_type="reduce" with default axis (0)
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "reduce" to a scalar return
        assert_type(df.apply(returns_scalar, result_type="reduce"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "reduce" to a series return
        assert_type(df.apply(returns_series, result_type="reduce"), pd.Series),
        pd.Series,  # This technically returns a pd.Series[pd.Series], but typing does not support that
    )
    check(
        assert_type(df.apply(returns_listlike_of_3, result_type="reduce"), pd.Series),
        pd.Series,
    )
    check(
        assert_type(df.apply(returns_dict, result_type="reduce"), pd.Series), pd.Series
    )

    # Check various return types for default result_type (None) with axis=1
    check(
        assert_type(df.apply(returns_scalar, axis=1), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(assert_type(df.apply(returns_series, axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.apply(returns_listlike_of_3, axis=1), pd.Series), pd.Series)
    check(assert_type(df.apply(returns_dict, axis=1), pd.Series), pd.Series)

    # Check various return types for result_type="expand" with axis=1
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "expand" to a scalar return
        assert_type(
            df.apply(returns_scalar, axis=1, result_type="expand"), "pd.Series[int]"
        ),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.apply(returns_series, axis=1, result_type="expand"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.apply(returns_listlike_of_3, axis=1, result_type="expand"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.apply(returns_dict, axis=1, result_type="expand"), pd.DataFrame),
        pd.DataFrame,
    )

    # Check various return types for result_type="reduce" with axis=1
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "reduce" to a scalar return
        assert_type(
            df.apply(returns_scalar, axis=1, result_type="reduce"), "pd.Series[int]"
        ),
        pd.Series,
        np.integer,
    )
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "reduce" to a series return
        assert_type(
            df.apply(returns_series, axis=1, result_type="reduce"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.apply(returns_listlike_of_3, axis=1, result_type="reduce"), pd.Series
        ),
        pd.Series,
    )
    check(
        assert_type(df.apply(returns_dict, axis=1, result_type="reduce"), pd.Series),
        pd.Series,
    )

    # Check various return types for result_type="broadcast" with axis=0 and axis=1
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "broadcast" to a scalar return
        assert_type(df.apply(returns_scalar, result_type="broadcast"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.apply(returns_series, result_type="broadcast"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        # Can only broadcast a list-like of 2 elements, not 3, because there are 2 rows
        assert_type(
            df.apply(returns_listlike_of_2, result_type="broadcast"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        # Note that technically it does not make sense
        # to pass a result_type of "broadcast" to a scalar return
        assert_type(
            df.apply(returns_scalar, axis=1, result_type="broadcast"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.apply(returns_series, axis=1, result_type="broadcast"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            # Can only broadcast a list-like of 3 elements, not 2,
            # as there are 3 columns
            df.apply(returns_listlike_of_3, axis=1, result_type="broadcast"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    # Since dicts will be assigned to elements of np.ndarray inside broadcasting,
    # we need to use a DataFrame with object dtype to make the assignment possible.
    df2 = pd.DataFrame({"col1": ["a", "b"], "col2": ["c", "d"]})
    check(
        assert_type(df2.apply(returns_dict, result_type="broadcast"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df2.apply(returns_dict, axis=1, result_type="broadcast"), pd.DataFrame
        ),
        pd.DataFrame,
    )

    # Test various other positional/keyword argument combinations
    # to ensure all overloads are supported
    check(
        assert_type(df.apply(returns_scalar, axis=0), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.apply(returns_scalar, axis=0, result_type=None), "pd.Series[int]"
        ),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(df.apply(returns_scalar, 0, False, None), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.apply(returns_scalar, 0, False, result_type=None), "pd.Series[int]"
        ),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.apply(returns_scalar, 0, raw=False, result_type=None), "pd.Series[int]"
        ),
        pd.Series,
        np.integer,
    )


def test_types_map() -> None:
    # GH774
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df.map(lambda x: x**2)
    df.map(np.exp)
    df.map(str)
    # na_action parameter was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.map(np.exp, na_action="ignore")
    df.map(str, na_action=None)


def test_types_element_wise_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})
    df2 = pd.DataFrame(data={"col1": [10, 20], "col3": [3, 4]})

    check(assert_type(df + df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df - df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sub(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df * df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mul(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df / df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div(df2, fill_value=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df / [2, 2], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div([2, 2], fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df // df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv(df2, fill_value=0), pd.DataFrame), pd.DataFrame)
    check(assert_type(df // [2, 2], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv([2, 2], fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df % df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mod(df2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df2**df, pd.DataFrame), pd.DataFrame)
    check(assert_type(df2.pow(df, fill_value=0), pd.DataFrame), pd.DataFrame)

    # divmod operation was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    check(
        assert_type(divmod(df, df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )
    check(
        assert_type(df.__divmod__(df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )
    check(
        assert_type(df.__rdivmod__(df2), tuple[pd.DataFrame, pd.DataFrame]),
        tuple,
        pd.DataFrame,
    )


def test_types_scalar_arithmetic() -> None:
    df = pd.DataFrame(data={"col1": [2, 1], "col2": [3, 4]})

    check(assert_type(df + 1, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add(1, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df - 1, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.sub(1, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df * 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mul(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df / 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.div(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df // 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.floordiv(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df % 2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.mod(2, fill_value=0), pd.DataFrame), pd.DataFrame)

    check(assert_type(df**2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df**0, pd.DataFrame), pd.DataFrame)
    check(assert_type(df**0.213, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.pow(0.5), pd.DataFrame), pd.DataFrame)


def test_types_melt() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    df.melt()
    df.melt(id_vars=["col1"], value_vars=["col2"])
    df.melt(
        id_vars=["col1"],
        value_vars=["col2"],
        var_name="someVariable",
        value_name="someValue",
    )

    pd.melt(df)
    pd.melt(df, id_vars=["col1"], value_vars=["col2"])
    pd.melt(
        df,
        id_vars=["col1"],
        value_vars=["col2"],
        var_name="someVariable",
        value_name="someValue",
    )


def test_types_pivot() -> None:
    df = pd.DataFrame(
        data={
            "col1": ["first", "second", "third", "fourth"],
            "col2": [50, 70, 56, 111],
            "col3": ["A", "B", "B", "A"],
            "col4": [100, 102, 500, 600],
        }
    )
    check(
        assert_type(
            df.pivot(index="col1", columns="col3", values="col2"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.pivot(index="col1", columns="col3"), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            df.pivot(index="col1", columns="col3", values=["col2", "col4"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(assert_type(df.pivot(columns="col3"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.pivot(columns="col3", values="col2"), pd.DataFrame), pd.DataFrame
    )


def test_types_pivot_table() -> None:
    df = pd.DataFrame(
        data={
            "col1": ["first", "second", "third", "fourth"],
            "col2": [50, 70, 56, 111],
            "col3": ["A", "B", "C", "D"],
            "col4": [100, 102, 500, 600],
        }
    )
    check(
        assert_type(
            df.pivot_table(index="col1", columns="col3", values=["col2", "col4"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_pivot_table_sort():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})

    check(
        assert_type(
            df.pivot_table(values="a", index="b", columns="c", sort=True), pd.DataFrame
        ),
        pd.DataFrame,
    )


def test_types_groupby_as_index() -> None:
    """Test type of groupby.size method depending on `as_index`."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(
            df.groupby("a", as_index=False).size(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby("a", as_index=True).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )
    check(
        assert_type(
            df.groupby("a").size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_as_index_list() -> None:
    """Test type of groupby.size method depending on list of grouper GH1045."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 2]})
    check(
        assert_type(
            df.groupby(["a", "b"], as_index=False).size(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby(["a", "b"], as_index=True).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )
    check(
        assert_type(
            df.groupby(["a", "b"]).size(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_as_index_value_counts() -> None:
    """Test type of groupby.value_counts method depending on `as_index`."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(
            df.groupby("a", as_index=False).value_counts(),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby("a", as_index=True).value_counts(),
            "pd.Series[int]",
        ),
        pd.Series,
    )


def test_types_groupby_size() -> None:
    """Test for GH886."""
    data = [
        {"date": "2023-12-01", "val": 12},
        {"date": "2023-12-02", "val": 2},
        {"date": "2023-12-03", "val": 1},
        {"date": "2023-12-03", "val": 10},
    ]

    df = pd.DataFrame(data)
    groupby = df.groupby("date")
    size = groupby.size()
    frame = size.to_frame()
    check(assert_type(frame.reset_index(), pd.DataFrame), pd.DataFrame)


def test_types_groupby() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]})
    df.index.name = "ind"
    df.groupby(by="col1")
    df.groupby(level="ind")
    df.groupby(by="col1", sort=False, as_index=True)
    df.groupby(by=["col1", "col2"])
    # GH 284
    df.groupby(df["col1"] > 2)
    df.groupby([df["col1"] > 2, df["col2"] % 2 == 1])
    df.groupby(lambda x: x)
    df.groupby([lambda x: x % 2, lambda x: x % 3])
    df.groupby(np.array([1, 0, 1]))
    df.groupby([np.array([1, 0, 0]), np.array([0, 0, 1])])
    df.groupby({1: 1, 2: 2, 3: 3})
    df.groupby([{1: 1, 2: 1, 3: 2}, {1: 1, 2: 2, 3: 2}])
    df.groupby(df.index)
    df.groupby([pd.Index([1, 0, 0]), pd.Index([0, 0, 1])])
    df.groupby(pd.Grouper(level=0))
    df.groupby([pd.Grouper(level=0), pd.Grouper(key="col1")])

    check(assert_type(df.groupby(by="col1").agg("sum"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby(level="ind").aggregate("sum"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.groupby(by="col1", sort=False, as_index=True).transform(
                lambda x: x.max()
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.groupby(by=["col1", "col2"]).count(), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(
            df.groupby(by=["col1", "col2"]).filter(lambda x: x["col1"] > 0),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.groupby(by=["col1", "col2"]).nunique(), pd.DataFrame),
        pd.DataFrame,
    )
    with pytest_warns_bounded(
        FutureWarning,
        "(The provided callable <built-in function sum> is currently using|The behavior of DataFrame.sum with)",
        upper="2.2.99",
    ):
        with pytest_warns_bounded(
            DeprecationWarning,
            "DataFrameGroupBy.apply operated on the grouping columns",
            upper="2.2.99",
        ):
            if PD_LTE_22:
                check(
                    assert_type(df.groupby(by="col1").apply(sum), pd.DataFrame),
                    pd.DataFrame,
                )
    check(assert_type(df.groupby("col1").transform("sum"), pd.DataFrame), pd.DataFrame)
    s1 = df.set_index("col1")["col2"]
    check(assert_type(s1, pd.Series), pd.Series)
    check(assert_type(s1.groupby("col1").transform("sum"), pd.Series), pd.Series)


def test_types_groupby_methods() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]})
    check(assert_type(df.groupby("col1").sum(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").prod(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").sample(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").count(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1").value_counts(normalize=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(
            df.groupby("col1").value_counts(subset=None, normalize=True),
            "pd.Series[float]",
        ),
        pd.Series,
        float,
    )
    check(assert_type(df.groupby("col1").idxmax(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").idxmin(), pd.DataFrame), pd.DataFrame)


def test_types_groupby_agg() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0], 0: [-1, -1, -1]}
    )
    check(assert_type(df.groupby("col1").agg("min"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1").agg(["min", "max"]), pd.DataFrame), pd.DataFrame
    )
    agg_dict1 = {"col2": "min", "col3": "max", 0: "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict1), pd.DataFrame), pd.DataFrame)

    def wrapped_min(x: Any) -> Any:
        return x.min()

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(df.groupby("col1")["col3"].agg(min), pd.Series), pd.Series)
        check(
            assert_type(df.groupby("col1")["col3"].agg([min, max]), pd.DataFrame),
            pd.DataFrame,
        )
        check(assert_type(df.groupby("col1").agg(min), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(df.groupby("col1").agg([min, max]), pd.DataFrame), pd.DataFrame
        )
        agg_dict2 = {"col2": min, "col3": max, 0: min}
        check(
            assert_type(df.groupby("col1").agg(agg_dict2), pd.DataFrame), pd.DataFrame
        )

        # Here, MyPy infers dict[object, object], so it must be explicitly annotated
        agg_dict3: dict[str | int, str | Callable] = {
            "col2": min,
            "col3": "max",
            0: wrapped_min,
        }
        check(
            assert_type(df.groupby("col1").agg(agg_dict3), pd.DataFrame), pd.DataFrame
        )
    agg_dict4 = {"col2": "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict4), pd.DataFrame), pd.DataFrame)
    agg_dict5 = {0: "sum"}
    check(assert_type(df.groupby("col1").agg(agg_dict5), pd.DataFrame), pd.DataFrame)
    named_agg = pd.NamedAgg(column="col2", aggfunc="max")
    check(
        assert_type(df.groupby("col1").agg(new_col=named_agg), pd.DataFrame),
        pd.DataFrame,
    )
    # GH#187
    cols: list[str] = ["col1", "col2"]
    check(assert_type(df.groupby(by=cols).sum(), pd.DataFrame), pd.DataFrame)

    cols_opt: list[str | None] = ["col1", "col2"]
    check(assert_type(df.groupby(by=cols_opt).sum(), pd.DataFrame), pd.DataFrame)

    cols_mixed: list[str | int] = ["col1", 0]
    check(assert_type(df.groupby(by=cols_mixed).sum(), pd.DataFrame), pd.DataFrame)
    # GH 736
    check(assert_type(df.groupby(by="col1").aggregate("size"), pd.Series), pd.Series)
    check(assert_type(df.groupby(by="col1").agg("size"), pd.Series), pd.Series)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_group_by_with_dropna_keyword() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]}
    )
    df.groupby(by="col2", dropna=True).sum()
    df.groupby(by="col2", dropna=False).sum()
    df.groupby(by="col2").sum()


def test_types_groupby_any() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, 1, 2],
            "col2": [True, False, False],
            "col3": [False, False, False],
        }
    )
    check(assert_type(df.groupby("col1").any(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby("col1").all(), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type(df.groupby("col1")["col2"].any(), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_types_groupby_iter() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    series_groupby = pd.Series([True, True, False], dtype=bool)
    first_group = next(iter(df.groupby(series_groupby)))
    check(
        assert_type(first_group[0], bool),
        bool,
    )
    check(
        assert_type(first_group[1], pd.DataFrame),
        pd.DataFrame,
    )


def test_types_groupby_level() -> None:
    # GH 836
    data = {
        "col1": [0, 0, 0],
        "col2": [0, 1, 0],
        "col3": [1, 2, 3],
        "col4": [1, 2, 3],
    }
    df = pd.DataFrame(data=data).set_index(["col1", "col2", "col3"])
    check(
        assert_type(df.groupby(level=["col1", "col2"]).sum(), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_merge() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df2 = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [0, 1, 0]})
    df.merge(df2)
    df.merge(df2, on="col1")
    df.merge(df2, on="col1", how="left")
    df.merge(df2, on=["col1", "col2"], how="left")
    df.merge(df2, on=("col1", "col2"), how="left")
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=(None, "s"))
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=("t", "s"))
    df.merge(df2, on=("col1", "col2"), how="left", suffixes=("a", None))
    df.merge(df2, how="cross")  # GH 289
    columns = ["col1", "col2"]
    df.merge(df2, on=columns)


def test_types_plot() -> None:
    if TYPE_CHECKING:  # skip pytest
        df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
        df.plot.hist()
        df.plot.scatter(x="col2", y="col1")


def test_types_window() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.expanding()
    if TYPE_CHECKING_INVALID_USAGE:
        df.expanding(axis=1)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        df.rolling(2, axis=1, center=True)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
        df.expanding(axis=1, center=True)  # type: ignore[arg-type, call-arg] # pyright: ignore[reportCallIssue]

    df.rolling(2)

    check(
        assert_type(df.rolling(2).agg("max"), pd.DataFrame),
        pd.DataFrame,
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.2.99",
    ):
        check(
            assert_type(df.rolling(2).agg(max), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(df.rolling(2).agg([max, min]), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(df.rolling(2).agg({"col2": max}), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(df.rolling(2).agg({"col2": [max, min]}), pd.DataFrame),
            pd.DataFrame,
        )

    check(
        assert_type(df.rolling(2).agg(["max", "min"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.rolling(2).agg({"col2": "max"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.rolling(2).agg({"col2": ["max", "min"]}), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_cov() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    df.cov()
    df.cov(min_periods=1)
    # ddof param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.cov(ddof=2)


def test_types_to_numpy() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    check(assert_type(df.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(df.to_numpy(dtype="str", copy=True), np.ndarray), np.ndarray)
    # na_value param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    check(assert_type(df.to_numpy(na_value=0), np.ndarray), np.ndarray)

    df = pd.DataFrame(data={"col1": [1, 1, 2]}, dtype=np.complex128)
    check(assert_type(df.to_numpy(na_value=0), np.ndarray), np.ndarray)
    check(assert_type(df.to_numpy(na_value=np.int32(4)), np.ndarray), np.ndarray)
    check(assert_type(df.to_numpy(na_value=np.float16(3.68)), np.ndarray), np.ndarray)
    check(
        assert_type(df.to_numpy(na_value=np.complex128(3.8, -493.2)), np.ndarray),
        np.ndarray,
    )


def test_to_markdown() -> None:
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    check(assert_type(df.to_markdown(), str), str)
    check(assert_type(df.to_markdown(None), str), str)
    check(assert_type(df.to_markdown(buf=None, mode="wt"), str), str)
    # index param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    check(assert_type(df.to_markdown(index=False), str), str)
    with ensure_clean() as path:
        check(assert_type(df.to_markdown(path), None), type(None))
    with ensure_clean() as path:
        check(assert_type(df.to_markdown(Path(path)), None), type(None))
    sio = io.StringIO()
    check(assert_type(df.to_markdown(sio), None), type(None))


def test_types_to_feather() -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(data={"col1": [1, 1, 2], "col2": [3, 4, 5]})
    with ensure_clean() as path:
        df.to_feather(path)
        # kwargs for pyarrow.feather.write_feather added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
        df.to_feather(path, compression="zstd", compression_level=3, chunksize=2)

        # to_feather has been able to accept a buffer since pandas 1.0.0
        # See https://pandas.pydata.org/docs/whatsnew/v1.0.0.html
        # Docstring and type were updated in 1.2.0.
        # https://github.com/pandas-dev/pandas/pull/35408
        with open(path, mode="wb") as file:
            df.to_feather(file)


# compare() method added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_compare() -> None:
    df1 = pd.DataFrame(
        data={"col1": [1, 1, 2, 1], "col2": [2, None, 1, 2], "col3": [3, 4, 3, 2]}
    )
    df2 = pd.DataFrame(
        data={"col1": [1, 2, 5, 6], "col2": [3, 4, 1, 1], "col3": [3, 4, 3, 2]}
    )
    df1.compare(df2)
    df2.compare(df1, align_axis=0, keep_shape=True, keep_equal=True)


def test_types_agg() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    check(assert_type(df.agg("min"), pd.Series), pd.Series)
    check(assert_type(df.agg(["min", "max"]), pd.DataFrame), pd.DataFrame)

    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <(built-in function (min|max|mean)|function mean at 0x\w+)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(df.agg(min), pd.Series), pd.Series)
        check(assert_type(df.agg([min, max]), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(
                df.agg(x=("A", max), y=("B", "min"), z=("C", np.mean)), pd.DataFrame
            ),
            pd.DataFrame,
        )
    check(
        assert_type(df.agg({"A": ["min", "max"], "B": "min"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.agg({"A": ["mean"]}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.agg("mean", axis=1), pd.Series), pd.Series)
    check(assert_type(df.agg({"A": "mean"}), pd.Series), pd.Series)
    check(assert_type(df.agg({"A": "mean", "B": "sum"}), pd.Series), pd.Series)


def test_types_aggregate() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    check(assert_type(df.aggregate("min"), pd.Series), pd.Series)
    check(assert_type(df.aggregate(["min", "max"]), pd.DataFrame), pd.DataFrame)
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(df.aggregate(min), pd.Series), pd.Series)
        check(assert_type(df.aggregate([min, max]), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(df.aggregate({"A": [min, max], "B": min}), pd.DataFrame),
            pd.DataFrame,
        )

    check(
        assert_type(df.aggregate({"A": ["min", "max"], "B": "min"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.aggregate({"A": ["mean"]}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.aggregate("mean", axis=1), pd.Series), pd.Series)
    check(assert_type(df.aggregate({"A": "mean"}), pd.Series), pd.Series)
    check(assert_type(df.aggregate({"A": "mean", "B": "sum"}), pd.Series), pd.Series)


def test_types_transform() -> None:
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["A", "B", "C"])
    check(assert_type(df.transform("abs"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.transform(abs), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.transform(["abs", "sqrt"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.transform([abs, np.sqrt]), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.transform({"A": ["abs", "sqrt"], "B": "abs"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.transform({"A": [abs, np.sqrt], "B": abs}), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_describe() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, 2, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.describe()
    df.describe(percentiles=[0.5], include="all")
    df.describe(exclude=[np.number])


def test_types_to_string() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, None, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.to_string(
        index=True,
        col_space=2,
        header=["a", "b"],
        na_rep="0",
        justify="left",
        max_rows=2,
        min_rows=0,
        max_cols=2,
        show_dimensions=True,
        line_width=3,
    )
    # col_space accepting list or dict added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_string(col_space=[1, 2])
    df.to_string(col_space={"col1": 1, "col2": 3})


def test_dataframe_to_string_float_fmt() -> None:
    """Test the different argument types for float_format."""
    df = pd.DataFrame(
        {"values": [2.304, 1.1, 3487392, 13.4732894237, 14.3, 18.0, 17.434, 19.3]}
    )
    check(assert_type(df.to_string(), str), str)

    def _formatter(x) -> str:
        return f"{x:.2f}"

    check(assert_type(df.to_string(float_format=_formatter), str), str)
    check(
        assert_type(
            df.to_string(float_format=EngFormatter(accuracy=2, use_eng_prefix=False)),
            str,
        ),
        str,
    )
    check(assert_type(df.to_string(float_format="%.2f"), str), str)


def test_types_to_html() -> None:
    df = pd.DataFrame(
        data={
            "col1": [1, None, -4],
            "col2": [
                np.datetime64("2000-01-01"),
                np.datetime64("2010-01-01"),
                np.datetime64("2010-01-01"),
            ],
        }
    )
    df.to_html(
        index=True,
        col_space=2,
        header=True,
        na_rep="0",
        justify="left",
        max_rows=2,
        max_cols=2,
        show_dimensions=True,
    )
    # col_space accepting list or dict added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    df.to_html(col_space=[1, 2])
    df.to_html(col_space={"col1": 1, "col2": 3})


def test_types_resample() -> None:
    df = pd.DataFrame({"values": [2, 11, 3, 13, 14, 18, 17, 19]})
    df["date"] = pd.date_range("01/01/2018", periods=8, freq="W")
    with pytest_warns_bounded(
        FutureWarning,
        "'M' is deprecated",
        lower="2.1.99",
        upper="2.2.99",
        upper_exception=ValueError,
    ):
        df.resample("M", on="date")
    df.resample("20min", origin="epoch", offset=pd.Timedelta(2, "minutes"), on="date")
    df.resample("20min", origin="epoch", offset=datetime.timedelta(2), on="date")
    df.resample(pd.Timedelta(20, "minutes"), origin="epoch", on="date")
    df.resample(datetime.timedelta(minutes=20), origin="epoch", on="date")


def test_types_to_dict() -> None:
    data = pd.DataFrame({"a": [1], "b": [2]})
    data.to_dict(orient="records")
    data.to_dict(orient="dict")
    data.to_dict(orient="list")
    data.to_dict(orient="series")
    data.to_dict(orient="split")
    data.to_dict(orient="index")

    # orient param accepting "tight" added in 1.4.0 https://pandas.pydata.org/docs/whatsnew/v1.4.0.html
    data.to_dict(orient="tight")


def test_types_from_dict() -> None:
    check(
        assert_type(
            pd.DataFrame.from_dict(
                {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict({1: [3, 2, 1, 0], 2: ["a", "b", "c", "d"]}),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict({"a": {1: 2}, "b": {3: 4, 1: 4}}, orient="index"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict({"a": {"row1": 2}, "b": {"row2": 4, "row1": 4}}),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict({"a": (1, 2, 3), "b": (2, 4, 5)}), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict(
                data={"col_1": {"a": 1}, "col_2": {"a": 1, "b": 2}}, orient="columns"
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    # orient param accepting "tight" added in 1.4.0 https://pandas.pydata.org/docs/whatsnew/v1.4.0.html
    check(
        assert_type(
            pd.DataFrame.from_dict(
                data={
                    "index": [("a", "b"), ("a", "c")],
                    "columns": [("x", 1), ("y", 2)],
                    "data": [[1, 3], [2, 4]],
                    "index_names": ["n1", "n2"],
                    "column_names": ["z1", "z2"],
                },
                orient="tight",
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    # added following #896
    data = {"l1": [1, 2, 3], "l2": [4, 5, 6]}
    # testing `dtype`
    check(
        assert_type(
            pd.DataFrame.from_dict(data, orient="index", dtype="float"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict(data, orient="index", dtype=float), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict(data, orient="index", dtype=None), pd.DataFrame
        ),
        pd.DataFrame,
    )
    # testing `columns`
    check(
        assert_type(
            pd.DataFrame.from_dict(data, orient="index", columns=["a", "b", "c"]),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            pd.DataFrame.from_dict(
                data, orient="index", columns=[1.0, 2, datetime.datetime.now()]
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        check(
            assert_type(  # type: ignore[assert-type]
                pd.DataFrame.from_dict(data, orient="columns", columns=["a", "b", "c"]),  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]
                pd.DataFrame,
            ),
            pd.DataFrame,
        )


def test_pipe() -> None:
    def resampler_foo(resampler: Resampler[pd.DataFrame]) -> pd.DataFrame:
        assert isinstance(resampler, Resampler)
        return pd.DataFrame(resampler)

    with pytest_warns_bounded(
        FutureWarning,
        "'M' is deprecated",
        lower="2.1.99",
        upper="2.2.99",
        upper_exception=ValueError,
    ):
        (
            pd.DataFrame(
                {
                    "price": [10, 11, 9, 13, 14, 18, 17, 19],
                    "volume": [50, 60, 40, 100, 50, 100, 40, 50],
                }
            )
            .assign(week_starting=pd.date_range("01/01/2018", periods=8, freq="W"))
            .resample("M", on="week_starting")
            .pipe(resampler_foo)
        )

    val = (
        pd.DataFrame(
            {
                "price": [10, 11, 9, 13, 14, 18, 17, 19],
                "volume": [50, 60, 40, 100, 50, 100, 40, 50],
            }
        )
        .assign(week_starting=pd.date_range("01/01/2018", periods=8, freq="W"))
        .resample("MS", on="week_starting")
        .pipe(resampler_foo)
    )

    def foo(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(df)

    df = pd.DataFrame({"a": [1], "b": [2]})
    check(assert_type(val, pd.DataFrame), pd.DataFrame)

    check(assert_type(df.pipe(foo), pd.DataFrame), pd.DataFrame)

    def bar(val: Styler) -> Styler:
        return val

    check(assert_type(df.style.pipe(bar), Styler), Styler)

    def baz(val: Styler) -> str:
        return val.to_latex()

    check(assert_type(df.style.pipe(baz), str), str)

    def qux(
        df: pd.DataFrame,
        positional_only: int,
        /,
        argument_1: list[float],
        argument_2: str,
        *,
        keyword_only: tuple[int, int],
    ) -> pd.DataFrame:
        return pd.DataFrame(df)

    check(
        assert_type(
            df.pipe(qux, 1, [1.0, 2.0], argument_2="hi", keyword_only=(1, 2)),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        df.pipe(  # pyright: ignore[reportCallIssue]
            qux,
            "a",  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            [1.0, 2.0],
            argument_2="hi",
            keyword_only=(1, 2),
        )
        df.pipe(  # pyright: ignore[reportCallIssue]
            qux,
            1,
            [1.0, "b"],  # type: ignore[list-item] # pyright: ignore[reportArgumentType]
            argument_2="hi",
            keyword_only=(1, 2),
        )
        df.pipe(  # pyright: ignore[reportCallIssue]
            qux,
            1,
            [1.0, 2.0],
            argument_2=11,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
            keyword_only=(1, 2),
        )
        df.pipe(  # pyright: ignore[reportCallIssue]
            qux,
            1,
            [1.0, 2.0],
            argument_2="hi",
            keyword_only=(1,),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        )
        df.pipe(  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
            qux,
            1,
            [1.0, 2.0],
            argument_3="hi",  # pyright: ignore[reportCallIssue]
            keyword_only=(1, 2),
        )
        df.pipe(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
            qux,
            1,
            [1.0, 2.0],
            11,
            (1, 2),  # pyright: ignore[reportCallIssue]
        )
        df.pipe(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
            qux,
            positional_only=1,  # pyright: ignore[reportCallIssue]
            argument_1=[1.0, 2.0],
            argument_2=11,
            keyword_only=(1, 2),
        )

    def dataframe_not_first_arg(_: int, df: pd.DataFrame) -> pd.DataFrame:
        return df

    check(
        assert_type(
            df.pipe(
                (
                    dataframe_not_first_arg,
                    "df",
                ),
                1,
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        df.pipe(  # pyright: ignore[reportCallIssue]
            (
                dataframe_not_first_arg,  # type: ignore[arg-type]
                1,  # pyright: ignore[reportArgumentType]
            ),
            1,
        )
        df.pipe(  # pyright: ignore[reportCallIssue]
            (
                1,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
                "df",
            ),
            1,
        )


# set_flags() method added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
def test_types_set_flags() -> None:
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(
        allows_duplicate_labels=False
    )
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"]).set_flags(
        allows_duplicate_labels=True
    )
    pd.DataFrame([[1, 2], [8, 9]], columns=["A", "A"])


def test_types_to_parquet() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("fastparquet")
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"]).set_flags(
        allows_duplicate_labels=False
    )
    with ensure_clean() as path:
        df.to_parquet(Path(path))
        # to_parquet() returns bytes when no path given since 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
        check(assert_type(df.to_parquet(), bytes), bytes)


def test_types_to_latex() -> None:
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    df.to_latex(
        columns=["A"], label="some_label", caption="some_caption", multirow=True
    )
    df.to_latex(escape=False, decimal=",", column_format="r")
    # position param was added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
    df.to_latex(position="some")
    df.to_latex(caption=("cap1", "cap2"))


def test_types_explode() -> None:
    df = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    check(assert_type(df.explode("A"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.explode("A", ignore_index=False), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.explode("A", ignore_index=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.explode(["A", "B"]), pd.DataFrame), pd.DataFrame)


def test_types_rename() -> None:
    df = pd.DataFrame(columns=["a"])
    col_map = {"a": "b"}
    df.rename(columns=col_map)
    df.rename(columns={"a": "b"})
    df.rename(columns={1: "b"})
    # Apparently all of these calls are accepted by pandas
    df.rename(columns={None: "b"})
    df.rename(columns={"": "b"})
    df.rename(columns={(2, 1): "b"})
    df.rename(columns=lambda s: s.upper())


def test_types_rename_axis() -> None:
    df = pd.DataFrame({"col_name": [1, 2, 3]})
    df.index.name = "a"
    df.columns.name = "b"

    # Rename axes with `mapper` and `axis`
    check(assert_type(df.rename_axis("A"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.rename_axis(["A"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.rename_axis(None), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.rename_axis("B", axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.rename_axis(["B"], axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.rename_axis(None, axis=1), pd.DataFrame), pd.DataFrame)

    # Rename axes with `index` and `columns`
    check(
        assert_type(df.rename_axis(index="A", columns="B"), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.rename_axis(index=["A"], columns=["B"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.rename_axis(index={"a": "A"}, columns={"b": "B"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.rename_axis(
                index=lambda name: name.upper(),
                columns=lambda name: name.upper(),
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.rename_axis(index=None, columns=None), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_eq() -> None:
    df1 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    check(assert_type(df1 == 1, pd.DataFrame), pd.DataFrame)
    df2 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    check(assert_type(df1 == df2, pd.DataFrame), pd.DataFrame)


def test_types_as_type() -> None:
    df1 = pd.DataFrame([[1, 2], [8, 9]], columns=["A", "B"])
    check(assert_type(df1.astype({"A": "int32"}), pd.DataFrame), pd.DataFrame)


def test_types_dot() -> None:
    df1 = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    df2 = pd.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
    s1 = pd.Series([1, 1, 2, 1])
    np_array = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
    check(assert_type(df1 @ df2, pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.dot(df2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1 @ np_array, pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.dot(np_array), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1 @ s1, pd.Series), pd.Series)
    check(assert_type(df1.dot(s1), pd.Series), pd.Series)


def test_types_regressions() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/32
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5, 6]})
    df2: pd.DataFrame = df.astype(int)

    # https://github.com/microsoft/python-type-stubs/issues/38
    check(
        assert_type(pd.DataFrame({"x": [12, 34], "y": [78, 9]}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.sort_values(["x", "y"], ascending=[True, False]), pd.DataFrame),
        pd.DataFrame,
    )

    # https://github.com/microsoft/python-type-stubs/issues/55
    df3 = pd.DataFrame([["a", 1], ["b", 2]], columns=["let", "num"]).set_index("let")
    df4 = df3.reset_index()
    check(assert_type(df4, pd.DataFrame), pd.DataFrame)
    check(assert_type(df4[["num"]], pd.DataFrame), pd.DataFrame)

    # https://github.com/microsoft/python-type-stubs/issues/58
    df1 = pd.DataFrame(columns=["a", "b", "c"])
    df2 = pd.DataFrame(columns=["a", "c"])
    check(assert_type(df1.drop(columns=df2.columns), pd.DataFrame), pd.DataFrame)

    # https://github.com/microsoft/python-type-stubs/issues/60
    df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["let", "num"]).set_index("let")
    s2 = df1["num"]
    check(
        assert_type(pd.merge(s2, df1, left_index=True, right_index=True), pd.DataFrame),
        pd.DataFrame,
    )

    # https://github.com/microsoft/python-type-stubs/issues/62
    df7: pd.DataFrame = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index(["a", "b", "c"]))
    index: pd.Index = pd.Index(["b"])
    check(assert_type(df7.loc[index], pd.DataFrame), pd.DataFrame)

    # https://github.com/microsoft/python-type-stubs/issues/31
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    check(assert_type(df.iloc[:, [0]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.iloc[:, 0], pd.Series), pd.Series)

    df = pd.DataFrame(
        {
            "a_col": list(range(10)),
            "a_nother": list(range(10)),
            "b_col": list(range(10)),
        }
    )
    df.loc[:, lambda df: df.columns.str.startswith("a_")]

    df = df[::-1]

    # https://github.com/microsoft/python-type-stubs/issues/69
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([4, 5, 6])
    df = pd.concat([s1, s2], axis=1)
    ts1 = pd.concat([s1, s2], axis=0)
    ts2 = pd.concat([s1, s2])

    check(assert_type(ts1, pd.Series), pd.Series)
    check(assert_type(ts2, pd.Series), pd.Series)

    # https://github.com/microsoft/python-type-stubs/issues/110
    check(assert_type(pd.Timestamp("2021-01-01"), pd.Timestamp), datetime.date)
    tslist = list(pd.to_datetime(["2022-01-01", "2022-01-02"]))
    check(assert_type(tslist, list[pd.Timestamp]), list, pd.Timestamp)
    sseries = pd.Series(tslist)
    with pytest_warns_bounded(FutureWarning, "'d' is deprecated", lower="2.2.99"):
        sseries + pd.Timedelta(1, "d")

    check(
        assert_type(sseries + pd.Timedelta(1, "D"), TimestampSeries),
        pd.Series,
        Timestamp,
    )

    # https://github.com/microsoft/pylance-release/issues/2133
    with pytest_warns_bounded(
        FutureWarning,
        "'H' is deprecated",
        lower="2.1.99",
        upper="2.2.99",
        upper_exception=ValueError,
    ):
        pd.date_range(start="2021-12-01", periods=24, freq="H")

    dr = pd.date_range(start="2021-12-01", periods=24, freq="h")
    check(assert_type(dr.strftime("%H:%M:%S"), pd.Index), pd.Index, str)

    # https://github.com/microsoft/python-type-stubs/issues/115
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})
    pd.DatetimeIndex(
        data=df["A"],
        tz=None,
        ambiguous="NaT",
        copy=True,
    )


def test_read_csv() -> None:
    with ensure_clean() as path:
        Path(path).write_text("A,B\n1,2")
        check(assert_type(pd.read_csv(path), pd.DataFrame), pd.DataFrame)
        check(
            assert_type(pd.read_csv(path, iterator=False), pd.DataFrame), pd.DataFrame
        )
        check(
            assert_type(
                pd.read_csv(path, iterator=False, chunksize=None), pd.DataFrame
            ),
            pd.DataFrame,
        )

        with check(
            assert_type(pd.read_csv(path, chunksize=1), TextFileReader), TextFileReader
        ):
            pass
        with check(
            assert_type(pd.read_csv(path, iterator=False, chunksize=1), TextFileReader),
            TextFileReader,
        ):
            pass
        with check(
            assert_type(pd.read_csv(path, iterator=True), TextFileReader),
            TextFileReader,
        ):
            pass
        with check(
            assert_type(
                pd.read_csv(path, iterator=True, chunksize=None), TextFileReader
            ),
            TextFileReader,
        ):
            pass
        with check(
            assert_type(pd.read_csv(path, iterator=True, chunksize=1), TextFileReader),
            TextFileReader,
        ):
            pass

        # https://github.com/microsoft/python-type-stubs/issues/118
        check(
            assert_type(pd.read_csv(path, storage_options=None), pd.DataFrame),
            pd.DataFrame,
        )

        # Allow a variety of dict types for the converters parameter
        converters1 = {"A": str, "B": str}
        check(
            assert_type(pd.read_csv(path, converters=converters1), pd.DataFrame),
            pd.DataFrame,
        )
        converters2 = {"A": str, "B": float}
        check(
            assert_type(pd.read_csv(path, converters=converters2), pd.DataFrame),
            pd.DataFrame,
        )
        converters3 = {0: str, 1: str}
        check(
            assert_type(pd.read_csv(path, converters=converters3), pd.DataFrame),
            pd.DataFrame,
        )
        converters4 = {0: str, 1: float}
        check(
            assert_type(pd.read_csv(path, converters=converters4), pd.DataFrame),
            pd.DataFrame,
        )
        converters5: dict[int | str, Callable[[str], Any]] = {
            0: str,
            "A": float,
        }
        check(
            assert_type(pd.read_csv(path, converters=converters5), pd.DataFrame),
            pd.DataFrame,
        )

        class ReadCsvKwargs(TypedDict):
            converters: dict[int, Callable[[str], Any]]

        read_csv_kwargs: ReadCsvKwargs = {"converters": {0: int}}

        check(
            assert_type(pd.read_csv(path, **read_csv_kwargs), pd.DataFrame),
            pd.DataFrame,
        )

        # Check value covariance for various other parameters too (these only accept a str key)
        na_values = {"A": ["1"], "B": ["1"]}
        check(
            assert_type(pd.read_csv(path, na_values=na_values), pd.DataFrame),
            pd.DataFrame,
        )

    # There are several possible inputs for parse_dates
    with ensure_clean() as path:
        Path(path).write_text("Date,Year,Month,Day\n20221125,2022,11,25")
        parse_dates_1 = ["Date"]
        check(
            assert_type(pd.read_csv(path, parse_dates=parse_dates_1), pd.DataFrame),
            pd.DataFrame,
        )
        check(
            assert_type(
                pd.read_csv(path, index_col="Date", parse_dates=True), pd.DataFrame
            ),
            pd.DataFrame,
        )
        if PD_LTE_22:
            parse_dates_2 = {"combined_date": ["Year", "Month", "Day"]}
            with pytest_warns_bounded(
                FutureWarning,
                "Support for nested sequences",
                lower="2.1.99",
            ):
                check(
                    assert_type(
                        pd.read_csv(path, parse_dates=parse_dates_2), pd.DataFrame
                    ),
                    pd.DataFrame,
                )
            parse_dates_3 = {"combined_date": [1, 2, 3]}
            with pytest_warns_bounded(
                FutureWarning, "Support for nested sequences", lower="2.1.99"
            ):
                check(
                    assert_type(
                        pd.read_csv(path, parse_dates=parse_dates_3), pd.DataFrame
                    ),
                    pd.DataFrame,
                )
            # MyPy calls this Dict[str, object] by default which necessitates the explicit annotation (Pyright does not)
            parse_dates_4: dict[str, list[str | int]] = {
                "combined_date": [1, "Month", 3]
            }
            with pytest_warns_bounded(
                FutureWarning, "Support for nested sequences", lower="2.1.99"
            ):
                check(
                    assert_type(
                        pd.read_csv(path, parse_dates=parse_dates_4), pd.DataFrame
                    ),
                    pd.DataFrame,
                )

            parse_dates_6 = [[1, 2, 3]]
            with pytest_warns_bounded(
                FutureWarning,
                "Support for nested sequences",
                lower="2.1.99",
            ):
                check(
                    assert_type(
                        pd.read_csv(path, parse_dates=parse_dates_6), pd.DataFrame
                    ),
                    pd.DataFrame,
                )
        parse_dates_5 = [0]
        check(
            assert_type(pd.read_csv(path, parse_dates=parse_dates_5), pd.DataFrame),
            pd.DataFrame,
        )


def test_groupby_series_methods() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3], "y": [10, 20, 30, 40, 50]})
    gb = df.groupby("x")["y"]
    check(assert_type(gb.describe(), pd.DataFrame), pd.DataFrame)
    gb.count().loc[2]
    gb.pct_change().loc[2]
    gb.bfill().loc[2]
    gb.cummax().loc[2]
    gb.cummin().loc[2]
    gb.cumprod().loc[2]
    gb.cumsum().loc[2]
    gb.ffill().loc[2]
    gb.first().loc[2]
    gb.head().loc[2]
    gb.last().loc[2]
    gb.max().loc[2]
    gb.mean().loc[2]
    gb.median().loc[2]
    gb.min().loc[2]
    gb.nlargest().loc[2]
    gb.nsmallest().loc[2]
    gb.nth(0).loc[1]


def test_dataframe_pct_change() -> None:
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3], "y": [10, 20, 30, 40, 50]})
    check(assert_type(df.pct_change(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.pct_change(fill_method=None), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.pct_change(axis="columns", periods=-1), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.pct_change(fill_value=0), pd.DataFrame), pd.DataFrame)


def test_indexslice_setitem():
    df = pd.DataFrame(
        {"x": [1, 2, 2, 3], "y": [1, 2, 3, 4], "z": [10, 20, 30, 40]}
    ).set_index(["x", "y"])
    s = pd.Series([-1, -2])
    df.loc[pd.IndexSlice[2, :]] = s.values
    df.loc[pd.IndexSlice[2, :], "z"] = [200, 300]
    # GH 314
    df.loc[pd.IndexSlice[pd.Index([2, 3]), :], "z"] = 99


def test_indexslice_getitem():
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


def test_compute_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    s: pd.Series = pd.Series([10, 20, 30, 40])
    check(assert_type(df["x"] + s.values, pd.Series), pd.Series, np.int64)


# https://github.com/microsoft/python-type-stubs/issues/164
def test_sum_get_add() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    s = df["x"]
    check(assert_type(s, pd.Series), pd.Series)
    summer = df.sum(axis=1)
    check(assert_type(summer, pd.Series), pd.Series)

    check(assert_type(s + summer, pd.Series), pd.Series)
    check(assert_type(s + df["y"], pd.Series), pd.Series)
    check(assert_type(summer + summer, pd.Series), pd.Series)


def test_getset_untyped() -> None:
    result: int = 10
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    # Tests that Dataframe.__getitem__ needs to return untyped series.
    result = df["x"].max()


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


def test_frame_getitem_isin() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    check(assert_type(df[df.index.isin([1, 3, 5])], pd.DataFrame), pd.DataFrame)


def test_to_excel() -> None:
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    with ensure_clean() as path:
        df.to_excel(path, engine="openpyxl")
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
    with ensure_clean() as path:
        df.to_excel(Path(path), engine="openpyxl")
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
    with ensure_clean() as path:
        df.to_excel(path, engine="openpyxl", startrow=1, startcol=1, header=False)
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
    with ensure_clean() as path:
        df.to_excel(path, engine="openpyxl", sheet_name="sheet", index=False)
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
    with ensure_clean() as path:
        df.to_excel(path, engine="openpyxl", header=["x", "y"])
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)
    with ensure_clean() as path:
        df.to_excel(path, engine="openpyxl", columns=["col1"])
        check(assert_type(pd.read_excel(path), pd.DataFrame), pd.DataFrame)


def test_join() -> None:
    float_frame = pd.DataFrame(getSeriesData())
    # GH 29
    left = float_frame["A"].to_frame()
    seriesB = float_frame["B"]
    frameCD = float_frame[["C", "D"]]
    right: list[pd.Series | pd.DataFrame] = [seriesB, frameCD]
    check(assert_type(left.join(right), pd.DataFrame), pd.DataFrame)
    check(assert_type(left.join(right, validate="1:1"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(left.join(right, validate="one_to_one"), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(left.join(right, validate="1:m"), pd.DataFrame), pd.DataFrame)


def test_types_join() -> None:
    df1 = pd.DataFrame({"A": [1, 2], "B": ["test", "test"]})
    df2 = pd.DataFrame({"C": [2, 3], "D": ["test", "test"]})
    check(assert_type(df1.join(df2, how="cross"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.join(df2, how="inner"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.join(df2, how="outer"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.join(df2, how="left"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df1.join(df2, how="right"), pd.DataFrame), pd.DataFrame)


def test_types_ffill() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.ffill(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.ffill(inplace=False), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.ffill(inplace=False, limit_area="inside"), pd.DataFrame),
        pd.DataFrame,
    )
    assert assert_type(df.ffill(inplace=True), None) is None
    assert assert_type(df.ffill(inplace=True, limit_area="outside"), None) is None


def test_types_bfill() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.bfill(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.bfill(inplace=False), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.bfill(inplace=False, limit_area="inside"), pd.DataFrame),
        pd.DataFrame,
    )
    assert assert_type(df.bfill(inplace=True), None) is None
    assert assert_type(df.bfill(inplace=True, limit_area="outside"), None) is None


def test_types_replace() -> None:
    # GH 44
    df = pd.DataFrame([[1, 2, 3]])
    check(assert_type(df.replace(1, 2), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(1, 2, inplace=False), pd.DataFrame), pd.DataFrame)
    assert assert_type(df.replace(1, 2, inplace=True), None) is None


def test_dataframe_replace() -> None:
    df = pd.DataFrame({"col1": ["a", "ab", "ba"], "col2": [0, 1, 2]})
    pattern = re.compile(r"^a.*")
    replace_dict_scalar = {0: 1}
    replace_dict_per_column = {"col2": {0: 1}}
    check(assert_type(df.replace("a", "x"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(pattern, "x"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace("a", "x", regex=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(pattern, "x"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(regex="a", value="x"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(regex=pattern, value="x"), pd.DataFrame), pd.DataFrame)

    check(assert_type(df.replace(["a"], ["x"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace([pattern], ["x"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(regex=["a"], value=["x"]), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.replace(regex=[pattern], value=["x"]), pd.DataFrame),
        pd.DataFrame,
    )

    check(assert_type(df.replace({"a": "x"}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(replace_dict_scalar), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace({pattern: "x"}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(pd.Series({"a": "x"})), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(regex={"a": "x"}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(regex={pattern: "x"}), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.replace(regex=pd.Series({"a": "x"})), pd.DataFrame), pd.DataFrame
    )

    check(
        assert_type(df.replace({"col1": "a"}, {"col1": "x"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.replace({"col1": pattern}, {"col1": "x"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(pd.Series({"col1": "a"}), pd.Series({"col1": "x"})), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df.replace(regex={"col1": "a"}, value={"col1": "x"}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(regex={"col1": pattern}, value={"col1": "x"}), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(regex=pd.Series({"col1": "a"}), value=pd.Series({"col1": "x"})),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(
        assert_type(df.replace({"col1": ["a"]}, {"col1": ["x"]}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.replace({"col1": [pattern]}, {"col1": ["x"]}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(pd.Series({"col1": ["a"]}), pd.Series({"col1": ["x"]})),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(regex={"col1": ["a"]}, value={"col1": ["x"]}), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(regex={"col1": [pattern]}, value={"col1": ["x"]}), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.replace(
                regex=pd.Series({"col1": ["a"]}), value=pd.Series({"col1": ["x"]})
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )

    check(assert_type(df.replace({"col1": {"a": "x"}}), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace(replace_dict_per_column), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.replace({"col1": {pattern: "x"}}), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.replace({"col1": pd.Series({"a": "x"})}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.replace(regex={"col1": {"a": "x"}}), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(df.replace(regex={"col1": {pattern: "x"}}), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.replace(regex={"col1": pd.Series({"a": "x"})}), pd.DataFrame),
        pd.DataFrame,
    )


def test_loop_dataframe() -> None:
    # GH 70
    df = pd.DataFrame({"x": [1, 2, 3]})
    for c in df:
        check(assert_type(df[c], pd.Series), pd.Series)


def test_groupby_index() -> None:
    # GH 42
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0]}
    ).set_index("col1")
    check(assert_type(df.groupby(df.index).min(), pd.DataFrame), pd.DataFrame)


def test_iloc_npint() -> None:
    # GH 69
    df = pd.DataFrame({"a": [10, 20, 30], "b": [20, 40, 60], "c": [30, 60, 90]})
    iloc = np.argmin(np.random.standard_normal(3))
    df.iloc[iloc]


# https://github.com/pandas-dev/pandas-stubs/issues/143
def test_iloc_tuple() -> None:
    df = pd.DataFrame({"Char": ["A", "B", "C"], "Number": [1, 2, 3]})
    df = df.iloc[0:2,]


def test_set_columns() -> None:
    # GH 73
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.0, 1, 1]})
    # Next lines should work, but it is a mypy bug
    # https://github.com/python/mypy/issues/3004
    # pyright accepts this, so we only type check for pyright,
    # and also test the code with pytest
    df.columns = ["c", "d"]  # type: ignore[assignment]
    df.columns = [1, 2]  # type: ignore[assignment]
    df.columns = [1, "a"]  # type: ignore[assignment]
    df.columns = np.array([1, 2])  # type: ignore[assignment]
    df.columns = pd.Series([1, 2])  # type: ignore[assignment]
    df.columns = np.array([1, "a"])  # type: ignore[assignment]
    df.columns = pd.Series([1, "a"])  # type: ignore[assignment]
    df.columns = (1, 2)  # type: ignore[assignment]
    df.columns = (1, "a")  # type: ignore[assignment]
    if TYPE_CHECKING_INVALID_USAGE:
        df.columns = "abc"  # type: ignore[assignment] # pyright: ignore[reportAttributeAccessIssue]


def test_frame_index_numpy() -> None:
    # GH 80
    i = np.array([1.0, 2.0])
    pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"], index=i)


def test_frame_stack() -> None:
    multicol2 = pd.MultiIndex.from_tuples([("weight", "kg"), ("height", "m")])
    df_multi_level_cols2 = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=["cat", "dog"], columns=multicol2
    )

    with pytest_warns_bounded(
        FutureWarning,
        "The previous implementation of stack is deprecated",
        upper="2.2.99",
    ):
        check(
            assert_type(
                df_multi_level_cols2.stack(0), Union[pd.DataFrame, "pd.Series[Any]"]
            ),
            pd.DataFrame,
        )
        check(
            assert_type(
                df_multi_level_cols2.stack([0, 1]),
                Union[pd.DataFrame, "pd.Series[Any]"],
            ),
            pd.Series,
        )
        if PD_LTE_22:
            check(
                assert_type(
                    df_multi_level_cols2.stack(0, future_stack=False),
                    Union[pd.DataFrame, "pd.Series[Any]"],
                ),
                pd.DataFrame,
            )
            check(
                assert_type(
                    df_multi_level_cols2.stack(0, dropna=True, sort=True),
                    Union[pd.DataFrame, "pd.Series[Any]"],
                ),
                pd.DataFrame,
            )


def test_frame_reindex() -> None:
    # GH 84
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    df.reindex([2, 1, 0])


def test_frame_reindex_like() -> None:
    # GH 84
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
    other = pd.DataFrame({"a": [1, 2]}, index=[1, 0])

    with pytest_warns_bounded(
        FutureWarning,
        "the 'method' keyword is deprecated and will be removed in a future version. Please take steps to stop the use of 'method'",
        lower="2.2.99",
        upper="3.0.99",
    ):
        check(
            assert_type(
                df.reindex_like(other, method="nearest", tolerance=[0.5, 0.2]),
                pd.DataFrame,
            ),
            pd.DataFrame,
        )


def test_frame_ndarray_assignmment() -> None:
    # GH 100
    df_a = pd.DataFrame({"a": [0.0] * 10})
    df_a.iloc[:, :] = np.array([[-1.0]] * 10)

    df_b = pd.DataFrame({"a": [0.0] * 10, "b": [1.0] * 10})
    df_b.iloc[:, :] = np.array([[-1.0, np.inf]] * 10)


def test_not_hashable() -> None:
    # GH 113
    assert assert_type(pd.DataFrame.__hash__, None) is None
    assert assert_type(pd.DataFrame().__hash__, None) is None
    assert assert_type(pd.Series.__hash__, None) is None
    assert assert_type(pd.Series([], dtype=object).__hash__, None) is None
    assert assert_type(pd.Index.__hash__, None) is None
    assert assert_type(pd.Index([]).__hash__, None) is None

    def test_func(_: Hashable):
        pass

    if TYPE_CHECKING_INVALID_USAGE:
        test_func(pd.DataFrame())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        test_func(pd.Series([], dtype=object))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        test_func(pd.Index([]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


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
    check(assert_type(df.loc[str_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[bytes_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[date], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[datetime_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[timedelta], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[int_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[float_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[complex_], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[timestamp], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[pd_timedelta], Union[pd.Series, pd.DataFrame]), pd.Series)
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
    check(assert_type(df2.loc[str_], Union[pd.Series, pd.DataFrame]), pd.DataFrame)

    df3 = pd.DataFrame({"x": range(2)}, index=pd.Index(["a", "b"]))
    check(assert_type(df3.loc[str_], Union[pd.Series, pd.DataFrame]), pd.Series)


def test_boolean_loc() -> None:
    # Booleans can only be used in loc when the index is boolean
    df = pd.DataFrame([[0, 1], [1, 0]], columns=[True, False], index=[True, False])
    check(assert_type(df.loc[True], Union[pd.Series, pd.DataFrame]), pd.Series)
    check(assert_type(df.loc[:, False], pd.Series), pd.Series)


def test_groupby_result() -> None:
    # GH 142
    df = pd.DataFrame({"a": [0, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    iterator = df.groupby(["a", "b"]).__iter__()
    assert_type(iterator, Iterator[tuple[tuple, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[tuple, pd.DataFrame])

    if PD_LTE_22:
        check(assert_type(index, tuple), tuple, np.integer)
    else:
        check(assert_type(index, tuple), tuple, int)

    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    iterator2 = df.groupby("a").__iter__()
    assert_type(iterator2, Iterator[tuple[Scalar, pd.DataFrame]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[Scalar, pd.DataFrame])

    check(assert_type(index2, Scalar), int)
    check(assert_type(value2, pd.DataFrame), pd.DataFrame)

    # GH 674
    # grouping by pd.MultiIndex should always resolve to a tuple as well
    multi_index = pd.MultiIndex.from_frame(df[["a", "b"]])
    iterator3 = df.groupby(multi_index).__iter__()
    assert_type(iterator3, Iterator[tuple[tuple, pd.DataFrame]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[tuple, pd.DataFrame])

    check(assert_type(index3, tuple), tuple, int)
    check(assert_type(value3, pd.DataFrame), pd.DataFrame)

    # Want to make sure these cases are differentiated
    for (k1, k2), g in df.groupby(["a", "b"]):
        pass

    for kk, g in df.groupby("a"):
        pass

    for (k1, k2), g in df.groupby(multi_index):
        pass


def test_groupby_result_for_scalar_indexes() -> None:
    # GH 674
    dates = pd.date_range("2020-01-01", "2020-12-31")
    df = pd.DataFrame({"date": dates, "days": 1})
    period_index = pd.PeriodIndex(df.date, freq="M")
    iterator = df.groupby(period_index).__iter__()
    assert_type(iterator, Iterator[tuple[pd.Period, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[pd.Period, pd.DataFrame])

    check(assert_type(index, pd.Period), pd.Period)
    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    dt_index = pd.DatetimeIndex(dates)
    iterator2 = df.groupby(dt_index).__iter__()
    assert_type(iterator2, Iterator[tuple[pd.Timestamp, pd.DataFrame]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[pd.Timestamp, pd.DataFrame])

    check(assert_type(index2, pd.Timestamp), pd.Timestamp)
    check(assert_type(value2, pd.DataFrame), pd.DataFrame)

    tdelta_index = pd.TimedeltaIndex(dates - pd.Timestamp("2020-01-01"))
    iterator3 = df.groupby(tdelta_index).__iter__()
    assert_type(iterator3, Iterator[tuple[pd.Timedelta, pd.DataFrame]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[pd.Timedelta, pd.DataFrame])

    check(assert_type(index3, pd.Timedelta), pd.Timedelta)
    check(assert_type(value3, pd.DataFrame), pd.DataFrame)

    intervals: list[pd.Interval[pd.Timestamp]] = [
        pd.Interval(date, date + pd.DateOffset(days=1), closed="left") for date in dates
    ]
    interval_index = pd.IntervalIndex(intervals)
    assert_type(interval_index, "pd.IntervalIndex[pd.Interval[pd.Timestamp]]")
    iterator4 = df.groupby(interval_index).__iter__()
    assert_type(iterator4, Iterator[tuple["pd.Interval[pd.Timestamp]", pd.DataFrame]])
    index4, value4 = next(iterator4)
    assert_type((index4, value4), tuple["pd.Interval[pd.Timestamp]", pd.DataFrame])

    check(assert_type(index4, "pd.Interval[pd.Timestamp]"), pd.Interval)
    check(assert_type(value4, pd.DataFrame), pd.DataFrame)

    for p, g in df.groupby(period_index):
        pass

    for dt, g in df.groupby(dt_index):
        pass

    for tdelta, g in df.groupby(tdelta_index):
        pass

    for interval, g in df.groupby(interval_index):
        pass


def test_groupby_result_for_ambiguous_indexes() -> None:
    # GH 674
    df = pd.DataFrame({"a": [0, 1, 2], "b": [4, 5, 6], "c": [7, 8, 9]})
    # this will use pd.Index which is ambiguous
    iterator = df.groupby(df.index).__iter__()
    assert_type(iterator, Iterator[tuple[Any, pd.DataFrame]])
    index, value = next(iterator)
    assert_type((index, value), tuple[Any, pd.DataFrame])

    check(assert_type(index, Any), int)
    check(assert_type(value, pd.DataFrame), pd.DataFrame)

    # categorical indexes are also ambiguous

    # https://github.com/pandas-dev/pandas/issues/54054 needs to be fixed
    with pytest_warns_bounded(
        FutureWarning,
        "The default of observed=False is deprecated",
        upper="2.2.99",
    ):
        categorical_index = pd.CategoricalIndex(df.a)
        iterator2 = df.groupby(categorical_index).__iter__()
        assert_type(iterator2, Iterator[tuple[Any, pd.DataFrame]])
        index2, value2 = next(iterator2)
        assert_type((index2, value2), tuple[Any, pd.DataFrame])

        check(assert_type(index2, Any), int)
        check(assert_type(value2, pd.DataFrame), pd.DataFrame)


def test_setitem_list():
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


def test_groupby_apply() -> None:
    # GH 167
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    def sum_mean(x: pd.DataFrame) -> float:
        return x.sum().mean()

    with pytest_warns_bounded(
        DeprecationWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        upper="2.99",
    ):
        check(
            assert_type(df.groupby("col1").apply(sum_mean), pd.Series),
            pd.Series,
        )

    lfunc: Callable[[pd.DataFrame], float] = lambda x: x.sum().mean()
    with pytest_warns_bounded(
        DeprecationWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        upper="2.99",
    ):
        check(assert_type(df.groupby("col1").apply(lfunc), pd.Series), pd.Series)

    def sum_to_list(x: pd.DataFrame) -> list:
        return x.sum().tolist()

    with pytest_warns_bounded(
        DeprecationWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        upper="2.99",
    ):
        check(assert_type(df.groupby("col1").apply(sum_to_list), pd.Series), pd.Series)

    def sum_to_series(x: pd.DataFrame) -> pd.Series:
        return x.sum()

    with pytest_warns_bounded(
        DeprecationWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        upper="2.99",
    ):
        check(
            assert_type(df.groupby("col1").apply(sum_to_series), pd.DataFrame),
            pd.DataFrame,
        )

    def sample_to_df(x: pd.DataFrame) -> pd.DataFrame:
        return x.sample()

    with pytest_warns_bounded(
        DeprecationWarning,
        "DataFrameGroupBy.apply operated on the grouping columns.",
        upper="2.99",
    ):
        check(
            assert_type(
                df.groupby("col1", group_keys=False).apply(sample_to_df), pd.DataFrame
            ),
            pd.DataFrame,
        )


def test_resample() -> None:
    # GH 181
    N = 10
    x = [x for x in range(N)]
    index = pd.date_range("1/1/2000", periods=N, freq="min")
    x = [x for x in range(N)]
    df = pd.DataFrame({"a": x, "b": x, "c": x}, index=index)
    check(assert_type(df.resample("2min").std(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").var(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").quantile(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").sum(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").prod(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").min(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").max(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").first(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").last(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").mean(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").sem(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").median(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.resample("2min").ohlc(), pd.DataFrame), pd.DataFrame)


def test_squeeze() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    check(
        assert_type(df1.squeeze(), Union[pd.DataFrame, pd.Series, Scalar]), pd.DataFrame
    )
    df2 = pd.DataFrame({"a": [1, 2]})
    check(assert_type(df2.squeeze(), Union[pd.DataFrame, pd.Series, Scalar]), pd.Series)
    df3 = pd.DataFrame({"a": [1], "b": [2]})
    check(
        assert_type(df3.squeeze(), Union[pd.DataFrame, pd.Series, Scalar]),
        pd.Series,
        np.integer,
    )
    df4 = pd.DataFrame({"a": [1]})
    check(
        assert_type(df4.squeeze(), Union[pd.DataFrame, pd.Series, Scalar]), np.integer
    )


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


def test_loclist() -> None:
    # GH 189
    df = pd.DataFrame({1: [1, 2], None: 5}, columns=pd.Index([1, None], dtype=object))

    check(assert_type(df.loc[:, [None]], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.loc[:, [1]], pd.DataFrame), pd.DataFrame)


def test_dict_items() -> None:
    # GH 180
    x = {"a": [1]}
    check(assert_type(pd.DataFrame(x.items()), pd.DataFrame), pd.DataFrame)


def test_generic() -> None:
    # GH 197
    T = TypeVar("T")

    class MyDataFrame(pd.DataFrame, Generic[T]): ...

    def func() -> MyDataFrame[int]:
        return MyDataFrame[int]({"foo": [1, 2, 3]})

    func()


def test_to_xarray() -> None:
    check(assert_type(DF.to_xarray(), xr.Dataset), xr.Dataset)


def test_to_records() -> None:
    check(assert_type(DF.to_records(False, "int8"), np.recarray), np.recarray)
    check(
        assert_type(DF.to_records(False, index_dtypes=np.int8), np.recarray),
        np.recarray,
    )
    check(
        assert_type(
            DF.to_records(False, {"col1": np.int8, "col2": np.int16}), np.recarray
        ),
        np.recarray,
    )
    dtypes = {"col1": np.int8, "col2": np.int16}
    check(
        assert_type(DF.to_records(False, dtypes), np.recarray),
        np.recarray,
    )


def test_to_dict() -> None:
    check(assert_type(DF.to_dict(), dict[Hashable, Any]), dict)
    check(assert_type(DF.to_dict("split"), dict[Hashable, Any]), dict)

    target: MutableMapping = defaultdict(list)
    check(
        assert_type(DF.to_dict(into=target), MutableMapping[Hashable, Any]), defaultdict
    )
    target = defaultdict(list)
    check(
        assert_type(DF.to_dict("tight", into=target), MutableMapping[Hashable, Any]),
        defaultdict,
    )
    target = defaultdict(list)
    check(assert_type(DF.to_dict("records"), list[dict[Hashable, Any]]), list)
    check(
        assert_type(
            DF.to_dict("records", into=target), list[MutableMapping[Hashable, Any]]
        ),
        list,
    )
    if TYPE_CHECKING_INVALID_USAGE:

        def test(mapping: Mapping) -> None:  # pyright: ignore[reportUnusedFunction]
            DF.to_dict(  # type: ignore[call-overload]
                into=mapping  # pyright: ignore[reportArgumentType,reportCallIssue]
            )


def test_neg() -> None:
    # GH 253
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(-df, pd.DataFrame), pd.DataFrame)


def test_pos() -> None:
    # GH 253
    df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    check(assert_type(+df, pd.DataFrame), pd.DataFrame)


def test_getattr() -> None:
    # GH 261
    df = pd.DataFrame({"a": [1, 2]})
    check(assert_type(df.a, pd.Series), pd.Series)


def test_xs_key() -> None:
    # GH 214
    mi = pd.MultiIndex.from_product([[0, 1], [0, 1]], names=["foo", "bar"])
    df = pd.DataFrame({"x": [10, 20, 30, 40], "y": [50, 60, 70, 80]}, index=mi)
    check(
        assert_type(df.xs(0, level="foo"), Union[pd.DataFrame, pd.Series]), pd.DataFrame
    )


def test_loc_slice() -> None:
    # GH 277
    df1 = pd.DataFrame(
        {"x": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=["num", "let"]),
    )
    check(assert_type(df1.loc[1, :], Union[pd.Series, pd.DataFrame]), pd.DataFrame)


def test_where() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def cond1(x: int) -> bool:
        return x % 2 == 0

    check(assert_type(df.where(cond1), pd.DataFrame), pd.DataFrame)

    def cond2(x: pd.DataFrame) -> pd.DataFrame:
        return x > 1

    check(assert_type(df.where(cond2), pd.DataFrame), pd.DataFrame)

    cond3 = pd.DataFrame({"a": [True, True, False], "b": [False, False, False]})
    check(assert_type(df.where(cond3), pd.DataFrame), pd.DataFrame)


def test_mask() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def cond1(x: int) -> bool:
        return x % 2 == 0

    check(assert_type(df.mask(cond1), pd.DataFrame), pd.DataFrame)


def test_setitem_loc() -> None:
    # GH 254
    df = pd.DataFrame.from_dict(
        {view: (True, True, True) for view in ["A", "B", "C"]}, orient="index"
    )
    df.loc[["A", "C"]] = False
    my_arr = ["A", "C"]
    df.loc[my_arr] = False


def test_replace_na() -> None:
    # GH 262
    frame = pd.DataFrame(["N/A", "foo", "bar"])
    check(assert_type(frame.replace("N/A", pd.NA), pd.DataFrame), pd.DataFrame)


def test_isetframe() -> None:
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    check(assert_type(frame.isetitem(0, 10), None), type(None))
    check(assert_type(frame.isetitem([0], [10, 12]), None), type(None))


def test_reset_index_150_changes() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[-10, -9, -8, -7])
    check(
        assert_type(
            frame.reset_index(allow_duplicates=True, names="idx"), pd.DataFrame
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            frame.reset_index(allow_duplicates=True, names=["idx"]), pd.DataFrame
        ),
        pd.DataFrame,
    )


def test_compare_150_changes() -> None:
    frame_a = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[-10, -9, -8, -7])
    frame_b = pd.DataFrame({"a": [1, 2, 4, 3]}, index=[-10, -9, -8, -7])
    check(
        assert_type(
            frame_a.compare(frame_b, result_names=("one", "the_other")), pd.DataFrame
        ),
        pd.DataFrame,
    )


def test_quantile_150_changes() -> None:
    frame = pd.DataFrame(getSeriesData())
    check(assert_type(frame.quantile(0.5, method="single"), pd.Series), pd.Series)
    check(
        assert_type(
            frame.quantile([0.25, 0.5, 0.75], interpolation="nearest", method="table"),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )


def test_resample_150_changes() -> None:
    idx = pd.date_range("2020-1-1", periods=700)
    frame = pd.DataFrame(np.random.standard_normal((700, 1)), index=idx, columns=["a"])
    with pytest_warns_bounded(
        FutureWarning,
        "'M' is deprecated",
        lower="2.1.99",
        upper="2.2.99",
        upper_exception=ValueError,
    ):
        frame.resample("M", group_keys=True)

    resampler = frame.resample("MS", group_keys=True)
    check(
        assert_type(resampler, "DatetimeIndexResampler[pd.DataFrame]"),
        DatetimeIndexResampler,
    )

    def f(s: pd.DataFrame) -> pd.Series:
        return s.mean()

    check(assert_type(resampler.apply(f), pd.DataFrame), pd.DataFrame)


def test_df_accepting_dicts_iterator() -> None:
    # GH 392
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 5}]
    check(assert_type(pd.DataFrame(iter(data)), pd.DataFrame), pd.DataFrame)


def test_series_added_in_astype() -> None:
    # GH410
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    check(assert_type(df.astype(df.dtypes), pd.DataFrame), pd.DataFrame)


def test_series_groupby_and_value_counts() -> None:
    df = pd.DataFrame(
        {
            "Animal": ["Falcon", "Falcon", "Parrot", "Parrot"],
            "Max Speed": [380, 370, 24, 26],
        }
    )
    c1 = df.groupby("Animal")["Max Speed"].value_counts()
    c2 = df.groupby("Animal")["Max Speed"].value_counts(normalize=True)
    check(assert_type(c1, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(c2, "pd.Series[float]"), pd.Series, float)


def test_axes_as_tuple() -> None:
    # GH 384
    index = (3, 5, 7)
    columns = ["a", "b", "c"]
    df = pd.DataFrame(data=1, index=index, columns=columns)
    check(assert_type(df, pd.DataFrame), pd.DataFrame)


def test_astype_dict() -> None:
    # GH 447
    df = pd.DataFrame({"a": [1, 2, 3], 43: [4, 5, 6]})
    columns_types = {"a": "int", 43: "float"}
    de = df.astype(columns_types)
    check(assert_type(de, pd.DataFrame), pd.DataFrame)
    check(assert_type(df.astype({"a": "int", 43: "float"}), pd.DataFrame), pd.DataFrame)


def test_setitem_none() -> None:
    df = pd.DataFrame(
        {"A": [1, 2, 3], "B": ["abc", "def", "ghi"]}, index=["x", "y", "z"]
    )
    df.loc["x", "B"] = None
    df.iloc[2, 0] = None
    sb = pd.Series([1, 2, 3], dtype=int)
    sb.loc["y"] = None
    sb.iloc[0] = None


def test_groupby_and_transform() -> None:
    df = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
            "B": ["one", "one", "two", "three", "two", "two"],
            "C": [1, 5, 5, 2, 5, 5],
            "D": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        }
    )
    ser = pd.Series(
        [390.0, 350.0, 30.0, 20.0],
        index=["Falcon", "Falcon", "Parrot", "Parrot"],
        name="Max Speed",
    )
    grouped = df.groupby("A")[["C", "D"]]
    grouped1 = ser.groupby(ser > 100)
    c1 = grouped.transform("sum")
    c2 = grouped.transform(lambda x: (x - x.mean()) / x.std())
    c3 = grouped1.transform("cumsum")
    c4 = grouped1.transform(lambda x: x.max() - x.min())
    check(assert_type(c1, pd.DataFrame), pd.DataFrame)
    check(assert_type(c2, pd.DataFrame), pd.DataFrame)
    check(assert_type(c3, pd.Series), pd.Series)
    check(assert_type(c4, pd.Series), pd.Series)


def test_getattr_and_dataframe_groupby() -> None:
    df = pd.DataFrame(
        data={"col1": [1, 1, 2], "col2": [3, 4, 5], "col3": [0, 1, 0], 0: [-1, -1, -1]}
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(df.groupby("col1").col3.agg(min), pd.Series), pd.Series)
        check(
            assert_type(df.groupby("col1").col3.agg([min, max]), pd.DataFrame),
            pd.DataFrame,
        )


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


def test_frame_dropna_subset() -> None:
    # GH 434
    data = {"col1": [1, 3, 4], "col2": [2, 3, 5], "col3": [2, 4, 4]}
    df = pd.DataFrame(data)
    check(
        assert_type(df.dropna(subset=df.columns.drop("col1")), pd.DataFrame),
        pd.DataFrame,
    )


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


def test_npint_loc_indexer() -> None:
    # GH 508

    df = pd.DataFrame(dict(x=[1, 2, 3]), index=np.array([10, 20, 30], dtype="uint64"))

    def get_NDArray(df: pd.DataFrame, key: npt.NDArray[np.uint64]) -> pd.DataFrame:
        df2 = df.loc[key]
        return df2

    a: npt.NDArray[np.uint64] = np.array([10, 30], dtype="uint64")
    check(assert_type(get_NDArray(df, a), pd.DataFrame), pd.DataFrame)


def test_in_columns() -> None:
    # GH 532 (PR)
    df = pd.DataFrame(np.random.random((3, 4)), columns=["cat", "dog", "rat", "pig"])
    cols = [c for c in df.columns if "at" in c]
    check(assert_type(cols, list[str]), list, str)
    check(assert_type(df.loc[:, cols], pd.DataFrame), pd.DataFrame)
    check(assert_type(df[cols], pd.DataFrame), pd.DataFrame)
    check(assert_type(df.groupby(by=cols).sum(), pd.DataFrame), pd.DataFrame)


def test_loc_list_str() -> None:
    # GH 1162 (PR)
    df = pd.DataFrame(
        [[1, 2], [4, 5], [7, 8]],
        index=["cobra", "viper", "sidewinder"],
        columns=["max_speed", "shield"],
    )

    result = df.loc[["viper", "sidewinder"]]
    check(assert_type(result, pd.DataFrame), pd.DataFrame)


def test_insert_newvalues() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    ab = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    ef = pd.DataFrame({"z": [4, 5, 6]})
    assert assert_type(df.insert(loc=0, column="b", value=None), None) is None
    assert assert_type(ab.insert(loc=0, column="newcol", value=[99, 99]), None) is None
    assert assert_type(ef.insert(loc=0, column="g", value=4), None) is None


def test_astype() -> None:
    s = pd.DataFrame({"d": [1, 2]})
    ab = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

    check(assert_type(s.astype(int), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype(pd.Int64Dtype()), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype(str), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype(bytes), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype(pd.Float64Dtype()), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype(complex), "pd.DataFrame"), pd.DataFrame)
    check(
        assert_type(ab.astype({"col1": "int32", "col2": str}), "pd.DataFrame"),
        pd.DataFrame,
    )
    check(assert_type(s.astype(pd.CategoricalDtype()), "pd.DataFrame"), pd.DataFrame)
    check(assert_type(s.astype("category"), "pd.DataFrame"), pd.DataFrame)  # GH 559

    population_dict = {
        "California": 38332521,
        "Texas": 26448193,
        "New York": 19651127,
        "Florida": 19552860,
        "Illinois": 12882135,
    }
    area_dict = {
        "California": 423967,
        "Texas": 695662,
        "New York": 141297,
        "Florida": 170312,
        "Illinois": 149995,
    }
    population = pd.Series(population_dict)
    area = pd.Series(area_dict)

    states = pd.DataFrame({"population": population, "area": area})
    check(assert_type(states.astype(object), pd.DataFrame), pd.DataFrame, object)


def test_xs_frame_new() -> None:
    d = {
        "num_legs": [4, 4, 2, 2],
        "num_wings": [0, 0, 2, 2],
        "class": ["mammal", "mammal", "mammal", "bird"],
        "animal": ["cat", "dog", "bat", "penguin"],
        "locomotion": ["walks", "walks", "flies", "walks"],
    }
    df = pd.DataFrame(data=d)
    df = df.set_index(["class", "animal", "locomotion"])
    s1 = df.xs("mammal", axis=0)
    s2 = df.xs("num_wings", axis=1)
    check(assert_type(s1, Union[pd.Series, pd.DataFrame]), pd.DataFrame)
    check(assert_type(s2, Union[pd.Series, pd.DataFrame]), pd.Series)


def test_align() -> None:
    df0 = pd.DataFrame(
        data=np.array(
            [
                ["A0", "A1", "A2", "A3"],
                ["B0", "B1", "B2", "B3"],
                ["C0", "C1", "C2", "C3"],
            ]
        ).T,
        index=[0, 1, 2, 3],
        columns=["A", "B", "C"],
    )

    s0 = pd.Series(data={0: "1", 3: "3", 5: "5"})
    aligned_df0, aligned_s0 = df0.align(s0, axis="index")
    check(assert_type(aligned_df0, pd.DataFrame), pd.DataFrame)
    check(assert_type(aligned_s0, "pd.Series[str]"), pd.Series, str)
    aligned_df0, aligned_s0 = df0.align(s0, axis="index", fill_value=0)
    check(assert_type(aligned_df0, pd.DataFrame), pd.DataFrame)
    check(assert_type(aligned_s0, "pd.Series[str]"), pd.Series, str)

    s1 = pd.Series(data={"A": "A", "D": "D"})
    aligned_df0, aligned_s1 = df0.align(s1, axis="columns")
    check(assert_type(aligned_df0, pd.DataFrame), pd.DataFrame)
    check(assert_type(aligned_s1, "pd.Series[str]"), pd.Series, str)

    df1 = pd.DataFrame(
        data=np.array(
            [
                ["A1", "A3", "A5"],
                ["D1", "D3", "D5"],
            ]
        ).T,
        index=[1, 3, 5],
        columns=["A", "D"],
    )
    aligned_df0, aligned_df1 = df0.align(df1)
    check(assert_type(aligned_df0, pd.DataFrame), pd.DataFrame)
    check(assert_type(aligned_df1, pd.DataFrame), pd.DataFrame)


def test_loc_returns_series() -> None:
    df1 = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[10, 20, 30, 40])
    df2 = df1.loc[10, :]
    check(assert_type(df2, Union[pd.Series, pd.DataFrame]), pd.Series)


def test_to_dict_index() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [9, 10]})
    check(
        assert_type(
            df.to_dict(orient="records", index=True), list[dict[Hashable, Any]]
        ),
        list,
    )
    check(assert_type(df.to_dict(orient="dict", index=True), dict[Hashable, Any]), dict)
    check(
        assert_type(df.to_dict(orient="series", index=True), dict[Hashable, Any]), dict
    )
    check(
        assert_type(df.to_dict(orient="index", index=True), dict[Hashable, Any]), dict
    )
    check(
        assert_type(df.to_dict(orient="split", index=True), dict[Hashable, Any]), dict
    )
    check(
        assert_type(df.to_dict(orient="tight", index=True), dict[Hashable, Any]), dict
    )
    check(
        assert_type(df.to_dict(orient="tight", index=False), dict[Hashable, Any]), dict
    )
    check(
        assert_type(df.to_dict(orient="split", index=False), dict[Hashable, Any]), dict
    )
    if TYPE_CHECKING_INVALID_USAGE:
        check(assert_type(df.to_dict(orient="records", index=False), list[dict[Hashable, Any]]), list)  # type: ignore[assert-type, call-overload] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]
        check(assert_type(df.to_dict(orient="dict", index=False), dict[Hashable, Any]), dict)  # type: ignore[assert-type, call-overload] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]
        check(assert_type(df.to_dict(orient="series", index=False), dict[Hashable, Any]), dict)  # type: ignore[assert-type, call-overload] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]
        check(assert_type(df.to_dict(orient="index", index=False), dict[Hashable, Any]), dict)  # type: ignore[assert-type, call-overload] # pyright: ignore[reportArgumentType,reportAssertTypeFailure,reportCallIssue]


def test_suffix_prefix_index() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    check(assert_type(df.add_suffix("_col", axis=1), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add_suffix("_col", axis="index"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.add_prefix("_col", axis="index"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.add_prefix("_col", axis="columns"), pd.DataFrame), pd.DataFrame
    )


def test_convert_dtypes_convert_floating() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    dfn = df.convert_dtypes(convert_floating=False)
    check(assert_type(dfn, pd.DataFrame), pd.DataFrame)


def test_convert_dtypes_dtype_backend() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    dfn = df.convert_dtypes(dtype_backend="numpy_nullable")
    check(assert_type(dfn, pd.DataFrame), pd.DataFrame)


def test_select_dtypes() -> None:
    df = pd.DataFrame({"a": [1, 2] * 3, "b": [True, False] * 3, "c": [1.0, 2.0] * 3})
    check(assert_type(df.select_dtypes("number"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.select_dtypes("integer"), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.select_dtypes(np.number), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.select_dtypes(object), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.select_dtypes(include="bool"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.select_dtypes(include=["float64"], exclude=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.select_dtypes(exclude=["int64"], include=None), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(df.select_dtypes(exclude=["int64", object]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(
            df.select_dtypes(
                exclude=[
                    np.datetime64,
                    "datetime64",
                    "datetime",
                    np.timedelta64,
                    "timedelta",
                    "timedelta64",
                    "category",
                    "datetimetz",
                    "datetime64[ns]",
                ]
            ),
            pd.DataFrame,
        ),
        pd.DataFrame,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        # include and exclude shall not be both empty
        assert_never(df.select_dtypes([], []))
        assert_never(df.select_dtypes())
        # str like dtypes are not allowed
        assert_never(df.select_dtypes(str))
        assert_never(df.select_dtypes(exclude=str))
        assert_never(df.select_dtypes(None, str))


def test_to_json_mode() -> None:
    df = pd.DataFrame(
        [["a", "b"], ["c", "d"]],
        index=["row 1", "row 2"],
        columns=["col 1", "col 2"],
    )
    result = df.to_json(orient="records", lines=True, mode="a")
    result1 = df.to_json(orient="split", mode="w")
    result2 = df.to_json(orient="columns", mode="w")
    result4 = df.to_json(orient="records", mode="w")
    check(assert_type(result, str), str)
    check(assert_type(result1, str), str)
    check(assert_type(result2, str), str)
    check(assert_type(result4, str), str)
    if TYPE_CHECKING_INVALID_USAGE:
        result3 = df.to_json(orient="records", lines=False, mode="a")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_interpolate_inplace() -> None:
    # GH 691
    df = pd.DataFrame({"a": range(3)})
    check(assert_type(df.interpolate(method="linear"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(df.interpolate(method="linear", inplace=False), pd.DataFrame),
        pd.DataFrame,
    )
    check(assert_type(df.interpolate(method="linear", inplace=True), None), type(None))


def test_getitem_generator() -> None:
    # GH 685
    check(assert_type(DF[(f"col{i+1}" for i in range(2))], pd.DataFrame), pd.DataFrame)


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

    df["x"] = df["y"] + pd.Timedelta(days=3)
    df.loc[ind, :] = pd.NaT
    df.iloc[[0, 2], :] = pd.NaT


def test_itertuples() -> None:
    # GH 822
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    for item in df.itertuples():
        check(assert_type(item, _PandasNamedTuple), tuple)
        assert_type(item.a, Scalar)


def test_get() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    # Get single column
    check(assert_type(df.get("a"), Union[pd.Series, None]), pd.Series, np.int64)
    check(assert_type(df.get("z"), Union[pd.Series, None]), type(None))
    check(
        assert_type(df.get("a", default=None), Union[pd.Series, None]),
        pd.Series,
        np.int64,
    )
    check(assert_type(df.get("z", default=None), Union[pd.Series, None]), type(None))
    check(
        assert_type(df.get("a", default=1), Union[pd.Series, int]), pd.Series, np.int64
    )
    check(assert_type(df.get("z", default=1), Union[pd.Series, int]), int)

    # Get multiple columns
    check(assert_type(df.get(["a"]), Union[pd.DataFrame, None]), pd.DataFrame)
    check(assert_type(df.get(["a", "b"]), Union[pd.DataFrame, None]), pd.DataFrame)
    check(assert_type(df.get(["z"]), Union[pd.DataFrame, None]), type(None))
    check(
        assert_type(df.get(["a", "b"], default=None), Union[pd.DataFrame, None]),
        pd.DataFrame,
    )
    check(
        assert_type(df.get(["z"], default=None), Union[pd.DataFrame, None]), type(None)
    )
    check(
        assert_type(df.get(["a", "b"], default=1), Union[pd.DataFrame, int]),
        pd.DataFrame,
    )
    check(assert_type(df.get(["z"], default=1), Union[pd.DataFrame, int]), int)


def test_info() -> None:
    df = pd.DataFrame()

    check(assert_type(df.info(verbose=True), None), type(None))
    check(assert_type(df.info(verbose=False), None), type(None))
    check(assert_type(df.info(verbose=None), None), type(None))

    check(assert_type(df.info(buf=io.StringIO()), None), type(None))

    check(assert_type(df.info(max_cols=1), None), type(None))
    check(assert_type(df.info(max_cols=None), None), type(None))

    check(assert_type(df.info(memory_usage=True), None), type(None))
    check(assert_type(df.info(memory_usage=False), None), type(None))
    check(assert_type(df.info(memory_usage="deep"), None), type(None))
    check(assert_type(df.info(memory_usage=None), None), type(None))

    check(assert_type(df.info(show_counts=True), None), type(None))
    check(assert_type(df.info(show_counts=False), None), type(None))
    check(assert_type(df.info(show_counts=None), None), type(None))


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


def test_frame_bool_fails() -> None:
    # GH 663

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    try:
        # We want the type checker to tell us the next line is invalid
        # mypy doesn't seem to figure that out, but pyright does
        if df == "foo":  # pyright: ignore[reportGeneralTypeIssues]
            # Next line is unreachable.
            s = df["a"]
    except ValueError:
        pass


def test_frame_subclass() -> None:
    class MyClass(pd.DataFrame):
        @property
        def _constructor(self) -> type[MyClass]:
            return MyClass

    df = MyClass({"a": [1, 2, 3], "b": [4, 5, 6]})
    check(assert_type(df.iloc[1:2], MyClass), MyClass)
    check(assert_type(df.loc[:, ["a", "b"]], MyClass), MyClass)
    check(assert_type(df[["a", "b"]], MyClass), MyClass)


def test_hashable_args() -> None:
    # GH 1104
    df = pd.DataFrame([["abc"]], columns=["test"], index=["ind"])
    test = ["test"]

    with ensure_clean() as path:

        df.to_stata(path, version=117, convert_strl=test)
        df.to_stata(path, version=117, convert_strl=["test"])

        df.to_html(path, columns=test)
        df.to_html(path, columns=["test"])

        df.to_xml(path, attr_cols=test)
        df.to_xml(path, attr_cols=["test"])

        df.to_xml(path, elem_cols=test)
        df.to_xml(path, elem_cols=["test"])

    # Next lines should work, but it is a mypy bug
    # https://github.com/python/mypy/issues/3004
    # pyright accepts this, so we only type check for pyright,
    # and also test the code with pytest
    df.columns = test  # type: ignore[assignment]
    df.columns = ["test"]  # type: ignore[assignment]

    testDict = {"test": 1}
    with ensure_clean() as path:
        df.to_string(path, col_space=testDict)
        df.to_string(path, col_space={"test": 1})


# GH 906
@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor: ...


def test_transpose() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    check(assert_type(df.transpose(), pd.DataFrame), pd.DataFrame)
    check(assert_type(df.transpose(None), pd.DataFrame), pd.DataFrame)

    msg = "The copy keyword is deprecated and will be removed in a future"
    with pytest_warns_bounded(
        DeprecationWarning,
        msg,
        lower="2.2.99",
    ):
        check(assert_type(df.transpose(copy=True), pd.DataFrame), pd.DataFrame)


def test_combine() -> None:
    df1 = pd.DataFrame({"A": [0, 0], "B": [4, 4]})
    df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})
    take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
    assert_type(
        check(
            df1.combine(df2, take_smaller, fill_value=0, overwrite=False), pd.DataFrame
        ),
        pd.DataFrame,
    )


def test_df_loc_dict() -> None:
    """Test that we can set a dict to a df.loc result GH1203."""
    df = pd.DataFrame(columns=["X"])
    df.loc[0] = {"X": 0}
    check(assert_type(df, pd.DataFrame), pd.DataFrame)

    df.iloc[0] = {"X": 0}
    check(assert_type(df, pd.DataFrame), pd.DataFrame)


def test_unstack() -> None:
    """Test different types of argument for `fill_value` in DataFrame.unstack."""
    df = pd.DataFrame(
        [
            ["a", "b", pd.Timestamp(2021, 3, 2)],
            ["a", "a", pd.Timestamp(2023, 4, 2)],
            ["b", "b", pd.Timestamp(2024, 3, 2)],
        ]
    ).set_index([0, 1])
    df_sr = pd.DataFrame(
        [
            ["a", "b", "abc"],
            ["a", "a", "def"],
            ["b", "b", "ghi"],
        ]
    ).set_index([0, 1])
    df_flt = pd.DataFrame(
        [
            ["a", "b", 1],
            ["a", "a", 12],
            ["b", "b", 14],
        ]
    ).set_index([0, 1])

    check(assert_type(df.unstack(0), pd.DataFrame | pd.Series), pd.DataFrame)
    check(
        assert_type(
            df.unstack(1, fill_value=pd.Timestamp(2023, 4, 5)), pd.DataFrame | pd.Series
        ),
        pd.DataFrame,
    )
    check(
        assert_type(df_flt.unstack(1, fill_value=0.0), pd.DataFrame | pd.Series),
        pd.DataFrame,
    )
    check(
        assert_type(df_flt.unstack(1, fill_value=1), pd.DataFrame | pd.Series),
        pd.DataFrame,
    )
    check(
        assert_type(df_sr.unstack(1, fill_value="string"), pd.DataFrame | pd.Series),
        pd.DataFrame,
    )
    check(
        assert_type(df.unstack(0, sort=False), pd.DataFrame | pd.Series), pd.DataFrame
    )
    check(
        assert_type(
            df.unstack(1, fill_value=pd.Timestamp(2023, 4, 5), sort=True),
            pd.DataFrame | pd.Series,
        ),
        pd.DataFrame,
    )
    check(
        assert_type(
            df_flt.unstack(1, fill_value=0.0, sort=False), pd.DataFrame | pd.Series
        ),
        pd.DataFrame,
    )
