from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
from pandas._testing import ensure_clean
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.core.window import ExponentialMovingWindow
import pytest
from typing_extensions import (
    Self,
    TypeAlias,
    assert_type,
)
import xarray as xr

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    DtypeObj,
    Scalar,
)

from tests import (
    PD_LTE_20,
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests.extension.decimal.array import DecimalDtype

if TYPE_CHECKING:
    from pandas.core.series import (
        TimedeltaSeries,
        TimestampSeries,
    )
else:
    TimedeltaSeries: TypeAlias = pd.Series
    TimestampSeries: TypeAlias = pd.Series

if TYPE_CHECKING:
    from pandas._typing import np_ndarray_int  # noqa: F401


def test_types_init() -> None:
    pd.Series(1)
    pd.Series((1, 2, 3))
    pd.Series(np.array([1, 2, 3]))
    pd.Series(data=[1, 2, 3, 4], name="series")
    pd.Series(data=[1, 2, 3, 4], dtype=np.int8)
    pd.Series(data={"row1": [1, 2], "row2": [3, 4]})
    pd.Series(data=[1, 2, 3, 4], index=[4, 3, 2, 1], copy=True)
    # GH 90
    dt: pd.DatetimeIndex = pd.to_datetime(
        [1, 2], unit="D", origin=pd.Timestamp("01/01/2000")
    )
    pd.Series(data=dt, index=None)
    pd.Series(data=[1, 2, 3, 4], dtype=int, index=None)
    pd.Series(data={"row1": [1, 2], "row2": [3, 4]}, dtype=int, index=None)
    pd.Series(data=[1, 2, 3, 4], index=None)
    pd.Series(data={"row1": [1, 2], "row2": [3, 4]}, index=None)


def test_types_any() -> None:
    check(assert_type(pd.Series([False, False]).any(), bool), np.bool_)
    check(assert_type(pd.Series([False, False]).any(bool_only=False), bool), np.bool_)
    check(assert_type(pd.Series([np.nan]).any(skipna=False), bool), np.bool_)


def test_types_all() -> None:
    check(assert_type(pd.Series([False, False]).all(), bool), np.bool_)
    check(assert_type(pd.Series([False, False]).all(bool_only=False), bool), np.bool_)
    check(assert_type(pd.Series([np.nan]).all(skipna=False), bool), np.bool_)


def test_types_csv() -> None:
    s = pd.Series(data=[1, 2, 3])
    csv_df: str = s.to_csv()

    with ensure_clean() as path:
        s.to_csv(path)
        s2: pd.DataFrame = pd.read_csv(path)

    with ensure_clean() as path:
        s.to_csv(Path(path))
        s3: pd.DataFrame = pd.read_csv(Path(path))

    # This keyword was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    with ensure_clean() as path:
        s.to_csv(path, errors="replace")
        s4: pd.DataFrame = pd.read_csv(path)


def test_types_copy() -> None:
    s = pd.Series(data=[1, 2, 3, 4])
    check(assert_type(s.copy(), pd.Series), pd.Series, np.int64)


def test_types_select() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    s[0]
    s[1:]


def test_types_iloc_iat() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    s2 = pd.Series(data=[1, 2])
    s.loc["row1"]
    s.iat[0]
    s2.loc[0]
    s2.iat[0]


def test_types_loc_at() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    s2 = pd.Series(data=[1, 2])
    s.loc["row1"]
    s.at["row1"]
    s2.loc[1]
    s2.at[1]


def test_multiindex_loc() -> None:
    s = pd.Series(
        [1, 2, 3, 4], index=pd.MultiIndex.from_product([[1, 2], ["a", "b"]]), dtype=int
    )
    check(assert_type(s.loc[1, :], "pd.Series[int]"), pd.Series, np.int_)
    check(assert_type(s.loc[pd.Index([1]), :], "pd.Series[int]"), pd.Series, np.int_)
    check(assert_type(s.loc[1, "a"], int), np.int_)


def test_types_boolean_indexing() -> None:
    s = pd.Series([0, 1, 2])
    s[s > 1]
    s[s]


def test_types_df_to_df_comparison() -> None:
    s = pd.Series(data={"col1": [1, 2]})
    s2 = pd.Series(data={"col1": [3, 2]})
    res_gt: pd.Series = s > s2
    res_ge: pd.Series = s >= s2
    res_lt: pd.Series = s < s2
    res_le: pd.Series = s <= s2
    res_e: pd.Series = s == s2


def test_types_head_tail() -> None:
    s = pd.Series([0, 1, 2])
    s.head(1)
    s.tail(1)


def test_types_sample() -> None:
    s = pd.Series([0, 1, 2])
    s.sample(frac=0.5)
    s.sample(n=1)


def test_types_nlargest_nsmallest() -> None:
    s = pd.Series([0, 1, 2])
    s.nlargest(1)
    s.nlargest(1, "first")
    s.nsmallest(1, "last")
    s.nsmallest(1, "all")


def test_types_filter() -> None:
    s = pd.Series(data=[1, 2, 3, 4], index=["cow", "coal", "coalesce", ""])
    s.filter(items=["cow"])
    s.filter(regex="co.*")
    s.filter(like="al")


def test_types_setting() -> None:
    s = pd.Series([0, 1, 2])
    s[3] = 4
    s[s == 1] = 5
    s[:] = 3


def test_types_drop() -> None:
    s = pd.Series([0, 1, 2])
    check(assert_type(s.drop(0), pd.Series), pd.Series)
    check(assert_type(s.drop([0, 1]), pd.Series), pd.Series)
    check(assert_type(s.drop(0, axis=0), pd.Series), pd.Series)
    assert assert_type(s.drop([0, 1], inplace=True, errors="raise"), None) is None
    assert assert_type(s.drop([0, 1], inplace=True, errors="ignore"), None) is None
    # GH 302
    s = pd.Series([0, 1, 2])
    check(assert_type(s.drop(pd.Index([0, 1])), pd.Series), pd.Series)
    check(assert_type(s.drop(index=pd.Index([0, 1])), pd.Series), pd.Series)


def test_types_drop_multilevel() -> None:
    index = pd.MultiIndex(
        levels=[["top", "bottom"], ["first", "second", "third"]],
        codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
    )
    s = pd.Series(data=[1, 2, 3, 4, 5, 6], index=index)
    res: pd.Series = s.drop(labels="first", level=1)


def test_types_dropna() -> None:
    s = pd.Series([1, np.nan, np.nan])
    check(assert_type(s.dropna(), pd.Series), pd.Series)
    assert assert_type(s.dropna(axis=0, inplace=True), None) is None


def test_pop() -> None:
    # Testing pop support for hashable types
    # Due to the bug in https://github.com/pandas-dev/pandas-stubs/issues/627
    class MyEnum(Enum):
        FIRST = "tayyar"
        SECOND = "haydar"

    s = pd.Series([3.2, 4.3], index=[MyEnum.FIRST, MyEnum.SECOND], dtype=float)
    res = s.pop(MyEnum.FIRST)
    check(assert_type(res, float), np.float64)

    s2 = pd.Series([3, 5], index=["alibaba", "zuhuratbaba"], dtype=int)
    check(assert_type(s2.pop("alibaba"), int), np.int_)


def test_types_fillna() -> None:
    s = pd.Series([1, np.nan, np.nan, 3])
    check(assert_type(s.fillna(0), pd.Series), pd.Series)
    check(assert_type(s.fillna(0, axis="index"), pd.Series), pd.Series)
    check(assert_type(s.fillna(method="backfill", axis=0), pd.Series), pd.Series)
    assert assert_type(s.fillna(method="bfill", inplace=True), None) is None
    check(assert_type(s.fillna(method="pad"), pd.Series), pd.Series)
    check(assert_type(s.fillna(method="ffill", limit=1), pd.Series), pd.Series)
    # GH 263
    check(assert_type(s.fillna(pd.NA), pd.Series), pd.Series)


def test_types_sort_index() -> None:
    s = pd.Series([1, 2, 3], index=[2, 3, 1])
    check(assert_type(s.sort_index(), pd.Series), pd.Series)
    check(assert_type(s.sort_index(ascending=False), pd.Series), pd.Series)
    assert assert_type(s.sort_index(ascending=False, inplace=True), None) is None
    check(assert_type(s.sort_index(kind="mergesort"), pd.Series), pd.Series)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_index_with_key() -> None:
    s = pd.Series([1, 2, 3], index=["a", "B", "c"])
    res: pd.Series = s.sort_index(key=lambda k: k.str.lower())


def test_types_sort_values() -> None:
    s = pd.Series([4, 2, 1, 3])
    check(assert_type(s.sort_values(), pd.Series), pd.Series)
    if TYPE_CHECKING_INVALID_USAGE:
        check(assert_type(s.sort_values(0), pd.Series), pd.Series)  # type: ignore[assert-type,call-overload] # pyright: ignore[reportGeneralTypeIssues]
    check(assert_type(s.sort_values(axis=0), pd.Series), pd.Series)
    check(assert_type(s.sort_values(ascending=False), pd.Series), pd.Series)
    assert assert_type(s.sort_values(inplace=True, kind="quicksort"), None) is None
    check(assert_type(s.sort_values(na_position="last"), pd.Series), pd.Series)
    check(assert_type(s.sort_values(ignore_index=True), pd.Series), pd.Series)


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_values_with_key() -> None:
    s = pd.Series([1, 2, 3], index=[2, 3, 1])
    res: pd.Series = s.sort_values(key=lambda k: -k)


def test_types_shift() -> None:
    s = pd.Series([1, 2, 3])
    s.shift()
    s.shift(axis=0, periods=1)
    s.shift(-1, fill_value=0)


def test_types_rank() -> None:
    s = pd.Series([1, 1, 2, 5, 6, np.nan])
    s.rank()
    s.rank(axis=0, na_option="bottom")
    s.rank(method="min", pct=True)
    s.rank(method="dense", ascending=True)
    s.rank(method="first", numeric_only=True)
    s2 = pd.Series([1, 1, 2, 5, 6, np.nan])
    s2.rank(method="first", numeric_only=True)


def test_types_mean() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    f1: float = s.mean()
    s1: pd.Series = s.groupby(level=0).mean()
    f2: float = s.mean(skipna=False)
    f3: float = s.mean(numeric_only=False)


def test_types_median() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    f1: float = s.median()
    s1: pd.Series = s.groupby(level=0).median()
    f2: float = s.median(skipna=False)
    f3: float = s.median(numeric_only=False)


def test_types_sum() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    s.sum()
    s.groupby(level=0).sum()
    s.sum(skipna=False)
    s.sum(numeric_only=False)
    s.sum(min_count=4)

    # Note:
    # 1. Return types of `series.groupby(...).sum(...)` are NOT tested
    #    (waiting for stubs).
    # 2. Runtime return types of `series.sum(min_count=...)` are NOT
    #    tested (because of potential `nan`s).

    s0 = assert_type(pd.Series([1, 2, 3, np.nan]), "pd.Series")
    check(assert_type(s0.sum(), "Any"), np.float64)
    check(assert_type(s0.sum(skipna=False), "Any"), np.float64)
    check(assert_type(s0.sum(numeric_only=False), "Any"), np.float64)
    assert_type(s0.sum(min_count=4), "Any")

    s1 = assert_type(pd.Series([False, True], dtype=bool), "pd.Series[bool]")
    check(assert_type(s1.sum(), "int"), np.int64)
    check(assert_type(s1.sum(skipna=False), "int"), np.int64)
    check(assert_type(s1.sum(numeric_only=False), "int"), np.int64)
    assert_type(s1.sum(min_count=4), "int")

    s2 = assert_type(pd.Series([0, 1], dtype=int), "pd.Series[int]")
    check(assert_type(s2.sum(), "int"), np.int64)
    check(assert_type(s2.sum(skipna=False), "int"), np.int64)
    check(assert_type(s2.sum(numeric_only=False), "int"), np.int64)
    assert_type(s2.sum(min_count=4), "int")

    s3 = assert_type(pd.Series([1, 2, 3, np.nan], dtype=float), "pd.Series[float]")
    check(assert_type(s3.sum(), "float"), np.float64)
    check(assert_type(s3.sum(skipna=False), "float"), np.float64)
    check(assert_type(s3.sum(numeric_only=False), "float"), np.float64)
    assert_type(s3.sum(min_count=4), "float")


def test_types_cumsum() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    s.cumsum()
    s.cumsum(axis=0)
    s.cumsum(skipna=False)


def test_types_min() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    s.min()
    s.min(axis=0)
    s.groupby(level=0).min()
    s.min(skipna=False)


def test_types_max() -> None:
    s = pd.Series([1, 2, 3, np.nan])
    s.max()
    s.max(axis=0)
    s.groupby(level=0).max()
    s.max(skipna=False)


def test_types_quantile() -> None:
    s = pd.Series([1, 2, 3, 10])
    s.quantile([0.25, 0.5])
    s.quantile(0.75)
    s.quantile()
    s.quantile(interpolation="nearest")


def test_types_clip() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.clip(lower=0, upper=5)
    s.clip(lower=0, upper=5, inplace=True)


def test_types_abs() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.abs()


def test_types_var() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.var()
    s.var(axis=0, ddof=1)
    s.var(skipna=True, numeric_only=False)


def test_types_std() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.std()
    s.std(axis=0, ddof=1)
    s.std(skipna=True, numeric_only=False)


def test_types_idxmin() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.idxmin()
    s.idxmin(axis=0)


def test_types_idxmax() -> None:
    s = pd.Series([-10, 2, 3, 10])
    s.idxmax()
    s.idxmax(axis=0)


def test_types_value_counts() -> None:
    s = pd.Series(["a", "b"])
    check(assert_type(s.value_counts(), "pd.Series[int]"), pd.Series, np.int64)


def test_types_unique() -> None:
    s = pd.Series([-10, 2, 2, 3, 10, 10])
    s.unique()


def test_types_apply() -> None:
    s = pd.Series([-10, 2, 2, 3.4, 10, 10])

    def square(x: float) -> float:
        return x**2

    check(assert_type(s.apply(square), pd.Series), pd.Series, float)
    check(assert_type(s.apply(np.exp), pd.Series), pd.Series, float)
    check(assert_type(s.apply(str), pd.Series), pd.Series, str)

    def makeseries(x: float) -> pd.Series:
        return pd.Series([x, 2 * x])

    check(assert_type(s.apply(makeseries), pd.DataFrame), pd.DataFrame)

    # GH 293

    def retseries(x: float) -> float:
        return x

    check(assert_type(s.apply(retseries).tolist(), list), list)

    def retlist(x: float) -> list:
        return [x]

    check(assert_type(s.apply(retlist), pd.Series), pd.Series, list)

    def get_depth(url: str) -> int:
        return len(url)

    ss = s.astype(str)
    check(assert_type(ss.apply(get_depth), pd.Series), pd.Series, np.int64)


def test_types_element_wise_arithmetic() -> None:
    s = pd.Series([0, 1, -10])
    s2 = pd.Series([7, -5, 10])

    res_add1: pd.Series = s + s2
    res_add2: pd.Series = s.add(s2, fill_value=0)

    res_sub: pd.Series = s - s2
    res_sub2: pd.Series = s.sub(s2, fill_value=0)

    res_mul: pd.Series = s * s2
    res_mul2: pd.Series = s.mul(s2, fill_value=0)

    res_div: pd.Series = s / s2
    res_div2: pd.Series = s.div(s2, fill_value=0)

    res_floordiv: pd.Series = s // s2
    res_floordiv2: pd.Series = s.floordiv(s2, fill_value=0)

    res_mod: pd.Series = s % s2
    res_mod2: pd.Series = s.mod(s2, fill_value=0)

    res_pow: pd.Series = s ** s2.abs()
    res_pow2: pd.Series = s.pow(s2.abs(), fill_value=0)

    check(assert_type(divmod(s, s2), Tuple[pd.Series, pd.Series]), tuple)


def test_types_scalar_arithmetic() -> None:
    s = pd.Series([0, 1, -10])

    res_add1: pd.Series = s + 1
    res_add2: pd.Series = s.add(1, fill_value=0)

    res_sub: pd.Series = s - 1
    res_sub2: pd.Series = s.sub(1, fill_value=0)

    res_mul: pd.Series = s * 2
    res_mul2: pd.Series = s.mul(2, fill_value=0)

    res_div: pd.Series = s / 2
    res_div2: pd.Series = s.div(2, fill_value=0)

    res_floordiv: pd.Series = s // 2
    res_floordiv2: pd.Series = s.floordiv(2, fill_value=0)

    res_mod: pd.Series = s % 2
    res_mod2: pd.Series = s.mod(2, fill_value=0)

    res_pow: pd.Series = s**2
    res_pow1: pd.Series = s**0
    res_pow2: pd.Series = s**0.213
    res_pow3: pd.Series = s.pow(0.5)


# GH 103
def test_types_complex_arithmetic() -> None:
    c = 1 + 1j
    s = pd.Series([1.0, 2.0, 3.0])
    x = s + c
    y = s - c


def test_types_groupby() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"])
    s.groupby(["a", "b", "a", "b"])
    s.groupby(level=0)
    s.groupby(s > 2)
    # GH 284
    s.groupby([s > 2, s % 2 == 1])
    s.groupby(lambda x: x)
    s.groupby([lambda x: x, lambda x: x.replace("a", "b")])
    s.groupby(np.array([1, 0, 1, 0]))
    s.groupby([np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])])
    s.groupby({"a": 1, "b": 2})
    s.groupby([{"a": 1, "b": 3}, {"a": 1, "b": 1}])
    s.groupby(s.index)
    s.groupby([pd.Index([1, 0, 0, 0]), pd.Index([0, 0, 1, 0])])
    s.groupby(pd.Grouper(level=0))
    s.groupby([pd.Grouper(level=0), pd.Grouper(level=0)])


def test_types_groupby_methods() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"], dtype=int)
    check(assert_type(s.groupby(level=0).sum(), "pd.Series[int]"), pd.Series, np.int_)
    check(assert_type(s.groupby(level=0).prod(), "pd.Series[int]"), pd.Series, np.int_)
    check(assert_type(s.groupby(level=0).sem(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.groupby(level=0).std(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.groupby(level=0).var(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.groupby(level=0).tail(), "pd.Series[int]"), pd.Series, np.int_)
    check(assert_type(s.groupby(level=0).unique(), pd.Series), pd.Series)
    check(assert_type(s.groupby(level=0).idxmax(), pd.Series), pd.Series)
    check(assert_type(s.groupby(level=0).idxmin(), pd.Series), pd.Series)


def test_groupby_result() -> None:
    # GH 142
    # since there are no columns in a Series, groupby name only works
    # with a named index, we use a MultiIndex, so we can group by more
    # than one level and test the non-scalar case
    multi_index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=["a", "b"])
    s = pd.Series([0, 1, 2], index=multi_index, dtype=int)
    iterator = s.groupby(["a", "b"]).__iter__()
    assert_type(iterator, Iterator[Tuple[Tuple, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), Tuple[Tuple, "pd.Series[int]"])

    check(assert_type(index, Tuple), tuple, np.integer)
    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    iterator2 = s.groupby("a").__iter__()
    assert_type(iterator2, Iterator[Tuple[Scalar, "pd.Series[int]"]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), Tuple[Scalar, "pd.Series[int]"])

    check(assert_type(index2, Scalar), int)
    check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)

    # GH 674
    # grouping by pd.MultiIndex should always resolve to a tuple as well
    iterator3 = s.groupby(multi_index).__iter__()
    assert_type(iterator3, Iterator[Tuple[Tuple, "pd.Series[int]"]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), Tuple[Tuple, "pd.Series[int]"])

    check(assert_type(index3, Tuple), tuple, int)
    check(assert_type(value3, "pd.Series[int]"), pd.Series, np.integer)

    # Want to make sure these cases are differentiated
    for (k1, k2), g in s.groupby(["a", "b"]):
        pass

    for kk, g in s.groupby("a"):
        pass

    for (k1, k2), g in s.groupby(multi_index):
        pass


def test_groupby_result_for_scalar_indexes() -> None:
    # GH 674
    s = pd.Series([0, 1, 2], dtype=int)
    dates = pd.Series(
        [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-15"),
            pd.Timestamp("2020-02-01"),
        ],
        dtype="datetime64[ns]",
    )

    period_index = pd.PeriodIndex(dates, freq="M")
    iterator = s.groupby(period_index).__iter__()
    assert_type(iterator, Iterator[Tuple[pd.Period, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), Tuple[pd.Period, "pd.Series[int]"])

    check(assert_type(index, pd.Period), pd.Period)
    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    dt_index = pd.DatetimeIndex(dates)
    iterator2 = s.groupby(dt_index).__iter__()
    assert_type(iterator2, Iterator[Tuple[pd.Timestamp, "pd.Series[int]"]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), Tuple[pd.Timestamp, "pd.Series[int]"])

    check(assert_type(index2, pd.Timestamp), pd.Timestamp)
    check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)

    tdelta_index = pd.TimedeltaIndex(dates - pd.Timestamp("2020-01-01"))
    iterator3 = s.groupby(tdelta_index).__iter__()
    assert_type(iterator3, Iterator[Tuple[pd.Timedelta, "pd.Series[int]"]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), Tuple[pd.Timedelta, "pd.Series[int]"])

    check(assert_type(index3, pd.Timedelta), pd.Timedelta)
    check(assert_type(value3, "pd.Series[int]"), pd.Series, np.integer)

    intervals: list[pd.Interval[pd.Timestamp]] = [
        pd.Interval(date, date + pd.DateOffset(days=1), closed="left") for date in dates
    ]
    interval_index = pd.IntervalIndex(intervals)
    assert_type(interval_index, "pd.IntervalIndex[pd.Interval[pd.Timestamp]]")
    iterator4 = s.groupby(interval_index).__iter__()
    assert_type(
        iterator4, Iterator[Tuple["pd.Interval[pd.Timestamp]", "pd.Series[int]"]]
    )
    index4, value4 = next(iterator4)
    assert_type((index4, value4), Tuple["pd.Interval[pd.Timestamp]", "pd.Series[int]"])

    check(assert_type(index4, "pd.Interval[pd.Timestamp]"), pd.Interval)
    check(assert_type(value4, "pd.Series[int]"), pd.Series, np.integer)

    for p, g in s.groupby(period_index):
        pass

    for dt, g in s.groupby(dt_index):
        pass

    for tdelta, g in s.groupby(tdelta_index):
        pass

    for interval, g in s.groupby(interval_index):
        pass


def test_groupby_result_for_ambiguous_indexes() -> None:
    # GH 674
    s = pd.Series([0, 1, 2], index=["a", "b", "a"], dtype=int)
    # this will use pd.Index which is ambiguous
    iterator = s.groupby(s.index).__iter__()
    assert_type(iterator, Iterator[Tuple[Any, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), Tuple[Any, "pd.Series[int]"])

    check(assert_type(index, Any), str)
    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    # categorical indexes are also ambiguous
    categorical_index = pd.CategoricalIndex(s.index)
    iterator2 = s.groupby(categorical_index).__iter__()
    assert_type(iterator2, Iterator[Tuple[Any, "pd.Series[int]"]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), Tuple[Any, "pd.Series[int]"])

    check(assert_type(index2, Any), str)
    check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)


def test_types_groupby_agg() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"])
    check(assert_type(s.groupby(level=0).agg("sum"), pd.Series), pd.Series)
    check(assert_type(s.groupby(level=0).agg(sum), pd.Series), pd.Series)
    check(
        assert_type(s.groupby(level=0).agg(["min", "sum"]), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(s.groupby(level=0).agg([min, sum]), pd.DataFrame), pd.DataFrame)


def test_types_groupby_aggregate() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"])
    check(assert_type(s.groupby(level=0).aggregate("sum"), pd.Series), pd.Series)
    check(assert_type(s.groupby(level=0).aggregate(sum), pd.Series), pd.Series)
    check(
        assert_type(s.groupby(level=0).aggregate(["min", "sum"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(s.groupby(level=0).aggregate([min, sum]), pd.DataFrame),
        pd.DataFrame,
    )


# This added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_group_by_with_dropna_keyword() -> None:
    s = pd.Series([1, 2, 3, 3], index=["col1", "col2", "col3", np.nan])
    s.groupby(level=0, dropna=True).sum()
    s.groupby(level=0, dropna=False).sum()
    s.groupby(level=0).sum()


def test_types_groupby_iter() -> None:
    s = pd.Series([1, 1, 2], dtype=int)
    series_groupby = pd.Series([True, True, False], dtype=bool)
    first_group = next(iter(s.groupby(series_groupby)))
    check(
        assert_type(first_group[0], bool),
        bool,
    )
    check(assert_type(first_group[1], "pd.Series[int]"), pd.Series, np.integer)


def test_types_plot() -> None:
    s = pd.Series([0, 1, 1, 0, -10])
    if TYPE_CHECKING:  # skip pytest
        s.plot.hist()


def test_types_window() -> None:
    s = pd.Series([0, 1, 1, 0, 5, 1, -10])
    s.expanding()
    if PD_LTE_20:
        s.expanding(axis=0)
        s.rolling(2, axis=0, center=True)
    if TYPE_CHECKING_INVALID_USAGE:
        s.expanding(axis=0, center=True)  # type: ignore[call-arg] # pyright: ignore[reportGeneralTypeIssues]

    s.rolling(2)

    check(
        assert_type(s.rolling(2).agg("sum"), pd.Series),
        pd.Series,
    )
    check(
        assert_type(s.rolling(2).agg(sum), pd.Series),
        pd.Series,
    )
    check(
        assert_type(s.rolling(2).agg(["max", "min"]), pd.DataFrame),
        pd.DataFrame,
    )
    check(
        assert_type(s.rolling(2).agg([max, min]), pd.DataFrame),
        pd.DataFrame,
    )


def test_types_cov() -> None:
    s1 = pd.Series([0, 1, 1, 0, 5, 1, -10])
    s2 = pd.Series([0, 2, 12, -4, 7, 9, 2])
    s1.cov(s2)
    s1.cov(s2, min_periods=1)
    # ddof param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    s1.cov(s2, ddof=2)


def test_update() -> None:
    s1 = pd.Series([0, 1, 1, 0, 5, 1, -10])
    s1.update(pd.Series([0, 2, 12]))
    # Series.update() accepting objects that can be coerced to a Series was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    s1.update([1, 2, -4, 3])
    s1.update([1, "b", "c", "d"])
    s1.update({1: 9, 3: 4})


def test_to_markdown() -> None:
    pytest.importorskip("tabulate")
    s = pd.Series([0, 1, 1, 0, 5, 1, -10])
    s.to_markdown()
    s.to_markdown(buf=None, mode="wt")
    # index param was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    s.to_markdown(index=False)


# compare() method added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_compare() -> None:
    s1 = pd.Series([0, 1, 1, 0, 5, 1, -10])
    s2 = pd.Series([0, 2, 12, -4, 7, 9, 2])
    s1.compare(s2)
    s2.compare(s1, align_axis="columns", keep_shape=True, keep_equal=True)


def test_types_between() -> None:
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([0, 1, 2])
    s3 = pd.Series([2, 3, 4])
    check(assert_type(s1.between(0, 2), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s1.between([0, 1, 2], [2, 3, 4]), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s1.between(s2, s3), "pd.Series[bool]"), pd.Series, np.bool_)


def test_types_agg() -> None:
    s = pd.Series([1, 2, 3], index=["col1", "col2", "col3"])
    check(assert_type(s.agg("min"), Any), np.int64)
    check(assert_type(s.agg(min), Any), np.int64)
    check(assert_type(s.agg(["min", "max"]), pd.Series), pd.Series)
    check(assert_type(s.agg([min, max]), pd.Series), pd.Series)
    check(assert_type(s.agg({"a": "min"}), pd.Series), pd.Series)
    check(assert_type(s.agg({0: min}), pd.Series), pd.Series)
    check(assert_type(s.agg(x=max, y="min", z=np.mean), pd.Series), pd.Series)
    check(assert_type(s.agg("mean", axis=0), Any), np.float64)


def test_types_aggregate() -> None:
    s = pd.Series([1, 2, 3], index=["col1", "col2", "col3"])
    check(assert_type(s.aggregate("min"), Any), np.int64)
    check(assert_type(s.aggregate(min), Any), np.int64)
    check(assert_type(s.aggregate(["min", "max"]), pd.Series), pd.Series)
    check(assert_type(s.aggregate([min, max]), pd.Series), pd.Series)
    check(assert_type(s.aggregate({"a": "min"}), pd.Series), pd.Series)
    check(assert_type(s.aggregate({0: min}), pd.Series), pd.Series)


def test_types_transform() -> None:
    s = pd.Series([1, 2, 3], index=["col1", "col2", "col3"])
    check(assert_type(s.transform("abs"), pd.Series), pd.Series)
    check(assert_type(s.transform(abs), pd.Series), pd.Series)
    check(assert_type(s.transform(["abs", "sqrt"]), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.transform([abs, np.sqrt]), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.transform({"col1": ["abs", "sqrt"]}), pd.DataFrame), pd.DataFrame
    )
    check(
        assert_type(s.transform({"col1": [abs, np.sqrt]}), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(s.transform({"index": "abs"}), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.transform({"index": abs}), pd.DataFrame), pd.DataFrame)


def test_types_describe() -> None:
    s = pd.Series([1, 2, 3, np.datetime64("2000-01-01")])
    s.describe()
    s.describe(percentiles=[0.5], include="all")
    s.describe(exclude=np.number)


def test_types_resample() -> None:
    s = pd.Series(range(9), index=pd.date_range("1/1/2000", periods=9, freq="T"))
    s.resample("3T").sum()
    # origin and offset params added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    s.resample("20min", origin="epoch", offset=pd.Timedelta(value=2, unit="minutes"))


# set_flags() method added in 1.2.0 https://pandas.pydata.org/docs/whatsnew/v1.2.0.html
def test_types_set_flags() -> None:
    pd.Series([1, 2], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
    pd.Series([3, 4], index=["a", "a"]).set_flags(allows_duplicate_labels=True)
    pd.Series([5, 2], index=["a", "a"])


def test_types_getitem() -> None:
    s = pd.Series({"key": [0, 1, 2, 3]})
    key: list[int] = s["key"]
    s2 = pd.Series([0, 1, 2, 3])
    value: int = s2[0]
    s3: pd.Series = s[:2]


def test_types_eq() -> None:
    s1 = pd.Series([1, 2, 3])
    res1: pd.Series = s1 == 1
    s2 = pd.Series([1, 2, 4])
    res2: pd.Series = s1 == s2


def test_types_rename_axis() -> None:
    s: pd.Series = pd.Series([1, 2, 3]).rename_axis("A")


def test_types_values() -> None:
    n1: np.ndarray | ExtensionArray = pd.Series([1, 2, 3]).values
    n2: np.ndarray | ExtensionArray = pd.Series(list("aabc")).values
    n3: np.ndarray | ExtensionArray = pd.Series(list("aabc")).astype("category").values
    n4: np.ndarray | ExtensionArray = pd.Series(
        pd.date_range("20130101", periods=3, tz="US/Eastern")
    ).values


def test_types_rename() -> None:
    # Scalar
    s1 = pd.Series([1, 2, 3]).rename("A")
    check(assert_type(s1, pd.Series), pd.Series)
    # Hashable Sequence
    s2 = pd.Series([1, 2, 3]).rename(("A", "B"))
    check(assert_type(s2, pd.Series), pd.Series)

    # Optional
    s3 = pd.Series([1, 2, 3]).rename(None)
    check(assert_type(s3, pd.Series), pd.Series)

    # Functions
    def add1(x: int) -> int:
        return x + 1

    s4 = pd.Series([1, 2, 3]).rename(add1)
    check(assert_type(s4, pd.Series), pd.Series)

    # Dictionary
    s5 = pd.Series([1, 2, 3]).rename({1: 10})
    check(assert_type(s5, pd.Series), pd.Series)
    # inplace
    s6: None = pd.Series([1, 2, 3]).rename("A", inplace=True)

    if TYPE_CHECKING_INVALID_USAGE:
        s7 = pd.Series([1, 2, 3]).rename({1: [3, 4, 5]})  # type: ignore[dict-item] # pyright: ignore[reportGeneralTypeIssues]


def test_types_ne() -> None:
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([1, 2, 4])
    s3: pd.Series = s1 != s2


def test_types_bfill() -> None:
    s1 = pd.Series([1, 2, 3])
    check(assert_type(s1.bfill(), pd.Series), pd.Series)
    check(assert_type(s1.bfill(inplace=False), pd.Series), pd.Series)
    assert assert_type(s1.bfill(inplace=True), None) is None


def test_types_ewm() -> None:
    s1 = pd.Series([1, 2, 3])
    if PD_LTE_20:
        check(
            assert_type(
                s1.ewm(com=0.3, min_periods=0, adjust=False, ignore_na=True, axis=0),
                "ExponentialMovingWindow[pd.Series]",
            ),
            ExponentialMovingWindow,
        )
    check(
        assert_type(s1.ewm(alpha=0.4), "ExponentialMovingWindow[pd.Series]"),
        ExponentialMovingWindow,
    )
    check(
        assert_type(s1.ewm(span=1.6), "ExponentialMovingWindow[pd.Series]"),
        ExponentialMovingWindow,
    )
    check(
        assert_type(s1.ewm(halflife=0.7), "ExponentialMovingWindow[pd.Series]"),
        ExponentialMovingWindow,
    )
    check(
        assert_type(
            s1.ewm(com=0.3, min_periods=0, adjust=False, ignore_na=True),
            "ExponentialMovingWindow[pd.Series]",
        ),
        ExponentialMovingWindow,
    )


def test_types_ffill() -> None:
    s1 = pd.Series([1, 2, 3])
    check(assert_type(s1.ffill(), pd.Series), pd.Series)
    check(assert_type(s1.ffill(inplace=False), pd.Series), pd.Series)
    assert assert_type(s1.ffill(inplace=True), None) is None


def test_types_as_type() -> None:
    s1 = pd.Series([1, 2, 8, 9])
    s2: pd.Series = s1.astype("int32")


def test_types_dot() -> None:
    s1 = pd.Series([0, 1, 2, 3])
    s2 = pd.Series([-1, 2, -3, 4])
    df1 = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
    n1 = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
    sc1: Scalar = s1.dot(s2)
    sc2: Scalar = s1 @ s2
    s3: pd.Series = s1.dot(df1)
    s4: pd.Series = s1 @ df1
    n2: np.ndarray = s1.dot(n1)
    n3: np.ndarray = s1 @ n1


def test_series_loc_setitem() -> None:
    s = pd.Series([1, 2, 3, 4, 5])
    v = s.loc[[0, 2, 4]].values
    s.loc[[0, 2, 4]] = v


def test_series_min_max_sub_axis() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
    s1 = df.min(axis=1)
    s2 = df.max(axis=1)
    sa = s1 + s2
    ss = s1 - s2
    sm = s1 * s2
    sd = s1 / s2
    check(assert_type(sa, pd.Series), pd.Series)
    check(assert_type(ss, pd.Series), pd.Series)
    check(assert_type(sm, pd.Series), pd.Series)
    check(assert_type(sd, pd.Series), pd.Series)


def test_series_index_isin() -> None:
    s = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 2, 3, 3])
    t1 = s.loc[s.index.isin([1, 3])]
    t2 = s.loc[~s.index.isin([1, 3])]
    t3 = s[s.index.isin([1, 3])]
    t4 = s[~s.index.isin([1, 3])]
    check(assert_type(t1, pd.Series), pd.Series)
    check(assert_type(t2, pd.Series), pd.Series)
    check(assert_type(t3, pd.Series), pd.Series)
    check(assert_type(t4, pd.Series), pd.Series)


def test_series_invert() -> None:
    s1 = pd.Series([True, False, True])
    s2 = ~s1
    check(assert_type(s2, "pd.Series[bool]"), pd.Series, np.bool_)
    s3 = pd.Series([1, 2, 3])
    check(assert_type(s3[s2], pd.Series), pd.Series)
    check(assert_type(s3.loc[s2], pd.Series), pd.Series)


def test_series_multiindex_getitem() -> None:
    s = pd.Series(
        [1, 2, 3, 4], index=pd.MultiIndex.from_product([["a", "b"], ["x", "y"]])
    )
    s1: pd.Series = s["a", :]


def test_series_mul() -> None:
    s = pd.Series([1, 2, 3])
    sm = s * 4
    check(assert_type(sm, pd.Series), pd.Series)
    ss = s - 4
    check(assert_type(ss, pd.Series), pd.Series)
    sm2 = s * s
    check(assert_type(sm2, pd.Series), pd.Series)
    sp = s + 4
    check(assert_type(sp, pd.Series), pd.Series)


def test_reset_index() -> None:
    s = pd.Series(
        [1, 2, 3, 4],
        index=pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"]),
    )
    r1 = s.reset_index()
    check(assert_type(r1, pd.DataFrame), pd.DataFrame)
    r2 = s.reset_index(["ab"])
    check(assert_type(r2, pd.DataFrame), pd.DataFrame)
    r3 = s.reset_index("ab")
    check(assert_type(r3, pd.DataFrame), pd.DataFrame)
    r4 = s.reset_index(drop=True)
    check(assert_type(r4, pd.Series), pd.Series)
    r5 = s.reset_index(["ab"], drop=True)
    check(assert_type(r5, pd.Series), pd.Series)
    r6 = s.reset_index(["ab"], drop=True, allow_duplicates=True)
    check(assert_type(r6, pd.Series), pd.Series)
    assert assert_type(s.reset_index(inplace=True, drop=True), None) is None


def test_series_add_str() -> None:
    s = pd.Series(["abc", "def"])
    check(assert_type(s + "x", pd.Series), pd.Series)
    check(assert_type("x" + s, pd.Series), pd.Series)


def test_series_dtype() -> None:
    s = pd.Series(["abc", "def"], dtype=str)
    check(assert_type(s, "pd.Series[str]"), pd.Series, str)


def test_types_replace() -> None:
    # GH 44
    s = pd.Series([1, 2, 3])
    check(assert_type(s.replace(1, 2), pd.Series), pd.Series)
    check(assert_type(s.replace(1, 2, inplace=False), pd.Series), pd.Series)
    assert assert_type(s.replace(1, 2, inplace=True), None) is None


def test_cat_accessor() -> None:
    # GH 43
    s = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
    check(assert_type(s.cat.codes, "pd.Series[int]"), pd.Series, np.int8)
    # GH 139
    ser = pd.Series([1, 2, 3], name="A").astype("category")
    check(
        assert_type(ser.cat.set_categories([1, 2, 3]), pd.Series), pd.Series, np.int64
    )
    check(
        assert_type(ser.cat.reorder_categories([2, 3, 1], ordered=True), pd.Series),
        pd.Series,
        np.int64,
    )
    check(
        assert_type(ser.cat.rename_categories([1, 2, 3]), pd.Series),
        pd.Series,
        np.int64,
    )
    check(
        assert_type(ser.cat.remove_unused_categories(), pd.Series), pd.Series, np.int64
    )
    check(assert_type(ser.cat.remove_categories([2]), pd.Series), pd.Series, np.int64)
    check(assert_type(ser.cat.add_categories([4]), pd.Series), pd.Series, np.int64)
    check(assert_type(ser.cat.as_ordered(), pd.Series), pd.Series, np.int64)
    check(assert_type(ser.cat.as_unordered(), pd.Series), pd.Series, np.int64)


def test_cat_ctor_values() -> None:
    c1 = pd.Categorical(["a", "b", "a"])
    # GH 95
    c2 = pd.Categorical(pd.Series(["a", "b", "a"]))
    s: Sequence = cast(Sequence, ["a", "b", "a"])
    c3 = pd.Categorical(s)
    # GH 107
    c4 = pd.Categorical(np.array([1, 2, 3, 1, 1]))


def test_iloc_getitem_ndarray() -> None:
    # GH 85
    # GH 86
    indices_i8 = np.array([0, 1, 2, 3], dtype=np.int8)
    indices_i16 = np.array([0, 1, 2, 3], dtype=np.int16)
    indices_i32 = np.array([0, 1, 2, 3], dtype=np.int_)
    indices_i64 = np.array([0, 1, 2, 3], dtype=np.int64)

    indices_u8 = np.array([0, 1, 2, 3], dtype=np.uint8)
    indices_u16 = np.array([0, 1, 2, 3], dtype=np.uint16)
    indices_u32 = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices_u64 = np.array([0, 1, 2, 3], dtype=np.uint64)

    values_s = pd.Series(np.arange(10), name="a")

    check(assert_type(values_s.iloc[indices_i8], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_i16], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_i32], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_i64], pd.Series), pd.Series)

    check(assert_type(values_s.iloc[indices_u8], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_u16], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_u32], pd.Series), pd.Series)
    check(assert_type(values_s.iloc[indices_u64], pd.Series), pd.Series)


def test_iloc_setitem_ndarray() -> None:
    # GH 85
    # GH 86
    indices_i8 = np.array([0, 1, 2, 3], dtype=np.int8)
    indices_i16 = np.array([0, 1, 2, 3], dtype=np.int16)
    indices_i32 = np.array([0, 1, 2, 3], dtype=np.int_)
    indices_i64 = np.array([0, 1, 2, 3], dtype=np.int64)

    indices_u8 = np.array([0, 1, 2, 3], dtype=np.uint8)
    indices_u16 = np.array([0, 1, 2, 3], dtype=np.uint16)
    indices_u32 = np.array([0, 1, 2, 3], dtype=np.uint32)
    indices_u64 = np.array([0, 1, 2, 3], dtype=np.uint64)

    values_s = pd.Series(np.arange(10), name="a")

    values_s.iloc[indices_i8] = -1
    values_s.iloc[indices_i16] = -1
    values_s.iloc[indices_i32] = -1
    values_s.iloc[indices_i64] = -1

    values_s.iloc[indices_u8] = -1
    values_s.iloc[indices_u16] = -1
    values_s.iloc[indices_u32] = -1
    values_s.iloc[indices_u64] = -1


def test_types_iter() -> None:
    s = pd.Series([1, 2, 3], dtype=int)
    iterable: Iterable[int] = s
    assert_type(iter(s), Iterator[int])
    assert_type(next(iter(s)), int)


def test_types_to_list() -> None:
    s = pd.Series(["a", "b", "c"], dtype=str)
    check(assert_type(s.tolist(), List[str]), list, str)
    check(assert_type(s.to_list(), List[str]), list, str)


def test_types_to_dict() -> None:
    s = pd.Series(["a", "b", "c"], dtype=str)
    assert_type(s.to_dict(), Dict[Any, str])


def test_categorical_codes():
    # GH-111
    cat = pd.Categorical(["a", "b", "a"])
    assert_type(cat.codes, "np_ndarray_int")


def test_string_accessors():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    s2 = pd.Series([["apple", "banana"], ["cherry", "date"], [1, "eggplant"]])
    s3 = pd.Series(["a1", "b2", "c3"])
    check(assert_type(s.str.capitalize(), pd.Series), pd.Series)
    check(assert_type(s.str.casefold(), pd.Series), pd.Series)
    check(assert_type(s.str.cat(sep="X"), str), str)
    check(assert_type(s.str.center(10), pd.Series), pd.Series)
    check(assert_type(s.str.contains("a"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.count("pp"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.decode("utf-8"), pd.Series), pd.Series)
    check(assert_type(s.str.encode("latin-1"), pd.Series), pd.Series)
    check(assert_type(s.str.endswith("e"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.str.endswith(("e", "f")), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(assert_type(s3.str.extract(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
    check(assert_type(s3.str.extractall(r"([ab])?(\d)"), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.find("p"), pd.Series), pd.Series)
    check(assert_type(s.str.findall("pp"), pd.Series), pd.Series)
    check(assert_type(s.str.fullmatch("apple"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.get(2), pd.Series), pd.Series)
    check(assert_type(s.str.get_dummies(), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.index("p"), pd.Series), pd.Series)
    check(assert_type(s.str.isalnum(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isalpha(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isdecimal(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isdigit(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isnumeric(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.islower(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isspace(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.istitle(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.str.isupper(), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s2.str.join("-"), pd.Series), pd.Series)
    check(assert_type(s.str.len(), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.str.ljust(80), pd.Series), pd.Series)
    check(assert_type(s.str.lower(), pd.Series), pd.Series)
    check(assert_type(s.str.lstrip("a"), pd.Series), pd.Series)
    check(assert_type(s.str.match("pp"), pd.Series), pd.Series)
    check(assert_type(s.str.normalize("NFD"), pd.Series), pd.Series)
    check(assert_type(s.str.pad(80, "right"), pd.Series), pd.Series)
    check(assert_type(s.str.partition("p"), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.removeprefix("a"), pd.Series), pd.Series)
    check(assert_type(s.str.removesuffix("e"), pd.Series), pd.Series)
    check(assert_type(s.str.repeat(2), pd.Series), pd.Series)
    check(assert_type(s.str.replace("a", "X"), pd.Series), pd.Series)
    check(assert_type(s.str.rfind("e"), pd.Series), pd.Series)
    check(assert_type(s.str.rindex("p"), pd.Series), pd.Series)
    check(assert_type(s.str.rjust(80), pd.Series), pd.Series)
    check(assert_type(s.str.rpartition("p"), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.rsplit("a"), pd.Series), pd.Series)
    check(assert_type(s.str.rsplit("a", expand=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.rstrip(), pd.Series), pd.Series)
    check(assert_type(s.str.slice(0, 4, 2), pd.Series), pd.Series)
    check(assert_type(s.str.slice_replace(0, 2, "XX"), pd.Series), pd.Series)
    check(assert_type(s.str.split("a"), pd.Series), pd.Series)
    # GH 194
    check(assert_type(s.str.split("a", expand=True), pd.DataFrame), pd.DataFrame)
    check(assert_type(s.str.startswith("a"), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.str.startswith(("a", "b")), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s.str.strip(), pd.Series), pd.Series)
    check(assert_type(s.str.swapcase(), pd.Series), pd.Series)
    check(assert_type(s.str.title(), pd.Series), pd.Series)
    check(assert_type(s.str.translate(None), pd.Series), pd.Series)
    check(assert_type(s.str.upper(), pd.Series), pd.Series)
    check(assert_type(s.str.wrap(80), pd.Series), pd.Series)
    check(assert_type(s.str.zfill(10), pd.Series), pd.Series)


def test_series_overloads_cat():
    s = pd.Series(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(s.str.cat(sep=";"), str), str)
    check(assert_type(s.str.cat(None, sep=";"), str), str)
    check(
        assert_type(s.str.cat(["A", "B", "C", "D", "E", "F", "G"], sep=";"), pd.Series),
        pd.Series,
    )


def test_series_overloads_partition():
    s = pd.Series(
        [
            "ap;pl;ep",
            "ban;an;ap",
            "Che;rr;yp",
            "DA;TEp",
            "eGGp;LANT;p",
            "12;3p",
            "23.45p",
        ]
    )
    check(assert_type(s.str.partition(sep=";"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.partition(sep=";", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(s.str.partition(sep=";", expand=False), pd.Series), pd.Series)

    check(assert_type(s.str.rpartition(sep=";"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.rpartition(sep=";", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(s.str.rpartition(sep=";", expand=False), pd.Series), pd.Series)


def test_series_overloads_extract():
    s = pd.Series(
        ["appl;ep", "ban;anap", "Cherr;yp", "DATEp", "eGGp;LANTp", "12;3p", "23.45p"]
    )
    check(assert_type(s.str.extract(r"[ab](\d)"), pd.DataFrame), pd.DataFrame)
    check(
        assert_type(s.str.extract(r"[ab](\d)", expand=True), pd.DataFrame), pd.DataFrame
    )
    check(assert_type(s.str.extract(r"[ab](\d)", expand=False), pd.Series), pd.Series)
    check(
        assert_type(s.str.extract(r"[ab](\d)", re.IGNORECASE, False), pd.Series),
        pd.Series,
    )


def test_relops() -> None:
    # GH 175
    s: str = "abc"
    check(assert_type(pd.Series([s]) > s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([s]) < s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([s]) <= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([s]) >= s, "pd.Series[bool]"), pd.Series, np.bool_)

    b: bytes = b"def"
    check(assert_type(pd.Series([b]) > b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([b]) < b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([b]) <= b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([b]) >= b, "pd.Series[bool]"), pd.Series, np.bool_)

    dtd = datetime.date(2022, 7, 31)
    check(assert_type(pd.Series([dtd]) > dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([dtd]) < dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([dtd]) <= dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([dtd]) >= dtd, "pd.Series[bool]"), pd.Series, np.bool_)

    dtdt = datetime.datetime(2022, 7, 31, 8, 32, 21)
    check(assert_type(pd.Series([dtdt]) > dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([dtdt]) < dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(pd.Series([dtdt]) <= dtdt, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Series([dtdt]) >= dtdt, "pd.Series[bool]"), pd.Series, np.bool_
    )

    dttd = datetime.timedelta(seconds=10)
    check(assert_type(pd.Series([dttd]) > dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([dttd]) < dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(pd.Series([dttd]) <= dttd, "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(
        assert_type(pd.Series([dttd]) >= dttd, "pd.Series[bool]"), pd.Series, np.bool_
    )

    bo: bool = True
    check(assert_type(pd.Series([bo]) > bo, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([bo]) < bo, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([bo]) <= bo, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([bo]) >= bo, "pd.Series[bool]"), pd.Series, np.bool_)

    ai: int = 10
    check(assert_type(pd.Series([ai]) > ai, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ai]) < ai, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ai]) <= ai, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ai]) >= ai, "pd.Series[bool]"), pd.Series, np.bool_)

    af: float = 3.14
    check(assert_type(pd.Series([af]) > af, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([af]) < af, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([af]) <= af, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([af]) >= af, "pd.Series[bool]"), pd.Series, np.bool_)

    ac: complex = 1 + 2j
    check(assert_type(pd.Series([ac]) > ac, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ac]) < ac, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ac]) <= ac, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ac]) >= ac, "pd.Series[bool]"), pd.Series, np.bool_)

    ts = pd.Timestamp("2022-07-31 08:35:12")
    check(assert_type(pd.Series([ts]) > ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ts]) < ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ts]) <= ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([ts]) >= ts, "pd.Series[bool]"), pd.Series, np.bool_)

    td = pd.Timedelta(seconds=10)
    check(assert_type(pd.Series([td]) > td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([td]) < td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([td]) <= td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(pd.Series([td]) >= td, "pd.Series[bool]"), pd.Series, np.bool_)


def test_resample() -> None:
    # GH 181
    N = 10
    index = pd.date_range("1/1/2000", periods=N, freq="T")
    x = [x for x in range(N)]
    df = pd.Series(x, index=index)
    check(assert_type(df.resample("2T").std(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").var(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").quantile(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").sum(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").prod(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").min(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").max(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").first(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").last(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").mean(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").sem(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").median(), pd.Series), pd.Series)
    check(assert_type(df.resample("2T").ohlc(), pd.DataFrame), pd.DataFrame)


def test_to_xarray():
    s = pd.Series([1, 2])
    check(assert_type(s.to_xarray(), xr.DataArray), xr.DataArray)


def test_neg() -> None:
    # GH 253
    sr = pd.Series([1, 2, 3])
    sr_int = pd.Series([1, 2, 3], dtype=int)
    check(assert_type(-sr, pd.Series), pd.Series)
    check(assert_type(-sr_int, "pd.Series[int]"), pd.Series, np.int_)


def test_getattr() -> None:
    # GH 261
    series = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype=int)
    check(assert_type(series.a, int), np.integer)


def test_dtype_type() -> None:
    # GH 216
    s1 = pd.Series(["foo"], dtype="string")
    check(assert_type(s1.dtype, DtypeObj), ExtensionDtype)
    check(assert_type(s1.dtype.kind, str), str)

    s2 = pd.Series([1], dtype="Int64")
    check(assert_type(s2.dtype, DtypeObj), ExtensionDtype)
    check(assert_type(s2.dtype.kind, str), str)

    s3 = pd.Series([1, 2, 3])
    check(assert_type(s3.dtype, DtypeObj), np.dtype)
    check(assert_type(s3.dtype.kind, str), str)


def test_types_to_numpy() -> None:
    s = pd.Series(["a", "b", "c"], dtype=str)
    check(assert_type(s.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(s.to_numpy(dtype="str", copy=True), np.ndarray), np.ndarray)
    check(assert_type(s.to_numpy(na_value=0), np.ndarray), np.ndarray)


def test_where() -> None:
    s = pd.Series([1, 2, 3], dtype=int)

    def cond1(x: int) -> bool:
        return x % 2 == 0

    check(assert_type(s.where(cond1, other=0), "pd.Series[int]"), pd.Series, np.int_)

    def cond2(x: pd.Series[int]) -> pd.Series[bool]:
        return x > 1

    check(assert_type(s.where(cond2, other=0), "pd.Series[int]"), pd.Series, np.int_)

    cond3 = pd.Series([False, True, True])
    check(assert_type(s.where(cond3, other=0), "pd.Series[int]"), pd.Series, np.int_)


def test_bitwise_operators() -> None:
    s = pd.Series([1, 2, 3, 4], dtype=int)
    s2 = pd.Series([9, 10, 11, 12], dtype=int)
    # for issue #348 (bitwise operators on Series should support int)
    # The bitwise integers return platform-dependent numpy integers in the Series
    check(assert_type(s & 3, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(3 & s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s | 3, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(3 | s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s ^ 3, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(3 ^ s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s & s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2 & s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s | s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2 | s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s ^ s2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2 ^ s, "pd.Series[int]"), pd.Series, np.integer)

    check(assert_type(s & [1, 2, 3, 4], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([1, 2, 3, 4] & s, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(s | [1, 2, 3, 4], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([1, 2, 3, 4] | s, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(s ^ [1, 2, 3, 4], "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type([1, 2, 3, 4] ^ s, "pd.Series[bool]"), pd.Series, np.bool_)


def test_logical_operators() -> None:
    # GH 380
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})

    check(
        assert_type((df["a"] >= 2) & (df["b"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type((df["a"] >= 2) | (df["b"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type((df["a"] >= 2) ^ (df["b"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type((df["a"] >= 2) & True, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type((df["a"] >= 2) | True, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type((df["a"] >= 2) ^ True, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(True & (df["a"] >= 2), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(True | (df["a"] >= 2), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(True ^ (df["a"] >= 2), "pd.Series[bool]"), pd.Series, np.bool_)

    check(
        assert_type((df["a"] >= 2) ^ [True, False, True], "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type((df["a"] >= 2) & [True, False, True], "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type((df["a"] >= 2) | [True, False, True], "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type([True, False, True] & (df["a"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type([True, False, True] | (df["a"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(
        assert_type([True, False, True] ^ (df["a"] >= 2), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )


def test_AnyArrayLike_and_clip() -> None:
    ser = pd.Series([1, 2, 3])
    s1 = ser.clip(lower=ser)
    s2 = ser.clip(upper=ser)
    check(assert_type(s1, pd.Series), pd.Series)
    check(assert_type(s2, pd.Series), pd.Series)


def test_pandera_generic() -> None:
    # GH 471
    T = TypeVar("T")

    class MySeries(pd.Series, Generic[T]):
        def __new__(cls, *args, **kwargs) -> Self:
            return object.__new__(cls)

    def func() -> MySeries[float]:
        return MySeries[float]([1, 2, 3])

    result = func()
    assert result.iloc[1] == 2


def test_change_to_dict_return_type() -> None:
    id = [1, 2, 3]
    value = ["a", "b", "c"]
    df = pd.DataFrame(zip(id, value), columns=["id", "value"])
    fd = df.set_index("id")["value"].to_dict()
    check(assert_type(fd, Dict[Any, Any]), dict)


def test_updated_astype() -> None:
    s = pd.Series([3, 4, 5])
    s1 = pd.Series(True)

    # Boolean types

    # Builtin bool types
    check(assert_type(s.astype(bool), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.astype("bool"), "pd.Series[bool]"), pd.Series, np.bool_)
    # Pandas nullable boolean types
    check(
        assert_type(s1.astype(pd.BooleanDtype()), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s1.astype("boolean"), "pd.Series[bool]"), pd.Series, np.bool_)
    # Numpy bool type
    check(assert_type(s.astype(np.bool_), "pd.Series[bool]"), pd.Series, np.bool_)

    # Integer types

    # Builtin integer types
    check(assert_type(s.astype(int), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.astype("int"), "pd.Series[int]"), pd.Series, np.integer)
    # Pandas nullable integer types
    check(assert_type(s.astype(pd.Int8Dtype()), "pd.Series[int]"), pd.Series, np.int8)
    check(assert_type(s.astype(pd.Int16Dtype()), "pd.Series[int]"), pd.Series, np.int16)
    check(assert_type(s.astype(pd.Int32Dtype()), "pd.Series[int]"), pd.Series, np.int32)
    check(assert_type(s.astype(pd.Int64Dtype()), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.astype("Int8"), "pd.Series[int]"), pd.Series, np.int8)
    check(assert_type(s.astype("Int16"), "pd.Series[int]"), pd.Series, np.int16)
    check(assert_type(s.astype("Int32"), "pd.Series[int]"), pd.Series, np.int32)
    check(assert_type(s.astype("Int64"), "pd.Series[int]"), pd.Series, np.int64)
    # Numpy signed integer types
    check(assert_type(s.astype(np.byte), "pd.Series[int]"), pd.Series, np.byte)
    check(assert_type(s.astype(np.int8), "pd.Series[int]"), pd.Series, np.int8)
    check(assert_type(s.astype(np.int16), "pd.Series[int]"), pd.Series, np.int16)
    check(assert_type(s.astype(np.int32), "pd.Series[int]"), pd.Series, np.int32)
    check(assert_type(s.astype(np.int64), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.astype(np.intp), "pd.Series[int]"), pd.Series, np.intp)
    check(assert_type(s.astype("byte"), "pd.Series[int]"), pd.Series, np.byte)
    check(assert_type(s.astype("int8"), "pd.Series[int]"), pd.Series, np.int8)
    check(assert_type(s.astype("int16"), "pd.Series[int]"), pd.Series, np.int16)
    check(assert_type(s.astype("int32"), "pd.Series[int]"), pd.Series, np.int32)
    check(assert_type(s.astype("int64"), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s.astype("intp"), "pd.Series[int]"), pd.Series, np.intp)
    # Numpy unsigned integer types
    check(assert_type(s.astype(np.ubyte), "pd.Series[int]"), pd.Series, np.ubyte)
    check(assert_type(s.astype(np.uint8), "pd.Series[int]"), pd.Series, np.uint8)
    check(assert_type(s.astype(np.uint16), "pd.Series[int]"), pd.Series, np.uint16)
    check(assert_type(s.astype(np.uint32), "pd.Series[int]"), pd.Series, np.uint32)
    check(assert_type(s.astype(np.uint64), "pd.Series[int]"), pd.Series, np.uint64)
    check(assert_type(s.astype(np.uintp), "pd.Series[int]"), pd.Series, np.uintp)
    check(assert_type(s.astype("ubyte"), "pd.Series[int]"), pd.Series, np.ubyte)
    check(assert_type(s.astype("uint8"), "pd.Series[int]"), pd.Series, np.uint8)
    check(assert_type(s.astype("uint16"), "pd.Series[int]"), pd.Series, np.uint16)
    check(assert_type(s.astype("uint32"), "pd.Series[int]"), pd.Series, np.uint32)
    check(assert_type(s.astype("uint64"), "pd.Series[int]"), pd.Series, np.uint64)
    check(assert_type(s.astype("uintp"), "pd.Series[int]"), pd.Series, np.uintp)

    # String types

    # Builtin str types
    check(assert_type(s.astype(str), "pd.Series[str]"), pd.Series, str)
    check(assert_type(s.astype("str"), "pd.Series[str]"), pd.Series, str)
    # Pandas nullable string types
    check(assert_type(s.astype(pd.StringDtype()), "pd.Series[str]"), pd.Series, str)
    check(assert_type(s.astype("string"), "pd.Series[str]"), pd.Series, str)

    # Bytes types

    check(assert_type(s.astype(bytes), "pd.Series[bytes]"), pd.Series, bytes)

    # Float types

    # Builtin float types
    check(assert_type(s.astype(float), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.astype("float"), "pd.Series[float]"), pd.Series, float)
    # Pandas nullable float types
    check(
        assert_type(s.astype(pd.Float32Dtype()), "pd.Series[float]"),
        pd.Series,
        np.float32,
    )
    check(
        assert_type(s.astype(pd.Float64Dtype()), "pd.Series[float]"),
        pd.Series,
        np.float64,
    )
    check(assert_type(s.astype("Float32"), "pd.Series[float]"), pd.Series, np.float32)
    check(assert_type(s.astype("Float64"), "pd.Series[float]"), pd.Series, np.float64)
    # Numpy float types
    check(assert_type(s.astype(np.float16), "pd.Series[float]"), pd.Series, np.float16)
    check(assert_type(s.astype(np.float32), "pd.Series[float]"), pd.Series, np.float32)
    check(assert_type(s.astype(np.float64), "pd.Series[float]"), pd.Series, np.float64)
    check(assert_type(s.astype("float16"), "pd.Series[float]"), pd.Series, np.float16)
    check(assert_type(s.astype("float32"), "pd.Series[float]"), pd.Series, np.float32)
    check(assert_type(s.astype("float64"), "pd.Series[float]"), pd.Series, np.float64)

    # Complex types

    # Builtin complex types
    check(assert_type(s.astype(complex), "pd.Series[complex]"), pd.Series, complex)
    check(assert_type(s.astype("complex"), "pd.Series[complex]"), pd.Series, complex)
    # Numpy complex types
    check(
        assert_type(s.astype(np.complex64), "pd.Series[complex]"),
        pd.Series,
        np.complex64,
    )
    check(
        assert_type(s.astype(np.complex128), "pd.Series[complex]"),
        pd.Series,
        np.complex128,
    )
    check(
        assert_type(s.astype("complex64"), "pd.Series[complex]"),
        pd.Series,
        np.complex64,
    )
    check(
        assert_type(s.astype("complex128"), "pd.Series[complex]"),
        pd.Series,
        np.complex128,
    )

    check(
        assert_type(s.astype("timedelta64[Y]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[M]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[W]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[D]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[h]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[m]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[s]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[ms]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[us]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[μs]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[ns]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[ps]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[fs]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )
    check(
        assert_type(s.astype("timedelta64[as]"), TimedeltaSeries),
        pd.Series,
        Timedelta,
    )

    check(
        assert_type(s.astype("datetime64[Y]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[M]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[W]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[D]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[h]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[m]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[s]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[ms]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[us]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[μs]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[ns]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[ps]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[fs]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )
    check(
        assert_type(s.astype("datetime64[as]"), TimestampSeries),
        pd.Series,
        Timestamp,
    )

    orseries = pd.Series([Decimal(x) for x in [1, 2, 3]])
    newtype = DecimalDtype()
    decseries = orseries.astype(newtype)
    check(
        assert_type(decseries, pd.Series),
        pd.Series,
        Decimal,
    )

    s4 = pd.Series([1, 1])
    s5 = pd.Series([s4, 4])
    population_dict = {
        "California": 38332521,
        "Texas": 26448193,
        "New York": 19651127,
        "Florida": 19552860,
        "Illinois": 12882135,
    }
    population = pd.Series(population_dict)

    check(assert_type(s4.astype(object), pd.Series), pd.Series, object)
    check(assert_type(s5.astype(object), pd.Series), pd.Series, object)
    check(assert_type(population.astype(object), pd.Series), pd.Series, object)

    # Categorical
    check(
        assert_type(s.astype(pd.CategoricalDtype()), "pd.Series[Any]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(s.astype("category"), "pd.Series[Any]"),
        pd.Series,
        np.integer,
    )


def test_check_xs() -> None:
    s4 = pd.Series([1, 4])
    s4.xs(0, axis=0)
    check(assert_type(s4, pd.Series), pd.Series)


def test_types_apply_set() -> None:
    series_of_lists: pd.Series = pd.Series(
        {"list1": [1, 2, 3], "list2": ["a", "b", "c"], "list3": [True, False, True]}
    )
    check(assert_type(series_of_lists.apply(lambda x: set(x)), pd.Series), pd.Series)


def test_prefix_summix_axis() -> None:
    s = pd.Series([1, 2, 3, 4])
    check(assert_type(s.add_suffix("_item", axis=0), pd.Series), pd.Series)
    check(assert_type(s.add_suffix("_item", axis="index"), pd.Series), pd.Series)
    check(assert_type(s.add_prefix("_item", axis=0), pd.Series), pd.Series)
    check(assert_type(s.add_prefix("_item", axis="index"), pd.Series), pd.Series)

    if TYPE_CHECKING_INVALID_USAGE:
        check(assert_type(s.add_prefix("_item", axis=1), pd.Series), pd.Series)  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]
        check(assert_type(s.add_suffix("_item", axis="columns"), pd.Series), pd.Series)  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]


def test_convert_dtypes_dtype_backend() -> None:
    s = pd.Series([1, 2, 3, 4])
    s1 = s.convert_dtypes(dtype_backend="numpy_nullable")
    check(assert_type(s1, pd.Series), pd.Series)


def test_apply_returns_none() -> None:
    # GH 557
    s = pd.Series([1, 2, 3])
    check(assert_type(s.apply(lambda x: None), pd.Series), pd.Series)


def test_loc_callable() -> None:
    # GH 586
    s = pd.Series([1, 2])
    check(assert_type(s.loc[lambda x: x > 1], pd.Series), pd.Series)


def test_to_json_mode() -> None:
    s = pd.Series([1, 2, 3, 4])
    result = s.to_json(orient="records", lines=True, mode="a")
    result1 = s.to_json(orient="split", mode="w")
    result2 = s.to_json(orient="table", mode="w")
    result4 = s.to_json(orient="records", mode="w")
    check(assert_type(result, str), str)
    check(assert_type(result1, str), str)
    check(assert_type(result2, str), str)
    check(assert_type(result4, str), str)
    if TYPE_CHECKING_INVALID_USAGE:
        result3 = s.to_json(orient="records", lines=False, mode="a")  # type: ignore[call-overload] # pyright: ignore[reportGeneralTypeIssues]


def test_groupby_diff() -> None:
    # GH 658
    s = pd.Series([1, 2, 3, np.nan])
    check(assert_type(s.groupby(level=0).diff(), pd.Series), pd.Series)


def test_to_string() -> None:
    # GH 720
    s = pd.Series([1])
    check(
        assert_type(
            s.to_string(
                index=False, header=False, length=False, dtype=False, name=False
            ),
            str,
        ),
        str,
    )
