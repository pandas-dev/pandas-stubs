from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Sequence,
)
import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
import platform
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from pandas._testing import ensure_clean
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.window import ExponentialMovingWindow
import pytest
from typing_extensions import (
    Self,
    TypeAlias,
    assert_never,
    assert_type,
)
import xarray as xr

from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import YearEnd
from pandas._typing import (
    DtypeObj,
    Scalar,
)

from tests import (
    PD_LTE_22,
    TYPE_CHECKING_INVALID_USAGE,
    WINDOWS,
    check,
    pytest_warns_bounded,
)
from tests.extension.decimal.array import DecimalDtype

if TYPE_CHECKING:
    from pandas.core.series import (
        OffsetSeries,
        TimedeltaSeries,
        TimestampSeries,
    )
else:
    TimedeltaSeries: TypeAlias = pd.Series
    TimestampSeries: TypeAlias = pd.Series
    OffsetSeries: TypeAlias = pd.Series

if TYPE_CHECKING:
    from pandas._typing import (
        BooleanDtypeArg,
        BytesDtypeArg,
        CategoryDtypeArg,
        ComplexDtypeArg,
        FloatDtypeArg,
        IntDtypeArg,
        ObjectDtypeArg,
        StrDtypeArg,
        TimedeltaDtypeArg,
        TimestampDtypeArg,
        UIntDtypeArg,
        VoidDtypeArg,
    )
    from pandas._typing import np_ndarray_int  # noqa: F401

# Tests will use numpy 2.1 in python 3.10 or later
# From Numpy 2.1 __init__.pyi
_DTypeKind: TypeAlias = Literal[
    "b",  # boolean
    "i",  # signed integer
    "u",  # unsigned integer
    "f",  # floating-point
    "c",  # complex floating-point
    "m",  # timedelta64
    "M",  # datetime64
    "O",  # python object
    "S",  # byte-string (fixed-width)
    "U",  # unicode-string (fixed-width)
    "V",  # void
    "T",  # unicode-string (variable-width)
]


def test_types_init() -> None:
    pd.Series(1)
    pd.Series((1, 2, 3))
    pd.Series(np.array([1, 2, 3]))
    pd.Series(pd.NaT)
    pd.Series(pd.NA)
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

    groupby = pd.Series(np.array([1, 2])).groupby(level=0)
    resampler = pd.Series(np.array([1, 2]), index=dt).resample("1D")
    pd.Series(data=groupby)
    pd.Series(data=resampler)


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
    check(assert_type(s.copy(), "pd.Series[int]"), pd.Series, np.integer)


def test_types_select() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    if PD_LTE_22:
        # Not valid in 3.0
        with pytest_warns_bounded(
            FutureWarning,
            "Series.__getitem__ treating keys as positions is deprecated",
            lower="2.0.99",
        ):
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


def test_multiindex_loc_str_tuple() -> None:
    s = pd.Series(
        [1, 2, 3, 4, 5, 6],
        index=pd.MultiIndex.from_product([["A", "B"], ["c", "d", "e"]]),
        dtype=int,
    )
    check(assert_type(s.loc[("A", "c")], int), np.int_)
    check(
        assert_type(s.loc[[("A", "c"), ("B", "d")]], "pd.Series[int]"),
        pd.Series,
        np.int_,
    )


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
    check(assert_type(s.drop(0), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.drop([0, 1]), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.drop(0, axis=0), "pd.Series[int]"), pd.Series, np.integer)
    assert assert_type(s.drop([0, 1], inplace=True, errors="raise"), None) is None
    assert assert_type(s.drop([0, 1], inplace=True, errors="ignore"), None) is None
    # GH 302
    s = pd.Series([0, 1, 2])
    check(
        assert_type(s.drop(pd.Index([0, 1])), "pd.Series[int]"), pd.Series, np.integer
    )
    check(
        assert_type(s.drop(index=pd.Index([0, 1])), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


def test_arguments_drop() -> None:
    # GH 950
    if TYPE_CHECKING_INVALID_USAGE:
        s = pd.Series([0, 1, 2])
        res1 = s.drop()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
        res2 = s.drop([0], columns=["col1"])  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        res3 = s.drop([0], index=[0])  # type: ignore[call-overload] # pyright: ignore[reportCallIssue, reportArgumentType]
        # These should also fail, but `None` is Hasheable and i do not know how
        # to type hint a non-None hashable.
        # res4 = s.drop(columns=None)
        # res5 = s.drop(index=None)
        # res6 = s.drop(None)


def test_types_drop_multilevel() -> None:
    index = pd.MultiIndex(
        levels=[["top", "bottom"], ["first", "second", "third"]],
        codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
    )
    s = pd.Series(data=[1, 2, 3, 4, 5, 6], index=index)
    res: pd.Series = s.drop(labels="first", level=1)


def test_types_drop_duplicates() -> None:
    s = pd.Series([1.0, 2.0, 2.0])
    check(assert_type(s.drop_duplicates(), "pd.Series[float]"), pd.Series, float)
    assert assert_type(s.drop_duplicates(inplace=True), None) is None
    assert (
        assert_type(s.drop_duplicates(inplace=True, ignore_index=False), None) is None
    )


def test_types_dropna() -> None:
    s = pd.Series([1.0, np.nan, np.nan])
    check(assert_type(s.dropna(), "pd.Series[float]"), pd.Series, float)
    assert assert_type(s.dropna(axis=0, inplace=True), None) is None
    assert assert_type(s.dropna(axis=0, inplace=True, ignore_index=True), None) is None


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
    s = pd.Series([1.0, np.nan, np.nan, 3.0])
    check(assert_type(s.fillna(0), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.fillna(0, axis="index"), "pd.Series[float]"), pd.Series, float)
    check(
        assert_type(s.fillna(0, axis=0), "pd.Series[float]"),
        pd.Series,
        float,
    )
    assert assert_type(s.fillna(0, inplace=True), None) is None
    check(assert_type(s.fillna(0), "pd.Series[float]"), pd.Series, float)
    check(
        assert_type(s.fillna(0, limit=1), "pd.Series[float]"),
        pd.Series,
        float,
    )
    # GH 263
    check(assert_type(s.fillna(pd.NA), "pd.Series[float]"), pd.Series, float)


def test_types_sort_index() -> None:
    s = pd.Series([1, 2, 3], index=[2, 3, 1])
    check(assert_type(s.sort_index(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.sort_index(ascending=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    assert assert_type(s.sort_index(ascending=False, inplace=True), None) is None
    check(
        assert_type(s.sort_index(kind="mergesort"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


# This was added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
def test_types_sort_index_with_key() -> None:
    s = pd.Series([1, 2, 3], index=["a", "B", "c"])
    res: pd.Series = s.sort_index(key=lambda k: k.str.lower())


def test_types_sort_values() -> None:
    s = pd.Series([4, 2, 1, 3])
    check(assert_type(s.sort_values(), "pd.Series[int]"), pd.Series, np.integer)
    if TYPE_CHECKING_INVALID_USAGE:
        check(assert_type(s.sort_values(0), pd.Series), pd.Series)  # type: ignore[assert-type,call-overload] # pyright: ignore[reportAssertTypeFailure,reportCallIssue]
    check(assert_type(s.sort_values(axis=0), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.sort_values(ascending=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    assert assert_type(s.sort_values(inplace=True, kind="quicksort"), None) is None
    check(
        assert_type(s.sort_values(na_position="last"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(s.sort_values(ignore_index=True), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


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

    s0 = assert_type(pd.Series([1.0, 2.0, 3.0, np.nan]), "pd.Series[float]")
    check(assert_type(s0.sum(), float), np.float64)
    check(assert_type(s0.sum(skipna=False), float), np.float64)
    check(assert_type(s0.sum(numeric_only=False), float), np.float64)
    assert_type(s0.sum(min_count=4), float)

    s1 = assert_type(pd.Series([False, True], dtype=bool), "pd.Series[bool]")
    check(assert_type(s1.sum(), "int"), np.integer)
    check(assert_type(s1.sum(skipna=False), "int"), np.integer)
    check(assert_type(s1.sum(numeric_only=False), "int"), np.integer)
    assert_type(s1.sum(min_count=4), "int")

    s2 = assert_type(pd.Series([0, 1], dtype=int), "pd.Series[int]")
    check(assert_type(s2.sum(), "int"), np.integer)
    check(assert_type(s2.sum(skipna=False), "int"), np.integer)
    check(assert_type(s2.sum(numeric_only=False), "int"), np.integer)
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


def test_types_groupby_level() -> None:
    # GH 836
    index = pd.MultiIndex.from_tuples(
        [(0, 0, 1), (0, 1, 2), (0, 0, 3)], names=["col1", "col2", "col3"]
    )
    s = pd.Series([1, 2, 3], index=index)
    check(
        assert_type(s.groupby(level=["col1", "col2"]).sum(), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


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
    check(assert_type(s.value_counts(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.value_counts(normalize=True), "pd.Series[float]"),
        pd.Series,
        float,
    )


def test_types_unique() -> None:
    s = pd.Series([-10, 2, 2, 3, 10, 10])
    check(assert_type(s.unique(), np.ndarray), np.ndarray)


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
    check(assert_type(ss.apply(get_depth), pd.Series), pd.Series, np.integer)

    check(assert_type(s.apply(lambda x: pd.NA), pd.Series), pd.Series, NAType)


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

    check(assert_type(divmod(s, s2), tuple["pd.Series[int]", "pd.Series[int]"]), tuple)


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

    s2 = pd.Series(["w", "x", "y", "z"], index=[3, 4, 3, 4], dtype=str)
    check(
        assert_type(s2.groupby(level=0).count(), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


def test_groupby_result() -> None:
    # GH 142
    # since there are no columns in a Series, groupby name only works
    # with a named index, we use a MultiIndex, so we can group by more
    # than one level and test the non-scalar case
    multi_index = pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=["a", "b"])
    s = pd.Series([0, 1, 2], index=multi_index, dtype=int)
    iterator = s.groupby(["a", "b"]).__iter__()
    assert_type(iterator, Iterator[tuple[tuple, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), tuple[tuple, "pd.Series[int]"])

    if PD_LTE_22:
        check(assert_type(index, tuple), tuple, np.integer)
    else:
        check(assert_type(index, tuple), tuple, int)

    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    iterator2 = s.groupby("a").__iter__()
    assert_type(iterator2, Iterator[tuple[Scalar, "pd.Series[int]"]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[Scalar, "pd.Series[int]"])

    check(assert_type(index2, Scalar), int)
    check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)

    # GH 674
    # grouping by pd.MultiIndex should always resolve to a tuple as well
    iterator3 = s.groupby(multi_index).__iter__()
    assert_type(iterator3, Iterator[tuple[tuple, "pd.Series[int]"]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[tuple, "pd.Series[int]"])

    check(assert_type(index3, tuple), tuple, int)
    check(assert_type(value3, "pd.Series[int]"), pd.Series, np.integer)

    # Explicit by=None
    iterator4 = s.groupby(None, level=0).__iter__()
    assert_type(iterator4, Iterator[tuple[Scalar, "pd.Series[int]"]])
    index4, value4 = next(iterator4)
    assert_type((index4, value4), tuple[Scalar, "pd.Series[int]"])

    check(assert_type(index4, Scalar), int)
    check(assert_type(value4, "pd.Series[int]"), pd.Series, np.integer)

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
    assert_type(iterator, Iterator[tuple[pd.Period, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), tuple[pd.Period, "pd.Series[int]"])

    check(assert_type(index, pd.Period), pd.Period)
    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    dt_index = pd.DatetimeIndex(dates)
    iterator2 = s.groupby(dt_index).__iter__()
    assert_type(iterator2, Iterator[tuple[pd.Timestamp, "pd.Series[int]"]])
    index2, value2 = next(iterator2)
    assert_type((index2, value2), tuple[pd.Timestamp, "pd.Series[int]"])

    check(assert_type(index2, pd.Timestamp), pd.Timestamp)
    check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)

    tdelta_index = pd.TimedeltaIndex(dates - pd.Timestamp("2020-01-01"))
    iterator3 = s.groupby(tdelta_index).__iter__()
    assert_type(iterator3, Iterator[tuple[pd.Timedelta, "pd.Series[int]"]])
    index3, value3 = next(iterator3)
    assert_type((index3, value3), tuple[pd.Timedelta, "pd.Series[int]"])

    check(assert_type(index3, pd.Timedelta), pd.Timedelta)
    check(assert_type(value3, "pd.Series[int]"), pd.Series, np.integer)

    intervals: list[pd.Interval[pd.Timestamp]] = [
        pd.Interval(date, date + pd.DateOffset(days=1), closed="left") for date in dates
    ]
    interval_index = pd.IntervalIndex(intervals)
    assert_type(interval_index, "pd.IntervalIndex[pd.Interval[pd.Timestamp]]")
    iterator4 = s.groupby(interval_index).__iter__()
    assert_type(
        iterator4, Iterator[tuple["pd.Interval[pd.Timestamp]", "pd.Series[int]"]]
    )
    index4, value4 = next(iterator4)
    assert_type((index4, value4), tuple["pd.Interval[pd.Timestamp]", "pd.Series[int]"])

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
    assert_type(iterator, Iterator[tuple[Any, "pd.Series[int]"]])
    index, value = next(iterator)
    assert_type((index, value), tuple[Any, "pd.Series[int]"])

    check(assert_type(index, Any), str)
    check(assert_type(value, "pd.Series[int]"), pd.Series, np.integer)

    # categorical indexes are also ambiguous
    # https://github.com/pandas-dev/pandas/issues/54054 needs to be fixed
    with pytest_warns_bounded(
        FutureWarning,
        "The default of observed=False is deprecated",
        upper="2.2.99",
    ):
        categorical_index = pd.CategoricalIndex(s.index)
        iterator2 = s.groupby(categorical_index).__iter__()
        assert_type(iterator2, Iterator[tuple[Any, "pd.Series[int]"]])
        index2, value2 = next(iterator2)
        assert_type((index2, value2), tuple[Any, "pd.Series[int]"])

        check(assert_type(index2, Any), str)
        check(assert_type(value2, "pd.Series[int]"), pd.Series, np.integer)


def test_types_groupby_agg() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"])
    check(assert_type(s.groupby(level=0).agg("sum"), pd.Series), pd.Series)
    check(
        assert_type(s.groupby(level=0).agg(["min", "sum"]), pd.DataFrame), pd.DataFrame
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|sum)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(s.groupby(level=0).agg(sum), pd.Series), pd.Series)
        check(
            assert_type(s.groupby(level=0).agg([min, sum]), pd.DataFrame), pd.DataFrame
        )


def test_types_groupby_aggregate() -> None:
    s = pd.Series([4, 2, 1, 8], index=["a", "b", "a", "b"])
    check(assert_type(s.groupby(level=0).aggregate("sum"), pd.Series), pd.Series)
    check(
        assert_type(s.groupby(level=0).aggregate(["min", "sum"]), pd.DataFrame),
        pd.DataFrame,
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|sum)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(s.groupby(level=0).aggregate(sum), pd.Series), pd.Series)
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
    s.rolling(2, center=True)
    if TYPE_CHECKING_INVALID_USAGE:
        s.expanding(axis=0)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
        s.rolling(2, axis=0, center=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
        s.expanding(axis=0, center=True)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]

    s.rolling(2)

    check(
        assert_type(s.rolling(2).agg("sum"), pd.Series),
        pd.Series,
    )
    check(
        assert_type(s.rolling(2).agg(["max", "min"]), pd.DataFrame),
        pd.DataFrame,
    )
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max|sum)> is currently using",
        upper="2.2.99",
    ):
        check(
            assert_type(s.rolling(2).agg(sum), pd.Series),
            pd.Series,
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
    if TYPE_CHECKING_INVALID_USAGE:
        s1.update([1, "b", "c", "d"])  # type: ignore[list-item] # pyright: ignore[reportArgumentType]
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
    check(assert_type(s.agg("min"), int), np.integer)
    check(assert_type(s.agg(["min", "max"]), pd.Series), pd.Series, np.integer)
    check(assert_type(s.agg({"a": "min"}), pd.Series), pd.Series, np.integer)
    check(assert_type(s.agg("mean", axis=0), float), np.float64)
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <(built-in function (min|max|mean)|function mean at 0x\w+)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(s.agg(min), int), np.integer if PD_LTE_22 else int)
        check(assert_type(s.agg([min, max]), pd.Series), pd.Series, np.integer)
        check(assert_type(s.agg({0: min}), pd.Series), pd.Series, np.integer)
        check(
            assert_type(s.agg(x=max, y="min", z=np.mean), pd.Series),
            pd.Series,
            np.float64,
        )


def test_types_aggregate() -> None:
    s = pd.Series([1, 2, 3], index=["col1", "col2", "col3"])
    check(assert_type(s.aggregate("min"), int), np.integer)
    check(
        assert_type(s.aggregate(["min", "max"]), pd.Series),
        pd.Series,
        np.integer,
    )
    check(assert_type(s.aggregate({"a": "min"}), pd.Series), pd.Series, np.integer)
    with pytest_warns_bounded(
        FutureWarning,
        r"The provided callable <built-in function (min|max)> is currently using",
        upper="2.2.99",
    ):
        check(assert_type(s.aggregate(min), int), np.integer if PD_LTE_22 else int)
        check(
            assert_type(s.aggregate([min, max]), pd.Series),
            pd.Series,
            np.integer,
        )
        check(assert_type(s.aggregate({0: min}), pd.Series), pd.Series, np.integer)


def test_types_transform() -> None:
    s = pd.Series([1, 2, 3], index=["col1", "col2", "col3"])
    check(assert_type(s.transform("abs"), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.transform(abs), "pd.Series[int]"), pd.Series, np.integer)
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
    s = pd.Series(range(9), index=pd.date_range("1/1/2000", periods=9, freq="min"))
    s.resample("3min").sum()
    # origin and offset params added in 1.1.0 https://pandas.pydata.org/docs/whatsnew/v1.1.0.html
    s.resample("20min", origin="epoch", offset=pd.Timedelta(value=2, unit="minutes"))
    s.resample("20min", origin=datetime.datetime.now(), offset=datetime.timedelta(1))


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


def test_types_getitem_by_timestamp() -> None:
    index = pd.date_range("2018-01-01", periods=2, freq="D")
    series = pd.Series(range(2), index=index)
    check(assert_type(series[index[-1]], int), np.integer)


def test_types_eq() -> None:
    s1 = pd.Series([1, 2, 3])
    res1: pd.Series = s1 == 1
    s2 = pd.Series([1, 2, 4])
    res2: pd.Series = s1 == s2


def test_types_rename_axis() -> None:
    s = pd.Series([1, 2, 3])
    s.index.name = "a"

    # Rename index with `mapper`
    check(assert_type(s.rename_axis("A"), "pd.Series[int]"), pd.Series)
    check(assert_type(s.rename_axis(["A"]), "pd.Series[int]"), pd.Series)
    check(assert_type(s.rename_axis(None), "pd.Series[int]"), pd.Series)

    # Rename index with `index`
    check(assert_type(s.rename_axis(index="A"), "pd.Series[int]"), pd.Series)
    check(assert_type(s.rename_axis(index=["A"]), "pd.Series[int]"), pd.Series)
    check(assert_type(s.rename_axis(index={"a": "A"}), "pd.Series[int]"), pd.Series)
    check(
        assert_type(s.rename_axis(index=lambda name: name.upper()), "pd.Series[int]"),
        pd.Series,
    )
    check(assert_type(s.rename_axis(index=None), "pd.Series[int]"), pd.Series)

    if TYPE_CHECKING_INVALID_USAGE:
        s.rename_axis(columns="A")  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]


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
    check(assert_type(s1, "pd.Series[int]"), pd.Series, np.integer)
    # Hashable Sequence
    s2 = pd.Series([1, 2, 3]).rename(("A", "B"))
    check(assert_type(s2, "pd.Series[int]"), pd.Series, np.integer)

    # Optional
    s3 = pd.Series([1, 2, 3]).rename(None)
    check(assert_type(s3, "pd.Series[int]"), pd.Series, np.integer)

    # Functions
    def add1(x: int) -> int:
        return x + 1

    s4 = pd.Series([1, 2, 3]).rename(add1)
    check(assert_type(s4, "pd.Series[int]"), pd.Series, np.integer)

    # Dictionary
    s5 = pd.Series([1, 2, 3]).rename({1: 10})
    check(assert_type(s5, "pd.Series[int]"), pd.Series, np.integer)
    # inplace
    s6: None = pd.Series([1, 2, 3]).rename("A", inplace=True)

    if TYPE_CHECKING_INVALID_USAGE:
        s7 = pd.Series([1, 2, 3]).rename({1: [3, 4, 5]})  # type: ignore[dict-item] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_types_ne() -> None:
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([1, 2, 4])
    s3: pd.Series = s1 != s2


def test_types_bfill() -> None:
    s1 = pd.Series([1, 2, 3])
    check(assert_type(s1.bfill(), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s1.bfill(inplace=False), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.bfill(inplace=False, limit_area="inside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    assert assert_type(s1.bfill(inplace=True), None) is None
    assert assert_type(s1.bfill(inplace=True, limit_area="outside"), None) is None


def test_types_ewm() -> None:
    s1 = pd.Series([1, 2, 3])
    if TYPE_CHECKING_INVALID_USAGE:
        check(
            assert_type(
                s1.ewm(com=0.3, min_periods=0, adjust=False, ignore_na=True, axis=0),  # type: ignore[call-arg] # pyright: ignore[reportAssertTypeFailure,reportCallIssue]
                "ExponentialMovingWindow[pd.Series]",
            ),
            ExponentialMovingWindow,
        )
    check(
        assert_type(
            s1.ewm(com=0.3, min_periods=0, adjust=False, ignore_na=True),
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
    check(assert_type(s1.ffill(), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s1.ffill(inplace=False), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.ffill(inplace=False, limit_area="inside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    assert assert_type(s1.ffill(inplace=True), None) is None
    assert assert_type(s1.ffill(inplace=True, limit_area="outside"), None) is None


def test_types_as_type() -> None:
    s1 = pd.Series([1, 2, 8, 9])
    s2: pd.Series = s1.astype("int32")


def test_types_dot() -> None:
    """Test typing of multiplication methods (dot and @) for Series."""
    s1 = pd.Series([0, 1, 2, 3])
    s2 = pd.Series([-1, 2, -3, 4])
    df1 = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
    n1 = np.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
    check(assert_type(s1.dot(s2), Scalar), np.int64)
    check(assert_type(s1 @ s2, Scalar), np.int64)
    check(assert_type(s1.dot(df1), "pd.Series[int]"), pd.Series, np.int64)
    check(assert_type(s1 @ df1, pd.Series), pd.Series)
    check(assert_type(s1.dot(n1), np.ndarray), np.ndarray)
    check(assert_type(s1 @ n1, np.ndarray), np.ndarray)


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
    check(assert_type(t1, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(t2, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(t3, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(t4, "pd.Series[int]"), pd.Series, np.integer)


def test_series_invert() -> None:
    s1 = pd.Series([True, False, True])
    s2 = ~s1
    check(assert_type(s2, "pd.Series[bool]"), pd.Series, np.bool_)
    s3 = pd.Series([1, 2, 3])
    check(assert_type(s3[s2], "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s3.loc[s2], "pd.Series[int]"), pd.Series, np.integer)


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
    check(assert_type(sp, "pd.Series[int]"), pd.Series, np.integer)


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
    check(assert_type(r4, "pd.Series[int]"), pd.Series, np.integer)
    r5 = s.reset_index(["ab"], drop=True)
    check(assert_type(r5, "pd.Series[int]"), pd.Series, np.integer)
    r6 = s.reset_index(["ab"], drop=True, allow_duplicates=True)
    check(assert_type(r6, "pd.Series[int]"), pd.Series, np.integer)
    assert assert_type(s.reset_index(inplace=True, drop=True), None) is None


def test_series_add_str() -> None:
    s = pd.Series(["abc", "def"])
    check(assert_type(s + "x", "pd.Series[str]"), pd.Series, str)
    check(assert_type("x" + s, "pd.Series[str]"), pd.Series, str)


def test_series_dtype() -> None:
    s = pd.Series(["abc", "def"], dtype=str)
    check(assert_type(s, "pd.Series[str]"), pd.Series, str)


def test_types_replace() -> None:
    # GH 44
    s = pd.Series([1, 2, 3])
    check(assert_type(s.replace(1, 2), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s.replace(1, 2, inplace=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    assert assert_type(s.replace(1, 2, inplace=True), None) is None


def test_cat_accessor() -> None:
    # GH 43
    s: pd.Series[str] = pd.Series(
        pd.Categorical(["a", "b", "a"], categories=["a", "b"])
    )
    check(assert_type(s.cat.codes, "pd.Series[int]"), pd.Series, np.int8)
    # GH 139
    ser = pd.Series([1, 2, 3], name="A").astype("category")
    check(
        assert_type(ser.cat.set_categories([1, 2, 3]), pd.Series), pd.Series, np.integer
    )
    check(
        assert_type(ser.cat.reorder_categories([2, 3, 1], ordered=True), pd.Series),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(ser.cat.rename_categories([1, 2, 3]), pd.Series),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(ser.cat.remove_unused_categories(), pd.Series),
        pd.Series,
        np.integer,
    )
    check(assert_type(ser.cat.remove_categories([2]), pd.Series), pd.Series, np.integer)
    check(assert_type(ser.cat.add_categories([4]), pd.Series), pd.Series, np.integer)
    check(assert_type(ser.cat.as_ordered(), pd.Series), pd.Series, np.integer)
    check(assert_type(ser.cat.as_unordered(), pd.Series), pd.Series, np.integer)


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
    check(assert_type(s.tolist(), list[str]), list, str)
    check(assert_type(s.to_list(), list[str]), list, str)


def test_types_to_dict() -> None:
    s = pd.Series(["a", "b", "c"], dtype=str)
    assert_type(s.to_dict(), dict[Any, str])


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
    check(
        assert_type(s.str.contains(re.compile(r"a")), "pd.Series[bool]"),
        pd.Series,
        np.bool_,
    )
    check(assert_type(s.str.count("pp"), "pd.Series[int]"), pd.Series, np.integer)
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
    check(assert_type(s.str.len(), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s.str.ljust(80), pd.Series), pd.Series)
    check(assert_type(s.str.lower(), pd.Series), pd.Series)
    check(assert_type(s.str.lstrip("a"), pd.Series), pd.Series)
    check(assert_type(s.str.match("pp"), "pd.Series[bool]"), pd.Series, np.bool_)
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
    index = pd.date_range("1/1/2000", periods=N, freq="min")
    x = [x for x in range(N)]
    s = pd.Series(x, index=index, dtype=float)
    check(assert_type(s.resample("2min").std(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").var(), "pd.Series[float]"), pd.Series, float)
    check(
        assert_type(s.resample("2min").quantile(), "pd.Series[float]"), pd.Series, float
    )
    check(assert_type(s.resample("2min").sum(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").prod(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").min(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").max(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").first(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").last(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").mean(), "pd.Series[float]"), pd.Series, float)
    check(assert_type(s.resample("2min").sem(), "pd.Series[float]"), pd.Series, float)
    check(
        assert_type(s.resample("2min").median(), "pd.Series[float]"), pd.Series, float
    )
    check(assert_type(s.resample("2min").ohlc(), pd.DataFrame), pd.DataFrame)


def test_to_xarray():
    s = pd.Series([1, 2])
    check(assert_type(s.to_xarray(), xr.DataArray), xr.DataArray)


def test_neg() -> None:
    # GH 253
    sr = pd.Series([1, 2, 3])
    sr_int = pd.Series([1, 2, 3], dtype=int)
    check(assert_type(-sr, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(-sr_int, "pd.Series[int]"), pd.Series, np.integer)


def test_getattr() -> None:
    # GH 261
    series = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype=int)
    check(assert_type(series.a, int), np.integer)


def test_dtype_type() -> None:
    # GH 216
    s1 = pd.Series(["foo"], dtype="string")
    check(assert_type(s1.dtype, DtypeObj), ExtensionDtype)
    check(assert_type(s1.dtype.kind, _DTypeKind), str)

    s2 = pd.Series([1], dtype="Int64")
    check(assert_type(s2.dtype, DtypeObj), ExtensionDtype)
    check(assert_type(s2.dtype.kind, _DTypeKind), str)

    s3 = pd.Series([1, 2, 3])
    check(assert_type(s3.dtype, DtypeObj), np.dtype)
    check(assert_type(s3.dtype.kind, _DTypeKind), str)


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

    if PD_LTE_22:
        with pytest_warns_bounded(
            FutureWarning,
            r"Logical ops \(and, or, xor\) between Pandas objects and dtype-less sequences "
            r"\(e.g. list, tuple\) are deprecated",
            lower="2.0.99",
        ):
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

    if PD_LTE_22:
        with pytest_warns_bounded(
            FutureWarning,
            r"Logical ops \(and, or, xor\) between Pandas objects and dtype-less sequences "
            r"\(e.g. list, tuple\) are deprecated",
            lower="2.0.99",
        ):
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
    check(assert_type(s1, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2, "pd.Series[int]"), pd.Series, np.integer)


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
    check(assert_type(fd, dict[Any, Any]), dict)


ASTYPE_BOOL_ARGS: list[tuple[BooleanDtypeArg, type]] = [
    # python boolean
    (bool, np.bool_),
    ("bool", np.bool_),
    # pandas boolean
    (pd.BooleanDtype(), np.bool_),
    ("boolean", np.bool_),
    # numpy boolean type
    (np.bool_, np.bool_),
    ("bool_", np.bool_),
    ("?", np.bool_),
    ("b1", np.bool_),
    # pyarrow boolean type
    ("bool[pyarrow]", bool),
    ("boolean[pyarrow]", bool),
]

ASTYPE_INT_ARGS: list[tuple[IntDtypeArg, type]] = [
    # python int
    (int, np.integer),
    ("int", np.integer),
    # pandas Int8
    (pd.Int8Dtype(), np.int8),
    ("Int8", np.int8),
    # pandas Int16
    (pd.Int16Dtype(), np.int16),
    ("Int16", np.int16),
    # pandas Int32
    (pd.Int32Dtype(), np.int32),
    ("Int32", np.int32),
    # pandas Int64
    (pd.Int64Dtype(), np.int64),
    ("Int64", np.int64),
    # numpy int8
    (np.byte, np.byte),
    ("byte", np.byte),
    ("b", np.byte),
    ("int8", np.int8),
    ("i1", np.int8),
    # numpy int16
    (np.short, np.short),
    ("short", np.short),
    ("h", np.short),
    ("int16", np.int16),
    ("i2", np.int16),
    # numpy int32
    (np.intc, np.intc),
    ("intc", np.intc),
    ("i", np.intc),
    ("int32", np.int32),
    ("i4", np.int32),
    # numpy int64
    (np.int_, np.int_),
    ("int_", np.int_),
    ("int64", np.int64),
    ("i8", np.int64),
    # numpy extended int
    (np.longlong, np.longlong),
    ("longlong", np.longlong),
    ("q", np.longlong),
    # numpy signed pointer  (platform dependent one of int[8,16,32,64])
    (np.intp, np.intp),
    ("intp", np.intp),
    ("p", np.intp),
    # pyarrow integer types
    ("int8[pyarrow]", int),
    ("int16[pyarrow]", int),
    ("int32[pyarrow]", int),
    ("int64[pyarrow]", int),
]

ASTYPE_UINT_ARGS: list[tuple[UIntDtypeArg, type]] = [
    # pandas UInt8
    (pd.UInt8Dtype(), np.uint8),
    ("UInt8", np.uint8),
    # pandas UInt16
    (pd.UInt16Dtype(), np.uint16),
    ("UInt16", np.uint16),
    # pandas UInt32
    (pd.UInt32Dtype(), np.uint32),
    ("UInt32", np.uint32),
    # pandas UInt64
    (pd.UInt64Dtype(), np.uint64),
    ("UInt64", np.uint64),
    # numpy uint8
    (np.ubyte, np.ubyte),
    ("ubyte", np.ubyte),
    ("B", np.ubyte),
    ("uint8", np.uint8),
    ("u1", np.uint8),
    # numpy uint16
    (np.ushort, np.ushort),
    ("ushort", np.ushort),
    ("H", np.ushort),
    ("uint16", np.uint16),
    ("u2", np.uint16),
    # numpy uint32
    (np.uintc, np.uintc),
    ("uintc", np.uintc),
    ("I", np.uintc),
    ("uint32", np.uint32),
    ("u4", np.uint32),
    # numpy uint64
    (np.uint, np.uint),
    ("uint", np.uint),
    ("uint64", np.uint64),
    ("u8", np.uint64),
    # numpy extended uint
    (np.ulonglong, np.ulonglong),
    ("ulonglong", np.ulonglong),
    ("Q", np.ulonglong),
    # numpy unsigned pointer  (platform dependent one of uint[8,16,32,64])
    (np.uintp, np.uintp),
    ("uintp", np.uintp),
    ("P", np.uintp),
    # pyarrow unsigned integer types
    ("uint8[pyarrow]", int),
    ("uint16[pyarrow]", int),
    ("uint32[pyarrow]", int),
    ("uint64[pyarrow]", int),
]

ASTYPE_FLOAT_ARGS: list[tuple[FloatDtypeArg, type]] = [
    # python float
    (float, np.floating),
    ("float", np.floating),
    # pandas Float32
    (pd.Float32Dtype(), np.float32),
    ("Float32", np.float32),
    # pandas Float64
    (pd.Float64Dtype(), np.float64),
    ("Float64", np.float64),
    # numpy float16
    (np.half, np.half),
    ("half", np.half),
    ("e", np.half),
    ("float16", np.float16),
    ("f2", np.float16),
    # numpy float32
    (np.single, np.single),
    ("single", np.single),
    ("f", np.single),
    ("float32", np.float32),
    ("f4", np.float32),
    # numpy float64
    (np.double, np.double),
    ("double", np.double),
    ("d", np.double),
    ("float64", np.float64),
    ("f8", np.float64),
    # numpy float128
    (np.longdouble, np.longdouble),
    ("longdouble", np.longdouble),
    ("g", np.longdouble),
    ("f16", np.longdouble),
    # ("float96", np.longdouble),  # NOTE: unsupported
    ("float128", np.longdouble),  # NOTE: UNIX ONLY
    # pyarrow float32
    ("float32[pyarrow]", float),
    ("float[pyarrow]", float),
    # pyarrow float64
    ("float64[pyarrow]", float),
    ("double[pyarrow]", float),
]

ASTYPE_COMPLEX_ARGS: list[tuple[ComplexDtypeArg, type]] = [
    # python complex
    (complex, np.complexfloating),
    ("complex", np.complexfloating),
    # numpy complex64
    (np.csingle, np.csingle),
    ("csingle", np.csingle),
    ("F", np.csingle),
    ("complex64", np.complex64),
    ("c8", np.complex64),
    # numpy complex128
    (np.cdouble, np.cdouble),
    ("cdouble", np.cdouble),
    ("D", np.cdouble),
    ("complex128", np.complex128),
    ("c16", np.complex128),
    # numpy complex256
    (np.clongdouble, np.clongdouble),
    ("clongdouble", np.clongdouble),
    ("G", np.clongdouble),
    ("c32", np.clongdouble),
    # ("complex192", np.clongdouble),  # NOTE: unsupported
    ("complex256", np.clongdouble),  # NOTE: UNIX ONLY
]


ASTYPE_TIMESTAMP_ARGS: list[tuple[TimestampDtypeArg, type]] = [
    # numpy datetime64
    ("datetime64[Y]", datetime.datetime),
    ("datetime64[M]", datetime.datetime),
    ("datetime64[W]", datetime.datetime),
    ("datetime64[D]", datetime.datetime),
    ("datetime64[h]", datetime.datetime),
    ("datetime64[m]", datetime.datetime),
    ("datetime64[s]", datetime.datetime),
    ("datetime64[ms]", datetime.datetime),
    ("datetime64[us]", datetime.datetime),
    ("datetime64[μs]", datetime.datetime),
    ("datetime64[ns]", datetime.datetime),
    ("datetime64[ps]", datetime.datetime),
    ("datetime64[fs]", datetime.datetime),
    ("datetime64[as]", datetime.datetime),
    # numpy datetime64 type codes
    ("M8[Y]", datetime.datetime),
    ("M8[M]", datetime.datetime),
    ("M8[W]", datetime.datetime),
    ("M8[D]", datetime.datetime),
    ("M8[h]", datetime.datetime),
    ("M8[m]", datetime.datetime),
    ("M8[s]", datetime.datetime),
    ("M8[ms]", datetime.datetime),
    ("M8[us]", datetime.datetime),
    ("M8[μs]", datetime.datetime),
    ("M8[ns]", datetime.datetime),
    ("M8[ps]", datetime.datetime),
    ("M8[fs]", datetime.datetime),
    ("M8[as]", datetime.datetime),
    # little endian
    ("<M8[Y]", datetime.datetime),
    ("<M8[M]", datetime.datetime),
    ("<M8[W]", datetime.datetime),
    ("<M8[D]", datetime.datetime),
    ("<M8[h]", datetime.datetime),
    ("<M8[m]", datetime.datetime),
    ("<M8[s]", datetime.datetime),
    ("<M8[ms]", datetime.datetime),
    ("<M8[us]", datetime.datetime),
    ("<M8[μs]", datetime.datetime),
    ("<M8[ns]", datetime.datetime),
    ("<M8[ps]", datetime.datetime),
    ("<M8[fs]", datetime.datetime),
    ("<M8[as]", datetime.datetime),
    # pyarrow timestamp
    ("timestamp[s][pyarrow]", datetime.datetime),
    ("timestamp[ms][pyarrow]", datetime.datetime),
    ("timestamp[us][pyarrow]", datetime.datetime),
    ("timestamp[ns][pyarrow]", datetime.datetime),
    # pyarrow date
    ("date32[pyarrow]", datetime.date),
    ("date64[pyarrow]", datetime.date),
]


ASTYPE_TIMEDELTA_ARGS: list[tuple[TimedeltaDtypeArg, type]] = [
    # numpy timedelta64
    ("timedelta64[Y]", datetime.timedelta),
    ("timedelta64[M]", datetime.timedelta),
    ("timedelta64[W]", datetime.timedelta),
    ("timedelta64[D]", datetime.timedelta),
    ("timedelta64[h]", datetime.timedelta),
    ("timedelta64[m]", datetime.timedelta),
    ("timedelta64[s]", datetime.timedelta),
    ("timedelta64[ms]", datetime.timedelta),
    ("timedelta64[us]", datetime.timedelta),
    ("timedelta64[μs]", datetime.timedelta),
    ("timedelta64[ns]", datetime.timedelta),
    ("timedelta64[ps]", datetime.timedelta),
    ("timedelta64[fs]", datetime.timedelta),
    ("timedelta64[as]", datetime.timedelta),
    # numpy timedelta64 type codes
    ("m8[Y]", datetime.timedelta),
    ("m8[M]", datetime.timedelta),
    ("m8[W]", datetime.timedelta),
    ("m8[D]", datetime.timedelta),
    ("m8[h]", datetime.timedelta),
    ("m8[m]", datetime.timedelta),
    ("m8[s]", datetime.timedelta),
    ("m8[ms]", datetime.timedelta),
    ("m8[us]", datetime.timedelta),
    ("m8[μs]", datetime.timedelta),
    ("m8[ns]", datetime.timedelta),
    ("m8[ps]", datetime.timedelta),
    ("m8[fs]", datetime.timedelta),
    ("m8[as]", datetime.timedelta),
    # little endian
    ("<m8[Y]", datetime.timedelta),
    ("<m8[M]", datetime.timedelta),
    ("<m8[W]", datetime.timedelta),
    ("<m8[D]", datetime.timedelta),
    ("<m8[h]", datetime.timedelta),
    ("<m8[m]", datetime.timedelta),
    ("<m8[s]", datetime.timedelta),
    ("<m8[ms]", datetime.timedelta),
    ("<m8[us]", datetime.timedelta),
    ("<m8[μs]", datetime.timedelta),
    ("<m8[ns]", datetime.timedelta),
    ("<m8[ps]", datetime.timedelta),
    ("<m8[fs]", datetime.timedelta),
    ("<m8[as]", datetime.timedelta),
    # pyarrow duration
    ("duration[s][pyarrow]", datetime.timedelta),
    ("duration[ms][pyarrow]", datetime.timedelta),
    ("duration[us][pyarrow]", datetime.timedelta),
    ("duration[ns][pyarrow]", datetime.timedelta),
]


ASTYPE_STRING_ARGS: list[tuple[StrDtypeArg, type]] = [
    # python string
    (str, str),
    ("str", str),
    # pandas string
    (pd.StringDtype(), str),
    ("string", str),
    # numpy string
    (np.str_, str),
    ("str_", str),
    ("unicode", str),
    ("U", str),
    # pyarrow string
    ("string[pyarrow]", str),
]

ASTYPE_BYTES_ARGS: list[tuple[BytesDtypeArg, type]] = [
    # python bytes
    (bytes, bytes),
    ("bytes", bytes),
    # numpy bytes
    (np.bytes_, np.bytes_),
    ("bytes_", np.bytes_),
    ("S", np.bytes_),
    # pyarrow bytes
    ("binary[pyarrow]", bytes),
]

ASTYPE_CATEGORICAL_ARGS: list[tuple[CategoryDtypeArg, type]] = [
    # pandas category
    (pd.CategoricalDtype(), object),
    ("category", object),
    # pyarrow dictionary
    # ("dictionary[pyarrow]", "pd.Series[category]", Categorical),
]


ASTYPE_OBJECT_ARGS: list[tuple[ObjectDtypeArg, type]] = [
    # python object
    (object, object),
    # numpy object
    (np.object_, object),
    ("object", object),
    # "object_"  # NOTE: not assigned
    ("O", object),
]

ASTYPE_VOID_ARGS: list[tuple[VoidDtypeArg, type]] = [
    # numpy void
    (np.void, np.void),
    ("void", np.void),
    ("V", np.void),
]


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_BOOL_ARGS, ids=repr)
def test_astype_bool(cast_arg: BooleanDtypeArg, target_type: type) -> None:
    s = pd.Series([0, 1])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python boolean
        assert_type(s.astype(bool), "pd.Series[bool]")
        assert_type(s.astype("bool"), "pd.Series[bool]")
        # pandas boolean
        assert_type(s.astype(pd.BooleanDtype()), "pd.Series[bool]")
        assert_type(s.astype("boolean"), "pd.Series[bool]")
        # numpy boolean type
        assert_type(s.astype(np.bool_), "pd.Series[bool]")
        assert_type(s.astype("bool_"), "pd.Series[bool]")
        assert_type(s.astype("?"), "pd.Series[bool]")
        # pyarrow boolean type
        assert_type(s.astype("bool[pyarrow]"), "pd.Series[bool]")
        assert_type(s.astype("boolean[pyarrow]"), "pd.Series[bool]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_INT_ARGS, ids=repr)
def test_astype_int(cast_arg: IntDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if cast_arg in (np.longlong, "longlong", "q"):
        pytest.skip(
            "longlong is bugged, for details, see"
            "https://github.com/pandas-dev/pandas/issues/54252"
        )

    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python int
        assert_type(s.astype(int), "pd.Series[int]")
        assert_type(s.astype("int"), "pd.Series[int]")
        # pandas Int8
        assert_type(s.astype(pd.Int8Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int8"), "pd.Series[int]")
        # pandas Int16
        assert_type(s.astype(pd.Int16Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int16"), "pd.Series[int]")
        # pandas Int32
        assert_type(s.astype(pd.Int32Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int32"), "pd.Series[int]")
        # pandas Int64
        assert_type(s.astype(pd.Int64Dtype()), "pd.Series[int]")
        assert_type(s.astype("Int64"), "pd.Series[int]")
        # numpy int8
        assert_type(s.astype(np.byte), "pd.Series[int]")
        assert_type(s.astype("byte"), "pd.Series[int]")
        assert_type(s.astype("int8"), "pd.Series[int]")
        assert_type(s.astype("b"), "pd.Series[int]")
        assert_type(s.astype("i1"), "pd.Series[int]")
        # numpy int16
        assert_type(s.astype(np.short), "pd.Series[int]")
        assert_type(s.astype("short"), "pd.Series[int]")
        assert_type(s.astype("int16"), "pd.Series[int]")
        assert_type(s.astype("h"), "pd.Series[int]")
        assert_type(s.astype("i2"), "pd.Series[int]")
        # numpy int32
        assert_type(s.astype(np.intc), "pd.Series[int]")
        assert_type(s.astype("intc"), "pd.Series[int]")
        assert_type(s.astype("int32"), "pd.Series[int]")
        assert_type(s.astype("i"), "pd.Series[int]")
        assert_type(s.astype("i4"), "pd.Series[int]")
        # numpy int64
        assert_type(s.astype(np.int_), "pd.Series[int]")
        assert_type(s.astype("int_"), "pd.Series[int]")
        assert_type(s.astype("int64"), "pd.Series[int]")
        assert_type(s.astype("long"), "pd.Series[int]")
        assert_type(s.astype("l"), "pd.Series[int]")
        assert_type(s.astype("i8"), "pd.Series[int]")
        # numpy signed pointer
        assert_type(s.astype(np.intp), "pd.Series[int]")
        assert_type(s.astype("intp"), "pd.Series[int]")
        assert_type(s.astype("p"), "pd.Series[int]")
        # pyarrow integer types
        assert_type(s.astype("int8[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int16[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int32[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("int64[pyarrow]"), "pd.Series[int]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_UINT_ARGS, ids=repr)
def test_astype_uint(cast_arg: IntDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # pandas UInt8
        assert_type(s.astype(pd.UInt8Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt8"), "pd.Series[int]")
        # pandas UInt16
        assert_type(s.astype(pd.UInt16Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt16"), "pd.Series[int]")
        # pandas UInt32
        assert_type(s.astype(pd.UInt32Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt32"), "pd.Series[int]")
        # pandas UInt64
        assert_type(s.astype(pd.UInt64Dtype()), "pd.Series[int]")
        assert_type(s.astype("UInt64"), "pd.Series[int]")
        # numpy uint8
        assert_type(s.astype(np.ubyte), "pd.Series[int]")
        assert_type(s.astype("ubyte"), "pd.Series[int]")
        assert_type(s.astype("uint8"), "pd.Series[int]")
        assert_type(s.astype("B"), "pd.Series[int]")
        assert_type(s.astype("u1"), "pd.Series[int]")
        # numpy uint16
        assert_type(s.astype(np.ushort), "pd.Series[int]")
        assert_type(s.astype("ushort"), "pd.Series[int]")
        assert_type(s.astype("uint16"), "pd.Series[int]")
        assert_type(s.astype("H"), "pd.Series[int]")
        assert_type(s.astype("u2"), "pd.Series[int]")
        # numpy uint32
        assert_type(s.astype(np.uintc), "pd.Series[int]")
        assert_type(s.astype("uintc"), "pd.Series[int]")
        assert_type(s.astype("uint32"), "pd.Series[int]")
        assert_type(s.astype("I"), "pd.Series[int]")
        assert_type(s.astype("u4"), "pd.Series[int]")
        # numpy uint64
        assert_type(s.astype(np.uint), "pd.Series[int]")
        assert_type(s.astype("uint"), "pd.Series[int]")
        assert_type(s.astype("uint64"), "pd.Series[int]")
        assert_type(s.astype("ulong"), "pd.Series[int]")
        assert_type(s.astype("L"), "pd.Series[int]")
        assert_type(s.astype("u8"), "pd.Series[int]")
        # numpy unsigned pointer
        assert_type(s.astype(np.uintp), "pd.Series[int]")
        assert_type(s.astype("uintp"), "pd.Series[int]")
        assert_type(s.astype("P"), "pd.Series[int]")
        # pyarrow unsigned integer types
        assert_type(s.astype("uint8[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint16[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint32[pyarrow]"), "pd.Series[int]")
        assert_type(s.astype("uint64[pyarrow]"), "pd.Series[int]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_FLOAT_ARGS, ids=repr)
def test_astype_float(cast_arg: FloatDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if platform.system() == "Windows" and cast_arg in ("f16", "float128"):
        with pytest.raises(TypeError):
            s.astype(cast_arg)
        pytest.skip("Windows does not support float128")

    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python float
        assert_type(s.astype(float), "pd.Series[float]")
        assert_type(s.astype("float"), "pd.Series[float]")
        # pandas Float32
        assert_type(s.astype(pd.Float32Dtype()), "pd.Series[float]")
        assert_type(s.astype("Float32"), "pd.Series[float]")
        # pandas Float64
        assert_type(s.astype(pd.Float64Dtype()), "pd.Series[float]")
        assert_type(s.astype("Float64"), "pd.Series[float]")
        # numpy float16
        assert_type(s.astype(np.half), "pd.Series[float]")
        assert_type(s.astype("half"), "pd.Series[float]")
        assert_type(s.astype("float16"), "pd.Series[float]")
        assert_type(s.astype("e"), "pd.Series[float]")
        assert_type(s.astype("f2"), "pd.Series[float]")
        # numpy float32
        assert_type(s.astype(np.single), "pd.Series[float]")
        assert_type(s.astype("single"), "pd.Series[float]")
        assert_type(s.astype("float32"), "pd.Series[float]")
        assert_type(s.astype("f"), "pd.Series[float]")
        assert_type(s.astype("f4"), "pd.Series[float]")
        # numpy float64
        assert_type(s.astype(np.double), "pd.Series[float]")
        assert_type(s.astype("double"), "pd.Series[float]")
        assert_type(s.astype("float64"), "pd.Series[float]")
        assert_type(s.astype("d"), "pd.Series[float]")
        assert_type(s.astype("f8"), "pd.Series[float]")
        # numpy float128
        assert_type(s.astype(np.longdouble), "pd.Series[float]")
        assert_type(s.astype("longdouble"), "pd.Series[float]")
        assert_type(s.astype("float128"), "pd.Series[float]")
        assert_type(s.astype("g"), "pd.Series[float]")
        assert_type(s.astype("f16"), "pd.Series[float]")
        # pyarrow float32
        assert_type(s.astype("float32[pyarrow]"), "pd.Series[float]")
        assert_type(s.astype("float[pyarrow]"), "pd.Series[float]")
        # pyarrow float64
        assert_type(s.astype("float64[pyarrow]"), "pd.Series[float]")
        assert_type(s.astype("double[pyarrow]"), "pd.Series[float]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_COMPLEX_ARGS, ids=repr)
def test_astype_complex(cast_arg: ComplexDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if platform.system() == "Windows" and cast_arg in ("c32", "complex256"):
        with pytest.raises(TypeError):
            s.astype(cast_arg)
        pytest.skip("Windows does not support complex256")

    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        assert_type(s.astype(complex), "pd.Series[complex]")
        assert_type(s.astype("complex"), "pd.Series[complex]")
        # numpy complex64
        assert_type(s.astype(np.csingle), "pd.Series[complex]")
        assert_type(s.astype("csingle"), "pd.Series[complex]")
        assert_type(s.astype("complex64"), "pd.Series[complex]")
        assert_type(s.astype("F"), "pd.Series[complex]")
        assert_type(s.astype("c8"), "pd.Series[complex]")
        # numpy complex128
        assert_type(s.astype(np.cdouble), "pd.Series[complex]")
        assert_type(s.astype("cdouble"), "pd.Series[complex]")
        assert_type(s.astype("complex128"), "pd.Series[complex]")
        assert_type(s.astype("D"), "pd.Series[complex]")
        assert_type(s.astype("c16"), "pd.Series[complex]")
        # numpy complex256
        assert_type(s.astype(np.clongdouble), "pd.Series[complex]")
        assert_type(s.astype("clongdouble"), "pd.Series[complex]")
        assert_type(s.astype("complex256"), "pd.Series[complex]")
        assert_type(s.astype("G"), "pd.Series[complex]")
        assert_type(s.astype("c32"), "pd.Series[complex]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_TIMESTAMP_ARGS, ids=repr)
def test_astype_timestamp(cast_arg: TimestampDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])

    if cast_arg in ("date32[pyarrow]", "date64[pyarrow]"):
        x = pd.Series(pd.date_range("2000-01-01", "2000-02-01"))
        check(x.astype(cast_arg), TimestampSeries, target_type)
    else:
        check(s.astype(cast_arg), TimestampSeries, target_type)

    if TYPE_CHECKING:
        # numpy datetime64
        assert_type(s.astype("datetime64[Y]"), TimestampSeries)
        assert_type(s.astype("datetime64[M]"), TimestampSeries)
        assert_type(s.astype("datetime64[W]"), TimestampSeries)
        assert_type(s.astype("datetime64[D]"), TimestampSeries)
        assert_type(s.astype("datetime64[h]"), TimestampSeries)
        assert_type(s.astype("datetime64[m]"), TimestampSeries)
        assert_type(s.astype("datetime64[s]"), TimestampSeries)
        assert_type(s.astype("datetime64[ms]"), TimestampSeries)
        assert_type(s.astype("datetime64[us]"), TimestampSeries)
        assert_type(s.astype("datetime64[μs]"), TimestampSeries)
        assert_type(s.astype("datetime64[ns]"), TimestampSeries)
        assert_type(s.astype("datetime64[ps]"), TimestampSeries)
        assert_type(s.astype("datetime64[fs]"), TimestampSeries)
        assert_type(s.astype("datetime64[as]"), TimestampSeries)
        # numpy datetime64 type codes
        assert_type(s.astype("M8[Y]"), TimestampSeries)
        assert_type(s.astype("M8[M]"), TimestampSeries)
        assert_type(s.astype("M8[W]"), TimestampSeries)
        assert_type(s.astype("M8[D]"), TimestampSeries)
        assert_type(s.astype("M8[h]"), TimestampSeries)
        assert_type(s.astype("M8[m]"), TimestampSeries)
        assert_type(s.astype("M8[s]"), TimestampSeries)
        assert_type(s.astype("M8[ms]"), TimestampSeries)
        assert_type(s.astype("M8[us]"), TimestampSeries)
        assert_type(s.astype("M8[μs]"), TimestampSeries)
        assert_type(s.astype("M8[ns]"), TimestampSeries)
        assert_type(s.astype("M8[ps]"), TimestampSeries)
        assert_type(s.astype("M8[fs]"), TimestampSeries)
        assert_type(s.astype("M8[as]"), TimestampSeries)
        # numpy datetime64 type codes
        assert_type(s.astype("<M8[Y]"), TimestampSeries)
        assert_type(s.astype("<M8[M]"), TimestampSeries)
        assert_type(s.astype("<M8[W]"), TimestampSeries)
        assert_type(s.astype("<M8[D]"), TimestampSeries)
        assert_type(s.astype("<M8[h]"), TimestampSeries)
        assert_type(s.astype("<M8[m]"), TimestampSeries)
        assert_type(s.astype("<M8[s]"), TimestampSeries)
        assert_type(s.astype("<M8[ms]"), TimestampSeries)
        assert_type(s.astype("<M8[us]"), TimestampSeries)
        assert_type(s.astype("<M8[μs]"), TimestampSeries)
        assert_type(s.astype("<M8[ns]"), TimestampSeries)
        assert_type(s.astype("<M8[ps]"), TimestampSeries)
        assert_type(s.astype("<M8[fs]"), TimestampSeries)
        assert_type(s.astype("<M8[as]"), TimestampSeries)
        # pyarrow timestamp
        assert_type(s.astype("timestamp[s][pyarrow]"), TimestampSeries)
        assert_type(s.astype("timestamp[ms][pyarrow]"), TimestampSeries)
        assert_type(s.astype("timestamp[us][pyarrow]"), TimestampSeries)
        assert_type(s.astype("timestamp[ns][pyarrow]"), TimestampSeries)
        # pyarrow date
        assert_type(s.astype("date32[pyarrow]"), TimestampSeries)
        assert_type(s.astype("date64[pyarrow]"), TimestampSeries)


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_TIMEDELTA_ARGS, ids=repr)
def test_astype_timedelta(cast_arg: TimedeltaDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), TimedeltaSeries, target_type)

    if TYPE_CHECKING:
        assert_type(s.astype("timedelta64[Y]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[M]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[W]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[D]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[h]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[m]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[s]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[ms]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[us]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[μs]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[ns]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[ps]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[fs]"), "TimedeltaSeries")
        assert_type(s.astype("timedelta64[as]"), "TimedeltaSeries")
        # numpy timedelta64 type codes
        assert_type(s.astype("m8[Y]"), "TimedeltaSeries")
        assert_type(s.astype("m8[M]"), "TimedeltaSeries")
        assert_type(s.astype("m8[W]"), "TimedeltaSeries")
        assert_type(s.astype("m8[D]"), "TimedeltaSeries")
        assert_type(s.astype("m8[h]"), "TimedeltaSeries")
        assert_type(s.astype("m8[m]"), "TimedeltaSeries")
        assert_type(s.astype("m8[s]"), "TimedeltaSeries")
        assert_type(s.astype("m8[ms]"), "TimedeltaSeries")
        assert_type(s.astype("m8[us]"), "TimedeltaSeries")
        assert_type(s.astype("m8[μs]"), "TimedeltaSeries")
        assert_type(s.astype("m8[ns]"), "TimedeltaSeries")
        assert_type(s.astype("m8[ps]"), "TimedeltaSeries")
        assert_type(s.astype("m8[fs]"), "TimedeltaSeries")
        assert_type(s.astype("m8[as]"), "TimedeltaSeries")
        # numpy timedelta64 type codes
        assert_type(s.astype("<m8[Y]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[M]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[W]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[D]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[h]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[m]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[s]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[ms]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[us]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[μs]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[ns]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[ps]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[fs]"), "TimedeltaSeries")
        assert_type(s.astype("<m8[as]"), "TimedeltaSeries")
        # pyarrow duration
        assert_type(s.astype("duration[s][pyarrow]"), "TimedeltaSeries")
        assert_type(s.astype("duration[ms][pyarrow]"), "TimedeltaSeries")
        assert_type(s.astype("duration[us][pyarrow]"), "TimedeltaSeries")
        assert_type(s.astype("duration[ns][pyarrow]"), "TimedeltaSeries")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_STRING_ARGS, ids=repr)
def test_astype_string(cast_arg: StrDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python string
        assert_type(s.astype(str), "pd.Series[str]")
        assert_type(s.astype("str"), "pd.Series[str]")
        # pandas string
        assert_type(s.astype(pd.StringDtype()), "pd.Series[str]")
        assert_type(s.astype("string"), "pd.Series[str]")
        # numpy string
        assert_type(s.astype(np.str_), "pd.Series[str]")
        assert_type(s.astype("str_"), "pd.Series[str]")
        assert_type(s.astype("unicode"), "pd.Series[str]")
        assert_type(s.astype("U"), "pd.Series[str]")
        # pyarrow string
        assert_type(s.astype("string[pyarrow]"), "pd.Series[str]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_BYTES_ARGS, ids=repr)
def test_astype_bytes(cast_arg: BytesDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python bytes
        assert_type(s.astype(bytes), "pd.Series[bytes]")
        assert_type(s.astype("bytes"), "pd.Series[bytes]")
        # numpy bytes
        assert_type(s.astype(np.bytes_), "pd.Series[bytes]")
        assert_type(s.astype("bytes_"), "pd.Series[bytes]")
        assert_type(s.astype("S"), "pd.Series[bytes]")
        # pyarrow bytes
        assert_type(s.astype("binary[pyarrow]"), "pd.Series[bytes]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_CATEGORICAL_ARGS, ids=repr)
def test_astype_categorical(cast_arg: CategoryDtypeArg, target_type: type) -> None:
    s = pd.Series(["a", "b"])
    check(s.astype("category"), pd.Series, target_type)

    if TYPE_CHECKING:
        # pandas category
        assert_type(s.astype(pd.CategoricalDtype()), "pd.Series[pd.CategoricalDtype]")
        assert_type(s.astype("category"), "pd.Series[pd.CategoricalDtype]")
        # pyarrow dictionary
        # assert_type(s.astype("dictionary[pyarrow]"), "pd.Series[Categorical]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_OBJECT_ARGS, ids=repr)
def test_astype_object(cast_arg: ObjectDtypeArg, target_type: type) -> None:
    s = pd.Series([object(), 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python object
        assert_type(s.astype(object), "pd.Series[Any]")
        assert_type(s.astype("object"), "pd.Series[Any]")
        # numpy object
        assert_type(s.astype(np.object_), "pd.Series[Any]")
        # assert_type(s.astype("object_"), "pd.Series[Any]")  # NOTE: not assigned
        # assert_type(s.astype("object0"), "pd.Series[Any]")  # NOTE: not assigned
        assert_type(s.astype("O"), "pd.Series[Any]")


@pytest.mark.parametrize("cast_arg, target_type", ASTYPE_VOID_ARGS, ids=repr)
def test_astype_void(cast_arg: VoidDtypeArg, target_type: type) -> None:
    s = pd.Series([1, 2, 3])
    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # numpy void
        assert_type(s.astype(np.void), "pd.Series[Any]")
        assert_type(s.astype("void"), "pd.Series[Any]")
        assert_type(s.astype("V"), "pd.Series[Any]")


def test_astype_other() -> None:
    s = pd.Series([3, 4, 5])

    # Test incorrect Literal
    if TYPE_CHECKING_INVALID_USAGE:
        s.astype("foobar")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]

    # Test self-consistent with s.dtype (#747)
    # NOTE: https://github.com/python/typing/issues/801#issuecomment-1646171898
    check(assert_type(s.astype(s.dtype), "pd.Series[Any]"), pd.Series, np.integer)

    # test DecimalDtype
    orseries = pd.Series([Decimal(x) for x in [1, 2, 3]])
    newtype = DecimalDtype()
    decseries = orseries.astype(newtype)
    check(
        assert_type(decseries, pd.Series),
        pd.Series,
        Decimal,
    )

    # Test non-literal string
    # NOTE: currently unsupported! Enable in future.
    # string: str = "int"  # not Literal!
    # check(assert_type(s.astype(string), "pd.Series[Any]"), pd.Series, np.integer)


def test_all_astype_args_tested() -> None:
    """Check that all relevant numpy type aliases are tested."""
    NUMPY_ALIASES: set[str] = {k for k in np.sctypeDict if isinstance(k, str)}
    EXCLUDED_ALIASES = {
        "datetime64",
        "m",
        "m8",
        "timedelta64",
        "M",
        "M8",
        "object_",
        "object0",
        "a",  # deprecated in numpy 2.0
    }
    NON_NUMPY20_ALIASES = {
        "complex_",
        "unicode_",
        "uint0",
        "longfloat",
        "string_",
        "cfloat",
        "int0",
        "void0",
        "bytes0",
        "singlecomplex",
        "longcomplex",
        "bool8",
        "clongfloat",
        "str0",
        "float_",
        # Next 4 are excluded because results are incompatible between numpy 1.x
        # and 2.0, and it's not possible to do numpy version specific typing
        "long",
        "l",
        "ulong",
        "L",
    }
    TESTED_ASTYPE_ARGS: list[tuple[Any, type]] = (
        ASTYPE_BOOL_ARGS
        + ASTYPE_INT_ARGS
        + ASTYPE_UINT_ARGS
        + ASTYPE_FLOAT_ARGS
        + ASTYPE_COMPLEX_ARGS
        + ASTYPE_TIMEDELTA_ARGS
        + ASTYPE_TIMESTAMP_ARGS
        + ASTYPE_BYTES_ARGS
        + ASTYPE_STRING_ARGS
        + ASTYPE_CATEGORICAL_ARGS
        + ASTYPE_OBJECT_ARGS
        + ASTYPE_VOID_ARGS
    )

    TESTED_ALIASES: set[str] = {
        arg for arg, _ in TESTED_ASTYPE_ARGS if isinstance(arg, str)
    }
    UNTESTED_ALIASES = (
        NUMPY_ALIASES - TESTED_ALIASES - NON_NUMPY20_ALIASES
    ) - EXCLUDED_ALIASES
    assert not UNTESTED_ALIASES, f"{UNTESTED_ALIASES}"

    NUMPY_TYPES: set[type] = set(np.sctypeDict.values())
    EXCLUDED_TYPES: set[type] = {np.str_, np.object_, np.timedelta64, np.datetime64}
    TESTED_TYPES: set[type] = {t for _, t in TESTED_ASTYPE_ARGS}
    UNTESTED_TYPES = (NUMPY_TYPES - TESTED_TYPES) - EXCLUDED_TYPES
    assert not UNTESTED_TYPES, f"{UNTESTED_TYPES}"


def test_check_xs() -> None:
    s4 = pd.Series([1, 4])
    s4.xs(0, axis=0)
    check(assert_type(s4, "pd.Series[int]"), pd.Series, np.integer)


def test_types_apply_set() -> None:
    series_of_lists: pd.Series = pd.Series(
        {"list1": [1, 2, 3], "list2": ["a", "b", "c"], "list3": [True, False, True]}
    )
    check(assert_type(series_of_lists.apply(lambda x: set(x)), pd.Series), pd.Series)


def test_prefix_summix_axis() -> None:
    s = pd.Series([1, 2, 3, 4])
    check(
        assert_type(s.add_suffix("_item", axis=0), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(s.add_suffix("_item", axis="index"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(s.add_prefix("_item", axis=0), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(s.add_prefix("_item", axis="index"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        s.add_prefix("_item", axis=1)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        s.add_suffix("_item", axis="columns")  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_convert_dtypes_convert_floating() -> None:
    df = pd.Series([1, 2, 3, 4])
    dfn = df.convert_dtypes(convert_floating=False)
    check(assert_type(dfn, "pd.Series[int]"), pd.Series, np.integer)


def test_convert_dtypes_dtype_backend() -> None:
    s = pd.Series([1, 2, 3, 4])
    s1 = s.convert_dtypes(dtype_backend="numpy_nullable")
    check(assert_type(s1, "pd.Series[int]"), pd.Series, np.integer)


def test_apply_returns_none() -> None:
    # GH 557
    s = pd.Series([1, 2, 3])
    check(assert_type(s.apply(lambda x: None), pd.Series), pd.Series)


def test_loc_callable() -> None:
    # GH 586
    s = pd.Series([1, 2])
    check(assert_type(s.loc[lambda x: x > 1], "pd.Series[int]"), pd.Series, np.integer)


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
        result3 = s.to_json(orient="records", lines=False, mode="a")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_groupby_diff() -> None:
    # GH 658
    s = pd.Series([1.0, 2.0, 3.0, np.nan])
    check(
        assert_type(s.groupby(level=0).diff(), "pd.Series[float]"),
        pd.Series,
        float,
    )


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


def test_types_mask() -> None:
    s = pd.Series([1, 2, 3, 4, 5])

    # Test case with a boolean condition and a scalar value
    check(assert_type(s.mask(s > 3, 10), "pd.Series[int]"), pd.Series, np.integer)

    def cond(x: int) -> bool:
        return x % 2 == 0

    # Test case with a callable condition and a scalar value
    check(assert_type(s.mask(cond, 10), "pd.Series[int]"), pd.Series, np.integer)

    # Test case with a boolean condition and a callable
    def double(x):
        return x * 2

    check(assert_type(s.mask(s > 3, double), "pd.Series[int]"), pd.Series, np.integer)

    # Test cases with None and pd.NA as other
    check(assert_type(s.mask(s > 3, None), "pd.Series[int]"), pd.Series, np.float64)
    check(assert_type(s.mask(s > 3, pd.NA), "pd.Series[int]"), pd.Series, np.float64)


def test_timedelta_div() -> None:
    series = pd.Series([pd.Timedelta(days=1)])
    delta = datetime.timedelta(1)

    check(assert_type(series / delta, "pd.Series[float]"), pd.Series, float)
    check(assert_type(series / [delta], "pd.Series[float]"), pd.Series, float)
    check(assert_type(series / 1, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(series / [1], "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(series // delta, "pd.Series[int]"), pd.Series, np.longlong)
    check(assert_type(series // [delta], "pd.Series[int]"), pd.Series, int)
    check(assert_type(series // 1, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(series // [1], "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(delta / series, "pd.Series[float]"), pd.Series, float)
    check(assert_type([delta] / series, "pd.Series[float]"), pd.Series, float)
    check(assert_type(delta // series, "pd.Series[int]"), pd.Series, np.longlong)
    check(assert_type([delta] // series, "pd.Series[int]"), pd.Series, np.signedinteger)

    if TYPE_CHECKING_INVALID_USAGE:
        1 / series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        [1] / series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        1 // series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        [1] // series  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_rank() -> None:
    check(
        assert_type(pd.Series([1, 2]).rank(), "pd.Series[float]"), pd.Series, np.float64
    )


def test_series_setitem_multiindex() -> None:
    # GH 767
    df = (
        pd.DataFrame({"x": [1, 2, 3, 4]})
        .assign(y=lambda df: df["x"] * 10, z=lambda df: df["x"] * 100)
        .set_index(["x", "y"])
    )
    ind = pd.Index([2, 3])
    s = df["z"]

    s.loc[pd.IndexSlice[ind, :]] = 30


def test_series_setitem_na() -> None:
    # GH 743
    df = pd.DataFrame(
        {"x": [1, 2, 3], "y": pd.date_range("3/1/2023", "3/3/2023")},
        index=pd.Index(["a", "b", "c"]),
    ).convert_dtypes()

    ind = pd.Index(["a", "c"])
    s = df["x"].copy()

    s.loc[ind] = pd.NA
    s.iloc[[0, 2]] = pd.NA

    s2 = df["y"].copy()
    s2.loc[ind] = pd.NaT
    s2.iloc[[0, 2]] = pd.NaT


def test_round() -> None:
    # GH 791
    check(assert_type(round(pd.DataFrame([])), pd.DataFrame), pd.DataFrame)
    check(assert_type(round(pd.Series([1], dtype=int)), "pd.Series[int]"), pd.Series)


def test_get() -> None:
    s_int = pd.Series([1, 2, 3], index=[1, 2, 3])

    check(assert_type(s_int.get(1), Union[int, None]), np.int64)
    check(assert_type(s_int.get(99), Union[int, None]), type(None))
    check(assert_type(s_int.get(1, default=None), Union[int, None]), np.int64)
    check(assert_type(s_int.get(99, default=None), Union[int, None]), type(None))
    check(assert_type(s_int.get(1, default=2), int), np.int64)
    check(assert_type(s_int.get(99, default="a"), Union[int, str]), str)

    s_str = pd.Series(list("abc"), index=list("abc"))

    check(assert_type(s_str.get("a"), Union[str, None]), str)
    check(assert_type(s_str.get("z"), Union[str, None]), type(None))
    check(assert_type(s_str.get("a", default=None), Union[str, None]), str)
    check(assert_type(s_str.get("z", default=None), Union[str, None]), type(None))
    check(assert_type(s_str.get("a", default="b"), str), str)
    check(assert_type(s_str.get("z", default=True), Union[str, bool]), bool)


def test_series_new_empty() -> None:
    # GH 826
    check(assert_type(pd.Series(), "pd.Series[Any]"), pd.Series)


def test_series_mapping() -> None:
    # GH 831
    check(
        assert_type(
            pd.Series(
                {
                    pd.Timestamp(2023, 1, 2): "b",
                }
            ),
            "pd.Series[str]",
        ),
        pd.Series,
        str,
    )

    check(
        assert_type(
            pd.Series(
                {
                    ("a", "b"): "c",
                }
            ),
            "pd.Series[str]",
        ),
        pd.Series,
        str,
    )


def test_timedeltaseries_operators() -> None:
    series = pd.Series([pd.Timedelta(days=1)])
    check(
        assert_type(series + datetime.datetime.now(), TimestampSeries),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(series + datetime.timedelta(1), TimedeltaSeries),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(datetime.datetime.now() + series, TimestampSeries),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(series - datetime.timedelta(1), TimedeltaSeries),
        pd.Series,
        pd.Timedelta,
    )


def test_timestamp_series() -> None:
    series = pd.Series([pd.Timestamp(2024, 4, 4)])
    check(
        assert_type(series + YearEnd(0), TimestampSeries),
        TimestampSeries,
        pd.Timestamp,
    )
    check(
        assert_type(series - YearEnd(0), TimestampSeries),
        TimestampSeries,
        pd.Timestamp,
    )


def test_pipe() -> None:
    ser = pd.Series(range(10))

    def first_arg_series(
        ser: pd.Series,
        positional_only: int,
        /,
        argument_1: list[float],
        argument_2: str,
        *,
        keyword_only: tuple[int, int],
    ) -> pd.Series:
        return ser

    check(
        assert_type(
            ser.pipe(
                first_arg_series,
                1,
                [1.0, 2.0],
                argument_2="hi",
                keyword_only=(1, 2),
            ),
            pd.Series,
        ),
        pd.Series,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        ser.pipe(
            first_arg_series,
            "a",  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
            [1.0, 2.0],
            argument_2="hi",
            keyword_only=(1, 2),
        )
        ser.pipe(
            first_arg_series,
            1,
            [1.0, "b"],  # type: ignore[list-item] # pyright: ignore[reportArgumentType,reportCallIssue]
            argument_2="hi",
            keyword_only=(1, 2),
        )
        ser.pipe(
            first_arg_series,
            1,
            [1.0, 2.0],
            argument_2=11,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
            keyword_only=(1, 2),
        )
        ser.pipe(
            first_arg_series,
            1,
            [1.0, 2.0],
            argument_2="hi",
            keyword_only=(1,),  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        )
        ser.pipe(  # type: ignore[call-arg]
            first_arg_series,
            1,
            [1.0, 2.0],
            argument_3="hi",  # pyright: ignore[reportCallIssue]
            keyword_only=(1, 2),
        )
        ser.pipe(  # type: ignore[call-overload]
            first_arg_series,
            1,
            [1.0, 2.0],
            11,
            (1, 2),  # pyright: ignore[reportCallIssue]
        )
        ser.pipe(  # type: ignore[call-overload]
            first_arg_series,
            positional_only=1,  # pyright: ignore[reportCallIssue]
            argument_1=[1.0, 2.0],
            argument_2=11,
            keyword_only=(1, 2),
        )

    def first_arg_not_series(argument_1: int, ser: pd.Series) -> pd.Series:
        return ser

    check(
        assert_type(
            ser.pipe(
                (first_arg_not_series, "ser"),
                1,
            ),
            pd.Series,
        ),
        pd.Series,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        ser.pipe(
            (
                first_arg_not_series,  # type: ignore[arg-type]
                1,  # pyright: ignore[reportArgumentType,reportCallIssue]
            ),
            1,
        )
        ser.pipe(
            (
                1,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
                "df",
            ),
            1,
        )


def test_series_apply() -> None:
    s = pd.Series(["A", "B", "AB"])
    check(assert_type(s.apply(tuple), "pd.Series[Any]"), pd.Series)
    check(assert_type(s.apply(list), "pd.Series[Any]"), pd.Series)
    check(assert_type(s.apply(set), "pd.Series[Any]"), pd.Series)
    check(assert_type(s.apply(frozenset), "pd.Series[Any]"), pd.Series)


def test_diff() -> None:
    s = pd.Series([1, 1, 2, 3, 5, 8])
    # int -> float
    check(assert_type(s.diff(), "pd.Series[float]"), pd.Series, float)
    # unint -> float
    check(assert_type(s.astype(np.uint32).diff(), "pd.Series[float]"), pd.Series, float)
    # float -> float
    check(assert_type(s.astype(float).diff(), "pd.Series[float]"), pd.Series, float)
    # datetime.date -> timeDelta
    check(
        assert_type(
            pd.Series(
                [datetime.datetime.now().date(), datetime.datetime.now().date()]
            ).diff(),
            "TimedeltaSeries",
        ),
        pd.Series,
        pd.Timedelta,
        index_to_check_for_type=-1,
    )
    # timestamp -> timedelta
    times = pd.Series([pd.Timestamp(0), pd.Timestamp(1)])
    check(
        assert_type(times.diff(), "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
        index_to_check_for_type=-1,
    )
    # timedelta -> timedelta64
    check(
        assert_type(
            pd.Series([pd.Timedelta(0), pd.Timedelta(1)]).diff(), "TimedeltaSeries"
        ),
        pd.Series,
        pd.Timedelta,
        index_to_check_for_type=-1,
    )
    # period -> object
    if WINDOWS:
        with pytest_warns_bounded(
            RuntimeWarning, "overflow encountered in scalar multiply"
        ):
            check(
                assert_type(
                    pd.Series(
                        pd.period_range(start="2017-01-01", end="2017-02-01", freq="D")
                    ).diff(),
                    "OffsetSeries",
                ),
                pd.Series,
                BaseOffset,
                index_to_check_for_type=-1,
            )
    else:
        check(
            assert_type(
                pd.Series(
                    pd.period_range(start="2017-01-01", end="2017-02-01", freq="D")
                ).diff(),
                "OffsetSeries",
            ),
            pd.Series,
            BaseOffset,
            index_to_check_for_type=-1,
        )
    # bool -> object
    check(
        assert_type(
            pd.Series([True, True, False, False, True]).diff(),
            "pd.Series[type[object]]",
        ),
        pd.Series,
        object,
    )
    # object -> object
    check(
        assert_type(s.astype(object).diff(), "pd.Series[type[object]]"),
        pd.Series,
        object,
    )
    # complex -> complex
    check(
        assert_type(s.astype(complex).diff(), "pd.Series[complex]"), pd.Series, complex
    )
    if TYPE_CHECKING_INVALID_USAGE:
        # interval -> TypeError: IntervalArray has no 'diff' method. Convert to a suitable dtype prior to calling 'diff'.
        assert_never(pd.Series([pd.Interval(0, 2), pd.Interval(1, 4)]).diff())


def test_diff_never1() -> None:
    s = pd.Series([1, 1, 2, 3, 5, 8])
    if TYPE_CHECKING_INVALID_USAGE:
        # bytes -> numpy.core._exceptions._UFuncNoLoopError: ufunc 'subtract' did not contain a loop with signature matching types (dtype('S21'), dtype('S21')) -> None
        assert_never(s.astype(bytes).diff())


def test_diff_never2() -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        # dtype -> TypeError: unsupported operand type(s) for -: 'type' and 'type'
        assert_never(pd.Series([str, int, bool]).diff())


def test_diff_never3() -> None:
    if TYPE_CHECKING_INVALID_USAGE:
        # str -> TypeError: unsupported operand type(s) for -: 'str' and 'str'
        assert_never(pd.Series(["a", "b"]).diff())


def test_operator_constistency() -> None:
    # created for #748
    s = pd.Series([1, 2, 3])
    check(
        assert_type(s * np.timedelta64(1, "s"), "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(np.timedelta64(1, "s") * s, "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(s.mul(np.timedelta64(1, "s")), "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(s.rmul(np.timedelta64(1, "s")), "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
    )


def test_map() -> None:
    s = pd.Series([1, 2, 3])

    mapping = {1: "a", 2: "b", 3: "c"}
    check(
        assert_type(s.map(mapping, na_action="ignore"), "pd.Series[str]"),
        pd.Series,
        str,
    )

    def callable(x: int) -> str:
        return str(x)

    check(
        assert_type(s.map(callable, na_action="ignore"), "pd.Series[str]"),
        pd.Series,
        str,
    )

    series = pd.Series(["a", "b", "c"])
    check(
        assert_type(s.map(series, na_action="ignore"), "pd.Series[str]"), pd.Series, str
    )


def test_map_na() -> None:
    s: pd.Series[int] = pd.Series([1, pd.NA, 3])

    mapping = {1: "a", 2: "b", 3: "c"}
    check(assert_type(s.map(mapping, na_action=None), "pd.Series[str]"), pd.Series, str)

    def callable(x: int | NAType) -> str | NAType:
        if isinstance(x, int):
            return str(x)
        return x

    check(
        assert_type(s.map(callable, na_action=None), "pd.Series[str]"), pd.Series, str
    )

    series = pd.Series(["a", "b", "c"])
    check(assert_type(s.map(series, na_action=None), "pd.Series[str]"), pd.Series, str)


def test_case_when() -> None:
    c = pd.Series([6, 7, 8, 9], name="c")
    a = pd.Series([0, 0, 1, 2])
    b = pd.Series([0, 3, 4, 5])
    r = c.case_when(
        caselist=[
            (a.gt(0), a),
            (b.gt(0), b),
        ]
    )
    check(assert_type(r, pd.Series), pd.Series)


def test_series_unique_timestamp() -> None:
    """Test type return of Series.unique on Series[datetime64[ns]]."""
    sr = pd.Series(pd.bdate_range("2023-10-10", "2023-10-15"))
    check(assert_type(sr.unique(), DatetimeArray), DatetimeArray)


def test_series_unique_timedelta() -> None:
    """Test type return of Series.unique on Series[timedeta64[ns]]."""
    sr = pd.Series([pd.Timedelta("1 days"), pd.Timedelta("3 days")])
    check(assert_type(sr.unique(), TimedeltaArray), TimedeltaArray)
