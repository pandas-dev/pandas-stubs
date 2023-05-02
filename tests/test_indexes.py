from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Tuple,
    Union,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from pandas._typing import Dtype  # noqa: F401
from pandas._typing import Scalar

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

if TYPE_CHECKING:
    from pandas.core.indexes.base import (
        _ComplexIndexType,
        _FloatIndexType,
        _IntIndexType,
    )
else:
    from pandas.core.indexes.base import (
        Index as _ComplexIndexType,
        Index as _FloatIndexType,
        Index as _IntIndexType,
    )


def test_index_unique() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    check(assert_type(ind, pd.Index), pd.Index)
    i2 = ind.unique()
    check(assert_type(i2, pd.Index), pd.Index)


def test_index_duplicated() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=pd.Index([1, 2, 3, 2]))
    ind = df.index
    duplicated = ind.duplicated("first")
    check(assert_type(duplicated, npt.NDArray[np.bool_]), np.ndarray, np.bool_)


def test_index_isin() -> None:
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    check(assert_type(isin, npt.NDArray[np.bool_]), np.ndarray, np.bool_)


def test_index_astype() -> None:
    indi = pd.Index([1, 2, 3])
    inds = pd.Index(["a", "b", "c"])
    indc = indi.astype(inds.dtype)
    check(assert_type(indc, pd.Index), pd.Index)
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    mia = mi.astype(object)  # object is only valid parameter for MultiIndex.astype()
    check(assert_type(mia, pd.MultiIndex), pd.MultiIndex)
    check(
        assert_type(mi.to_frame(name=[3, 7], allow_duplicates=True), pd.DataFrame),
        pd.DataFrame,
    )


def test_multiindex_get_level_values() -> None:
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    check(assert_type(i1, pd.Index), pd.Index)


def test_index_tolist() -> None:
    i1 = pd.Index([1, 2, 3])
    check(assert_type(i1.tolist(), list), list, int)
    check(assert_type(i1.to_list(), list), list, int)


def test_column_getitem() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199#issuecomment-1132806594
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    column = df.columns[0]
    check(assert_type(column, Scalar), str)
    check(assert_type(df[column], pd.Series), pd.Series, np.int64)


def test_column_contains() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199
    df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "E": [3, 4]})

    collist = [column for column in df.columns]

    collist2 = [column for column in df.columns[df.columns.str.contains("A|B")]]

    length = len(df.columns[df.columns.str.contains("A|B")])


def test_column_sequence() -> None:
    df = pd.DataFrame([1, 2, 3])
    col_list = list(df.columns)
    check(
        assert_type(col_list, list),
        list,
        int,
    )


def test_difference_none() -> None:
    # https://github.com/pandas-dev/pandas-stubs/issues/17
    ind = pd.Index([1, 2, 3])
    check(assert_type(ind.difference([1, None]), pd.Index), pd.Index)
    # GH 253
    check(assert_type(ind.difference([1]), pd.Index), pd.Index)


def test_str_split() -> None:
    # GH 194
    ind = pd.Index(["a-b", "c-d"])
    check(assert_type(ind.str.split("-"), pd.Index), pd.Index)
    check(assert_type(ind.str.split("-", expand=True), pd.MultiIndex), pd.MultiIndex)


def test_index_dropna():
    idx = pd.Index([1, 2])

    check(assert_type(idx.dropna(how="all"), pd.Index), pd.Index)
    check(assert_type(idx.dropna(how="any"), pd.Index), pd.Index)

    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]])

    check(assert_type(midx.dropna(how="all"), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(midx.dropna(how="any"), pd.MultiIndex), pd.MultiIndex)


def test_index_neg():
    # GH 253
    idx = pd.Index([1, 2])
    check(assert_type(-idx, pd.Index), pd.Index)


def test_types_to_numpy() -> None:
    idx = pd.Index([1, 2])
    check(assert_type(idx.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(dtype="int", copy=True), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(na_value=0), np.ndarray), np.ndarray)


def test_index_arithmetic() -> None:
    # GH 287
    idx = pd.Index([1, 2.2, 3], dtype=float)
    check(assert_type(idx + 3, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(idx - 3, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(idx * 3, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(idx / 3, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(idx // 3, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(3 + idx, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(3 - idx, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(3 * idx, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(3 / idx, "_FloatIndexType"), _FloatIndexType, np.float64)
    check(assert_type(3 // idx, "_FloatIndexType"), _FloatIndexType, np.float64)


def test_index_relops() -> None:
    # GH 265
    data = pd.date_range("2022-01-01", "2022-01-31", freq="D")
    x = pd.Timestamp("2022-01-17")
    idx = pd.Index(data, name="date")
    check(assert_type(data[x <= idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x < idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x >= idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x > idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx < x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx >= x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx > x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx <= x], pd.DatetimeIndex), pd.DatetimeIndex)

    dt_idx = pd.DatetimeIndex(data, name="date")
    check(assert_type(data[x <= dt_idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x >= dt_idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x < dt_idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x > dt_idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[dt_idx <= x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[dt_idx >= x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[dt_idx < x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[dt_idx > x], pd.DatetimeIndex), pd.DatetimeIndex)

    ind = pd.Index([1, 2, 3])
    check(assert_type(ind <= 2, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(ind >= 2, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(ind < 2, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(ind > 2, npt.NDArray[np.bool_]), np.ndarray, np.bool_)


def test_range_index_union():
    check(
        assert_type(
            pd.RangeIndex(0, 10).union(pd.RangeIndex(10, 20)),
            Union[pd.Index, _IntIndexType, pd.RangeIndex],
        ),
        pd.RangeIndex,
    )
    check(
        assert_type(
            pd.RangeIndex(0, 10).union([11, 12, 13]),
            Union[pd.Index, _IntIndexType, pd.RangeIndex],
        ),
        pd.Index,
    )
    check(
        assert_type(
            pd.RangeIndex(0, 10).union(["a", "b", "c"]),
            Union[pd.Index, _IntIndexType, pd.RangeIndex],
        ),
        pd.Index,
    )


def test_interval_range():
    check(
        assert_type(pd.interval_range(0, 10), "pd.IntervalIndex[pd.Interval[int]]"),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(0, 10, name="something", closed="both"),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(pd.interval_range(0.0, 10), "pd.IntervalIndex[pd.Interval[float]]"),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(dt.datetime(2000, 1, 1), dt.datetime(2010, 1, 1), 5),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                np.datetime64("2000-01-01"), np.datetime64("2020-01-01"), 5
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(pd.Timestamp(2000, 1, 1), pd.Timestamp(2010, 1, 1), 5),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timestamp(2000, 1, 1), pd.Timestamp(2010, 1, 1), freq="1M"
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timestamp(2000, 1, 1),
                pd.Timestamp(2010, 1, 1),
                freq=pd.DateOffset(months=2),
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timestamp(2000, 1, 1),
                pd.Timestamp(2010, 1, 1),
                freq=pd.Timedelta(days=30),
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timestamp(2000, 1, 1),
                pd.Timestamp(2010, 1, 1),
                freq=dt.timedelta(days=30),
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(pd.Timestamp(2000, 1, 1), dt.datetime(2010, 1, 1), 5),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )

    check(
        assert_type(
            pd.interval_range(pd.Timedelta("1D"), pd.Timedelta("10D")),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timedelta("1D"), pd.Timedelta("10D"), freq=pd.Timedelta("2D")
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(
                pd.Timedelta("1D"), pd.Timedelta("10D"), freq=dt.timedelta(days=2)
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(end=pd.Timedelta("10D"), periods=10, freq="D"),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.interval_range(start=pd.Timedelta("1D"), periods=10, freq="D"),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )


def test_interval_index_breaks():
    check(
        assert_type(
            pd.IntervalIndex.from_breaks([1, 2, 3, 4]),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks([1.0, 2.0, 3.0, 4.0]),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(
                [pd.Timestamp(2000, 1, 1), pd.Timestamp(2000, 1, 2)]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks([pd.Timedelta(1, "D"), pd.Timedelta(2, "D")]),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )

    check(
        assert_type(
            pd.IntervalIndex.from_breaks(np.array([1, 2, 3, 4], dtype=np.int64)),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            ),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    np_ndarray_dt64 = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ],
        dtype=np.datetime64,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(np_ndarray_dt64),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    np_ndarray_td64 = np.array(
        [
            np.timedelta64(1, "D"),
            np.timedelta64(2, "D"),
            np.timedelta64(3, "D"),
            np.timedelta64(4, "D"),
        ],
        dtype=np.timedelta64,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(np_ndarray_td64),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(pd.Series([1, 2, 3, 4], dtype=int)),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
    )
    pd_series_float = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(pd_series_float),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    timestamp_series = pd.Series(pd.date_range("2000-01-01", "2003-01-01", freq="D"))
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(timestamp_series),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_breaks(
                [
                    dt.datetime(2000, 1, 1),
                    dt.datetime(2001, 1, 1),
                    dt.datetime(2002, 1, 1),
                    dt.datetime(2003, 1, 1),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )


def test_interval_index_arrays():
    check(
        assert_type(
            pd.IntervalIndex.from_arrays([1, 2, 3, 4], [2, 3, 4, 5]),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_arrays([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(
                np.array([1, 2, 3, 4], dtype=np.int64),
                np.array([2, 3, 4, 5], dtype=np.int64),
            ),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            ),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    left_dt64_arr: npt.NDArray[np.datetime64] = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ],
        dtype="datetime64[ns]",
    )
    right_dt_arr: npt.NDArray[np.datetime64] = np.array(
        [
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
            np.datetime64("2004-01-01"),
        ],
        dtype="datetime64[ns]",
    )
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(left_dt64_arr, right_dt_arr),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )

    check(
        assert_type(
            pd.IntervalIndex.from_arrays(
                pd.Series([1, 2, 3, 4], dtype=int), pd.Series([2, 3, 4, 5], dtype=int)
            ),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    series_float_left = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    series_float_right = pd.Series([2.0, 3.0, 4.0, 5.0], dtype=float)
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(series_float_left, series_float_right),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    left_s_ts = pd.Series(pd.date_range("2000-01-01", "2003-01-01", freq="Y"))
    right_s_ts = pd.Series(pd.date_range("2001-01-01", "2004-01-01", freq="Y"))
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(left_s_ts, right_s_ts),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_arrays(
                [
                    dt.datetime(2000, 1, 1),
                    dt.datetime(2001, 1, 1),
                    dt.datetime(2002, 1, 1),
                    dt.datetime(2003, 1, 1),
                ],
                [
                    dt.datetime(2001, 1, 1),
                    dt.datetime(2002, 1, 1),
                    dt.datetime(2003, 1, 1),
                    dt.datetime(2004, 1, 1),
                ],
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )


def test_interval_index_tuples():
    check(
        assert_type(
            pd.IntervalIndex.from_tuples([(1, 2), (2, 3)]),
            "pd.IntervalIndex[pd.Interval[int]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples([(1.0, 2.0), (2.0, 3.0)]),
            "pd.IntervalIndex[pd.Interval[float]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (pd.Timestamp(2000, 1, 1), pd.Timestamp(2001, 1, 1)),
                    (pd.Timestamp(2001, 1, 1), pd.Timestamp(2002, 1, 1)),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (dt.datetime(2000, 1, 1), dt.datetime(2001, 1, 1)),
                    (dt.datetime(2001, 1, 1), dt.datetime(2002, 1, 1)),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (np.datetime64("2000-01-01"), np.datetime64("2001-01-01")),
                    (np.datetime64("2001-01-01"), np.datetime64("2002-01-01")),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (pd.Timedelta(1, "D"), pd.Timedelta(2, "D")),
                    (pd.Timedelta(2, "D"), pd.Timedelta(3, "D")),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (dt.timedelta(days=1), dt.timedelta(days=2)),
                    (dt.timedelta(days=2), dt.timedelta(days=3)),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(
        assert_type(
            pd.IntervalIndex.from_tuples(
                [
                    (np.timedelta64(1, "D"), np.timedelta64(2, "D")),
                    (np.timedelta64(2, "D"), np.timedelta64(3, "D")),
                ]
            ),
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )


def test_sorted_and_list() -> None:
    # GH 497
    i1 = pd.Index([3, 2, 1])
    check(
        assert_type(
            sorted(i1),
            list,
        ),
        list,
    )
    check(
        assert_type(
            list(i1),
            list,
        ),
        list,
    )


def test_index_operators() -> None:
    # GH 405
    i1 = pd.Index([1, 2, 3])
    i2 = pd.Index([4, 5, 6])

    check(assert_type(i1 + i2, pd.Index), pd.Index)
    check(assert_type(i1 + 10, pd.Index), pd.Index)
    check(assert_type(10 + i1, pd.Index), pd.Index)
    check(assert_type(i1 - i2, pd.Index), pd.Index)
    check(assert_type(i1 - 10, pd.Index), pd.Index)
    check(assert_type(10 - i1, pd.Index), pd.Index)
    check(assert_type(i1 * i2, pd.Index), pd.Index)
    check(assert_type(i1 * 10, pd.Index), pd.Index)
    check(assert_type(10 * i1, pd.Index), pd.Index)
    check(assert_type(i1 / i2, pd.Index), pd.Index)
    check(assert_type(i1 / 10, pd.Index), pd.Index)
    check(assert_type(10 / i1, pd.Index), pd.Index)
    check(assert_type(i1 // i2, pd.Index), pd.Index)
    check(assert_type(i1 // 10, pd.Index), pd.Index)
    check(assert_type(10 // i1, pd.Index), pd.Index)
    check(assert_type(i1**i2, pd.Index), pd.Index)
    check(assert_type(i1**2, pd.Index), pd.Index)
    check(assert_type(2**i1, pd.Index), pd.Index)
    check(assert_type(i1 % i2, pd.Index), pd.Index)
    check(assert_type(i1 % 10, pd.Index), pd.Index)
    check(assert_type(10 % i1, pd.Index), pd.Index)
    check(assert_type(divmod(i1, i2), Tuple[pd.Index, pd.Index]), tuple)
    check(assert_type(divmod(i1, 10), Tuple[pd.Index, pd.Index]), tuple)
    check(assert_type(divmod(10, i1), Tuple[pd.Index, pd.Index]), tuple)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(
            i1 & i2,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1 & 10,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10 & i1,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(
            i1 | i2,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1 | 10,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10 | i1,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(
            i1 ^ i2,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1 ^ 10,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10 ^ i1,  # type:ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
            Never,
        )


def test_getitem() -> None:
    # GH 536
    ip = pd.period_range(start="2022-06-01", periods=10)
    check(assert_type(ip, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(ip[0], pd.Period), pd.Period)
    check(assert_type(ip[[0, 2, 4]], pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    idt = pd.DatetimeIndex(["2022-08-14", "2022-08-20", "2022-08-24"])
    check(assert_type(idt, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(idt[0], pd.Timestamp), pd.Timestamp)
    check(assert_type(idt[[0, 2]], pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    itd = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    check(assert_type(itd, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(itd[0], pd.Timedelta), pd.Timedelta)
    check(
        assert_type(itd[[0, 2, 4]], pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta
    )

    iini = pd.interval_range(0, 10)
    check(
        assert_type(iini, "pd.IntervalIndex[pd.Interval[int]]"),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(assert_type(iini[0], "pd.Interval[int]"), pd.Interval)
    check(
        assert_type(iini[[0, 2, 4]], "pd.IntervalIndex[pd.Interval[int]]"),
        pd.IntervalIndex,
        pd.Interval,
    )

    iinf = pd.interval_range(0.0, 10)
    check(
        assert_type(iinf, "pd.IntervalIndex[pd.Interval[float]]"),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(assert_type(iinf[0], "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(iinf[[0, 2, 4]], "pd.IntervalIndex[pd.Interval[float]]"),
        pd.IntervalIndex,
        pd.Interval,
    )

    iints = pd.interval_range(dt.datetime(2000, 1, 1), dt.datetime(2010, 1, 1), 5)
    check(
        assert_type(
            iints,
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(assert_type(iints[0], "pd.Interval[pd.Timestamp]"), pd.Interval)
    check(
        assert_type(iints[[0, 2, 4]], "pd.IntervalIndex[pd.Interval[pd.Timestamp]]"),
        pd.IntervalIndex,
        pd.Interval,
    )

    iintd = pd.interval_range(pd.Timedelta("1D"), pd.Timedelta("10D"))
    check(
        assert_type(
            iintd,
            "pd.IntervalIndex[pd.Interval[pd.Timedelta]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )
    check(assert_type(iintd[0], "pd.Interval[pd.Timedelta]"), pd.Interval)
    check(
        assert_type(iintd[[0, 2, 4]], "pd.IntervalIndex[pd.Interval[pd.Timedelta]]"),
        pd.IntervalIndex,
        pd.Interval,
    )

    iri = pd.RangeIndex(0, 10)
    check(assert_type(iri, pd.RangeIndex), pd.RangeIndex, int)
    check(assert_type(iri[0], int), int)
    check(assert_type(iri[[0, 2, 4]], pd.Index), pd.Index, np.int64)

    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    check(assert_type(mi, pd.MultiIndex), pd.MultiIndex)
    check(assert_type(mi[0], tuple), tuple)
    check(assert_type(mi[[0, 2]], pd.MultiIndex), pd.MultiIndex, tuple)

    i0 = pd.Index(["a", "b", "c"])
    check(assert_type(i0, pd.Index), pd.Index)
    check(assert_type(i0[0], Scalar), str)
    check(assert_type(i0[[0, 2]], pd.Index), pd.Index, str)


def test_multiindex_dtypes():
    # GH-597
    mi = pd.MultiIndex.from_tuples([(1, 2.0), (2, 3.0)], names=["foo", "bar"])
    check(assert_type(mi.dtypes, "pd.Series[Dtype]"), pd.Series)


def test_index_constructors():
    # See if we can pick up the different index types in 2.0
    # Eventually should be using a generic index
    ilist = [1, 2, 3]
    check(
        assert_type(pd.Index(ilist, dtype="int"), _IntIndexType), pd.Index, np.integer
    )
    check(assert_type(pd.Index(ilist, dtype=int), _IntIndexType), pd.Index, np.integer)
    check(assert_type(pd.Index(ilist, dtype=np.int8), _IntIndexType), pd.Index, np.int8)
    check(
        assert_type(pd.Index(ilist, dtype=np.int16), _IntIndexType), pd.Index, np.int16
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int32), _IntIndexType), pd.Index, np.int32
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int64), _IntIndexType), pd.Index, np.int64
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint8), _IntIndexType), pd.Index, np.uint8
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint16), _IntIndexType),
        pd.Index,
        np.uint16,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint32), _IntIndexType),
        pd.Index,
        np.uint32,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint64), _IntIndexType),
        pd.Index,
        np.uint64,
    )

    flist = [1.1, 2.2, 3.3]
    check(
        assert_type(pd.Index(flist, dtype="float"), _FloatIndexType),
        pd.Index,
        np.float64,
    )
    check(
        assert_type(pd.Index(flist, dtype=float), _FloatIndexType), pd.Index, np.float64
    )
    check(
        assert_type(pd.Index(flist, dtype=np.float32), _FloatIndexType),
        pd.Index,
        np.float32,
    )
    check(
        assert_type(pd.Index(flist, dtype=np.float64), _FloatIndexType),
        pd.Index,
        np.float64,
    )

    clist = [1 + 1j, 2 + 2j, 3 + 4j]
    check(
        assert_type(pd.Index(clist, dtype="complex"), _ComplexIndexType),
        pd.Index,
        complex,
    )
    check(
        assert_type(pd.Index(clist, dtype=complex), _ComplexIndexType),
        pd.Index,
        complex,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        # This should be detected by the type checker, but for it to work,
        # we need to change the last overload of __new__ in core/indexes/base.pyi
        # to specify all the possible dtype options.  For right now, we will leave the
        # test here as a reminder that we would like this to be seen as incorrect usage.
        pd.Index(flist, dtype=np.float16)
