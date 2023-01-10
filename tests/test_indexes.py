from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Hashable,
    List,
    Tuple,
    Union,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.core.indexes.numeric import NumericIndex
from typing_extensions import (
    Never,
    assert_type,
)

from pandas._typing import Scalar

if TYPE_CHECKING:
    from pandas._typing import IndexIterScalar

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
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
    check(assert_type(df[column], pd.Series), pd.Series, int)


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
        assert_type(col_list, List[Union["IndexIterScalar", Tuple[Hashable, ...]]]),
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
    check(assert_type(idx + 3, NumericIndex), NumericIndex)
    check(assert_type(idx - 3, NumericIndex), NumericIndex)
    check(assert_type(idx * 3, NumericIndex), NumericIndex)
    check(assert_type(idx / 3, NumericIndex), NumericIndex)
    check(assert_type(idx // 3, NumericIndex), NumericIndex)
    check(assert_type(3 + idx, NumericIndex), NumericIndex)
    check(assert_type(3 - idx, NumericIndex), NumericIndex)
    check(assert_type(3 * idx, NumericIndex), NumericIndex)
    check(assert_type(3 / idx, NumericIndex), NumericIndex)
    check(assert_type(3 // idx, NumericIndex), NumericIndex)


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
    with pytest_warns_bounded(
        FutureWarning,
        match="pandas.Int64Index",
        upper="1.5.99",
        upper_exception=AttributeError,
    ):
        check(
            assert_type(
                pd.RangeIndex(0, 10).union(pd.RangeIndex(10, 20)),
                Union[pd.Index, pd.Int64Index, pd.RangeIndex],
            ),
            pd.RangeIndex,
        )
        check(
            assert_type(
                pd.RangeIndex(0, 10).union([11, 12, 13]),
                Union[pd.Index, pd.Int64Index, pd.RangeIndex],
            ),
            pd.Int64Index,
        )
        check(
            assert_type(
                pd.RangeIndex(0, 10).union(["a", "b", "c"]),
                Union[pd.Index, pd.Int64Index, pd.RangeIndex],
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
            List[Union["IndexIterScalar", Tuple[Hashable, ...]]],
        ),
        list,
    )
    check(
        assert_type(
            list(i1),
            List[Union["IndexIterScalar", Tuple[Hashable, ...]]],
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
        assert_type(i1 & i2, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(i1 & 10, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(10 & i1, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(i1 | i2, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(i1 | 10, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(10 | i1, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(i1 ^ i2, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(i1 ^ 10, Never)  # pyright: ignore[reportGeneralTypeIssues]
        assert_type(10 ^ i1, Never)  # pyright: ignore[reportGeneralTypeIssues]
