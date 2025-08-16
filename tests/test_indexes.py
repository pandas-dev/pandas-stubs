from __future__ import annotations

from collections.abc import Hashable
import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    cast,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.base import Index
from typing_extensions import (
    Never,
    assert_type,
)

if TYPE_CHECKING:
    from tests import Dtype  # noqa: F401

from tests import (
    PD_LTE_23,
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
    check(
        assert_type(mi.to_frame(name=[3, 7], allow_duplicates=True), pd.DataFrame),
        pd.DataFrame,
    )

    df = pd.DataFrame({"a": [1, 2, 3]})
    check(
        assert_type(
            pd.MultiIndex.from_product([["x", "y"], df.columns]), pd.MultiIndex
        ),
        pd.MultiIndex,
    )
    check(
        assert_type(
            pd.MultiIndex.from_product([["x", "y"], pd.Series([1, 2])]), pd.MultiIndex
        ),
        pd.MultiIndex,
    )


def test_multiindex_get_level_values() -> None:
    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    i1 = mi.get_level_values("ab")
    check(assert_type(i1, pd.Index), pd.Index)


def test_multiindex_constructors() -> None:
    check(
        assert_type(
            pd.MultiIndex([[1], [4]], codes=[[0], [0]], name=["a", "b"]), pd.MultiIndex
        ),
        pd.MultiIndex,
    )
    check(
        assert_type(
            pd.MultiIndex(
                [[1], [4]],
                codes=[[0], [0]],
                names=["a", "b"],
                sortorder=0,
                copy=True,
                verify_integrity=True,
            ),
            pd.MultiIndex,
        ),
        pd.MultiIndex,
    )
    check(
        assert_type(pd.MultiIndex.from_arrays([[1], [4]]), pd.MultiIndex), pd.MultiIndex
    )
    check(
        assert_type(
            pd.MultiIndex.from_arrays([np.arange(3), np.arange(3)]), pd.MultiIndex
        ),
        pd.MultiIndex,
    )
    check(
        assert_type(pd.MultiIndex.from_tuples(zip([1, 2], [3, 4])), pd.MultiIndex),
        pd.MultiIndex,
    )
    check(
        assert_type(pd.MultiIndex.from_tuples([(1, 3), (2, 4)]), pd.MultiIndex),
        pd.MultiIndex,
    )
    check(
        assert_type(pd.MultiIndex.from_frame(pd.DataFrame({"a": [1]})), pd.MultiIndex),
        pd.MultiIndex,
    )


def test_index_tolist() -> None:
    i1 = pd.Index([1, 2, 3])
    check(assert_type(i1.tolist(), list[int]), list, int)
    check(assert_type(i1.to_list(), list[int]), list, int)


def test_column_getitem() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199#issuecomment-1132806594
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    column = df.columns[0]
    check(assert_type(column, str), str)
    check(assert_type(df[column], pd.Series), pd.Series, np.integer)


def test_column_contains() -> None:
    # https://github.com/microsoft/python-type-stubs/issues/199
    df = pd.DataFrame({"A": [1, 2], "B": ["c", "d"], "E": [3, 4]})

    collist = [column for column in df.columns]
    check(assert_type(collist, list[str]), list, str)

    collist2 = [column for column in df.columns[df.columns.str.contains("A|B")]]
    check(assert_type(collist2, list[str]), list, str)

    length = len(df.columns[df.columns.str.contains("A|B")])
    check(assert_type(length, int), int)


def test_column_sequence() -> None:
    df = pd.DataFrame([1, 2, 3])
    col_list = list(df.columns)
    check(
        assert_type(col_list, list[str]),
        list,
        int,
    )


def test_difference_none() -> None:
    # https://github.com/pandas-dev/pandas-stubs/issues/17
    ind = pd.Index([1, 2, 3])
    check(assert_type(ind.difference([1, None]), "pd.Index[int]"), pd.Index)
    # GH 253
    check(assert_type(ind.difference([1]), "pd.Index[int]"), pd.Index)

    # check with sort parameter
    check(assert_type(ind.difference([1, None], sort=False), "pd.Index[int]"), pd.Index)
    check(assert_type(ind.difference([1], sort=True), "pd.Index[int]"), pd.Index)


def test_str_split() -> None:
    # GH 194
    ind = pd.Index(["a-b", "c-d"])
    check(assert_type(ind.str.split("-"), "pd.Index[list[str]]"), pd.Index, list)
    check(assert_type(ind.str.split("-", expand=True), pd.MultiIndex), pd.MultiIndex)
    check(
        assert_type(ind.str.split("-", expand=False), "pd.Index[list[str]]"),
        pd.Index,
        list,
    )


def test_str_rsplit() -> None:
    # GH 1074
    ind = pd.Index(["a-b", "c-d"])
    check(assert_type(ind.str.rsplit("-"), "pd.Index[list[str]]"), pd.Index, list)
    check(assert_type(ind.str.rsplit("-", expand=True), pd.MultiIndex), pd.MultiIndex)
    check(
        assert_type(ind.str.rsplit("-", expand=False), "pd.Index[list[str]]"),
        pd.Index,
        list,
    )


def test_str_match() -> None:
    i = pd.Index(
        ["applep", "bananap", "Cherryp", "DATEp", "eGGpLANTp", "123p", "23.45p"]
    )
    check(assert_type(i.str.match("pp"), npt.NDArray[np.bool_]), np.ndarray, np.bool_)


def test_index_rename() -> None:
    """Test that index rename returns an element of type Index."""
    ind = pd.Index([1, 2, 3], name="foo")
    ind2 = ind.rename("goo")
    check(assert_type(ind2, "pd.Index[int]"), pd.Index, np.integer)


def test_index_rename_inplace() -> None:
    """Test that index rename in-place does not return anything (None)."""
    ind = pd.Index([1, 2, 3], name="foo")
    ind2 = ind.rename("goo", inplace=True)
    check(assert_type(ind2, None), type(None))
    assert ind2 is None


def test_index_dropna():
    idx = pd.Index([1, 2])

    check(assert_type(idx.dropna(how="all"), "pd.Index[int]"), pd.Index)
    check(assert_type(idx.dropna(how="any"), "pd.Index[int]"), pd.Index)

    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]])

    check(assert_type(midx.dropna(how="all"), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(midx.dropna(how="any"), pd.MultiIndex), pd.MultiIndex)


def test_index_neg():
    # GH 253
    idx = pd.Index([1, 2])
    check(assert_type(-idx, "pd.Index[int]"), pd.Index)


def test_types_to_numpy() -> None:
    idx = pd.Index([1, 2])
    check(assert_type(idx.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(dtype="int", copy=True), np.ndarray), np.ndarray)
    check(assert_type(idx.to_numpy(na_value=0), np.ndarray), np.ndarray)


def test_index_arithmetic() -> None:
    # GH 287
    idx = pd.Index([1, 2.2, 3], dtype=float)
    check(assert_type(idx + 3, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(idx - 3, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(idx * 3, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(idx / 3, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(idx // 3, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(3 + idx, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(3 - idx, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(3 * idx, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(3 / idx, "pd.Index[float]"), pd.Index, np.float64)
    check(assert_type(3 // idx, "pd.Index[float]"), pd.Index, np.float64)


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
            Union[pd.Index, "pd.Index[int]", pd.RangeIndex],
        ),
        pd.RangeIndex,
    )
    check(
        assert_type(
            pd.RangeIndex(0, 10).union([11, 12, 13]),
            Union[pd.Index, "pd.Index[int]", pd.RangeIndex],
        ),
        pd.Index,
    )
    check(
        assert_type(
            pd.RangeIndex(0, 10).union(["a", "b", "c"]),
            Union[pd.Index, "pd.Index[int]", pd.RangeIndex],
        ),
        pd.Index,
    )


def test_index_union_sort() -> None:
    """Test sort argument in pd.Index.union GH1264."""
    check(
        assert_type(pd.Index(["e", "f"]).union(["a", "b", "c"], sort=True), pd.Index),
        pd.Index,
    )
    check(
        assert_type(pd.Index(["e", "f"]).union(["a", "b", "c"], sort=False), pd.Index),
        pd.Index,
    )


def test_range_index_start_stop_step():
    idx = pd.RangeIndex(3)
    check(assert_type(idx.start, int), int)
    check(assert_type(idx.stop, int), int)
    check(assert_type(idx.step, int), int)


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
    with pytest_warns_bounded(
        FutureWarning,
        "'M' is deprecated",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
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
    with pytest_warns_bounded(
        FutureWarning,
        "'Y' is deprecated",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
        pd.Series(pd.date_range("2000-01-01", "2003-01-01", freq="Y"))
        pd.Series(pd.date_range("2001-01-01", "2004-01-01", freq="Y"))

    left_s_ts = pd.Series(pd.date_range("2000-01-01", "2003-01-01", freq="YS"))
    right_s_ts = pd.Series(pd.date_range("2001-01-01", "2004-01-01", freq="YS"))
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
    check(assert_type(sorted(i1), list[int]), list, int)
    check(assert_type(list(i1), list[int]), list, int)


def test_index_operators() -> None:
    # GH 405
    i1 = pd.Index([1, 2, 3])
    i2 = pd.Index([4, 5, 6])

    check(assert_type(i1 + i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 + 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 + i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 - i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 - 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 - i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 * i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 * 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 * i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 / i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 / 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 / i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 // i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 // 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 // i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1**i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1**2, "pd.Index[int]"), pd.Index)
    check(assert_type(2**i1, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 % i2, "pd.Index[int]"), pd.Index)
    check(assert_type(i1 % 10, "pd.Index[int]"), pd.Index)
    check(assert_type(10 % i1, "pd.Index[int]"), pd.Index)
    check(assert_type(divmod(i1, i2), tuple["pd.Index[int]", "pd.Index[int]"]), tuple)
    check(assert_type(divmod(i1, 10), tuple["pd.Index[int]", "pd.Index[int]"]), tuple)
    check(assert_type(divmod(10, i1), tuple["pd.Index[int]", "pd.Index[int]"]), tuple)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(
            i1
            & i2,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1
            & 10,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10
            & i1,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(
            i1
            | i2,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1
            | 10,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10
            | i1,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(
            i1
            ^ i2,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            i1
            ^ 10,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
            Never,
        )
        assert_type(  # type: ignore[assert-type]
            10
            ^ i1,  # type:ignore[operator] # pyright: ignore[reportAssertTypeFailure,reportOperatorIssue]
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
    check(
        assert_type(iri[[0, 2, 4]], pd.Index),
        pd.Index,
        np.integer if PD_LTE_23 else int,
    )

    mi = pd.MultiIndex.from_product([["a", "b"], ["c", "d"]], names=["ab", "cd"])
    check(assert_type(mi, pd.MultiIndex), pd.MultiIndex)
    check(assert_type(mi[0], tuple), tuple)
    check(assert_type(mi[[0, 2]], pd.MultiIndex), pd.MultiIndex, tuple)

    i0 = pd.Index(["a", "b", "c"])
    check(assert_type(i0, "pd.Index[str]"), pd.Index)
    check(assert_type(i0[0], str), str)
    check(assert_type(i0[[0, 2]], "pd.Index[str]"), pd.Index, str)


def test_append_mix() -> None:
    """Test pd.Index.append with mixed types"""
    first = pd.Index([1])
    second = pd.Index(["a"])
    third = pd.Index([1, "a"])
    check(assert_type(first.append(second), pd.Index), pd.Index)
    check(assert_type(first.append([second]), pd.Index), pd.Index)

    check(assert_type(first.append(third), pd.Index), pd.Index)
    check(assert_type(first.append([third]), pd.Index), pd.Index)
    check(assert_type(first.append([second, third]), pd.Index), pd.Index)

    check(assert_type(third.append([]), pd.Index), pd.Index)
    check(assert_type(third.append(cast("list[Index[Any]]", [])), pd.Index), pd.Index)
    check(assert_type(third.append([first]), pd.Index), pd.Index)
    check(assert_type(third.append([first, second]), pd.Index), pd.Index)


def test_append_int() -> None:
    """Test pd.Index[int].append"""
    first = pd.Index([1])
    second = pd.Index([2])
    check(assert_type(first.append([]), "pd.Index[int]"), pd.Index, np.int64)
    check(assert_type(first.append(second), "pd.Index[int]"), pd.Index, np.int64)
    check(assert_type(first.append([second]), "pd.Index[int]"), pd.Index, np.int64)


def test_append_str() -> None:
    """Test pd.Index[str].append"""
    first = pd.Index(["str"])
    second = pd.Index(["rts"])
    check(assert_type(first.append([]), "pd.Index[str]"), pd.Index, str)
    check(assert_type(first.append(second), "pd.Index[str]"), pd.Index, str)
    check(assert_type(first.append([second]), "pd.Index[str]"), pd.Index, str)


def test_range_index_range() -> None:
    """Test that pd.RangeIndex can be initialized from range."""
    iri = pd.RangeIndex(range(5))
    check(assert_type(iri, pd.RangeIndex), pd.RangeIndex, int)


def test_multiindex_dtypes():
    # GH-597
    mi = pd.MultiIndex.from_tuples([(1, 2.0), (2, 3.0)], names=["foo", "bar"])
    check(assert_type(mi.dtypes, "pd.Series[Dtype]"), pd.Series)


def test_index_constructors():
    # See if we can pick up the different index types in 2.0
    # Eventually should be using a generic index
    ilist = [1, 2, 3]
    check(
        assert_type(pd.Index(ilist, dtype="int"), "pd.Index[int]"), pd.Index, np.integer
    )
    check(
        assert_type(pd.Index(ilist, dtype=int), "pd.Index[int]"), pd.Index, np.integer
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int8), "pd.Index[int]"), pd.Index, np.int8
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int16), "pd.Index[int]"),
        pd.Index,
        np.int16,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int32), "pd.Index[int]"),
        pd.Index,
        np.int32,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.int64), "pd.Index[int]"),
        pd.Index,
        np.int64,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint8), "pd.Index[int]"),
        pd.Index,
        np.uint8,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint16), "pd.Index[int]"),
        pd.Index,
        np.uint16,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint32), "pd.Index[int]"),
        pd.Index,
        np.uint32,
    )
    check(
        assert_type(pd.Index(ilist, dtype=np.uint64), "pd.Index[int]"),
        pd.Index,
        np.uint64,
    )

    flist = [1.1, 2.2, 3.3]
    check(
        assert_type(pd.Index(flist, dtype="float"), "pd.Index[float]"),
        pd.Index,
        np.float64,
    )
    check(
        assert_type(pd.Index(flist, dtype=float), "pd.Index[float]"),
        pd.Index,
        np.float64,
    )
    check(
        assert_type(pd.Index(flist, dtype=np.float32), "pd.Index[float]"),
        pd.Index,
        np.float32,
    )
    check(
        assert_type(pd.Index(flist, dtype=np.float64), "pd.Index[float]"),
        pd.Index,
        np.float64,
    )

    clist = [1 + 1j, 2 + 2j, 3 + 4j]
    check(
        assert_type(pd.Index(clist, dtype="complex"), "pd.Index[complex]"),
        pd.Index,
        complex,
    )
    check(
        assert_type(pd.Index(clist, dtype=complex), "pd.Index[complex]"),
        pd.Index,
        complex,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        # This should be detected by the type checker, but for it to work,
        # we need to change the last overload of __new__ in core/indexes/base.pyi
        # to specify all the possible dtype options.  For right now, we will leave the
        # test here as a reminder that we would like this to be seen as incorrect usage.
        pd.Index(flist, dtype=np.float16)


def test_datetime_index_constructor() -> None:
    check(assert_type(pd.DatetimeIndex(["2020"]), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(pd.DatetimeIndex(["2020"], name="ts"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(pd.DatetimeIndex(["2020"], freq="D"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(pd.DatetimeIndex(["2020"], tz="Asia/Kathmandu"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )

    # https://github.com/microsoft/python-type-stubs/issues/115
    df = pd.DataFrame({"A": [1, 2, 3], "B": [5, 6, 7]})

    check(
        assert_type(
            pd.DatetimeIndex(data=df["A"], tz=None, ambiguous="NaT", copy=True),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )


def test_iter() -> None:
    # GH 723
    with pytest_warns_bounded(
        FutureWarning,
        "'H' is deprecated",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
        for ts in pd.date_range(start="1/1/2023", end="1/08/2023", freq="6H"):
            check(assert_type(ts, pd.Timestamp), pd.Timestamp)


def test_intersection() -> None:
    # GH 744
    index = pd.DatetimeIndex(["2022-01-01"])
    check(assert_type(index.intersection(index), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(index.intersection([pd.Timestamp("1/1/2023")]), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )


def test_annotate() -> None:
    # GH 502
    df = pd.DataFrame({"a": [1, 2]})

    columns: pd.Index[str] = df.columns
    for column in columns:
        check(assert_type(column, str), str)

    names: list[str] = list(df.columns)
    for name in names:
        check(assert_type(name, str), str)


def test_new() -> None:
    check(assert_type(pd.Index([1]), "pd.Index[int]"), pd.Index, np.integer)
    check(assert_type(pd.Index([1], dtype=float), "pd.Index[float]"), pd.Index, float)
    check(
        assert_type(pd.Index([pd.Timestamp(0)]), pd.DatetimeIndex),
        pd.DatetimeIndex,
        pd.Timestamp,
    )
    check(
        assert_type(pd.Index([pd.Timedelta(0)]), pd.TimedeltaIndex),
        pd.TimedeltaIndex,
        pd.Timedelta,
    )
    check(
        assert_type(pd.Index([pd.Period("2012-1-1", freq="D")]), pd.PeriodIndex),
        pd.PeriodIndex,
        pd.Period,
    )
    check(
        assert_type(
            pd.Index([pd.Interval(pd.Timestamp(0), pd.Timestamp(1))]),
            "pd.IntervalIndex[pd.Interval[pd.Timestamp]]",
        ),
        pd.IntervalIndex,
        pd.Interval,
    )


def test_timedelta_div() -> None:
    index = pd.Index([pd.Timedelta(days=1)], dtype="timedelta64[s]")
    delta = dt.timedelta(1)

    check(assert_type(index / delta, "pd.Index[float]"), pd.Index, float)
    check(assert_type(index / [delta], "pd.Index[float]"), pd.Index, float)
    check(assert_type(index / 1, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(index / [1], pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(index // delta, "pd.Index[int]"), pd.Index, np.longlong)
    check(assert_type(index // [delta], "pd.Index[int]"), pd.Index, int)
    check(assert_type(index // 1, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(index // [1], pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    check(assert_type(delta / index, "pd.Index[float]"), pd.Index, float)
    check(assert_type([delta] / index, "pd.Index[float]"), pd.Index, float)
    check(assert_type(delta // index, "pd.Index[int]"), pd.Index, np.longlong)
    check(assert_type([delta] // index, "pd.Index[int]"), pd.Index, np.signedinteger)

    if TYPE_CHECKING_INVALID_USAGE:
        1 / index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        [1] / index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        1 // index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        [1] // index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_datetime_operators_builtin() -> None:
    time = pd.date_range("2022-01-01", "2022-01-31", freq="D")
    check(assert_type(time + dt.timedelta(0), pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(time - dt.timedelta(0), pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(time - dt.datetime.now(), pd.TimedeltaIndex), pd.TimedeltaIndex)

    delta = check(assert_type(time - time, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(delta + dt.timedelta(0), pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(dt.datetime.now() + delta, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(delta - dt.timedelta(0), pd.TimedeltaIndex), pd.TimedeltaIndex)


def test_get_loc() -> None:
    unique_index = pd.Index(list("abc"))
    check(
        assert_type(
            unique_index.get_loc("b"), Union[int, slice, npt.NDArray[np.bool_]]
        ),
        int,
    )

    monotonic_index = pd.Index(list("abbc"))
    check(
        assert_type(
            monotonic_index.get_loc("b"), Union[int, slice, npt.NDArray[np.bool_]]
        ),
        slice,
    )

    non_monotonic_index = pd.Index(list("abcb"))
    check(
        assert_type(
            non_monotonic_index.get_loc("b"), Union[int, slice, npt.NDArray[np.bool_]]
        ),
        np.ndarray,
        np.bool_,
    )


def test_value_counts() -> None:
    nmi = pd.Index(list("abcb"))
    check(assert_type(nmi.value_counts(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(nmi.value_counts(normalize=True), "pd.Series[float]"),
        pd.Series,
        float,
    )


def test_index_factorize() -> None:
    """Test Index.factorize method."""
    codes, idx_uniques = pd.Index(["b", "b", "a", "c", "b"]).factorize()
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(idx_uniques, np.ndarray | Index | Categorical), pd.Index)

    codes, idx_uniques = pd.Index(["b", "b", "a", "c", "b"]).factorize(
        use_na_sentinel=False
    )
    check(assert_type(codes, np.ndarray), np.ndarray)
    check(assert_type(idx_uniques, np.ndarray | Index | Categorical), pd.Index)


def test_disallow_empty_index() -> None:
    # From GH 826
    if TYPE_CHECKING_INVALID_USAGE:
        i0 = pd.Index()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]


def test_datetime_index_max_min_reductions() -> None:
    dtidx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    check(assert_type(dtidx.argmax(), np.int64), np.int64)
    check(assert_type(dtidx.argmin(), np.int64), np.int64)
    check(assert_type(dtidx.max(), pd.Timestamp), pd.Timestamp)
    check(assert_type(dtidx.min(), pd.Timestamp), pd.Timestamp)


def test_periodindex_shift() -> None:
    ind = pd.period_range(start="2022-06-01", periods=10)
    check(assert_type(ind.shift(1), pd.PeriodIndex), pd.PeriodIndex)


def test_datetimeindex_shift() -> None:
    ind = pd.date_range("2023-01-01", "2023-02-01")
    check(assert_type(ind.shift(1), pd.DatetimeIndex), pd.DatetimeIndex)


def test_timedeltaindex_shift() -> None:
    ind = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    check(assert_type(ind.shift(1), pd.TimedeltaIndex), pd.TimedeltaIndex)


def test_index_insert() -> None:
    """Test the return type of Index.insert GH1196."""
    idx = pd.Index([1, 2, 3, 4, 5])
    check(assert_type(idx.insert(2, 3), "pd.Index[int]"), pd.Index, np.integer)

    ind = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    check(
        assert_type(ind.insert(2, pd.Timedelta("1D")), pd.TimedeltaIndex),
        pd.TimedeltaIndex,
    )

    dt_ind = pd.date_range("2023-01-01", "2023-02-01")
    check(
        assert_type(dt_ind.insert(2, pd.Timestamp(2024, 3, 5)), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )


def test_index_delete() -> None:
    """Test the return type of Index.delete GH1196."""
    idx = pd.Index([1, 2, 3, 4, 5])
    check(assert_type(idx.delete(2), "pd.Index[int]"), pd.Index, np.integer)

    ind = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    check(assert_type(ind.delete(2), pd.TimedeltaIndex), pd.TimedeltaIndex)

    dt_ind = pd.date_range("2023-01-01", "2023-02-01")
    check(assert_type(dt_ind.delete(2), pd.DatetimeIndex), pd.DatetimeIndex)


def test_index_dict() -> None:
    """Test passing an ordered iterables to Index and subclasses constructor GH828."""
    check(
        assert_type(pd.Index({"Jan. 1, 2008": "New Year’s Day"}), "pd.Index[str]"),
        pd.Index,
        str,
    )
    check(
        assert_type(
            pd.DatetimeIndex({"Jan. 1, 2008": "New Year’s Day"}), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.TimedeltaIndex({pd.Timedelta(days=1): "New Year’s Day"}),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )


def test_index_infer_objects() -> None:
    """Test infer_objects method on Index."""
    df = pd.DataFrame({"A": ["a", 1, 2, 3]})
    idx = df.set_index("A").index[1:]
    check(assert_type(idx.infer_objects(), pd.Index), pd.Index)


def test_multiindex_range() -> None:
    """Test using range in `MultiIndex.from_product` GH1285."""
    midx = pd.MultiIndex.from_product(
        [range(3), range(5)],
    )
    check(assert_type(midx, pd.MultiIndex), pd.MultiIndex)

    midx_mixed_types = pd.MultiIndex.from_product(
        [range(3), pd.Series([2, 3, 5])],
    )
    check(assert_type(midx_mixed_types, pd.MultiIndex), pd.MultiIndex)


def test_index_naming() -> None:
    """
    Test index names type both for the getter and the setter.
    The names of an index should be settable with a sequence (not str) and names
    property is a list[Hashable | None] (FrozenList).
    """
    df = pd.DataFrame({"a": ["a", "b", "c"], "i": [10, 11, 12]})

    df.index.names = ["idx"]
    check(assert_type(df.index.names, list[Hashable | None]), list)
    df.index.names = [3]
    check(assert_type(df.index.names, list[Hashable | None]), list)
    df.index.names = ("idx2",)
    check(assert_type(df.index.names, list[Hashable | None]), list)
    df.index.names = [None]
    check(assert_type(df.index.names, list[Hashable | None]), list)
    df.index.names = (None,)
    check(assert_type(df.index.names, list[Hashable | None]), list)
