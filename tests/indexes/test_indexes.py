from __future__ import annotations

from collections.abc import Hashable
import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.base import Index
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    PD_LTE_23,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    np_1darray,
    np_1darray_bool,
    np_1darray_int64,
    np_1darray_intp,
    np_ndarray_dt,
    pytest_warns_bounded,
)

if TYPE_CHECKING:
    from tests import Dtype  # noqa: F401


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
    check(assert_type(duplicated, np_1darray_bool), np_1darray_bool)


def test_index_isin() -> None:
    ind = pd.Index([1, 2, 3, 4, 5])
    isin = ind.isin([2, 4])
    check(assert_type(isin, np_1darray_bool), np_1darray_bool)


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

    collist = list(df.columns)
    check(assert_type(collist, list[str]), list, str)

    collist2 = list(df.columns[df.columns.str.contains("A|B")])
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


def test_index_dropna() -> None:
    idx = pd.Index([1, 2])

    check(assert_type(idx.dropna(how="all"), "pd.Index[int]"), pd.Index)
    check(assert_type(idx.dropna(how="any"), "pd.Index[int]"), pd.Index)

    midx = pd.MultiIndex.from_arrays([[1, 2], [3, 4]])

    check(assert_type(midx.dropna(how="all"), pd.MultiIndex), pd.MultiIndex)
    check(assert_type(midx.dropna(how="any"), pd.MultiIndex), pd.MultiIndex)


def test_index_neg() -> None:
    # GH 253
    idx = pd.Index([1, 2])
    check(assert_type(-idx, "pd.Index[int]"), pd.Index)


def test_types_to_numpy() -> None:
    idx = pd.Index([1, 2])
    check(assert_type(idx.to_numpy(), np_1darray), np_1darray)
    check(assert_type(idx.to_numpy(dtype="int", copy=True), np_1darray), np_1darray)
    check(assert_type(idx.to_numpy(na_value=0), np_1darray), np_1darray)

    r_idx = pd.RangeIndex(2)
    check(assert_type(r_idx.to_numpy(), np_1darray_int64), np_1darray_int64)
    check(assert_type(r_idx.to_numpy(na_value=0), np_1darray_int64), np_1darray_int64)
    check(
        assert_type(r_idx.to_numpy(dtype="int", copy=True), np_1darray),
        np_1darray,
        dtype=np.integer,
    )
    check(
        assert_type(r_idx.to_numpy(dtype=np.int32), np_1darray[np.int32]),
        np_1darray[np.int32],
    )


def test_range_index_union() -> None:
    check(
        assert_type(
            pd.RangeIndex(0, 10).union(pd.RangeIndex(10, 20)),
            "pd.Index[int] | pd.RangeIndex",
        ),
        pd.RangeIndex,
    )
    check(
        assert_type(
            pd.RangeIndex(0, 10).union([11, 12, 13]), "pd.Index[int] | pd.RangeIndex"
        ),
        pd.Index,
        np.integer,
    )
    check(assert_type(pd.RangeIndex(0, 10).union(["a", "b", "c"]), pd.Index), pd.Index)


def test_index_union_sort() -> None:
    """Test sort argument in pd.Index.union GH1264."""
    check(
        assert_type(
            pd.Index(["e", "f"]).union(["a", "b", "c"], sort=True), "pd.Index[str]"
        ),
        pd.Index,
        str,
    )
    check(
        assert_type(
            pd.Index(["e", "f"]).union(["a", "b", "c"], sort=False), "pd.Index[str]"
        ),
        pd.Index,
        str,
    )


def test_range_index_start_stop_step() -> None:
    idx = pd.RangeIndex(3)
    check(assert_type(idx.start, int), int)
    check(assert_type(idx.stop, int), int)
    check(assert_type(idx.step, int), int)


def test_interval_range() -> None:
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


def test_interval_index_breaks() -> None:
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


def test_interval_index_arrays() -> None:
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
    left_dt64_arr: np_ndarray_dt = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ],
        dtype="datetime64[ns]",
    )
    right_dt_arr: np_ndarray_dt = np.array(
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


def test_interval_index_tuples() -> None:
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


def test_multiindex_dtypes() -> None:
    # GH-597
    mi = pd.MultiIndex.from_tuples([(1, 2.0), (2, 3.0)], names=["foo", "bar"])
    check(assert_type(mi.dtypes, "pd.Series[Dtype]"), pd.Series)


def test_index_constructors() -> None:
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
        assert_type(unique_index.get_loc("b"), int | slice | np_1darray_bool),
        int,
    )

    monotonic_index = pd.Index(list("abbc"))
    check(
        assert_type(monotonic_index.get_loc("b"), int | slice | np_1darray_bool),
        slice,
    )

    non_monotonic_index = pd.Index(list("abcb"))
    check(
        assert_type(non_monotonic_index.get_loc("b"), int | slice | np_1darray_bool),
        np_1darray_bool,
    )

    i1, i2, i3 = pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(0, 2)
    unique_interval_index = pd.IntervalIndex([i1, i2])
    check(
        assert_type(unique_interval_index.get_loc(i1), int | slice | np_1darray_bool),
        np.int64,
    )
    overlap_interval_index = pd.IntervalIndex([i1, i2, i3])
    check(
        assert_type(overlap_interval_index.get_loc(1), int | slice | np_1darray_bool),
        np_1darray_bool,
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
    check(assert_type(codes, np_1darray), np_1darray)
    check(assert_type(idx_uniques, np_1darray | Index | Categorical), pd.Index)

    codes, idx_uniques = pd.Index(["b", "b", "a", "c", "b"]).factorize(
        use_na_sentinel=False
    )
    check(assert_type(codes, np_1darray), np_1darray)
    check(assert_type(idx_uniques, np_1darray | Index | Categorical), pd.Index)


def test_index_categorical() -> None:
    """Test creating an index with Categorical type GH1383."""
    sr = pd.Index([1], dtype="category")
    check(assert_type(sr, CategoricalIndex), CategoricalIndex)


def test_disallow_empty_index() -> None:
    # From GH 826
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = pd.Index()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]


def test_periodindex_shift() -> None:
    ind = pd.period_range(start="2022-06-01", periods=10)
    check(assert_type(ind.shift(1), pd.PeriodIndex), pd.PeriodIndex)


def test_timedeltaindex_shift() -> None:
    ind = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    # broken on 3.0.0.dev0 as of 20250813, fix with pandas-dev/pandas/issues/62094
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


def test_index_searchsorted() -> None:
    idx = pd.Index([1, 2, 3])
    check(assert_type(idx.searchsorted(1), np.intp), np.intp)
    check(assert_type(idx.searchsorted([1]), np_1darray_intp), np_1darray_intp)
    check(assert_type(idx.searchsorted(range(1, 2)), np_1darray_intp), np_1darray_intp)
    check(
        assert_type(idx.searchsorted(pd.Series([1])), np_1darray_intp), np_1darray_intp
    )
    check(
        assert_type(idx.searchsorted(np.array([1])), np_1darray_intp), np_1darray_intp
    )
    check(assert_type(idx.searchsorted(1, side="left"), np.intp), np.intp)
    check(assert_type(idx.searchsorted(1, sorter=[1, 0, 2]), np.intp), np.intp)
    check(assert_type(idx.searchsorted(1, sorter=range(3)), np.intp), np.intp)


def test_period_index_constructor() -> None:
    check(
        assert_type(pd.PeriodIndex(["2000"], dtype="period[D]"), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.PeriodIndex(["2000"], freq="D", name="foo", copy=True), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )


def test_period_index_asof_locs() -> None:
    idx = pd.PeriodIndex(["2000", "2001"], freq="D")
    where = pd.DatetimeIndex(["2023-05-30 00:12:00", "2023-06-01 00:00:00"])
    mask = np.ones(2, dtype=bool)
    check(assert_type(idx.asof_locs(where, mask), np_1darray_intp), np_1darray_intp)


def test_array_property() -> None:
    """Test that Index.array and semantic Index.array return ExtensionArray and its subclasses"""
    # casting due to pandas-dev/pandas-stubs#1383
    check(
        assert_type(Index([1], dtype="category").array, pd.Categorical),
        pd.Categorical,
        int,
    )
    check(
        assert_type(pd.interval_range(0, 1).array, IntervalArray),
        IntervalArray,
        pd.Interval,
    )

    # Test with pd.to_datetime().array - this is the main issue reported
    arr = pd.to_datetime(["2020-01-01", "2020-01-02"]).array
    check(assert_type(arr, DatetimeArray), DatetimeArray, pd.Timestamp)

    # Test with DatetimeIndex constructor directly
    dt_index = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    check(assert_type(dt_index.array, DatetimeArray), DatetimeArray, pd.Timestamp)

    check(
        assert_type(pd.to_timedelta(["1s"]).array, TimedeltaArray),
        TimedeltaArray,
        pd.Timedelta,
    )
    check(assert_type(Index([1]).array, ExtensionArray), ExtensionArray, np.integer)


def test_to_series() -> None:
    """Test that Index.to_series return typed Series"""
    check(
        assert_type(pd.interval_range(0, 1).to_series(), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(
            pd.date_range(start="2022-06-01", periods=10).to_series(),
            "pd.Series[pd.Timestamp]",
        ),
        pd.Series,
        pd.Timestamp,
    )

    check(
        assert_type(
            pd.timedelta_range(start="1 day", periods=10).to_series(),
            "pd.Series[pd.Timedelta]",
        ),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.period_range(start="2022-06-01", periods=10).to_series(),
            "pd.Series[pd.Period]",
        ),
        pd.Series,
        pd.Period,
    )

    check(
        assert_type(Index([True]).to_series(), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(assert_type(Index([1]).to_series(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(Index([1.0]).to_series(), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )
    check(
        assert_type(Index([1j]).to_series(), "pd.Series[complex]"),
        pd.Series,
        np.complexfloating,
    )
    check(assert_type(Index(["1"]).to_series(), "pd.Series[str]"), pd.Series, str)


def test_multiindex_union() -> None:
    """Test that MultiIndex.union returns MultiIndex"""
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["let", "num"])
    mi2 = pd.MultiIndex.from_product([["a", "b"], [3, 4]], names=["let", "num"])

    check(assert_type(mi.union(mi2), "pd.MultiIndex"), pd.MultiIndex)
    check(assert_type(mi.union([("c", 3), ("d", 4)]), "pd.MultiIndex"), pd.MultiIndex)


def test_multiindex_swaplevel() -> None:
    """Test that MultiIndex.swaplevel returns MultiIndex"""
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["let", "num"])
    check(assert_type(mi.swaplevel(0, 1), "pd.MultiIndex"), pd.MultiIndex)


def test_index_where() -> None:
    """Test Index.where with multiple types of other GH1419."""
    idx = pd.Index(range(48))
    mask = np.ones(48, dtype=bool)
    val_idx = idx.where(mask, idx)
    check(assert_type(val_idx, "pd.Index[int]"), pd.Index, int)

    val_sr = idx.where(mask, (idx).to_series())
    check(assert_type(val_sr, "pd.Index[int]"), pd.Index, int)


def test_datetimeindex_where() -> None:
    """Test DatetimeIndex.where with multiple types of other GH1419."""
    datetime_index = pd.date_range(start="2025-01-01", freq="h", periods=48)
    mask = np.ones(48, dtype=bool)
    val_idx = datetime_index.where(mask, datetime_index - pd.Timedelta(days=1))
    check(assert_type(val_idx, DatetimeIndex), DatetimeIndex)

    val_sr = datetime_index.where(
        mask, (datetime_index - pd.Timedelta(days=1)).to_series()
    )
    check(assert_type(val_sr, DatetimeIndex), DatetimeIndex)

    val_idx_scalar = datetime_index.where(mask, pd.Index([0, 1]))
    check(assert_type(val_idx_scalar, pd.Index), pd.Index)

    val_sr_scalar = datetime_index.where(mask, pd.Series([0, 1]))
    check(assert_type(val_sr_scalar, pd.Index), pd.Index)

    val_scalar = datetime_index.where(mask, 1)
    check(assert_type(val_scalar, pd.Index), pd.Index)

    val_range = pd.RangeIndex(2).where(pd.Series([True, False]), 3)
    check(assert_type(val_range, pd.Index), pd.RangeIndex)


def test_index_set_names() -> None:
    """Test Index.where with multiple types of other GH1419."""
    idx = pd.Index([1, 2])
    check(assert_type(idx.set_names("chinchilla"), "pd.Index[int]"), pd.Index, np.int64)
    check(assert_type(idx.set_names((0,)), "pd.Index[int]"), pd.Index, np.int64)
    check(
        assert_type(idx.set_names(["chinchilla"]), "pd.Index[int]"), pd.Index, np.int64
    )

    mi = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=["elk", "owl"])
    mi.set_names(["beluga", "pig"])
    mi.set_names({"elk": "beluga", "owl": "pig"})
