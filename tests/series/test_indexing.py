from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    check,
    pytest_warns_bounded,
)


def test_types_select() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    if PD_LTE_23:
        # Not valid in 3.0
        with pytest_warns_bounded(
            FutureWarning,
            "Series.__getitem__ treating keys as positions is deprecated",
            lower="2.0.99",
        ):
            s[0]
    check(assert_type(s[1:], "pd.Series[int]"), pd.Series, np.integer)


def test_types_iloc_iat() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    s2 = pd.Series(data=[1, 2])
    s.loc["row1"]
    s.iat[0]
    s.iat[0] = 999
    s2.loc[0]
    s2.iat[0]
    s2.iat[0] = None


def test_types_loc_at() -> None:
    s = pd.Series(data={"row1": 1, "row2": 2})
    s2 = pd.Series(data=[1, 2])
    s.loc["row1"]
    s.at["row1"]
    s.at["row1"] = 9
    s2.loc[1]
    s2.at[1]
    s2.at[1] = 99


def test_types_getitem() -> None:
    s = pd.Series({"key": [0, 1, 2, 3]})
    check(assert_type(s["key"], Any), list)
    s2 = pd.Series([0, 1, 2, 3])
    check(assert_type(s2[0], int), np.integer)
    check(assert_type(s[:2], pd.Series), pd.Series)


def test_types_getitem_by_timestamp() -> None:
    index = pd.date_range("2018-01-01", periods=2, freq="D")
    series = pd.Series(range(2), index=index)
    check(assert_type(series[index[-1]], int), np.integer)


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


def test_series_loc_setitem() -> None:
    s = pd.Series([1, 2, 3, 4, 5])
    v = s.loc[[0, 2, 4]].values
    s.loc[[0, 2, 4]] = v


def test_series_isin() -> None:
    s = pd.Series([1, 2, 3, 4, 5])
    check(assert_type(s.isin([3, 4]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.isin({3, 4}), "pd.Series[bool]"), pd.Series, np.bool_)
    check(
        assert_type(s.isin(pd.Series([3, 4])), "pd.Series[bool]"), pd.Series, np.bool_
    )
    check(assert_type(s.isin(pd.Index([3, 4])), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s.isin(iter([3, "4"])), "pd.Series[bool]"), pd.Series, np.bool_)


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
    _s1: pd.Series = s["a", :]


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


def test_loc_callable() -> None:
    # GH 586
    s = pd.Series([1, 2])
    check(assert_type(s.loc[lambda x: x > 1], "pd.Series[int]"), pd.Series, np.integer)


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


def test_slice_timestamp() -> None:
    dti = pd.date_range("1/1/2025", "2/28/2025")

    s = pd.Series(list(range(len(dti))), index=dti)

    # For `s1`, see discussion in GH 397.  Needs mypy fix.
    # s1 = s.loc["2025-01-15":"2025-01-20"]

    # GH 397
    check(
        assert_type(
            s.loc[pd.Timestamp("2025-01-15") : pd.Timestamp("2025-01-20")],
            "pd.Series[int]",
        ),
        pd.Series,
        np.integer,
    )


def test_series_single_slice() -> None:
    # GH 572
    s = pd.Series([1, 2, 3])
    check(assert_type(s.loc[:], "pd.Series[int]"), pd.Series, np.integer)

    s.loc[:] = 1 + s


def test_series_index_timestamp() -> None:
    # GH 620
    dt1 = pd.to_datetime("2023-05-01")
    dt2 = pd.to_datetime("2023-05-02")
    s = pd.Series([1, 2], index=[dt1, dt2])
    check(assert_type(s[dt1], int), np.integer)
    check(assert_type(s.loc[[dt1]], "pd.Series[int]"), pd.Series, np.integer)
