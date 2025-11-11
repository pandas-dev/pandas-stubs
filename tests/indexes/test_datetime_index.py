from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    check,
    np_1darray_bool,
    np_1darray_intp,
)


def test_index_relops() -> None:
    # GH 265
    data = check(
        assert_type(
            pd.date_range("2022-01-01", "2022-01-31", freq="D"), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    x = pd.Timestamp("2022-01-17")
    idx = check(
        assert_type(pd.Index(data, name="date"), pd.DatetimeIndex), pd.DatetimeIndex
    )
    check(assert_type(data[x <= idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x < idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x >= idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[x > idx], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx <= x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx < x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx >= x], pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(data[idx > x], pd.DatetimeIndex), pd.DatetimeIndex)

    ind = pd.Index([1, 2, 3])
    check(assert_type(ind <= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind < 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind >= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind > 2, np_1darray_bool), np_1darray_bool)


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


def test_intersection() -> None:
    # GH 744
    index = pd.DatetimeIndex(["2022-01-01"])
    check(assert_type(index.intersection(index), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(index.intersection([pd.Timestamp("1/1/2023")]), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )


def test_datetime_index_max_min_reductions() -> None:
    dtidx = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    check(assert_type(dtidx.argmax(), np.int64), np.int64)
    check(assert_type(dtidx.argmin(), np.int64), np.int64)
    check(assert_type(dtidx.max(), pd.Timestamp), pd.Timestamp)
    check(assert_type(dtidx.min(), pd.Timestamp), pd.Timestamp)


def test_datetimeindex_shift() -> None:
    ind = pd.date_range("2023-01-01", "2023-02-01")
    check(assert_type(ind.shift(1), pd.DatetimeIndex), pd.DatetimeIndex)


def test_datetimeindex_indexer_at_time() -> None:
    dti = pd.date_range("2023-01-01", "2023-02-01")
    check(assert_type(dti.indexer_at_time("10:00"), np_1darray_intp), np_1darray_intp)
    check(assert_type(dti.indexer_at_time(time(10)), np_1darray_intp), np_1darray_intp)


def test_datetimeindex_indexer_between_time() -> None:
    dti = pd.date_range("2023-01-01", "2023-02-01")
    check(
        assert_type(
            dti.indexer_between_time(
                "10:00", time(11), include_start=False, include_end=True
            ),
            np_1darray_intp,
        ),
        np_1darray_intp,
    )
    check(
        assert_type(dti.indexer_between_time(time(10), "11:00"), np_1darray_intp),
        np_1darray_intp,
    )


def test_datetimeindex_snap() -> None:
    dti = pd.date_range("2023-01-01", "2023-02-01")
    check(assert_type(dti.snap("MS"), pd.DatetimeIndex), pd.DatetimeIndex)
