from __future__ import annotations

from typing import assert_type

import pandas as pd

from tests import check


def test_datetime_index_relops() -> None:
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
