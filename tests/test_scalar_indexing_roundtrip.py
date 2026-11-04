from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1.0, 2.0]),
        pd.Series([True, False]),
        pd.Series([pd.Period("2026-01", freq="M"), pd.Period("2026-02", freq="M")]),
        pd.Series([pd.Interval(0, 1), pd.Interval(1, 2)]),
        pd.Series([pd.NA, pd.NA], dtype="Float64"),
        pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
        pd.Series([None, None], dtype=object),
    ],
)
def test_scalar_indexing_roundtrip(series: pd.Series) -> None:
    series.iat[0] = series.iat[1]
    series.at[0] = series.at[1]

    frame = pd.DataFrame({"a": series})
    frame["b"] = series.iat[0]
    frame.loc[0, "a"] = series.iat[1]
    frame.iloc[0, 0] = series.iat[1]
    frame.iat[0, 0] = frame.iat[1, 0]
    frame.at[0, "a"] = frame.at[1, "a"]
    pd.testing.assert_series_equal(frame["a"], series, check_names=False)
    pd.testing.assert_series_equal(
        frame["b"], series, check_dtype=False, check_names=False
    )

    scalar = series.iat[0]
    missing: bool = pd.isna(scalar)
    present: bool = pd.notna(scalar)
    assert missing == pd.isnull(scalar)
    assert present == pd.notnull(scalar)
    assert missing != present


def check_nullable_period_narrowing(value: pd.Period | None) -> None:
    if pd.notna(value):
        assert_type(value, pd.Period)
    if not pd.isna(value):
        assert_type(value, pd.Period)


def check_nullable_numpy_bool_narrowing(value: np.bool_ | None) -> None:
    if pd.notna(value):
        assert_type(value, np.bool_)
    if not pd.isna(value):
        assert_type(value, np.bool_)


def check_nullable_interval_narrowing(value: pd.Interval[int] | None) -> None:
    if pd.notna(value):
        assert_type(value, pd.Interval[int])
    if not pd.isna(value):
        assert_type(value, pd.Interval[int])
