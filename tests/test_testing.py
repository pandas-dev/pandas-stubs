from __future__ import annotations

import pandas as pd
from pandas.testing import (
    assert_frame_equal,
    assert_series_equal,
)

from tests import TYPE_CHECKING_INVALID_USAGE


def test_types_assert_series_equal() -> None:
    s1 = pd.Series([0, 1, 1, 0])
    s2 = pd.Series([0, 1, 1, 0])
    assert_series_equal(left=s1, right=s2)
    assert_series_equal(
        s1, s2, check_freq=False, check_categorical=True, check_flags=True
    )

    assert_series_equal(s1, s2, check_like=True)
    # GH 417
    assert_series_equal(s1, s2, check_index=False)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_series_equal(s1, s2, check_dtype=True, check_less_precise=True, check_names=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        assert_series_equal(s1, s2, check_datetimelike_compat=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

    if TYPE_CHECKING_INVALID_USAGE:
        assert_series_equal(s1, s2, check_dtype=True, check_less_precise=True, check_names=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        assert_series_equal(s1, s2, check_datetimelike_compat=True)  # type: ignore[call-overload] # pyright: ignore[reportCallIssue] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_assert_frame_equal() -> None:
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [1, 2, 3]})
    # GH 56
    assert_frame_equal(df1, df2, check_index_type=False)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_frame_equal(df1, df2, check_datetimelike_compat=True)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[unexpected-keyword] # ty: ignore[unknown-argument]


def test_assert_datetime_array_equal() -> None:
    from pandas._testing import assert_datetime_array_equal
    from pandas.core.arrays.datetimes import DatetimeArray

    dta1 = pd.array([pd.Timestamp("2020-01-01")])
    dta2 = pd.array([pd.Timestamp("2020-01-01")])
    assert isinstance(dta1, DatetimeArray)
    assert isinstance(dta2, DatetimeArray)
    assert_datetime_array_equal(dta1, dta2)
    assert_datetime_array_equal(dta1, dta2, obj="CustomName")

    if TYPE_CHECKING_INVALID_USAGE:
        assert_datetime_array_equal(dta1, dta2, check_freq=True)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[unexpected-keyword] # ty: ignore[unknown-argument]


def test_assert_timedelta_array_equal() -> None:
    from pandas._testing import assert_timedelta_array_equal
    from pandas.core.arrays.timedeltas import TimedeltaArray

    tda1 = pd.array([pd.Timedelta("1 day")])
    tda2 = pd.array([pd.Timedelta("1 day")])
    assert isinstance(tda1, TimedeltaArray)
    assert isinstance(tda2, TimedeltaArray)
    assert_timedelta_array_equal(tda1, tda2)
    assert_timedelta_array_equal(tda1, tda2, obj="CustomName")

    if TYPE_CHECKING_INVALID_USAGE:
        assert_timedelta_array_equal(tda1, tda2, check_freq=True)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[unexpected-keyword] # ty: ignore[unknown-argument]
