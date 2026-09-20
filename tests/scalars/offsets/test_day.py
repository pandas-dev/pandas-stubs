from typing import assert_type

import pandas as pd

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    Day,
    Hour,
)


def test_day_is_not_a_tick() -> None:
    """`Day` is a `SingleConstructorOffset`, not a `Tick`, so sub-day operators reject it."""
    check(assert_type(pd.Timedelta(Hour()), pd.Timedelta), pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        pd.Timedelta(Day())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]


def test_day_stays_an_offset() -> None:
    """`Day` is still a `BaseOffset`, so it still applies to datetimes and combines with itself."""
    timestamp = pd.Timestamp("2026-01-01")
    check(assert_type(Day() + timestamp, pd.Timestamp), pd.Timestamp)
    check(assert_type(timestamp + Day(), pd.Timestamp), pd.Timestamp)
    check(assert_type(Day(1) + Day(2), Day), Day)
    check(
        assert_type(
            pd.date_range("2026-01-01", periods=2, freq=Day()), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
