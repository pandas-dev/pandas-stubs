from typing import assert_type

import pandas as pd

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BaseOffset,
    Day,
    Hour,
    Tick,
)


def test_day_type() -> None:
    """`Day` is a `SingleConstructorOffset` (a `BaseOffset`), not a `Tick`."""
    # positive: `Day()` is exactly `Day`, not a base
    check(assert_type(Day(), Day), Day)
    _day_is_offset: BaseOffset = Day()
    if TYPE_CHECKING_INVALID_USAGE:
        pd.Timedelta(Day())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        _day_is_not_tick: Tick = Day()  # type: ignore[assignment] # pyright: ignore[reportAssignmentType] # pyrefly: ignore[bad-assignment] # ty: ignore[invalid-assignment]


def test_hour_type() -> None:
    """`Hour` is a `Tick` (a `BaseOffset`), so `Timedelta` accepts it."""
    check(assert_type(Hour(), Hour), Hour)
    _hour_is_tick: Tick = Hour()
    check(assert_type(pd.Timedelta(Hour()), pd.Timedelta), pd.Timedelta)
