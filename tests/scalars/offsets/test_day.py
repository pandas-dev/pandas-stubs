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


def test_day_is_not_a_tick() -> None:
    """`Day` is a `SingleConstructorOffset`, not a `Tick`, so sub-day operators reject it."""
    check(assert_type(pd.Timedelta(Hour()), pd.Timedelta), pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        pd.Timedelta(Day())  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        _0: Tick = Day()  # type: ignore[assignment] # pyright: ignore[reportAssignmentType] # pyrefly: ignore[bad-assignment] # ty: ignore[invalid-assignment]


def test_day_is_still_a_base_offset() -> None:
    """`Day` still inherits `BaseOffset`, so the hierarchy change is confined to `Tick`."""
    # The assignment is the assertion: it only type checks because `Day` is a
    # `BaseOffset`.  `assert_type` cannot express it, as ty narrows `offset` to `Day`.
    offset: BaseOffset = Day()
    check(offset, Day)
