import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BaseOffset,
    BusinessDay,
    CustomBusinessDay,
    DateOffset,
    Day,
    Easter,
    FY5253,
    Hour,
    Micro,
    Milli,
    Minute,
    MonthEnd,
    Nano,
    QuarterEnd,
    Second,
    SemiMonthEnd,
    Tick,
    Week,
    WeekOfMonth,
    YearEnd,
)


@pytest.fixture
def timestamp() -> pd.Timestamp:
    return pd.Timestamp("2026-01-01")


@pytest.fixture
def anchors() -> list[BaseOffset]:
    """Offsets that shift to an anchor, so they reject durations and other offsets."""
    return [
        DateOffset(),
        Easter(),
        MonthEnd(),
        QuarterEnd(),
        SemiMonthEnd(),
        WeekOfMonth(),
        YearEnd(),
        FY5253(),
    ]


def test_anchor_datetime_like(anchors: list[BaseOffset], timestamp: pd.Timestamp) -> None:
    """Every anchor offset maps a datetime-like operand to a `Timestamp` in both directions."""
    for anchor in anchors:
        check(assert_type(anchor + timestamp, pd.Timestamp), pd.Timestamp)
        check(assert_type(timestamp + anchor, pd.Timestamp), pd.Timestamp)
        check(assert_type(anchor + dt.date(2026, 1, 1), pd.Timestamp), pd.Timestamp)
        check(
            assert_type(anchor + np.datetime64("2026-01-01"), pd.Timestamp), pd.Timestamp
        )
        check(assert_type(anchor.__radd__(timestamp), pd.Timestamp), pd.Timestamp)


def test_anchor_nat(anchors: list[BaseOffset]) -> None:
    """`NaT` propagates through an anchor offset rather than becoming a `Timestamp`."""
    for anchor in anchors:
        check(assert_type(anchor + pd.NaT, NaTType), NaTType)
        check(assert_type(pd.NaT + anchor, NaTType), NaTType)


def test_anchor_rejects_durations() -> None:
    """Anchor offsets cannot absorb a duration or another offset (runtime `TypeError`)."""
    anchor: BaseOffset = MonthEnd()
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = MonthEnd() + dt.timedelta(hours=2)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _1 = dt.timedelta(hours=2) + MonthEnd()  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _2 = MonthEnd() + pd.Timedelta("2h")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _3 = Easter() + Day()  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _4 = MonthEnd() + MonthEnd()  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        # The same contract holds for a `BaseOffset`-typed receiver.
        _5 = anchor + dt.timedelta(hours=2)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_tick_arithmetic(timestamp: pd.Timestamp) -> None:
    """Ticks accept durations and other ticks; same-class addition preserves the class."""
    check(assert_type(Hour(1) + Hour(2), Hour), Hour)
    check(assert_type(Minute(1) + Minute(2), Minute), Minute)
    check(assert_type(Second(1) + Second(2), Second), Second)
    check(assert_type(Milli(1) + Milli(2), Milli), Milli)
    check(assert_type(Micro(1) + Micro(2), Micro), Micro)
    check(assert_type(Nano(1) + Nano(2), Nano), Nano)
    # Mixing ticks normalizes to the coarsest non-zero unit, which is not static.
    check(assert_type(Hour(1) + Minute(30), Tick), Tick)
    check(assert_type(Hour(1) + Minute(60), Tick), Tick)
    check(assert_type(timestamp + Hour(1), pd.Timestamp), pd.Timestamp)
    check(assert_type(Hour(1) + timestamp, pd.Timestamp), pd.Timestamp)
    check(assert_type(Hour(1) + pd.NaT, NaTType), NaTType)
    check(
        assert_type(Hour(1) + dt.timedelta(hours=2), pd.Timedelta), pd.Timedelta
    )
    check(assert_type(Hour(1) + np.timedelta64(2, "h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(Hour(1) + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(Hour(1) + Day(), pd.Timedelta), pd.Timedelta)
    check(assert_type(Hour(1) + Week(), pd.Timedelta), pd.Timedelta)
    check(assert_type(Hour(1) + BusinessDay(), BusinessDay), BusinessDay)
    check(assert_type(Hour(1).__radd__(Hour(2)), Hour), Hour)
    check(assert_type(Hour(1).__radd__(pd.Timedelta("2h")), pd.Timedelta), pd.Timedelta)


def test_day_arithmetic(timestamp: pd.Timestamp) -> None:
    """`Day` preserves a duration's own type and collapses day-or-finer offsets."""
    check(assert_type(Day(1) + Day(2), Day), Day)
    check(assert_type(timestamp + Day(), pd.Timestamp), pd.Timestamp)
    check(assert_type(Day() + timestamp, pd.Timestamp), pd.Timestamp)
    check(assert_type(Day() + pd.NaT, NaTType), NaTType)
    check(assert_type(Day() + dt.timedelta(hours=2), dt.timedelta), dt.timedelta)
    check(assert_type(Day() + np.timedelta64(2, "h"), np.timedelta64), np.timedelta64)
    check(assert_type(Day() + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(Day() + Hour(), pd.Timedelta), pd.Timedelta)
    check(assert_type(Day() + Week(), dt.timedelta), dt.timedelta)
    check(assert_type(Day() + BusinessDay(), BusinessDay), BusinessDay)
    check(assert_type(dt.timedelta(hours=2) + Day(), dt.timedelta), dt.timedelta)
    check(assert_type(Day().__radd__(Hour()), pd.Timedelta), pd.Timedelta)


def test_week_arithmetic() -> None:
    """`Week` folds day-and-coarser operands back to a plain `timedelta`."""
    check(assert_type(Week() + dt.timedelta(hours=2), dt.timedelta), dt.timedelta)
    check(assert_type(Week() + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)
    # Unlike `Day`, a `np.timedelta64` operand degenerates to a plain `timedelta`.
    check(assert_type(Week() + np.timedelta64(2, "h"), dt.timedelta), dt.timedelta)
    check(assert_type(Week() + Hour(), pd.Timedelta), pd.Timedelta)
    check(assert_type(Week() + Day(), dt.timedelta), dt.timedelta)
    check(assert_type(Week() + Week(), dt.timedelta), dt.timedelta)
    check(assert_type(Week() + BusinessDay(), BusinessDay), BusinessDay)


@pytest.mark.parametrize("business", [BusinessDay(), CustomBusinessDay()])
def test_business_day_arithmetic(business: BusinessDay) -> None:
    """A duration or day-or-finer offset folds into `offset=`, yielding a `BusinessDay`."""
    check(assert_type(business + Day(), BusinessDay), BusinessDay)
    check(assert_type(business + Hour(), BusinessDay), BusinessDay)
    check(assert_type(business + Week(), BusinessDay), BusinessDay)
    check(assert_type(business + dt.timedelta(hours=2), BusinessDay), BusinessDay)
    check(assert_type(business + pd.Timedelta("2h"), BusinessDay), BusinessDay)
    check(assert_type(Day() + business, BusinessDay), BusinessDay)
    check(assert_type(Hour() + business, BusinessDay), BusinessDay)
    check(assert_type(Week() + business, BusinessDay), BusinessDay)
    check(assert_type(business.__radd__(Hour()), BusinessDay), BusinessDay)
    check(assert_type(business.__radd__(pd.Timedelta("2h")), BusinessDay), BusinessDay)


def test_custom_business_day_returns_business_day() -> None:
    """`CustomBusinessDay` arithmetic returns a plain `BusinessDay` at runtime."""
    result = check(assert_type(CustomBusinessDay() + Day(), BusinessDay), BusinessDay)
    assert type(result) is BusinessDay


def test_business_day_rejects_business_day() -> None:
    """Two business-day offsets cannot be added (runtime `TypeError`)."""
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = BusinessDay() + BusinessDay()  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
