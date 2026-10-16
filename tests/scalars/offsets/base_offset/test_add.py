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
    FY5253,
    BaseOffset,
    DateOffset,
    Day,
    Easter,
    MonthEnd,
    QuarterEnd,
    SemiMonthEnd,
    WeekOfMonth,
    YearEnd,
)


@pytest.fixture
def timestamp() -> pd.Timestamp:
    return pd.Timestamp("2026-01-01")


@pytest.fixture
def anchors() -> list[BaseOffset]:
    """The anchor offsets under test."""
    # The `BaseOffset` annotation statically erases every element, so this fixture proves
    # the *base* contract over eight heterogeneous runtime instances.  Pinning a
    # concrete anchor class is `easter/test_add.py`'s job -- it is what would catch
    # `Easter` resolving incorrectly, and its object-array cases have no counterpart
    # here.  The two modules are complementary, not duplicative.
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


def test_anchor_datetime_like(
    anchors: list[BaseOffset], timestamp: pd.Timestamp
) -> None:
    """Every anchor offset maps a datetime-like operand to a `Timestamp` in both directions."""
    for anchor in anchors:
        check(assert_type(anchor + timestamp, pd.Timestamp), pd.Timestamp)
        check(assert_type(timestamp + anchor, pd.Timestamp), pd.Timestamp)
        check(assert_type(anchor + dt.date(2026, 1, 1), pd.Timestamp), pd.Timestamp)
        check(
            assert_type(anchor + np.datetime64("2026-01-01"), pd.Timestamp),
            pd.Timestamp,
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
