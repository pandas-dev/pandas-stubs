import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import check

from pandas.tseries.offsets import (
    FY5253,
    BaseOffset,
    DateOffset,
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
    """Offsets that shift to an anchor, so they share one datetime-like contract."""
    # The `BaseOffset` annotation statically erases every element, so this fixture proves
    # the *base* contract over eight heterogeneous runtime instances.
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
