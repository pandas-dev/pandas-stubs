import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import check

from pandas.tseries.offsets import (
    Easter,
    MonthEnd,
)

# Each concrete offset is named by its own annotation.  An annotation such as
# `list[BaseOffset]` would erase every element, collapsing all of these into one identical
# static check of the inherited `BaseOffset` signature.


@pytest.fixture
def timestamp() -> pd.Timestamp:
    return pd.Timestamp("2026-01-01")


@pytest.fixture
def easter() -> Easter:
    """The offset class with leaf narrowing, named so its exact type is preserved."""
    return Easter()


@pytest.fixture
def month_end() -> MonthEnd:
    """A plain anchor offset, named so its exact type is preserved."""
    return MonthEnd()


def test_easter_datetime_like(easter: Easter, timestamp: pd.Timestamp) -> None:
    """`Easter` maps every datetime-like operand to a `Timestamp` in both directions."""
    check(assert_type(easter + dt.datetime(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 1, 1) + easter, pd.Timestamp), pd.Timestamp)
    check(assert_type(easter + dt.date(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(easter + np.datetime64("2026-01-01"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(easter + timestamp, pd.Timestamp), pd.Timestamp)


def test_easter_nat(easter: Easter) -> None:
    """`NaT` propagates through `Easter` rather than becoming a `Timestamp`."""
    check(assert_type(easter + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + easter, NaTType), NaTType)


def test_month_end_datetime_like(month_end: MonthEnd, timestamp: pd.Timestamp) -> None:
    """`MonthEnd` shares the same datetime-like contract as any anchor offset."""
    check(assert_type(month_end + dt.datetime(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 1, 1) + month_end, pd.Timestamp), pd.Timestamp)
    check(assert_type(month_end + dt.date(2026, 1, 1), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(month_end + np.datetime64("2026-01-01"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(month_end + timestamp, pd.Timestamp), pd.Timestamp)


def test_month_end_nat(month_end: MonthEnd) -> None:
    """`NaT` propagates through `MonthEnd` rather than becoming a `Timestamp`."""
    check(assert_type(month_end + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + month_end, NaTType), NaTType)
