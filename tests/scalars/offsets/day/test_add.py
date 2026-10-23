import datetime as dt
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import check

from pandas.tseries.offsets import (
    BusinessDay,
    Day,
    Hour,
    Week,
)


@pytest.fixture
def left() -> Day:
    """The Day offset under test."""
    return Day()


def test_day_python_scalars(left: Day) -> None:
    """`Day` maps `date`/`datetime` operands to a `Timestamp` in both directions."""
    check(assert_type(left + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)


def test_day_numpy_scalars(left: Day) -> None:
    """`Day` maps a `np.datetime64` operand to a `Timestamp` in both directions."""
    check(assert_type(left + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + left, pd.Timestamp), pd.Timestamp)


def test_day_pandas_scalars(left: Day) -> None:
    """`Day` maps a `Timestamp` operand to a `Timestamp` and propagates `NaT`."""
    check(assert_type(left + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)


def test_day_durations(left: Day) -> None:
    """`Day` preserves the type of a duration operand, in both directions."""
    check(assert_type(left + dt.timedelta(hours=2), dt.timedelta), dt.timedelta)
    check(assert_type(dt.timedelta(hours=2) + left, dt.timedelta), dt.timedelta)
    check(assert_type(left + np.timedelta64(2, "h"), np.timedelta64), np.timedelta64)
    check(assert_type(left + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)


def test_day_offsets(left: Day) -> None:
    """`Day` collapses day-or-finer offsets, degenerates `Week`, and defers to `BusinessDay`."""
    check(assert_type(Day(1) + Day(2), Day), Day)
    check(assert_type(left + Hour(), pd.Timedelta), pd.Timedelta)
    check(assert_type(left + Week(), dt.timedelta), dt.timedelta)
    check(assert_type(left + BusinessDay(), BusinessDay), BusinessDay)
    check(assert_type(left.__radd__(Hour()), pd.Timedelta), pd.Timedelta)
