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
    Micro,
    Milli,
    Minute,
    Nano,
    Second,
    Tick,
    Week,
)


@pytest.fixture
def left() -> Tick:
    """The Tick offset under test."""
    return Hour()


def test_tick_python_scalars(left: Tick) -> None:
    """`Tick` maps `date`/`datetime` operands to a `Timestamp` and `timedelta` to a `Timedelta`."""
    check(assert_type(left + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.timedelta(hours=2), pd.Timedelta), pd.Timedelta)
    check(assert_type(dt.timedelta(hours=2) + left, pd.Timedelta), pd.Timedelta)


def test_tick_numpy_scalars(left: Tick) -> None:
    """`Tick` maps a `np.datetime64` operand to a `Timestamp` and `np.timedelta64` to a `Timedelta`."""
    check(assert_type(left + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + np.timedelta64(2, "h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(np.timedelta64(2, "h") + left, pd.Timedelta), pd.Timedelta)


def test_tick_pandas_scalars(left: Tick) -> None:
    """`Tick` maps `Timestamp` to itself, propagates `NaT`, and maps `Timedelta` to itself."""
    check(assert_type(left + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)
    check(assert_type(left + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("2h") + left, pd.Timedelta), pd.Timedelta)


def test_tick_same_class() -> None:
    """Adding two ticks of the same concrete class preserves that class."""
    check(assert_type(Hour(1) + Hour(2), Hour), Hour)
    check(assert_type(Minute(1) + Minute(2), Minute), Minute)
    check(assert_type(Second(1) + Second(2), Second), Second)
    check(assert_type(Milli(1) + Milli(2), Milli), Milli)
    check(assert_type(Micro(1) + Micro(2), Micro), Micro)
    check(assert_type(Nano(1) + Nano(2), Nano), Nano)


def test_tick_mixed_ticks() -> None:
    """Mixing tick classes normalizes to a `Tick`, whose exact class is not static."""
    check(assert_type(Hour(1) + Minute(30), Tick), Tick)
    check(assert_type(Hour(1) + Minute(60), Tick), Tick)


def test_tick_offsets(left: Tick) -> None:
    """A day-or-coarser offset added to a tick collapses to a `Timedelta`, except `BusinessDay`."""
    check(assert_type(left + Day(), pd.Timedelta), pd.Timedelta)
    check(assert_type(left + Week(), pd.Timedelta), pd.Timedelta)
    check(assert_type(left + BusinessDay(), BusinessDay), BusinessDay)


def test_tick_reflected(left: Tick) -> None:
    """A tick's reflected addition preserves an exact-class tick and maps durations to `Timedelta`."""
    check(assert_type(Hour(1).__radd__(Hour(2)), Hour), Hour)
    check(assert_type(left.__radd__(pd.Timedelta("2h")), pd.Timedelta), pd.Timedelta)
