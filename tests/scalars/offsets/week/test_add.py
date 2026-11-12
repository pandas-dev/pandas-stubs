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
def left() -> Week:
    """The Week offset under test."""
    return Week()


def test_week_python_scalars(left: Week) -> None:
    """`Week` maps `date`/`datetime` operands to a `Timestamp` and a `timedelta` to itself."""
    check(assert_type(left + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.timedelta(hours=2), dt.timedelta), dt.timedelta)


def test_week_numpy_scalars(left: Week) -> None:
    """`Week` maps `np.datetime64` to a `Timestamp`, while `np.timedelta64` degenerates."""
    check(assert_type(left + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + np.timedelta64(2, "h"), dt.timedelta), dt.timedelta)


def test_week_pandas_scalars(left: Week) -> None:
    """`Week` maps `Timestamp` to itself, propagates `NaT`, and maps `Timedelta` to itself."""
    check(assert_type(left + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)
    check(assert_type(left + pd.Timedelta("2h"), pd.Timedelta), pd.Timedelta)


def test_week_offsets(left: Week) -> None:
    """`Week` collapses day-or-finer offsets, degenerates day-and-coarser ones, and defers to `BusinessDay`."""
    check(assert_type(left + Hour(), pd.Timedelta), pd.Timedelta)
    check(assert_type(left + Day(), dt.timedelta), dt.timedelta)
    check(assert_type(left + Week(), dt.timedelta), dt.timedelta)
    check(assert_type(left + BusinessDay(), BusinessDay), BusinessDay)
