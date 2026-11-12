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
    BusinessDay,
    CustomBusinessDay,
    Day,
    Hour,
    Week,
)


@pytest.mark.parametrize("left", [BusinessDay(), CustomBusinessDay()])
def test_business_day_python_scalars(left: BusinessDay) -> None:
    """`BusinessDay` maps `date`/`datetime` operands to a `Timestamp` and folds a `timedelta`."""
    check(assert_type(left + dt.date(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.date(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.datetime(2026, 10, 2), pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2026, 10, 2) + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + dt.timedelta(hours=2), BusinessDay), BusinessDay)


@pytest.mark.parametrize("left", [BusinessDay(), CustomBusinessDay()])
def test_business_day_numpy_scalars(left: BusinessDay) -> None:
    """`BusinessDay` maps a `np.datetime64` operand to a `Timestamp` and folds `np.timedelta64`."""
    check(assert_type(left + np.datetime64("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(np.datetime64("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + np.timedelta64(2, "h"), BusinessDay), BusinessDay)


@pytest.mark.parametrize("left", [BusinessDay(), CustomBusinessDay()])
def test_business_day_pandas_scalars(left: BusinessDay) -> None:
    """`BusinessDay` maps `Timestamp` to itself, propagates `NaT`, and folds a `Timedelta`."""
    check(assert_type(left + pd.Timestamp("2026-10-02"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp("2026-10-02") + left, pd.Timestamp), pd.Timestamp)
    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)
    check(assert_type(left + pd.Timedelta("2h"), BusinessDay), BusinessDay)


@pytest.mark.parametrize("left", [BusinessDay(), CustomBusinessDay()])
def test_business_day_offsets(left: BusinessDay) -> None:
    """A day-or-finer offset folds into `offset=`, and either direction yields `BusinessDay`."""
    check(assert_type(left + Day(), BusinessDay), BusinessDay)
    check(assert_type(left + Hour(), BusinessDay), BusinessDay)
    check(assert_type(left + Week(), BusinessDay), BusinessDay)
    check(assert_type(Day() + left, BusinessDay), BusinessDay)
    check(assert_type(Hour() + left, BusinessDay), BusinessDay)
    check(assert_type(Week() + left, BusinessDay), BusinessDay)
    check(assert_type(left.__radd__(Hour()), BusinessDay), BusinessDay)
    check(assert_type(left.__radd__(pd.Timedelta("2h")), BusinessDay), BusinessDay)


def test_custom_business_day_returns_business_day() -> None:
    """`CustomBusinessDay` arithmetic returns a plain `BusinessDay` at runtime."""
    result = check(assert_type(CustomBusinessDay() + Day(), BusinessDay), BusinessDay)
    assert type(result) is BusinessDay


def test_business_day_rejects_business_day() -> None:
    """Two business-day offsets cannot be added (runtime `TypeError`)."""
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = BusinessDay() + BusinessDay()  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
