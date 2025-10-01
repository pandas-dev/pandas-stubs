from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.indexes.accessors import (
    DatetimeProperties,
    PeriodProperties,
    Properties,
    TimedeltaProperties,
)
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

if TYPE_CHECKING:
    from pandas.core.indexes.accessors import TimestampProperties  # noqa: F401


def test_dt_property() -> None:
    """Test the Series.dt property"""
    check(
        assert_type(pd.Series([pd.Timestamp(2025, 9, 28)]).dt, "TimestampProperties"),
        DatetimeProperties,
    )
    check(
        assert_type(pd.Series([pd.Timedelta(1, "s")]).dt, TimedeltaProperties),
        TimedeltaProperties,
    )
    check(
        assert_type(
            pd.period_range(start="2022-06-01", periods=10).to_series().dt,
            PeriodProperties,
        ),
        PeriodProperties,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        s = pd.DataFrame({"a": [1]})["a"]
        # python/mypy#19952: mypy believes Properties and its subclasses have a
        # conflict and gives Any for s.dt
        assert_type(s.dt, Properties)  # type: ignore[assert-type]
        _1 = pd.Series([1]).dt  # type: ignore[arg-type] # pyright: ignore[reportAttributeAccessIssue]


def test_array_property() -> None:
    """Test that Series.array returns ExtensionArray and its subclasses"""
    check(
        assert_type(
            pd.Series([1], dtype="category").array,
            pd.Categorical,
        ),
        pd.Categorical,
        int,
    )
    check(
        assert_type(pd.Series(pd.interval_range(0, 1)).array, IntervalArray),
        IntervalArray,
        pd.Interval,
    )
    check(
        assert_type(pd.Series([pd.Timestamp(2025, 9, 28)]).array, DatetimeArray),
        DatetimeArray,
        pd.Timestamp,
    )
    check(
        assert_type(pd.Series([pd.Timedelta(1, "s")]).array, TimedeltaArray),
        TimedeltaArray,
        pd.Timedelta,
    )
    check(assert_type(pd.Series([1]).array, ExtensionArray), ExtensionArray, np.integer)
    # python/mypy#19952: mypy believes ExtensionArray and its subclasses have a
    # conflict and gives Any for s.array
    check(assert_type(pd.Series([1, "s"]).array, ExtensionArray), ExtensionArray)  # type: ignore[assert-type]
