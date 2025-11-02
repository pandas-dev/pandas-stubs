from typing import TYPE_CHECKING

import numpy as np
from pandas.core.arrays import DatetimeArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.timedeltas import TimedeltaArray
from pandas.core.frame import DataFrame
from pandas.core.indexes.accessors import (
    DatetimeProperties,
    PeriodProperties,
    Properties,
    TimedeltaProperties,
)
from pandas.core.indexes.interval import interval_range
from pandas.core.indexes.period import period_range
from pandas.core.series import Series
from typing_extensions import assert_type

from pandas._libs.interval import Interval
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

if TYPE_CHECKING:
    from pandas.core.indexes.accessors import TimestampProperties  # noqa: F401


def test_property_dt() -> None:
    """Test the Series.dt property"""
    check(
        assert_type(Series([Timestamp(2025, 9, 28)]).dt, "TimestampProperties"),
        DatetimeProperties,
    )
    check(
        assert_type(Series([Timedelta(1, "s")]).dt, TimedeltaProperties),
        TimedeltaProperties,
    )
    check(
        assert_type(
            period_range(start="2022-06-01", periods=10).to_series().dt,
            PeriodProperties,
        ),
        PeriodProperties,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        s = DataFrame({"a": [1]})["a"]
        # python/mypy#19952: mypy believes Properties and its subclasses have a
        # conflict and gives Any for s.dt
        assert_type(s.dt, Properties)  # type: ignore[assert-type]
        _1 = Series([1]).dt  # type: ignore[arg-type] # pyright: ignore[reportAttributeAccessIssue]


def test_property_array() -> None:
    """Test that Series.array returns ExtensionArray and its subclasses"""
    check(
        assert_type(Series([1], dtype="category").array, Categorical), Categorical, int
    )
    check(
        assert_type(Series(interval_range(0, 1)).array, IntervalArray),
        IntervalArray,
        Interval,
    )
    check(
        assert_type(Series([Timestamp(2025, 9, 28)]).array, DatetimeArray),
        DatetimeArray,
        Timestamp,
    )
    check(
        assert_type(Series([Timedelta(1, "s")]).array, TimedeltaArray),
        TimedeltaArray,
        Timedelta,
    )
    check(
        assert_type(Series([1]).array, NumpyExtensionArray),
        NumpyExtensionArray,
        np.integer,
    )
    # python/mypy#19952: mypy believes ExtensionArray and its subclasses have a
    # conflict and gives Any for s.array
    # check(assert_type(Series([1, "s"]).array, NumpyExtensionArray), NumpyExtensionArray)  # type: ignore[assert-type]
