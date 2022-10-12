from datetime import (
    timedelta,
    timezone,
)

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check

from pandas.tseries.offsets import (
    BusinessDay,
    CustomBusinessDay,
    Day,
)


def test_datetimetz_dtype() -> None:
    check(
        assert_type(pd.DatetimeTZDtype(unit="ns", tz="UTC"), pd.DatetimeTZDtype),
        pd.DatetimeTZDtype,
    )
    check(
        assert_type(
            pd.DatetimeTZDtype(unit="ns", tz=timezone(timedelta(hours=1))),
            pd.DatetimeTZDtype,
        ),
        pd.DatetimeTZDtype,
    )


def test_period_dtype() -> None:
    check(assert_type(pd.PeriodDtype(freq="D"), pd.PeriodDtype), pd.PeriodDtype)
    check(assert_type(pd.PeriodDtype(freq=Day()), pd.PeriodDtype), pd.PeriodDtype)
    check(
        assert_type(pd.PeriodDtype(freq=BusinessDay()), pd.PeriodDtype), pd.PeriodDtype
    )
    check(
        assert_type(pd.PeriodDtype(freq=CustomBusinessDay()), pd.PeriodDtype),
        pd.PeriodDtype,
    )


def test_interval_dtype() -> None:
    check(
        assert_type(
            pd.Interval(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02")),
            "pd.Interval[pd.Timestamp]",
        ),
        pd.Interval,
    )
    check(
        assert_type(pd.Interval(1, 2, closed="left"), "pd.Interval[int]"), pd.Interval
    )
    check(
        assert_type(pd.Interval(1.0, 2.5, closed="right"), "pd.Interval[float]"),
        pd.Interval,
    )
    check(
        assert_type(pd.Interval(1.0, 2.5, closed="both"), "pd.Interval[float]"),
        pd.Interval,
    )
    check(
        assert_type(
            pd.Interval(
                pd.Timedelta("1 day"), pd.Timedelta("2 days"), closed="neither"
            ),
            "pd.Interval[pd.Timedelta]",
        ),
        pd.Interval,
    )


def test_int64_dtype() -> None:
    check(assert_type(pd.Int64Dtype(), pd.Int64Dtype), pd.Int64Dtype)


def test_categorical_dtype() -> None:
    check(
        assert_type(
            pd.CategoricalDtype(categories=["a", "b", "c"], ordered=True),
            pd.CategoricalDtype,
        ),
        pd.CategoricalDtype,
    )
    check(
        assert_type(pd.CategoricalDtype(categories=[1, 2, 3]), pd.CategoricalDtype),
        pd.CategoricalDtype,
    )


def test_sparse_dtype() -> None:
    check(assert_type(pd.SparseDtype(str), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(complex), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(bool), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(int), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.int64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(str), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(float), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.datetime64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.timedelta64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype("datetime64"), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(), pd.SparseDtype), pd.SparseDtype)


def test_string_dtype() -> None:
    check(assert_type(pd.StringDtype("pyarrow"), pd.StringDtype), pd.StringDtype)
    check(assert_type(pd.StringDtype("python"), pd.StringDtype), pd.StringDtype)


def test_boolean_dtype() -> None:
    check(assert_type(pd.BooleanDtype(), pd.BooleanDtype), pd.BooleanDtype)


def test_arrow_dtype() -> None:
    import pyarrow as pa

    check(assert_type(pd.ArrowDtype(pa.int64()), pd.ArrowDtype), pd.ArrowDtype)
