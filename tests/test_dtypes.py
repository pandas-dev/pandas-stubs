from __future__ import annotations

import datetime as dt
from datetime import (
    timedelta,
    timezone,
)
from typing import (
    Literal,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from pandas.core.arrays import BooleanArray  # noqa: F401
from pandas.core.arrays import IntegerArray  # noqa: F401
import pyarrow as pa
from typing_extensions import assert_type

from pandas._libs import NaTType
from pandas._libs.missing import NAType
from pandas._typing import Scalar

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BusinessDay,
    CustomBusinessDay,
    Day,
)


def test_datetimetz_dtype() -> None:
    dttz_dt = pd.DatetimeTZDtype(unit="ns", tz="UTC")
    check(assert_type(dttz_dt, pd.DatetimeTZDtype), pd.DatetimeTZDtype)
    check(
        assert_type(
            pd.DatetimeTZDtype(unit="ns", tz=timezone(timedelta(hours=1))),
            pd.DatetimeTZDtype,
        ),
        pd.DatetimeTZDtype,
    )
    check(assert_type(dttz_dt.unit, Literal["ns"]), str)
    check(assert_type(dttz_dt.tz, dt.tzinfo), dt.tzinfo)
    check(assert_type(dttz_dt.name, str), str)
    check(assert_type(dttz_dt.na_value, NaTType), NaTType)


def test_period_dtype() -> None:
    p_dt = pd.PeriodDtype(freq="D")
    check(assert_type(p_dt, pd.PeriodDtype), pd.PeriodDtype)
    check(assert_type(pd.PeriodDtype(freq=Day()), pd.PeriodDtype), pd.PeriodDtype)
    check(
        assert_type(pd.PeriodDtype(freq=BusinessDay()), pd.PeriodDtype), pd.PeriodDtype
    )
    if TYPE_CHECKING_INVALID_USAGE:
        pd.PeriodDtype(freq=CustomBusinessDay())  # TODO(raises on 2.1)
    check(
        assert_type(p_dt.freq, pd.tseries.offsets.BaseOffset),
        pd.tseries.offsets.DateOffset,
    )
    check(assert_type(p_dt.na_value, NaTType), NaTType)
    check(assert_type(p_dt.name, str), str)


def test_interval_dtype() -> None:
    i_dt = pd.IntervalDtype("int64")
    check(assert_type(i_dt, pd.IntervalDtype), pd.IntervalDtype)
    check(assert_type(pd.IntervalDtype(np.int64), pd.IntervalDtype), pd.IntervalDtype)
    check(assert_type(pd.IntervalDtype(float), pd.IntervalDtype), pd.IntervalDtype)
    check(assert_type(pd.IntervalDtype(complex), pd.IntervalDtype), pd.IntervalDtype)
    check(
        assert_type(pd.IntervalDtype(np.timedelta64), pd.IntervalDtype),
        pd.IntervalDtype,
    )
    check(
        assert_type(pd.IntervalDtype(np.datetime64), pd.IntervalDtype), pd.IntervalDtype
    )


def test_int64_dtype() -> None:
    check(assert_type(pd.Int8Dtype(), pd.Int8Dtype), pd.Int8Dtype)
    check(assert_type(pd.Int16Dtype(), pd.Int16Dtype), pd.Int16Dtype)
    check(assert_type(pd.Int32Dtype(), pd.Int32Dtype), pd.Int32Dtype)
    check(assert_type(pd.Int64Dtype(), pd.Int64Dtype), pd.Int64Dtype)
    check(assert_type(pd.UInt8Dtype(), pd.UInt8Dtype), pd.UInt8Dtype)
    check(assert_type(pd.UInt16Dtype(), pd.UInt16Dtype), pd.UInt16Dtype)
    check(assert_type(pd.UInt32Dtype(), pd.UInt32Dtype), pd.UInt32Dtype)
    check(assert_type(pd.UInt64Dtype(), pd.UInt64Dtype), pd.UInt64Dtype)

    i64dt = pd.Int64Dtype()
    check(assert_type(i64dt.itemsize, int), int)
    check(assert_type(i64dt.na_value, NAType), NAType)
    check(assert_type(i64dt.construct_array_type(), "type[IntegerArray]"), type)


def test_categorical_dtype() -> None:
    cdt = pd.CategoricalDtype(categories=["a", "b", "c"], ordered=True)
    check(assert_type(cdt, pd.CategoricalDtype), pd.CategoricalDtype)
    check(
        assert_type(pd.CategoricalDtype(categories=[1, 2, 3]), pd.CategoricalDtype),
        pd.CategoricalDtype,
    )
    check(assert_type(cdt.categories, pd.Index), pd.Index)
    assert check(assert_type(cdt.ordered, Optional[bool]), bool)


def test_sparse_dtype() -> None:
    s_dt = pd.SparseDtype("i4")
    check(assert_type(s_dt, pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(str), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(complex), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(bool), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.int64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(str), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(float), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.datetime64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(np.timedelta64), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype("datetime64"), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(pd.SparseDtype(), pd.SparseDtype), pd.SparseDtype)
    check(assert_type(s_dt.fill_value, Union[Scalar, None]), int)


def test_string_dtype() -> None:
    s_dt = pd.StringDtype("pyarrow")
    check(assert_type(pd.StringDtype(), pd.StringDtype), pd.StringDtype)
    check(assert_type(pd.StringDtype("pyarrow"), pd.StringDtype), pd.StringDtype)
    check(assert_type(pd.StringDtype("python"), pd.StringDtype), pd.StringDtype)
    check(assert_type(s_dt.na_value, NAType), NAType)


def test_boolean_dtype() -> None:
    b_dt = pd.BooleanDtype()
    check(assert_type(b_dt, pd.BooleanDtype), pd.BooleanDtype)
    check(assert_type(b_dt.na_value, NAType), NAType)
    check(assert_type(b_dt.construct_array_type(), "type[BooleanArray]"), type)


def test_arrow_dtype() -> None:
    a_dt = pd.ArrowDtype(pa.int64())
    check(assert_type(a_dt, pd.ArrowDtype), pd.ArrowDtype)
    check(assert_type(a_dt.pyarrow_dtype, pa.DataType), pa.DataType)
