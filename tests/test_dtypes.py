from __future__ import annotations

from datetime import (
    timedelta,
    timezone,
)
from typing import (
    TYPE_CHECKING,
    Literal,
)
from zoneinfo import (
    ZoneInfo,
    available_timezones,
)

import numpy as np
import pandas as pd
from pandas.api.types import is_any_real_numeric_dtype
from pandas.api.typing import (
    NaTType,
    NAType,
)
from pandas.core.arrays import (
    BooleanArray,
    IntegerArray,
)
import pyarrow as pa
import pytest
from typing_extensions import assert_type

from pandas._typing import Scalar

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import TimeUnit

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
    assert assert_type(dttz_dt.unit, TimeUnit) == "ns"
    check(assert_type(dttz_dt.tz, timezone), timezone)
    assert assert_type(dttz_dt.name, str) == "datetime64[ns, UTC]"
    check(assert_type(dttz_dt.na_value, NaTType), NaTType)

    assert assert_type(pd.DatetimeTZDtype("s", "PRC").unit, TimeUnit) == "s"
    assert assert_type(pd.DatetimeTZDtype("ms", "CST6CDT").unit, TimeUnit) == "ms"
    assert assert_type(pd.DatetimeTZDtype("us", "HST").unit, TimeUnit) == "us"

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = pd.DatetimeTZDtype()  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]
        _10 = pd.DatetimeTZDtype("us")  # type: ignore[call-overload] # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("key", available_timezones())
def test_datetimetz_dtype_tz(key: str) -> None:
    tz = ZoneInfo(key)
    assert assert_type(pd.DatetimeTZDtype(tz=tz).name, str) == f"datetime64[ns, {key}]"


def test_period_dtype() -> None:
    p_dt = pd.PeriodDtype(freq="D")
    check(assert_type(p_dt, pd.PeriodDtype), pd.PeriodDtype)
    check(assert_type(pd.PeriodDtype(freq=Day()), pd.PeriodDtype), pd.PeriodDtype)
    if TYPE_CHECKING_INVALID_USAGE:
        pd.PeriodDtype(
            freq=CustomBusinessDay()  # type:ignore[arg-type] # pyright: ignore[reportArgumentType]
        )
        pd.PeriodDtype(
            freq=BusinessDay()  # type:ignore[arg-type] # pyright: ignore[reportArgumentType]
        )
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
    check(assert_type(i64dt.construct_array_type(), type[IntegerArray]), type)


def test_categorical_dtype() -> None:
    cdt = pd.CategoricalDtype(categories=["a", "b", "c"], ordered=True)
    check(assert_type(cdt, pd.CategoricalDtype), pd.CategoricalDtype)
    check(
        assert_type(pd.CategoricalDtype(categories=[1, 2, 3]), pd.CategoricalDtype),
        pd.CategoricalDtype,
    )
    check(assert_type(cdt.categories, pd.Index), pd.Index)
    check(assert_type(cdt.ordered, bool | None), bool)


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
    check(assert_type(s_dt.fill_value, Scalar | None), int)


@pytest.mark.parametrize("storage", ["python", "pyarrow", None])
@pytest.mark.parametrize("na_value", [pd.NA, float("nan")])
def test_string_dtype(
    storage: Literal["python", "pyarrow"] | None, na_value: NAType | float
) -> None:
    s_dts = [pd.StringDtype(storage, na_value)]
    if storage is None:
        s_dts.append(pd.StringDtype(na_value=na_value))
        if na_value is pd.NA:
            s_dts.append(pd.StringDtype())
    if na_value is pd.NA:
        s_dts.append(pd.StringDtype(storage))
    for s_dt in s_dts:
        check(s_dt, pd.StringDtype)
        assert s_dt.storage in ({storage} if storage else {"python", "pyarrow"})
        check(assert_type(s_dt.na_value, NAType | float), type(na_value))

    if TYPE_CHECKING:
        assert_type(pd.StringDtype(), pd.StringDtype)
        assert_type(pd.StringDtype(None), pd.StringDtype)
        assert_type(pd.StringDtype("pyarrow"), pd.StringDtype[Literal["pyarrow"]])
        assert_type(pd.StringDtype("python"), pd.StringDtype[Literal["python"]])

        assert_type(pd.StringDtype().storage, Literal["python", "pyarrow"])
        assert_type(pd.StringDtype(None).storage, Literal["python", "pyarrow"])
        assert_type(pd.StringDtype("python").storage, Literal["python"])
        assert_type(pd.StringDtype("pyarrow").storage, Literal["pyarrow"])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = pd.StringDtype("invalid_storage")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        _1 = pd.StringDtype(na_value="invalid_na")  # type: ignore[call-overload] # pyright: ignore[reportArgumentType]


def test_boolean_dtype() -> None:
    b_dt = pd.BooleanDtype()
    check(assert_type(b_dt, pd.BooleanDtype), pd.BooleanDtype)
    check(assert_type(b_dt.na_value, NAType), NAType)
    check(assert_type(b_dt.construct_array_type(), type[BooleanArray]), type)


def test_arrow_dtype() -> None:
    a_dt = pd.ArrowDtype(pa.int64())
    check(assert_type(a_dt, pd.ArrowDtype), pd.ArrowDtype)
    check(assert_type(a_dt.pyarrow_dtype, pa.DataType), pa.DataType)


def test_is_any_real_numeric_dtype() -> None:
    check(assert_type(is_any_real_numeric_dtype(np.array([1, 2])), bool), bool)
    check(assert_type(is_any_real_numeric_dtype(int), bool), bool)
    check(assert_type(is_any_real_numeric_dtype(float), bool), bool)
