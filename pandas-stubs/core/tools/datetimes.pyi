from collections.abc import Sequence
from datetime import (
    date,
    datetime,
)
from typing import (
    Any,
    Literal,
    TypeAlias,
    TypedDict,
    overload,
)

import numpy as np
from pandas import (
    Index,
    Timestamp,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series

from pandas._libs.tslibs import NaTType
from pandas._typing import (
    AnyArrayLike,
    DictConvertible,
    RaiseCoerce,
    TimestampConvertibleTypes,
    np_ndarray_dt,
    np_ndarray_int64,
    np_ndarray_str,
)

ArrayConvertible: TypeAlias = list[Any] | tuple[Any, ...] | AnyArrayLike
Scalar: TypeAlias = float | str
DatetimeScalar: TypeAlias = Scalar | datetime | np.datetime64 | date

DatetimeScalarOrArrayConvertible: TypeAlias = DatetimeScalar | ArrayConvertible

DatetimeDictArg: TypeAlias = (
    list[int | str] | tuple[int | str, ...] | AnyArrayLike | int | str
)

class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg

class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg

@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: Literal["raise"] = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: Literal["D", "s", "ms", "us", "ns"] | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = "unix",
    cache: bool = True,
) -> Timestamp: ...
@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: Literal["coerce"],
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: Literal["D", "s", "ms", "us", "ns"] | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = "unix",
    cache: bool = True,
) -> Timestamp | NaTType: ...
@overload
def to_datetime(
    arg: Series | DictConvertible,
    errors: RaiseCoerce = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: Literal["D", "s", "ms", "us", "ns"] | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = "unix",
    cache: bool = True,
) -> Series[Timestamp]: ...
@overload
def to_datetime(
    arg: (
        Sequence[float | date]
        | list[str]
        | tuple[float | str | date, ...]
        | np_ndarray_dt
        | np_ndarray_str
        | np_ndarray_int64
        | Index
        | ExtensionArray
    ),
    errors: RaiseCoerce = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: Literal["D", "s", "ms", "us", "ns"] | None = None,
    origin: Literal["julian", "unix"] | TimestampConvertibleTypes = "unix",
    cache: bool = True,
) -> DatetimeIndex: ...
