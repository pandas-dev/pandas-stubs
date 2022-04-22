from datetime import datetime as datetime
from numpy import datetime64 as datetime64
import numpy as np

from pandas._typing import (
    ArrayLike as ArrayLike,
    Index as Index,
    AnyArrayLike as AnyArrayLike,
    DateTimeErrorChoices as DateTimeErrorChoices,
    ExtensionArray as ExtensionArray,
    Timestamp as Timestamp,
)
from pandas.core.dtypes.generic import ABCSeries as ABCSeries
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.indexes.datetimes import DatetimeIndex as DatetimeIndex
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.series import Series as Series
from typing import List, Optional, Tuple, TypedDict, Union, overload

ArrayConvertible = Union[List, Tuple, AnyArrayLike]
Scalar = Union[int, float, str]
DatetimeScalar = Union[Scalar, datetime]

DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]

DatetimeDictArg = Union[List[Scalar], Tuple[Scalar, ...], AnyArrayLike]

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

DictConvertible = Union[FulldatetimeDict, "DataFrame"]

def should_cache(arg: ArrayConvertible, unique_share: float = ..., check_count: Optional[int] = ...) -> bool: ...
@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Timestamp: ...
@overload
def to_datetime(
    arg: Series | DictConvertible,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Series[Timestamp]: ...
@overload
def to_datetime(
    arg: list | tuple | np.ndarray | Index | ExtensionArray,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex: ...
def to_time(arg, format=..., infer_time_format: bool = ..., errors: str = ...): ...
