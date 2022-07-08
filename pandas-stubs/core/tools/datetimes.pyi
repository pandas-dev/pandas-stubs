from datetime import datetime
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    overload,
)

import numpy as np
from numpy import datetime64
from pandas import (
    Index,
    Timestamp,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import (
    Series,
    TimestampSeries,
)

from pandas._libs.tslibs import NaTType
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DateTimeErrorChoices,
)

from pandas.core.dtypes.generic import ABCSeries

ArrayConvertible = Union[List, Tuple, AnyArrayLike]
Scalar = Union[int, float, str]
DatetimeScalar = Union[Scalar, datetime, np.datetime64]

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

DictConvertible = Union[FulldatetimeDict, DataFrame]

def should_cache(
    arg: ArrayConvertible, unique_share: float = ..., check_count: Optional[int] = ...
) -> bool: ...
@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: Literal["ignore", "raise"] = ...,
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
    arg: DatetimeScalar,
    errors: Literal["coerce"],
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Timestamp | NaTType: ...
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
) -> TimestampSeries: ...
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
