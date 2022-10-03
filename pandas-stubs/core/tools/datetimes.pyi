from datetime import datetime
from typing import (
    Literal,
    Sequence,
    TypedDict,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas import (
    Index,
    Timestamp,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import (
    Series,
    TimestampSeries,
)
from typing_extensions import TypeAlias

from pandas._libs.tslibs import NaTType
from pandas._typing import (
    AnyArrayLike,
    DateTimeErrorChoices,
    IgnoreRaise,
    npt,
)

ArrayConvertible: TypeAlias = Union[list, tuple, AnyArrayLike]
Scalar: TypeAlias = Union[float, str]
DatetimeScalar: TypeAlias = Union[Scalar, datetime, np.datetime64]

DatetimeScalarOrArrayConvertible: TypeAlias = Union[DatetimeScalar, ArrayConvertible]

DatetimeDictArg: TypeAlias = Union[list[Scalar], tuple[Scalar, ...], AnyArrayLike]

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

DictConvertible: TypeAlias = Union[FulldatetimeDict, DataFrame]

@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: IgnoreRaise = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin: int | Literal["julian", "unix"] | pd.Timestamp = ...,
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
    origin: Literal["julian", "unix"] | pd.Timestamp = ...,
    cache: bool = ...,
) -> Timestamp | NaTType: ...
@overload
def to_datetime(
    # TODO: Test dataframe return type
    arg: Series | DictConvertible,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin: int | Literal["julian", "unix"] | pd.Timestamp = ...,
    cache: bool = ...,
) -> TimestampSeries: ...
@overload
def to_datetime(
    # TODO: Test other types
    arg: Sequence[int | float | datetime]
    | list[str]
    | tuple[int | float | str | datetime, ...]
    | npt.NDArray[np.datetime64]
    | npt.NDArray[np.str_]
    | npt.NDArray[np.int_]
    | npt.NDArray[np.float_]
    | Index
    | ExtensionArray,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool | None = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    # TODO: Origin needs int in pandas docs
    origin: int | Literal["julian", "unix"] | pd.Timestamp = ...,
    cache: bool = ...,
) -> DatetimeIndex: ...
