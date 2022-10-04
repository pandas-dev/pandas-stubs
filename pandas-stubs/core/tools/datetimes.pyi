from datetime import (
    date,
    datetime,
)
from typing import (
    Literal,
    Sequence,
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
    DictConvertible,
    IgnoreRaise,
    npt,
)

ArrayConvertible: TypeAlias = Union[list, tuple, AnyArrayLike]
Scalar: TypeAlias = Union[float, str]
DatetimeScalar: TypeAlias = Union[Scalar, datetime, np.datetime64, date]

DatetimeScalarOrArrayConvertible: TypeAlias = Union[DatetimeScalar, ArrayConvertible]

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
    arg: Sequence[int | float | datetime]
    | list[str]
    | tuple[int | float | str | datetime, ...]
    | npt.NDArray[np.datetime64]
    | npt.NDArray[np.str_]
    | npt.NDArray[np.int_]
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
