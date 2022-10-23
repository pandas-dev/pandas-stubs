import datetime as dt
from typing import (
    Any,
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas import Index
from pandas.core.indexes.extension import ExtensionIndex
from pandas.core.series import (
    TimedeltaSeries,
    TimestampSeries,
)
from typing_extensions import TypeAlias

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin,
)
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
    DatetimeLike,
    DtypeArg,
    FillnaOptions,
    IntervalClosedType,
    Label,
    TimedeltaConvertibleTypes,
    np_ndarray_bool,
    npt,
)

from pandas.core.dtypes.dtypes import IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCSeries

_EdgesInt: TypeAlias = Union[
    Sequence[int],
    npt.NDArray[np.int64],
    npt.NDArray[np.int32],
    npt.NDArray[np.intp],
    pd.Series[int],
    pd.Int64Index,
]
_EdgesFloat: TypeAlias = Union[
    Sequence[float] | npt.NDArray[np.float64] | pd.Series[float] | pd.Float64Index,
]
_EdgesTimestamp: TypeAlias = Union[
    Sequence[DatetimeLike]
    | npt.NDArray[np.datetime64]
    | pd.Series[pd.Timestamp]
    | TimestampSeries
    | pd.DatetimeIndex
]
_EdgesTimedelta: TypeAlias = Union[
    Sequence[pd.Timedelta]
    | npt.NDArray[np.timedelta64]
    | pd.Series[pd.Timedelta]
    | TimedeltaSeries
    | pd.TimedeltaIndex
]

_IntervalT = TypeVar(
    "_IntervalT",
    Interval[int],
    Interval[float],
    Interval[pd.Timestamp],
    Interval[pd.Timedelta],
)

class IntervalIndex(IntervalMixin, ExtensionIndex, Generic[_IntervalT]):
    def __new__(
        cls,
        data: Sequence[_IntervalT],
        closed: IntervalClosedType = ...,
        dtype: IntervalDtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        verify_integrity: bool = ...,
    ) -> IntervalIndex[_IntervalT]: ...
    # ignore[misc] here due to overlap, e.g., Sequence[int] and Sequence[float]
    @overload
    @classmethod
    def from_breaks(  # type:ignore[misc]
        cls,
        breaks: _EdgesInt,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[int]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: _EdgesFloat,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[float]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: _EdgesTimestamp,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: _EdgesTimedelta,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[pd.Timedelta]]: ...
    # ignore[misc] here due to overlap, e.g., Sequence[int] and Sequence[float]
    @overload
    @classmethod
    def from_arrays(  # type:ignore[misc]
        cls,
        left: _EdgesInt,
        right: _EdgesInt,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[int]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: _EdgesFloat,
        right: _EdgesFloat,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[float]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: _EdgesTimestamp,
        right: _EdgesTimestamp,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: _EdgesTimedelta,
        right: _EdgesTimedelta,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[Interval[pd.Timedelta]]: ...
    @classmethod
    def from_tuples(
        cls,
        data: Sequence[tuple[int, int]]
        | Sequence[tuple[float, float]]
        | Sequence[tuple[DatetimeLike, DatetimeLike]]
        | npt.NDArray,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    def __contains__(self, key: Any) -> bool: ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> IntervalIndex: ...
    @property
    def inferred_type(self) -> str: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def is_overlapping(self) -> bool: ...
    # Note: tolerance no effect. It is included in all get_loc so
    # that signatures are consistent with base even though it is usually not used
    def get_loc(
        self,
        key: Label,
        method: FillnaOptions | Literal["nearest"] | None = ...,
        tolerance=...,
    ) -> int | slice | npt.NDArray[np.bool_]: ...
    def get_indexer(
        self,
        target: Index,
        method: FillnaOptions | Literal["nearest"] | None = ...,
        limit: int | None = ...,
        tolerance=...,
    ) -> npt.NDArray[np.intp]: ...
    def get_indexer_non_unique(
        self, target: Index
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]: ...
    @property
    def left(self) -> Index: ...
    @property
    def right(self) -> Index: ...
    @property
    def mid(self) -> Index: ...
    @property
    def length(self) -> Index: ...
    def get_value(self, series: ABCSeries, key): ...
    @property
    def is_all_dates(self) -> bool: ...
    @overload  # type: ignore[override]
    def __gt__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: object) -> bool: ...
    @overload  # type: ignore[override]
    def __ge__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: object) -> bool: ...
    @overload  # type: ignore[override]
    def __le__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: object) -> bool: ...
    @overload  # type: ignore[override]
    def __lt__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: object) -> bool: ...
    @overload  # type: ignore[override]
    def __eq__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: object) -> bool: ...
    @overload  # type: ignore[override]
    def __ne__(self, other: Interval | IntervalIndex) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: object) -> bool: ...

@overload
def interval_range(
    start: int | float | None = ...,
    end: int | float | None = ...,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex: ...
@overload
def interval_range(
    start: DatetimeLike | None = ...,
    end: DatetimeLike | None = ...,
    periods: int | None = ...,
    freq: str | DateOffset | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex: ...
