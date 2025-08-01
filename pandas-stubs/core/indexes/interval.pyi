from collections.abc import (
    Hashable,
    Sequence,
)
import datetime as dt
from typing import (
    Literal,
    final,
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
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._typing import (
    DatetimeLike,
    DtypeArg,
    FillnaOptions,
    IntervalClosedType,
    IntervalT,
    Label,
    MaskType,
    np_ndarray_anyint,
    np_ndarray_bool,
    npt,
)

from pandas.core.dtypes.dtypes import IntervalDtype as IntervalDtype

_EdgesInt: TypeAlias = (
    Sequence[int]
    | npt.NDArray[np.int64]
    | npt.NDArray[np.int32]
    | npt.NDArray[np.intp]
    | pd.Series[int]
    | Index[int]
)
_EdgesFloat: TypeAlias = (
    Sequence[float] | npt.NDArray[np.float64] | pd.Series[float] | Index[float]
)
_EdgesTimestamp: TypeAlias = (
    Sequence[DatetimeLike]
    | npt.NDArray[np.datetime64]
    | TimestampSeries
    | pd.DatetimeIndex
)
_EdgesTimedelta: TypeAlias = (
    Sequence[pd.Timedelta]
    | npt.NDArray[np.timedelta64]
    | TimedeltaSeries
    | pd.TimedeltaIndex
)
_TimestampLike: TypeAlias = pd.Timestamp | np.datetime64 | dt.datetime
_TimedeltaLike: TypeAlias = pd.Timedelta | np.timedelta64 | dt.timedelta

class IntervalIndex(ExtensionIndex[IntervalT], IntervalMixin):
    closed: IntervalClosedType

    def __new__(
        cls,
        data: Sequence[IntervalT],
        closed: IntervalClosedType = ...,
        dtype: IntervalDtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        verify_integrity: bool = ...,
    ) -> IntervalIndex[IntervalT]: ...
    @overload
    @classmethod
    def from_breaks(  # pyright: ignore[reportOverlappingOverload]
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
    @overload
    @classmethod
    def from_arrays(  # pyright: ignore[reportOverlappingOverload]
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
    @overload
    @classmethod
    def from_tuples(  # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[tuple[int, int]],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[pd.Interval[int]]: ...
    # Ignore misc here due to intentional overlap between int and float
    @overload
    @classmethod
    def from_tuples(
        cls,
        data: Sequence[tuple[float, float]],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[pd.Interval[float]]: ...
    @overload
    @classmethod
    def from_tuples(
        cls,
        data: Sequence[
            tuple[pd.Timestamp, pd.Timestamp]
            | tuple[dt.datetime, dt.datetime]
            | tuple[np.datetime64, np.datetime64]
        ],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[pd.Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_tuples(
        cls,
        data: Sequence[
            tuple[pd.Timedelta, pd.Timedelta]
            | tuple[dt.timedelta, dt.timedelta]
            | tuple[np.timedelta64, np.timedelta64]
        ],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex[pd.Interval[pd.Timedelta]]: ...
    def to_tuples(self, na_tuple: bool = ...) -> pd.Index: ...
    @overload
    def __contains__(self, key: IntervalT) -> bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __contains__(self, key: object) -> Literal[False]: ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> IntervalIndex: ...
    @property
    def inferred_type(self) -> str: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def is_overlapping(self) -> bool: ...
    def get_loc(self, key: Label) -> int | slice | npt.NDArray[np.bool_]: ...
    @final
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
    @overload  # type: ignore[override]
    def __getitem__(
        self,
        idx: (
            slice
            | np_ndarray_anyint
            | Sequence[int]
            | Index
            | MaskType
            | np_ndarray_bool
        ),
    ) -> IntervalIndex[IntervalT]: ...
    @overload
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, idx: int
    ) -> IntervalT: ...
    @overload  # type: ignore[override]
    def __gt__(
        self, other: IntervalT | IntervalIndex[IntervalT]
    ) -> np_ndarray_bool: ...
    @overload
    def __gt__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: pd.Series[IntervalT]
    ) -> pd.Series[bool]: ...
    @overload  # type: ignore[override]
    def __ge__(
        self, other: IntervalT | IntervalIndex[IntervalT]
    ) -> np_ndarray_bool: ...
    @overload
    def __ge__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: pd.Series[IntervalT]
    ) -> pd.Series[bool]: ...
    @overload  # type: ignore[override]
    def __le__(
        self, other: IntervalT | IntervalIndex[IntervalT]
    ) -> np_ndarray_bool: ...
    @overload
    def __le__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: pd.Series[IntervalT]
    ) -> pd.Series[bool]: ...
    @overload  # type: ignore[override]
    def __lt__(
        self, other: IntervalT | IntervalIndex[IntervalT]
    ) -> np_ndarray_bool: ...
    @overload
    def __lt__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: pd.Series[IntervalT]
    ) -> pd.Series[bool]: ...
    @overload  # type: ignore[override]
    def __eq__(self, other: IntervalT | IntervalIndex[IntervalT]) -> np_ndarray_bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __eq__(self, other: pd.Series[IntervalT]) -> pd.Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: object
    ) -> Literal[False]: ...
    @overload  # type: ignore[override]
    def __ne__(self, other: IntervalT | IntervalIndex[IntervalT]) -> np_ndarray_bool: ...  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
    @overload
    def __ne__(self, other: pd.Series[IntervalT]) -> pd.Series[bool]: ...  # type: ignore[overload-overlap]
    @overload
    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: object
    ) -> Literal[True]: ...

# misc here because int and float overlap but interval has distinct types
# int gets hit first and so the correct type is returned
@overload
def interval_range(  # pyright: ignore[reportOverlappingOverload]
    start: int = ...,
    end: int = ...,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[int]]: ...

# Overlaps since int is a subclass of float
@overload
def interval_range(  # pyright: ignore[reportOverlappingOverload]
    start: int,
    *,
    end: None = ...,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[int]]: ...
@overload
def interval_range(  # pyright: ignore[reportOverlappingOverload]
    *,
    start: None = ...,
    end: int,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[int]]: ...
@overload
def interval_range(
    start: float = ...,
    end: float = ...,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[float]]: ...
@overload
def interval_range(
    start: float,
    *,
    end: None = ...,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[float]]: ...
@overload
def interval_range(
    *,
    start: None = ...,
    end: float,
    periods: int | None = ...,
    freq: int | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[float]]: ...
@overload
def interval_range(
    start: _TimestampLike,
    end: _TimestampLike = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timestamp]]: ...
@overload
def interval_range(
    *,
    start: None = ...,
    end: _TimestampLike,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timestamp]]: ...
@overload
def interval_range(
    start: _TimestampLike,
    *,
    end: None = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timestamp]]: ...
@overload
def interval_range(
    start: _TimedeltaLike,
    end: _TimedeltaLike = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timedelta]]: ...
@overload
def interval_range(
    *,
    start: None = ...,
    end: _TimedeltaLike,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timedelta]]: ...
@overload
def interval_range(
    start: _TimedeltaLike,
    *,
    end: None = ...,
    periods: int | None = ...,
    freq: str | BaseOffset | pd.Timedelta | dt.timedelta | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex[Interval[pd.Timedelta]]: ...
