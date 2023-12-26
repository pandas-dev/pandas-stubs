from collections.abc import (
    Hashable,
    Sequence,
)
from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
from typing import (
    Literal,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
)
from pandas.core.indexes.accessors import DatetimeIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.series import (
    TimedeltaSeries,
    TimestampSeries,
)

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DateAndDatetimeLike,
    IntervalClosedType,
)

from pandas.core.dtypes.dtypes import DatetimeTZDtype

from pandas.tseries.offsets import BaseOffset

class DatetimeIndex(DatetimeTimedeltaMixin[Timestamp], DatetimeIndexProperties):
    def __init__(
        self,
        data: ArrayLike | AnyArrayLike | list | tuple,
        freq=...,
        tz=...,
        normalize: bool = ...,
        closed=...,
        ambiguous: str = ...,
        dayfirst: bool = ...,
        yearfirst: bool = ...,
        dtype=...,
        copy: bool = ...,
        name=...,
    ) -> None: ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __reduce__(self): ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload
    def __add__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __add__(
        self, other: timedelta | Timedelta | TimedeltaIndex
    ) -> DatetimeIndex: ...
    @overload
    def __sub__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __sub__(
        self, other: timedelta | Timedelta | TimedeltaIndex
    ) -> DatetimeIndex: ...
    @overload
    def __sub__(
        self, other: datetime | Timestamp | DatetimeIndex
    ) -> TimedeltaIndex: ...
    def to_series(self, index=..., name=...) -> TimestampSeries: ...
    def snap(self, freq: str = ...): ...
    def get_value(self, series, key): ...
    def slice_indexer(self, start=..., end=..., step=...): ...
    def searchsorted(self, value, side: str = ..., sorter=...): ...
    @property
    def inferred_type(self) -> str: ...
    def insert(self, loc, item): ...
    def indexer_at_time(self, time, asof: bool = ...): ...
    def indexer_between_time(
        self, start_time, end_time, include_start: bool = ..., include_end: bool = ...
    ): ...
    def to_perioddelta(self, freq) -> TimedeltaIndex: ...
    def to_julian_date(self) -> Index[float]: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def tzinfo(self) -> tzinfo | None: ...
    @property
    def dtype(self) -> np.dtype | DatetimeTZDtype: ...

def date_range(
    start: str | DateAndDatetimeLike | None = ...,
    end: str | DateAndDatetimeLike | None = ...,
    periods: int | None = ...,
    freq: str | timedelta | Timedelta | BaseOffset = ...,
    tz: str | tzinfo = ...,
    normalize: bool = ...,
    name: Hashable | None = ...,
    inclusive: IntervalClosedType = ...,
    unit: Literal["s", "ms", "us", "ns"] | None = ...,
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = ...,
    end: str | DateAndDatetimeLike | None = ...,
    periods: int | None = ...,
    freq: str | timedelta | Timedelta | BaseOffset = ...,
    tz: str | tzinfo = ...,
    normalize: bool = ...,
    name: Hashable | None = ...,
    weekmask: str | None = ...,
    holidays: None = ...,
    inclusive: IntervalClosedType = ...,
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = ...,
    end: str | DateAndDatetimeLike | None = ...,
    periods: int | None = ...,
    *,
    freq: str | timedelta | Timedelta | BaseOffset,
    tz: str | tzinfo = ...,
    normalize: bool = ...,
    name: Hashable | None = ...,
    weekmask: str | None = ...,
    holidays: Sequence[str | DateAndDatetimeLike],
    inclusive: IntervalClosedType = ...,
) -> DatetimeIndex: ...
