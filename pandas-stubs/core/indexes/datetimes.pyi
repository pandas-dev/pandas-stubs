from __future__ import annotations

from datetime import tzinfo
from typing import (
    Optional,
    Union,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Timedelta,
    Timestamp,
)
from pandas.core.indexes.api import (
    Float64Index,
    PeriodIndex,
)
from pandas.core.indexes.datetimelike import (
    DatetimelikeDelegateMixin,
    DatetimeTimedeltaMixin,
)
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import (
    TimedeltaSeries,
    TimestampSeries,
)

from pandas._typing import np_ndarray_bool

class DatetimeDelegateMixin(DatetimelikeDelegateMixin): ...

class DatetimeIndex(DatetimeTimedeltaMixin, DatetimeDelegateMixin):
    tz: tzinfo | None
    def __init__(
        self,
        data=...,
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
    ): ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __reduce__(self): ...
    @overload
    def __add__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __add__(self, other: Timedelta | TimedeltaIndex) -> DatetimeIndex: ...
    def union_many(self, others): ...
    # overload needed because Index.to_series() and DatetimeIndex.to_series() have
    # different arguments
    def to_series(self, keep_tz=..., index=..., name=...) -> TimestampSeries: ...  # type: ignore[override]
    def snap(self, freq: str = ...): ...
    def get_value(self, series, key): ...
    def get_value_maybe_box(self, series, key): ...
    def get_loc(self, key, method=..., tolerance=...): ...
    def slice_indexer(self, start=..., end=..., step=..., kind=...): ...
    def searchsorted(self, value, side: str = ..., sorter=...): ...
    def is_type_compatible(self, typ) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def insert(self, loc, item): ...
    def indexer_at_time(self, time, asof: bool = ...): ...
    def indexer_between_time(
        self, start_time, end_time, include_start: bool = ..., include_end: bool = ...
    ): ...
    def strftime(self, date_format: str = ...) -> np.ndarray: ...
    def tz_convert(self, tz) -> DatetimeIndex: ...
    def tz_localize(self, tz, ambiguous=..., nonexistent=...) -> DatetimeIndex: ...
    def to_period(self, freq) -> PeriodIndex: ...
    def to_perioddelta(self, freq) -> TimedeltaIndex: ...
    def to_julian_date(self) -> Float64Index: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def tzinfo(self) -> tzinfo | None: ...
    def __lt__(self, other: Timestamp) -> np_ndarray_bool: ...
    def __le__(self, other: Timestamp) -> np_ndarray_bool: ...
    def __gt__(self, other: Timestamp) -> np_ndarray_bool: ...
    def __ge__(self, other: Timestamp) -> np_ndarray_bool: ...

def date_range(
    start=...,
    end=...,
    periods=...,
    freq=...,
    tz=...,
    normalize=...,
    name=...,
    closed=...,
    **kwargs,
) -> DatetimeIndex: ...
def bdate_range(
    start=...,
    end=...,
    periods=...,
    freq: str = ...,
    tz=...,
    normalize: bool = ...,
    name=...,
    weekmask=...,
    holidays=...,
    closed=...,
) -> DatetimeIndex: ...
