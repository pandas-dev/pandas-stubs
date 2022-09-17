from typing import overload

from pandas.core.indexes.accessors import TimedeltaIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import TimedeltaSeries

from pandas._libs import (
    Timedelta,
    Timestamp,
)
from pandas._typing import num

class TimedeltaIndex(DatetimeTimedeltaMixin, TimedeltaIndexProperties):
    def __new__(
        cls,
        data=...,
        unit=...,
        freq=...,
        closed=...,
        dtype=...,
        copy: bool = ...,
        name=...,
    ): ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload  # type: ignore[override]
    def __add__(self, other: DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __add__(self, other: Timedelta | TimedeltaIndex) -> TimedeltaIndex: ...
    def __radd__(self, other: Timestamp | DatetimeIndex) -> DatetimeIndex: ...  # type: ignore[override]
    def __sub__(self, other: Timedelta | TimedeltaIndex) -> TimedeltaIndex: ...  # type: ignore[override]
    def __mul__(self, other: num) -> TimedeltaIndex: ...  # type: ignore[override]
    def __truediv__(self, other: num) -> TimedeltaIndex: ...  # type: ignore[override]
    def astype(self, dtype, copy: bool = ...): ...
    def get_value(self, series, key): ...
    def get_loc(self, key, tolerance=...): ...
    def searchsorted(self, value, side: str = ..., sorter=...): ...
    def is_type_compatible(self, typ) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    def insert(self, loc, item): ...
    def to_series(self, index=..., name=...) -> TimedeltaSeries: ...

def timedelta_range(
    start=..., end=..., periods=..., freq=..., name=..., closed=...
) -> TimedeltaIndex: ...
