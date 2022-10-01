import datetime as dt
from typing import (
    Any,
    Hashable,
    Sequence,
)

import numpy as np
from pandas import Index
from pandas.core.indexes.extension import ExtensionIndex

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._libs.tslibs import BaseOffset
from pandas._typing import (
    DatetimeLike,
    DtypeArg,
    IntervalClosedType,
    Label,
    npt,
)

from pandas.core.dtypes.dtypes import IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCSeries

class IntervalIndex(IntervalMixin, ExtensionIndex):
    def __new__(
        cls,
        data,
        closed: IntervalClosedType = ...,
        dtype: IntervalDtype | None = ...,
        copy: bool = ...,
        name: Hashable = ...,
        verify_integrity: bool = ...,
    ): ...
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[int]
        | Sequence[float]
        | Sequence[DatetimeLike]
        | npt.NDArray[np.int_]
        | npt.NDArray[np.float_],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    def from_arrays(
        cls,
        left: Sequence[int]
        | Sequence[float]
        | Sequence[DatetimeLike]
        | npt.NDArray[np.int_]
        | npt.NDArray[np.float_],
        right: Sequence[int]
        | Sequence[float]
        | Sequence[DatetimeLike]
        | npt.NDArray[np.int_]
        | npt.NDArray[np.float_],
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
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
    def is_monotonic_decreasing(self) -> bool: ...
    def is_unique(self) -> bool: ...
    @property
    def is_overlapping(self) -> bool: ...
    # Note: tolerance removed as it has no effect
    def get_loc(
        self,
        key: Label,
        method: str | None = ...,
    ) -> int | slice | np.ndarray: ...
    def get_indexer(
        self,
        target: Index,
        method: str | None = ...,
        limit: int | None = ...,
        tolerance=...,
    ) -> np.ndarray: ...
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

def interval_range(
    start: int | float | DatetimeLike | None = ...,
    end: int | float | DatetimeLike | None = ...,
    periods: int | None = ...,
    freq: int | str | BaseOffset | None = ...,
    name: Hashable = ...,
    closed: IntervalClosedType = ...,
) -> IntervalIndex: ...
