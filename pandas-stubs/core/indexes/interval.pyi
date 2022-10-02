import datetime as dt
from typing import (
    Any,
    Hashable,
    Literal,
    Sequence,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas import Index
from pandas.core.indexes.extension import ExtensionIndex
from typing_extensions import TypeAlias

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
    DatetimeLike,
    DtypeArg,
    FillnaOptions,
    IntervalClosedType,
    Label,
    npt,
)

from pandas.core.dtypes.dtypes import IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCSeries

_Edges: TypeAlias = Union[
    Sequence[int],
    Sequence[float],
    Sequence[DatetimeLike],
    npt.NDArray[np.int_],
    npt.NDArray[np.float_],
    npt.NDArray[np.datetime64],
    pd.Series[int],
    pd.Series[float],
    pd.Series[pd.Timestamp],
    pd.Int64Index,
    pd.DatetimeIndex,
]

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
        breaks: _Edges,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    def from_arrays(
        cls,
        left: _Edges,
        right: _Edges,
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
