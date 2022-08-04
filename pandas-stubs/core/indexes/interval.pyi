from typing import Hashable

import numpy as np
from pandas.core.indexes.extension import ExtensionIndex

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._typing import (
    DtypeArg,
    IntervalClosedType,
)

from pandas.core.dtypes.dtypes import IntervalDtype as IntervalDtype
from pandas.core.dtypes.generic import ABCSeries as ABCSeries

class SetopCheck:
    op_name = ...
    def __init__(self, op_name) -> None: ...
    def __call__(self, setop): ...

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
        breaks,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    @classmethod
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType = ...,
        name: Hashable = ...,
        copy: bool = ...,
        dtype: IntervalDtype | None = ...,
    ) -> IntervalIndex: ...
    def astype(self, dtype: DtypeArg, copy: bool = ...) -> IntervalIndex: ...
    @property
    def inferred_type(self) -> str: ...
    def memory_usage(self, deep: bool = ...) -> int: ...
    @property
    def is_overlapping(self) -> bool: ...
    def get_loc(
        self, key, method: str | None = ..., tolerance=...
    ) -> int | slice | np.ndarray: ...
    def get_indexer(
        self,
        targetArrayLike,
        method: str | None = ...,
        limit: int | None = ...,
        tolerance=...,
    ) -> np.ndarray: ...
    def get_indexer_non_unique(
        self, targetArrayLike
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def get_value(self, series: ABCSeries, key): ...
    @property
    def is_all_dates(self) -> bool: ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...

def interval_range(
    start=..., end=..., periods=..., freq=..., name=..., closed: str = ...
): ...
