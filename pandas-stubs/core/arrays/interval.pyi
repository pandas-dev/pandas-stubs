import datetime as dt
from typing import (
    Generic,
    Literal,
    Sequence,
    overload,
)

import numpy as np
import pandas as pd
from pandas import Index
from pandas.core.arrays.base import ExtensionArray as ExtensionArray

from pandas._libs.interval import (
    Interval as Interval,
    IntervalMixin as IntervalMixin,
)
from pandas._typing import (
    Axis,
    IntervalT,
    npt,
)

from pandas.core.dtypes.generic import ABCExtensionArray

class IntervalArray(IntervalMixin, ExtensionArray, Generic[IntervalT]):
    ndim: int = ...
    can_hold_na: bool = ...
    def __new__(
        cls,
        data: Sequence[IntervalT] | IntervalArray[IntervalT],
        closed: Literal["left", "right", "both", "neither"] = ...,
        dtype: pd.IntervalDtype = ...,
        copy: bool = ...,
        verify_integrity: bool = ...,
    ) -> IntervalArray[IntervalT]: ...
    @overload
    @classmethod
    def from_breaks(
        cls, breaks: Sequence[int], closed: str = ..., copy: bool = ..., dtype=...
    ) -> IntervalArray[Interval[int]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls, breaks: Sequence[float], closed: str = ..., copy: bool = ..., dtype=...
    ) -> IntervalArray[Interval[float]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[pd.Timestamp | np.datetime64 | dt.datetime],
        closed: str = ...,
        copy: bool = ...,
        dtype=...,
    ) -> IntervalArray[Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[pd.Timedelta | np.timedelta64 | dt.timedelta],
        closed: str = ...,
        copy: bool = ...,
        dtype=...,
    ) -> IntervalArray[Interval[pd.Timedelta]]: ...
    @classmethod
    def from_arrays(
        cls, left, right, closed: str = ..., copy: bool = ..., dtype=...
    ) -> IntervalArray: ...
    @classmethod
    def from_tuples(
        cls, data, closed: str = ..., copy: bool = ..., dtype=...
    ) -> IntervalArray: ...
    def __iter__(self) -> IntervalT: ...
    def __len__(self) -> int: ...
    def __getitem__(self, value: IntervalT): ...
    def __setitem__(self, key: int, value: IntervalT) -> None: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def fillna(self, value=..., method=..., limit=...): ...
    @property
    def dtype(self): ...
    def astype(self, dtype, copy: bool = ...): ...
    def copy(self): ...
    def isna(self): ...
    @property
    def nbytes(self) -> int: ...
    @property
    def size(self) -> int: ...
    def shift(
        self, periods: int = ..., fill_value: object = ...
    ) -> ABCExtensionArray: ...
    def take(
        self, indices, *, allow_fill: bool = ..., fill_value=..., axis=..., **kwargs
    ): ...
    def value_counts(self, dropna: bool = ...): ...
    @property
    def left(self) -> Index: ...
    @property
    def right(self) -> Index: ...
    @property
    def closed(self) -> str: ...
    def set_closed(self, closed) -> IntervalArray[IntervalT]: ...
    @property
    def length(self) -> Index: ...
    @property
    def mid(self) -> Index: ...
    @property
    def is_non_overlapping_monotonic(self) -> bool: ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __arrow_array__(self, type=...): ...
    def to_tuples(self, na_tuple: bool = ...) -> npt.NDArray[np.object_]: ...
    def repeat(self, repeats, axis: Axis | None = ...): ...
    def contains(
        self, other: float | pd.Timestamp | pd.Timedelta
    ) -> npt.NDArray[np.bool_]: ...
    def overlaps(self, other: Interval) -> npt.NDArray[np.bool_]: ...
    @property
    def is_empty(self) -> npt.NDArray[np.bool_]: ...
