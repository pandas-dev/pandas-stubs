import datetime as dt
from typing import (
    Generic,
    Literal,
    Sequence,
    overload,
)

import numpy as np
import pandas as pd
from pandas import (
    Index,
    Series,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.base import IndexOpsMixin
from typing_extensions import Self

from pandas._libs.interval import (
    Interval,
    IntervalMixin,
)
from pandas._typing import (
    Axis,
    IntervalT,
    Scalar,
    TakeIndexer,
    npt,
)

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
        cls,
        breaks: Sequence[int] | npt.NDArray[np.integer] | IndexOpsMixin[int],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[int]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[float] | npt.NDArray[np.floating] | IndexOpsMixin[float],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[float]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[np.datetime64 | dt.datetime] | IndexOpsMixin[pd.Timestamp],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_breaks(
        cls,
        breaks: Sequence[np.timedelta64 | dt.timedelta] | IndexOpsMixin[pd.Timedelta],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[pd.Timedelta]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: Sequence[int] | npt.NDArray[np.integer] | IndexOpsMixin[int],
        right: Sequence[int] | npt.NDArray[np.integer] | IndexOpsMixin[int],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[int]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: Sequence[float] | npt.NDArray[np.floating] | IndexOpsMixin[float],
        right: Sequence[float] | npt.NDArray[np.floating] | IndexOpsMixin[float],
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[float]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: Sequence[np.datetime64 | dt.datetime] | pd.DatetimeIndex,
        right: Sequence[np.datetime64 | dt.datetime] | pd.DatetimeIndex,
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[pd.Timestamp]]: ...
    @overload
    @classmethod
    def from_arrays(
        cls,
        left: Sequence[np.timedelta64 | dt.timedelta] | pd.TimedeltaIndex,
        right: Sequence[np.timedelta64 | dt.timedelta] | pd.TimedeltaIndex,
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray[Interval[pd.Timedelta]]: ...
    @classmethod
    def from_tuples(
        cls,
        data,
        closed: Literal["left", "right", "both", "neither"] = ...,
        copy: bool = ...,
        dtype: pd.IntervalDtype = ...,
    ) -> IntervalArray: ...
    def __iter__(self) -> IntervalT: ...
    def __len__(self) -> int: ...
    def __getitem__(self, value: IntervalT): ...
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
    def shift(self, periods: int = ..., fill_value: object = ...) -> IntervalArray: ...
    def take(  # type: ignore[override]
        self: Self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = ...,
        fill_value=...,
        axis=...,
        **kwargs,
    ) -> Self: ...
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
    @property
    def is_empty(self) -> npt.NDArray[np.bool_]: ...
    @overload
    def contains(self, other: Series) -> Series[bool]: ...
    @overload
    def contains(
        self, other: Scalar | ExtensionArray | Index | np.ndarray
    ) -> npt.NDArray[np.bool_]: ...
    def overlaps(self, other: Interval) -> npt.NDArray[np.bool_]: ...
