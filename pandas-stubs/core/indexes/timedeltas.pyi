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
from pandas import (
    Index,
    Period,
)
from pandas.core.indexes.accessors import TimedeltaIndexProperties
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs import Timedelta
from pandas._typing import (
    AxesData,
    Frequency,
    TimedeltaConvertibleTypes,
    np_ndarray_td,
    num,
)

class TimedeltaIndex(
    DatetimeTimedeltaMixin[Timedelta, np.timedelta64], TimedeltaIndexProperties
):
    def __new__(
        cls,
        data: (
            Sequence[dt.timedelta | Timedelta | np.timedelta64 | float] | AxesData
        ) = ...,
        freq: Frequency = ...,
        closed: object = ...,
        dtype: Literal["<m8[ns]"] = ...,
        copy: bool = ...,
        name: str = ...,
    ) -> Self: ...
    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    @overload  # type: ignore[override]
    def __add__(self, other: Period) -> PeriodIndex: ...
    @overload
    def __add__(self, other: dt.datetime | DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Self
    ) -> Self: ...
    @overload  # type: ignore[override]
    def __radd__(self, other: Period) -> PeriodIndex: ...
    @overload
    def __radd__(self, other: dt.datetime | DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Self
    ) -> Self: ...
    def __sub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | np.timedelta64 | np_ndarray_td | Self
    ) -> Self: ...
    def __mul__(self, other: float) -> Self: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @overload  # type: ignore[override]
    def __truediv__(self, other: float | Sequence[float]) -> Self: ...
    @overload
    def __truediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Sequence[dt.timedelta]
    ) -> Index[float]: ...
    def __rtruediv__(self, other: dt.timedelta | Sequence[dt.timedelta]) -> Index[float]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @overload  # type: ignore[override]
    def __floordiv__(self, other: num | Sequence[float]) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Sequence[dt.timedelta]
    ) -> Index[int]: ...
    def __rfloordiv__(self, other: dt.timedelta | Sequence[dt.timedelta]) -> Index[int]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def searchsorted(self, value, side: str = ..., sorter=...): ...
    @property
    def inferred_type(self) -> str: ...
    @final
    def to_series(self, index=..., name: Hashable = ...) -> Series[Timedelta]: ...
    def shift(
        self, periods: int = 1, freq: Frequency | dt.timedelta | None = None
    ) -> Self: ...

@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    end: TimedeltaConvertibleTypes,
    *,
    freq: Frequency | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = ...,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    *,
    end: TimedeltaConvertibleTypes,
    periods: int,
    freq: Frequency | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = ...,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    *,
    periods: int,
    freq: Frequency | Timedelta | dt.timedelta | None = None,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = ...,
) -> TimedeltaIndex: ...
@overload
def timedelta_range(
    start: TimedeltaConvertibleTypes,
    end: TimedeltaConvertibleTypes,
    periods: int,
    *,
    name: Hashable | None = None,
    closed: Literal["left", "right"] | None = None,
    unit: None | str = ...,
) -> TimedeltaIndex: ...
