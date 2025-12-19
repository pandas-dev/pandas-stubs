from collections.abc import (
    Hashable,
    Sequence,
)
from datetime import (
    datetime,
    time,
    timedelta,
    tzinfo as _tzinfo,
)
import sys
from typing import (
    Any,
    Literal,
    final,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.indexes.accessors import DatetimeIndexProperties
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    AxesData,
    DateAndDatetimeLike,
    Frequency,
    IntervalClosedType,
    TimeUnit,
    TimeZones,
    np_1darray_intp,
    np_ndarray,
    np_ndarray_bool,
    np_ndarray_dt,
    np_ndarray_td,
)

from pandas.core.dtypes.dtypes import DatetimeTZDtype

from pandas.tseries.offsets import BaseOffset

class DatetimeIndex(
    DatetimeTimedeltaMixin[Timestamp, np.datetime64], DatetimeIndexProperties
):
    def __new__(
        cls,
        data: AxesData,
        freq: Frequency = ...,
        tz: TimeZones = ...,
        ambiguous: Literal["infer", "NaT", "raise"] | np_ndarray_bool = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: np.dtype[np.datetime64] | pd.DatetimeTZDtype | str | None = None,
        copy: bool = False,
        name: Hashable = None,
    ) -> Self: ...

    # various ignores needed for mypy, as we do want to restrict what can be used in
    # arithmetic for these types
    def __add__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: timedelta | BaseOffset
    ) -> Self: ...
    def __radd__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: timedelta | BaseOffset
    ) -> Self: ...
    @overload  # type: ignore[override]
    def __sub__(  # pyrefly: ignore[bad-override]
        self, other: datetime | np.datetime64 | np_ndarray_dt | Self
    ) -> TimedeltaIndex: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: timedelta | np.timedelta64 | np_ndarray_td | BaseOffset
    ) -> Self: ...
    def __truediv__(  # type: ignore[override] # pyrefly: ignore[bad-override]
        self, other: np_ndarray
    ) -> Never: ...
    def __rtruediv__(  # type: ignore[override] # pyrefly: ignore[bad-override]
        self, other: np_ndarray
    ) -> Never: ...
    @final
    def to_series(
        self, index: Index | None = None, name: Hashable | None = None
    ) -> Series[Timestamp]: ...
    def snap(self, freq: Frequency = "S") -> Self: ...
    @property
    def inferred_type(self) -> str: ...
    def indexer_at_time(
        self, time: str | time, asof: bool = False
    ) -> np_1darray_intp: ...
    def indexer_between_time(
        self,
        start_time: time | str,
        end_time: time | str,
        include_start: bool = True,
        include_end: bool = True,
    ) -> np_1darray_intp: ...
    def to_julian_date(self) -> Index[float]: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    if sys.version_info >= (3, 11):
        @property
        def dtype(self) -> np.dtype | DatetimeTZDtype: ...
    else:
        @property
        def dtype(self) -> np.dtype[Any] | DatetimeTZDtype: ...

    def shift(
        self, periods: int = 1, freq: Frequency | timedelta | None = None
    ) -> Self: ...

@overload
def date_range(
    start: str | DateAndDatetimeLike,
    end: str | DateAndDatetimeLike,
    freq: Frequency | timedelta | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    start: str | DateAndDatetimeLike,
    end: str | DateAndDatetimeLike,
    periods: int,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    start: str | DateAndDatetimeLike,
    *,
    periods: int,
    freq: Frequency | timedelta | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def date_range(
    *,
    end: str | DateAndDatetimeLike,
    periods: int,
    freq: Frequency | timedelta | None = None,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    inclusive: IntervalClosedType = "both",
    unit: TimeUnit | None = None,
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = None,
    end: str | DateAndDatetimeLike | None = None,
    periods: int | None = None,
    freq: Frequency | timedelta = "B",
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    weekmask: str | None = None,
    holidays: None = None,
    inclusive: IntervalClosedType = "both",
) -> DatetimeIndex: ...
@overload
def bdate_range(
    start: str | DateAndDatetimeLike | None = None,
    end: str | DateAndDatetimeLike | None = None,
    periods: int | None = None,
    *,
    freq: Frequency | timedelta,
    tz: TimeZones = None,
    normalize: bool = False,
    name: Hashable = None,
    weekmask: str | None = None,
    holidays: Sequence[str | DateAndDatetimeLike],
    inclusive: IntervalClosedType = "both",
) -> DatetimeIndex: ...
