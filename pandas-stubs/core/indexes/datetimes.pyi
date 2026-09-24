from collections.abc import (
    Hashable,
    Sequence,
)
from datetime import (
    time,
    timedelta,
)
from typing import (
    Literal,
    Never,
    Self,
    final,
    overload,
)

import numpy as np
import pandas as pd
from pandas._stubs_only import (
    ScalarArrayIndexDatetime,
    ScalarArrayIndexTimedelta,
)
from pandas.core.frame import DataFrame
from pandas.core.indexes.accessors import DatetimeIndexProperties
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import override

from pandas._libs.tslibs.offsets import BaseOffset
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
)

from pandas.core.dtypes.dtypes import DatetimeTZDtype

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
    @overload  # type: ignore[override]
    @override
    # pyrefly: ignore[bad-override]
    def __add__(self, other: np_ndarray_dt, /) -> Never: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexTimedelta | BaseOffset, /
    ) -> Self: ...
    @overload  # type: ignore[override]
    @override
    # pyrefly: ignore[bad-override]
    def __radd__(self, other: np_ndarray_dt, /) -> Never: ...
    @overload
    def __radd__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexTimedelta | BaseOffset, /
    ) -> Self: ...
    @overload  # type: ignore[override]
    @override
    def __sub__(  # pyrefly: ignore[bad-override]
        self, other: ScalarArrayIndexDatetime, /
    ) -> TimedeltaIndex: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexTimedelta | BaseOffset, /
    ) -> Self: ...
    @override
    def __rsub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexDatetime, /
    ) -> TimedeltaIndex: ...
    @override
    def __truediv__(self, other: np_ndarray, /) -> Never: ...  # type: ignore[override]
    @override
    def __rtruediv__(self, other: np_ndarray, /) -> Never: ...  # type: ignore[override]
    @final
    @override
    def to_series(
        self, index: Index | None = None, name: Hashable | None = None
    ) -> Series[Timestamp]: ...
    def snap(self, freq: Frequency = "S") -> Self: ...
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
    @override
    def dtype(self) -> np.dtype | DatetimeTZDtype: ...
    def shift(
        self, periods: int = 1, freq: Frequency | timedelta | None = None
    ) -> Self: ...
    @override
    def diff(self, periods: int = 1) -> TimedeltaIndex: ...

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
