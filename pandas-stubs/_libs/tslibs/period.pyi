import datetime
from typing import (
    Literal,
    Union,
    overload,
)

import numpy as np
from pandas import (
    DatetimeIndex,
    Index,
    PeriodIndex,
    Timedelta,
)
from typing_extensions import TypeAlias

from pandas._typing import npt

from .timestamps import Timestamp

class IncompatibleFrequency(ValueError): ...

from pandas._libs.tslibs.offsets import BaseOffset

_PeriodAddSub: TypeAlias = Union[
    Timedelta, datetime.timedelta, np.timedelta64, np.int64, int, BaseOffset
]

_PeriodFreqHow: TypeAlias = Literal[
    "S",
    "E",
    "Start",
    "Finish",
    "Begin",
    "End",
    "s",
    "e",
    "start",
    "finish",
    "begin",
    "end",
]

class PeriodMixin:
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def start_time(self) -> Timestamp: ...

class Period(PeriodMixin):
    def __init__(
        self,
        value: Period | str | None = ...,
        freq: str | BaseOffset | None = ...,
        ordinal: int | None = ...,
        year: int | None = ...,
        month: int | None = ...,
        quarter: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
    ) -> None: ...
    @overload
    def __sub__(self, other: _PeriodAddSub) -> Period: ...
    @overload
    def __sub__(self, other: Period) -> BaseOffset: ...
    @overload
    def __sub__(self, other: PeriodIndex) -> Index: ...
    @overload
    def __add__(self, other: _PeriodAddSub) -> Period: ...
    @overload
    def __add__(self, other: Index) -> PeriodIndex: ...

    # @overload
    # def __add__(self, other: Index) -> Period: ...
    @overload  # type: ignore[override]
    def __eq__(self, other: Period) -> bool: ...
    @overload
    def __eq__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    @overload
    def __ge__(self, other: Period) -> bool: ...
    @overload
    def __ge__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    @overload
    def __gt__(self, other: Period) -> bool: ...
    @overload
    def __gt__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    def __hash__(self) -> int: ...
    @overload
    def __le__(self, other: Period) -> bool: ...
    @overload
    def __le__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    @overload
    def __lt__(self, other: Period) -> bool: ...
    @overload
    def __lt__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    @overload  # type: ignore[override]
    def __ne__(self, other: Period) -> bool: ...
    @overload
    def __ne__(self, other: PeriodIndex) -> npt.NDArray[np.bool_]: ...
    # Ignored due to indecipherable error from mypy:
    # Forward operator "__add__" is not callable  [misc]
    @overload
    def __radd__(self, other: _PeriodAddSub) -> Period: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: Index) -> PeriodIndex: ...
    @property
    def day(self) -> int: ...
    @property
    def dayofweek(self) -> int: ...
    @property
    def dayofyear(self) -> int: ...
    @property
    def daysinmonth(self) -> int: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def freq(self) -> BaseOffset: ...
    @property
    def freqstr(self) -> str: ...
    @property
    def hour(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def qyear(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def start_time(self) -> Timestamp: ...
    @property
    def week(self) -> int: ...
    @property
    def weekday(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def year(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def day_of_week(self) -> int: ...
    def asfreq(self, freq: str | BaseOffset, how: _PeriodFreqHow = ...) -> Period: ...
    @classmethod
    def now(cls, freq: str | BaseOffset = ...) -> Period: ...
    def strftime(self, fmt: str) -> str: ...
    def to_timestamp(
        self,
        freq: str | BaseOffset | None = ...,
        how: _PeriodFreqHow = ...,
    ) -> Timestamp: ...
