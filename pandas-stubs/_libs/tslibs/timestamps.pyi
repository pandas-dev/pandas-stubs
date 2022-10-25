from datetime import (
    date as _date,
    datetime,
    time as _time,
    timedelta,
    tzinfo as _tzinfo,
)
from time import struct_time
from typing import (
    ClassVar,
    Literal,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from pandas import (
    DatetimeIndex,
    Index,
    TimedeltaIndex,
)
from pandas.core.series import (
    Series,
    TimedeltaSeries,
    TimestampSeries,
)
from typing_extensions import TypeAlias

from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    Tick,
    Timedelta,
)
from pandas._typing import (
    np_ndarray_bool,
    npt,
)

_DatetimeT = TypeVar("_DatetimeT", bound=datetime)
_Ambiguous: TypeAlias = Union[bool, Literal["raise", "NaT"]]
_Nonexistent: TypeAlias = Union[
    Literal["raise", "NaT", "shift_backward", "shift_forward"], Timedelta, timedelta
]

class Timestamp(datetime):
    min: ClassVar[Timestamp]
    max: ClassVar[Timestamp]

    resolution: ClassVar[Timedelta]
    value: int
    def __new__(
        cls: type[_DatetimeT],
        ts_input: np.integer | float | str | _date | datetime | np.datetime64 = ...,
        # Freq is deprecated but is left in to allow code like Timestamp(2000,1,1)
        # Removing it would make the other arguments position only
        freq: int | str | BaseOffset | None = ...,
        tz: str | _tzinfo | int | None = ...,
        unit: str | int | None = ...,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        nanosecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        *,
        fold: Literal[0, 1] | None = ...,
    ) -> _DatetimeT: ...
    # GH 46171
    # While Timestamp can return pd.NaT, having the constructor return
    # a Union with NaTType makes things awkward for users of pandas
    @property
    def year(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def day(self) -> int: ...
    @property
    def hour(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def microsecond(self) -> int: ...
    @property
    def nanosecond(self) -> int: ...
    @property
    def tzinfo(self) -> _tzinfo | None: ...
    @property
    def tz(self) -> _tzinfo | None: ...
    @property
    def fold(self) -> int: ...
    @classmethod
    def fromtimestamp(
        cls: type[_DatetimeT], t: float, tz: _tzinfo | str | None = ...
    ) -> _DatetimeT: ...
    @classmethod
    def utcfromtimestamp(cls: type[_DatetimeT], ts: float) -> _DatetimeT: ...
    @classmethod
    def today(cls: type[_DatetimeT], tz: _tzinfo | str | None = ...) -> _DatetimeT: ...
    @classmethod
    def fromordinal(
        cls: type[_DatetimeT],
        ordinal: int,
        *,
        tz: _tzinfo | str | None = ...,
    ) -> _DatetimeT: ...
    @classmethod
    def now(cls: type[_DatetimeT], tz: _tzinfo | str | None = ...) -> _DatetimeT: ...
    @classmethod
    def utcnow(cls: type[_DatetimeT]) -> _DatetimeT: ...
    # error: Signature of "combine" incompatible with supertype "datetime"
    @classmethod
    def combine(cls, date: _date, time: _time) -> datetime: ...  # type: ignore[override]
    @classmethod
    def fromisoformat(cls: type[_DatetimeT], date_string: str) -> _DatetimeT: ...
    def strftime(self, format: str) -> str: ...
    def __format__(self, fmt: str) -> str: ...
    def toordinal(self) -> int: ...
    def timetuple(self) -> struct_time: ...
    def timestamp(self) -> float: ...
    def utctimetuple(self) -> struct_time: ...
    def date(self) -> _date: ...
    def time(self) -> _time: ...
    def timetz(self) -> _time: ...
    # Override since fold is more precise than datetime.replace(fold:int)
    # Violation of Liskov substitution principle
    def replace(  # type:ignore[override]
        self,
        year: int | None = ...,
        month: int | None = ...,
        day: int | None = ...,
        hour: int | None = ...,
        minute: int | None = ...,
        second: int | None = ...,
        microsecond: int | None = ...,
        tzinfo: _tzinfo | None = ...,
        fold: Literal[0, 1] | None = ...,
    ) -> Timestamp: ...
    def astimezone(self: _DatetimeT, tz: _tzinfo | None = ...) -> _DatetimeT: ...
    def ctime(self) -> str: ...
    def isoformat(self, sep: str = ..., timespec: str = ...) -> str: ...
    @classmethod
    def strptime(cls, date_string: str, format: str) -> datetime: ...
    def utcoffset(self) -> timedelta | None: ...
    def tzname(self) -> str | None: ...
    def dst(self) -> timedelta | None: ...
    # Mypy complains Forward operator "<inequality op>" is not callable, so ignore misc
    # for le, lt ge and gt
    @overload  # type: ignore[override]
    def __le__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: Index | npt.NDArray[np.datetime64]) -> np_ndarray_bool: ...
    @overload
    def __le__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __lt__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: Index | npt.NDArray[np.datetime64]) -> np_ndarray_bool: ...
    @overload
    def __lt__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __ge__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: Index | npt.NDArray[np.datetime64]) -> np_ndarray_bool: ...
    @overload
    def __ge__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...
    @overload  # type: ignore[override]
    def __gt__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: Index | npt.NDArray[np.datetime64]) -> np_ndarray_bool: ...
    @overload
    def __gt__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...
    # error: Signature of "__add__" incompatible with supertype "date"/"datetime"
    @overload  # type: ignore[override]
    def __add__(
        self, other: npt.NDArray[np.timedelta64]
    ) -> npt.NDArray[np.datetime64]: ...
    @overload
    def __add__(
        self: _DatetimeT, other: timedelta | np.timedelta64 | Tick
    ) -> _DatetimeT: ...
    @overload
    def __add__(
        self, other: TimedeltaSeries | Series[Timedelta]
    ) -> TimestampSeries: ...
    @overload
    def __add__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(self: _DatetimeT, other: timedelta) -> _DatetimeT: ...
    @overload
    def __radd__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(
        self, other: npt.NDArray[np.timedelta64]
    ) -> npt.NDArray[np.datetime64]: ...
    # TODO: test dt64
    @overload  # type: ignore[override]
    def __sub__(self, other: Timestamp | datetime | np.datetime64) -> Timedelta: ...
    @overload
    def __sub__(
        self: _DatetimeT, other: timedelta | np.timedelta64 | Tick
    ) -> _DatetimeT: ...
    @overload
    def __sub__(self, other: TimedeltaIndex) -> DatetimeIndex: ...
    @overload
    def __sub__(self, other: TimedeltaSeries) -> TimestampSeries: ...
    @overload
    def __sub__(
        self, other: npt.NDArray[np.timedelta64]
    ) -> npt.NDArray[np.datetime64]: ...
    @overload
    def __eq__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: npt.NDArray[np.datetime64] | Index) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: object) -> Literal[False]: ...
    @overload
    def __ne__(self, other: Timestamp | datetime | np.datetime64) -> bool: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: TimestampSeries | Series[Timestamp]) -> Series[bool]: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: npt.NDArray[np.datetime64] | Index) -> np_ndarray_bool: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: object) -> Literal[True]: ...
    def __hash__(self) -> int: ...
    def weekday(self) -> int: ...
    def isoweekday(self) -> int: ...
    def isocalendar(self) -> tuple[int, int, int]: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def is_month_start(self) -> bool: ...
    @property
    def is_quarter_start(self) -> bool: ...
    @property
    def is_year_start(self) -> bool: ...
    @property
    def is_month_end(self) -> bool: ...
    @property
    def is_quarter_end(self) -> bool: ...
    @property
    def is_year_end(self) -> bool: ...
    def to_pydatetime(self, warn: bool = ...) -> datetime: ...
    def to_datetime64(self) -> np.datetime64: ...
    def to_period(self, freq: BaseOffset | str | None = ...) -> Period: ...
    def to_julian_date(self) -> np.float64: ...
    @property
    def asm8(self) -> np.datetime64: ...
    def tz_convert(self: _DatetimeT, tz: _tzinfo | str | None) -> _DatetimeT: ...
    # TODO: could return NaT?
    def tz_localize(
        self: _DatetimeT,
        tz: _tzinfo | str | None,
        ambiguous: _Ambiguous = ...,
        nonexistent: _Nonexistent = ...,
    ) -> _DatetimeT: ...
    def normalize(self: _DatetimeT) -> _DatetimeT: ...
    # TODO: round/floor/ceil could return NaT?
    def round(
        self: _DatetimeT,
        freq: str,
        ambiguous: _Ambiguous = ...,
        nonexistent: _Nonexistent = ...,
    ) -> _DatetimeT: ...
    def floor(
        self: _DatetimeT,
        freq: str,
        ambiguous: _Ambiguous = ...,
        nonexistent: _Nonexistent = ...,
    ) -> _DatetimeT: ...
    def ceil(
        self: _DatetimeT,
        freq: str,
        ambiguous: _Ambiguous = ...,
        nonexistent: _Nonexistent = ...,
    ) -> _DatetimeT: ...
    def day_name(self, locale: str | None = ...) -> str: ...
    def month_name(self, locale: str | None = ...) -> str: ...
    @property
    def day_of_week(self) -> int: ...
    @property
    def dayofweek(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def dayofyear(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def week(self) -> int: ...
    def to_numpy(self) -> np.datetime64: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def daysinmonth(self) -> int: ...
