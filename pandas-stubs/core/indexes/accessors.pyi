import datetime as dt
from datetime import tzinfo
from typing import (
    Generic,
    Literal,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from pandas import (
    Period,
    Timedelta,
)
from pandas.core.accessor import PandasDelegate
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
)
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)
from pandas.core.frame import DataFrame
from pandas.core.indexes.numeric import IntegerIndex
from pandas.core.series import (
    Series,
    TimestampSeries,
)
from pytz.tzinfo import BaseTzInfo

from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import np_ndarray_bool

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Series, orig) -> None: ...

_DTReturnType = TypeVar("_DTReturnType", Series[int], IntegerIndex)

class DatetimeFieldOps(Properties, Generic[_DTReturnType]):
    @property
    def year(self) -> _DTReturnType: ...
    @property
    def month(self) -> _DTReturnType: ...
    @property
    def day(self) -> _DTReturnType: ...
    @property
    def hour(self) -> _DTReturnType: ...
    @property
    def minute(self) -> _DTReturnType: ...
    @property
    def second(self) -> _DTReturnType: ...
    @property
    def weekofyear(self) -> _DTReturnType: ...
    @property
    def week(self) -> _DTReturnType: ...
    @property
    def weekday(self) -> _DTReturnType: ...
    @property
    def dayofweek(self) -> _DTReturnType: ...
    @property
    def day_of_week(self) -> _DTReturnType: ...
    @property
    def dayofyear(self) -> _DTReturnType: ...
    @property
    def day_of_year(self) -> _DTReturnType: ...
    @property
    def quarter(self) -> _DTReturnType: ...
    @property
    def days_in_month(self) -> _DTReturnType: ...
    @property
    def daysinmonth(self) -> _DTReturnType: ...
    @property
    def microsecond(self) -> _DTReturnType: ...
    @property
    def nanosecond(self) -> _DTReturnType: ...

class DatetimeAndPeriodProperties(DatetimeFieldOps[Series[int]]):
    @property
    def is_leap_year(self) -> Series[bool]: ...
    @property
    def freq(self) -> str | None: ...

class DatetimeProperties(DatetimeAndPeriodProperties):
    def to_pydatetime(self) -> np.ndarray: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def is_month_start(self) -> Series[bool]: ...
    @property
    def is_month_end(self) -> Series[bool]: ...
    @property
    def is_quarter_start(self) -> Series[bool]: ...
    @property
    def is_quarter_end(self) -> Series[bool]: ...
    @property
    def is_year_start(self) -> Series[bool]: ...
    @property
    def is_year_end(self) -> Series[bool]: ...
    @property
    def tz(self) -> tzinfo | BaseTzInfo | None: ...
    @property
    def date(self) -> Series[dt.date]: ...
    @property
    def time(self) -> Series[dt.time]: ...
    @property
    def timetz(self) -> Series[dt.time]: ...
    def to_period(self, freq: str | BaseOffset | None = ...) -> Series[Period]: ...
    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> DatetimeArray: ...
    def tz_convert(self, tz: str | None) -> TimestampSeries: ...
    def normalize(self) -> TimestampSeries: ...
    def strftime(self, date_format: str) -> Series[str]: ...
    # Ideally, the next 3 methods would return TimestampSeries, but because of
    # how Series.dt is hooked in, we don't know which kind of series was passed
    # in to the dt accessor
    def round(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...
    def floor(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...
    def ceil(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...
    def month_name(self, locale: str | None = ...) -> Series[str]: ...
    def day_name(self, locale: str | None = ...) -> Series[str]: ...

class TimedeltaProperties(Properties):
    def to_pytimedelta(self) -> np.ndarray: ...
    @property
    def components(self) -> DataFrame: ...
    @property
    def days(self) -> Series[int]: ...
    @property
    def seconds(self) -> Series[int]: ...
    @property
    def microseconds(self) -> Series[int]: ...
    @property
    def nanoseconds(self) -> Series[int]: ...
    def total_seconds(self) -> Series[float]: ...
    # Ideally, the next 3 methods would return TimedeltaSeries, but because of
    # how Series.dt is hooked in, we don't know which kind of series was passed
    # in to the dt accessor
    def round(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...
    def floor(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...
    def ceil(
        self,
        freq: str | BaseOffset | None,
        ambiguous: Literal["raise", "infer", "NaT"] | np_ndarray_bool = ...,
        nonexistent: Literal["shift_forward", "shift_backward", "NaT", "raise"]
        | Timedelta = ...,
    ) -> Series: ...

class PeriodProperties(DatetimeAndPeriodProperties):
    @property
    def start_time(self) -> TimestampSeries: ...
    @property
    def end_time(self) -> TimestampSeries: ...
    @property
    def qyear(self) -> Series[int]: ...
    def strftime(self, date_format: str) -> Series[str]: ...
    def to_timestamp(
        self,
        freq: str | DateOffset | None = ...,
        how: Literal["s", "e", "start", "end"] = ...,
    ) -> DatetimeArray: ...
    def asfreq(
        self,
        freq: str | DateOffset | None = ...,
        how: Literal["E", "END", "FINISH", "S", "START", "BEGIN"] = ...,
    ) -> PeriodArray: ...

class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    def __new__(cls, data: Series): ...
