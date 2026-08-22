from collections.abc import Sequence
from typing import Any

from pandas import PeriodDtype
from pandas.core.arrays.datetimelike import DatelikeOps
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.indexes.period import PeriodIndex
from pandas.core.series import Series
import pyarrow as pa

from pandas._libs.tslibs.period import (
    Period,
    PeriodMixin,
)
from pandas._typing import (
    DtypeArg,
    NpDtype,
    PeriodFrequency,
    np_1darray_bool,
    np_1darray_int64,
    np_1darray_object,
    np_ndarray_anyint,
)

class PeriodArray(DatelikeOps, PeriodMixin):
    __array_priority__: int = ...
    def __init__(
        self,
        values: (
            np_ndarray_anyint
            | PeriodArray
            | PeriodIndex
            | Series[Period]
            | Sequence[int]
        ),
        dtype: PeriodDtype | None = None,
        copy: bool = False,
    ) -> None: ...
    @property
    def dtype(self) -> PeriodDtype: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray_object: ...
    def __arrow_array__(
        self, type: DtypeArg | None = None
    ) -> pa.ExtensionArray[Any]: ...
    @property
    def year(self) -> np_1darray_int64: ...
    @property
    def month(self) -> np_1darray_int64: ...
    @property
    def day(self) -> np_1darray_int64: ...
    @property
    def hour(self) -> np_1darray_int64: ...
    @property
    def minute(self) -> np_1darray_int64: ...
    @property
    def second(self) -> np_1darray_int64: ...
    @property
    def weekofyear(self) -> np_1darray_int64: ...
    @property
    def week(self) -> np_1darray_int64: ...
    @property
    def dayofweek(self) -> np_1darray_int64: ...
    @property
    def weekday(self) -> np_1darray_int64: ...
    @property
    def dayofyear(self) -> np_1darray_int64: ...
    @property
    def day_of_year(self) -> np_1darray_int64: ...
    @property
    def quarter(self) -> np_1darray_int64: ...
    @property
    def qyear(self) -> np_1darray_int64: ...
    @property
    def days_in_month(self) -> np_1darray_int64: ...
    daysinmonth: np_1darray_int64 = ...
    @property
    def is_leap_year(self) -> np_1darray_bool: ...
    @property
    def start_time(self) -> DatetimeArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    @property
    def end_time(self) -> DatetimeArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    def to_timestamp(
        self, freq: PeriodFrequency | None = None, how: str = ...
    ) -> DatetimeArray: ...
    def asfreq(self, freq: str | None = ..., how: str = "E") -> PeriodArray: ...
