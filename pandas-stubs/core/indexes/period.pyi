from collections.abc import Hashable
import datetime
from typing import (
    overload,
)

import numpy as np
import pandas as pd
from pandas import Index
from pandas.core.indexes.accessors import PeriodIndexFieldOps
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.timedeltas import TimedeltaIndex
from typing_extensions import Self

from pandas._libs.tslibs import (
    NaTType,
    Period,
)
from pandas._libs.tslibs.period import _PeriodAddSub
from pandas._typing import (
    AxesData,
    Dtype,
    Frequency,
)

class PeriodIndex(DatetimeIndexOpsMixin[pd.Period, np.object_], PeriodIndexFieldOps):
    def __new__(
        cls,
        data: AxesData | None = None,
        freq: Frequency | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ): ...
    @property
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.object_]]: ...
    @overload
    def __sub__(self, other: Period) -> Index: ...
    @overload
    def __sub__(self, other: Self) -> Index: ...
    @overload
    def __sub__(self, other: _PeriodAddSub) -> Self: ...
    @overload
    def __sub__(self, other: NaTType) -> NaTType: ...
    @overload
    def __sub__(self, other: TimedeltaIndex | pd.Timedelta) -> Self: ...
    @overload  # type: ignore[override]
    def __rsub__(self, other: Period) -> Index: ...
    @overload
    def __rsub__(self, other: Self) -> Index: ...
    @overload
    def __rsub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: NaTType
    ) -> NaTType: ...
    def asof_locs(self, where, mask): ...
    @property
    def is_full(self) -> bool: ...
    @property
    def inferred_type(self) -> str: ...
    @property
    def freqstr(self) -> str: ...
    def shift(self, periods: int = 1, freq=...) -> Self: ...

def period_range(
    start: (
        str | datetime.datetime | datetime.date | pd.Timestamp | pd.Period | None
    ) = None,
    end: (
        str | datetime.datetime | datetime.date | pd.Timestamp | pd.Period | None
    ) = None,
    periods: int | None = None,
    freq: Frequency | None = None,
    name: Hashable | None = None,
) -> PeriodIndex: ...
