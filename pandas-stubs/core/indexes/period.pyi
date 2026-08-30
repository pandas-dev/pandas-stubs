from collections.abc import Hashable
import datetime
from typing import (
    Self,
    overload,
)

import numpy as np
import pandas as pd
from pandas import Index
from pandas._stubs_only import (
    PeriodAddSub,
    ScalarArrayIndexPeriod,
)
from pandas.core.indexes.accessors import PeriodIndexFieldOps
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.timedeltas import TimedeltaIndex
from typing_extensions import override

from pandas._libs.tslibs import (
    NaTType,
    Period,
)
from pandas._typing import (
    AxesData,
    Dtype,
    Frequency,
    np_1darray_intp,
    np_1darray_object,
    np_ndarray_bool,
)

class PeriodIndex(DatetimeIndexOpsMixin[Period, np.object_], PeriodIndexFieldOps):
    def __new__(
        cls,
        data: AxesData | None = None,
        freq: Frequency | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self: ...
    @property
    @override
    def values(self) -> np_1darray_object: ...
    @override
    def __add__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexPeriod, /
    ) -> Self: ...
    @override
    def __radd__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override] # ty: ignore[invalid-method-override]
        self, other: ScalarArrayIndexPeriod, /
    ) -> Self: ...
    @overload  # type: ignore[override]
    @override
    def __sub__(self, other: Period, /) -> Index: ...  # pyrefly: ignore[bad-override]
    @overload
    def __sub__(self, other: Self, /) -> Index: ...
    @overload
    def __sub__(self, other: PeriodAddSub, /) -> Self: ...
    @overload
    def __sub__(self, other: NaTType, /) -> NaTType: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: TimedeltaIndex | pd.Timedelta, /
    ) -> Self: ...
    @overload  # type: ignore[override]
    @override
    def __rsub__(self, other: Period, /) -> Index: ...  # pyrefly: ignore[bad-override]
    @overload
    def __rsub__(self, other: Self, /) -> Index: ...
    @overload
    def __rsub__(  # pyright: ignore[reportIncompatibleMethodOverride] # ty: ignore[invalid-method-override]
        self, other: NaTType, /
    ) -> NaTType: ...
    @override
    def asof_locs(
        self, where: pd.DatetimeIndex | Self, mask: np_ndarray_bool
    ) -> np_1darray_intp: ...
    @property
    def is_full(self) -> bool: ...
    @property
    @override
    def freqstr(self) -> str: ...
    def shift(self, periods: int = 1, freq: Frequency | None = None) -> Self: ...
    @override
    def diff(self, periods: int = 1) -> Index: ...

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
