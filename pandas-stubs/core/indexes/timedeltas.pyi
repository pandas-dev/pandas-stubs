from collections.abc import (
    Hashable,
    Sequence,
)
import datetime as dt
from typing import (
    Literal,
    TypeAlias,
    final,
    overload,
)

import numpy as np
from pandas.core.indexes.accessors import TimedeltaIndexProperties
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.series import Series
from typing_extensions import (
    Never,
    Self,
)

from pandas._libs import Timedelta
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.period import Period
from pandas._typing import (
    AxesData,
    Frequency,
    Just,
    TimedeltaConvertibleTypes,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_td,
    num,
)

_NUM_FACTOR: TypeAlias = Just[int] | Just[float] | np.integer | np.floating
_NUM_FACTOR_SEQ: TypeAlias = (
    _NUM_FACTOR
    | Sequence[_NUM_FACTOR]
    | np_ndarray_anyint
    | np_ndarray_float
    | Index[int]
    | Index[float]
)
_DT_FACTOR: TypeAlias = dt.timedelta | np.timedelta64 | Timedelta
_DT_FACTOR_SEQ: TypeAlias = _DT_FACTOR | Sequence[_DT_FACTOR] | np_ndarray_td

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
    # pyrefly: ignore  # bad-override
    def __add__(self, other: Period) -> PeriodIndex: ...
    @overload
    def __add__(self, other: dt.datetime | DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Self
    ) -> Self: ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __radd__(self, other: Period) -> PeriodIndex: ...
    @overload
    def __radd__(self, other: dt.datetime | DatetimeIndex) -> DatetimeIndex: ...
    @overload
    def __radd__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Self
    ) -> Self: ...
    def __sub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | np.timedelta64 | np_ndarray_td | BaseOffset | Self
    ) -> Self: ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __rsub__(
        self, other: dt.timedelta | np.timedelta64 | np_ndarray_td | BaseOffset | Self
    ) -> Self: ...
    @overload
    def __rsub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.datetime | np.datetime64 | np_ndarray_dt | DatetimeIndex
    ) -> DatetimeIndex: ...
    @overload  # type: ignore[override]
    def __mul__(self, other: np_ndarray_bool | np_ndarray_complex) -> Never: ...
    @overload
    def __mul__(self, other: _NUM_FACTOR_SEQ) -> Self: ...
    @overload  # type: ignore[override]
    def __rmul__(self, other: np_ndarray_bool | np_ndarray_complex) -> Never: ...
    @overload
    def __rmul__(self, other: _NUM_FACTOR_SEQ) -> Self: ...
    @overload  # type: ignore[override]
    def __truediv__(  # pyrefly: ignore[bad-override]
        self, other: np_ndarray_bool | np_ndarray_complex | np_ndarray_dt
    ) -> Never: ...
    @overload
    def __truediv__(self, other: _NUM_FACTOR_SEQ) -> Self: ...
    @overload
    def __truediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: _DT_FACTOR_SEQ | Self
    ) -> Index[float]: ...
    @overload  # type: ignore[override]
    def __rtruediv__(  # pyrefly: ignore[bad-override]
        self, other: np_ndarray_bool | np_ndarray_complex | np_ndarray_dt
    ) -> Never: ...
    @overload
    def __rtruediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: _DT_FACTOR_SEQ | Self
    ) -> Index[float]: ...
    @overload  # type: ignore[override]
    # pyrefly: ignore  # bad-override
    def __floordiv__(self, other: num | Sequence[float]) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: dt.timedelta | Sequence[dt.timedelta]
    ) -> Index[int]: ...
    def __rfloordiv__(self, other: dt.timedelta | Sequence[dt.timedelta]) -> Index[int]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    @property
    def inferred_type(self) -> str: ...
    @final
    def to_series(
        self, index: Index | None = None, name: Hashable | None = None
    ) -> Series[Timedelta]: ...
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
