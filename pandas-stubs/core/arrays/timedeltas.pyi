from collections.abc import Sequence
from datetime import timedelta

import numpy as np
from pandas.core.arrays.datetimelike import (
    DatetimeLikeArrayMixin,
    TimelikeOps,
)
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import np_1darray

from pandas.core.dtypes.base import ExtensionDtype

class TimedeltaArray(DatetimeLikeArrayMixin, TimelikeOps):
    __array_priority__: int = ...
    @property
    def dtype(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> np.dtypes.TimeDelta64DType: ...
    def __init__(self, values, dtype=..., freq=..., copy: bool = ...) -> None: ...
    @classmethod
    def _from_sequence(
        cls,
        data: (
            Sequence[timedelta | np.timedelta64]
            | np_1darray[np.timedelta64]
            | TimedeltaIndex
            | Series[Timedelta]
            | Self
        ),
        *,
        dtype: np.dtype | ExtensionDtype | None = None,
        copy: bool = True,
    ) -> TimedeltaArray: ...
