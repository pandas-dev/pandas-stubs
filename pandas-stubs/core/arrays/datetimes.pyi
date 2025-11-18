from collections.abc import Sequence
from datetime import datetime

import numpy as np
from pandas.core.arrays.datetimelike import (
    DatelikeOps,
    DatetimeLikeArrayMixin,
    TimelikeOps,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.tslibs.timestamps import Timestamp

class DatetimeArray(DatetimeLikeArrayMixin, TimelikeOps, DatelikeOps):
    __array_priority__: int = ...
    def __init__(
        self,
        values: (
            Sequence[datetime | np.datetime64]
            | np.typing.NDArray[np.datetime64]
            | DatetimeIndex
            | Series[Timestamp]
            | Self
        ),
        dtype: np.dtype | None = None,
        copy: bool = False,
    ) -> None: ...
    @property
    def dtype(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> np.dtypes.DateTime64DType: ...
