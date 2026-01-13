from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import numpy as np
from pandas.core.arrays.datetimelike import (
    DatetimeLikeArrayMixin,
    TimelikeOps,
)
from typing_extensions import Self

from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
    Frequency,
)

class TimedeltaArray(DatetimeLikeArrayMixin, TimelikeOps):
    __array_priority__: int = ...
    @property
    def dtype(self) -> np.dtypes.TimeDelta64DType: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # pyrefly: ignore[bad-override]
    def __init__(
        self,
        values: AnyArrayLike,
        dtype: DtypeArg | None = None,
        freq: Frequency | None = None,
        copy: bool = ...,
    ) -> None: ...
    # TODO: pandas-dev/pandas-stubs#1589 add testing to figure out the correct types
    # def sum(
    #     self,
    #     *,
    #     axis=...,
    #     dtype=...,
    #     out=...,
    #     keepdims: bool = ...,
    #     initial=...,
    #     skipna: bool = ...,
    #     min_count: int = ...,
    # ): ...
    # def std(
    #     self,
    #     *,
    #     axis=...,
    #     dtype=...,
    #     out=...,
    #     ddof: int = ...,
    #     keepdims: bool = ...,
    #     skipna: bool = ...,
    # ): ...
    # def median(
    #     self,
    #     *,
    #     axis=...,
    #     out=...,
    #     overwrite_input: bool = ...,
    #     keepdims: bool = ...,
    #     skipna: bool = ...,
    # ): ...
    def __mul__(self, other: Any) -> Self: ...
    __rmul__ = __mul__
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __floordiv__(self, other: Any) -> Any: ...
    def __rfloordiv__(self, other: Any) -> Any: ...
    def __mod__(self, other: Any) -> Any: ...
    def __rmod__(self, other: Any) -> Any: ...
    def __divmod__(self, other: Any) -> Any: ...
    def __rdivmod__(self, other: Any) -> Any: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def total_seconds(self) -> int: ...
    def to_pytimedelta(self) -> Sequence[timedelta]: ...
    days: int = ...
    seconds: int = ...
    microseconds: int = ...
    nanoseconds: int = ...
    @property
    def components(self) -> int: ...
