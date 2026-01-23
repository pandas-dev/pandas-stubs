from typing import (
    Any,
    Self,
    overload,
)

import numpy as np
from pandas.core.arrays.datetimelike import TimelikeOps
from pandas.core.frame import DataFrame

from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import DateOffset
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    AnyArrayLike,
    DtypeArg,
    Frequency,
    NpDtype,
    np_1darray_float,
    np_1darray_int32,
    np_1darray_int64,
    np_1darray_object,
    np_1darray_td,
)

class TimedeltaArray(TimelikeOps):
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
    @overload
    def __truediv__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Timedelta
    ) -> np_1darray_float: ...
    @overload
    def __truediv__(self, other: Any) -> Self: ...
    @overload
    def __rtruediv__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Timedelta
    ) -> np_1darray_float: ...
    @overload
    def __rtruediv__(self, other: Any) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Timedelta
    ) -> np_1darray_int64: ...
    @overload
    def __floordiv__(self, other: Any) -> Self: ...
    @overload
    def __rfloordiv__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Timedelta
    ) -> np_1darray_int64: ...
    @overload
    def __rfloordiv__(self, other: Any) -> Self: ...
    def __mod__(self, other: Any) -> Self: ...
    def __rmod__(self, other: Any) -> Self: ...
    def __divmod__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self, other: Any
    ) -> tuple[np_1darray_int64, TimedeltaArray]: ...
    def __rdivmod__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]  # ty: ignore[invalid-method-override]
        self, other: Any
    ) -> tuple[np_1darray_int64, TimedeltaArray]: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...
    def __abs__(self) -> Self: ...
    def total_seconds(self) -> np_1darray_float: ...
    def to_pytimedelta(self) -> np_1darray_object: ...
    @property
    def days(self) -> np_1darray_int64: ...
    @property
    def seconds(self) -> np_1darray_int32: ...
    @property
    def microseconds(self) -> np_1darray_int32: ...
    @property
    def nanoseconds(self) -> np_1darray_int32: ...
    @property
    def components(self) -> DataFrame: ...
    @property
    def freq(self) -> DateOffset | None: ...  # pyrefly: ignore[bad-override]
    @freq.setter  # type: ignore[override]
    def freq(  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-param-name-override]
        self, value: DateOffset
    ) -> None: ...
    def min(self, *, skipna: bool = True, **kwargs: Any) -> Timedelta | NaTType: ...
    def max(self, *, skipna: bool = True, **kwargs: Any) -> Timedelta | NaTType: ...
    def mean(self, *, skipna: bool = True, **kwargs: Any) -> Timedelta | NaTType: ...
    def median(self, *, skipna: bool = True, **kwargs: Any) -> Timedelta | NaTType: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray_td: ...
