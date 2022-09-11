from typing import (
    Callable,
    Generator,
    Generic,
    Hashable,
    Literal,
    Mapping,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy

from pandas._typing import (
    AxisType,
    NDFrameT,
    Scalar,
    npt,
)

_FrameGroupByFunc = (
    Callable[[DataFrame], Scalar]
    | Callable[[DataFrame], Series]
    | Callable[[DataFrame], DataFrame]
    | np.ufunc
)
_FrameGroupByFuncTypes = _FrameGroupByFunc | str | list[_FrameGroupByFunc | str]
_FrameGroupByFuncArgs = (
    _FrameGroupByFuncTypes | Mapping[Hashable, _FrameGroupByFuncTypes]
)

_SeriesGroupByFunc = Callable[[Series], Scalar] | Callable[[Series], Series] | np.ufunc
_SeriesGroupByFuncTypes = _SeriesGroupByFunc | str | list[_SeriesGroupByFunc | str]
_SeriesGroupByFuncArgs = (
    _SeriesGroupByFuncTypes | Mapping[Hashable, _SeriesGroupByFunc | str]
)

_Interpolation = Literal[
    "linear",
    "time",
    "index",
    "pad",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "spline",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

class Resampler(BaseGroupBy, Generic[NDFrameT]):
    def __getattr__(self, attr: str) -> SeriesGroupBy: ...
    def __iter__(self) -> Generator[tuple[Hashable, NDFrameT], None, None]: ...
    @property
    def obj(self) -> NDFrameT: ...
    @property
    def ax(self) -> Index: ...
    @overload
    def pipe(
        self: Resampler[DataFrame],
        func: Callable[..., DataFrame]
        | tuple[Callable[..., DataFrame], str]
        | Callable[..., Series]
        | tuple[Callable[..., Series], str],
        *args,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def pipe(
        self: Resampler[DataFrame],
        func: Callable[..., Scalar] | tuple[Callable[..., Scalar], str],
        *args,
        **kwargs,
    ) -> Series: ...
    @overload
    def pipe(
        self: Resampler[Series],
        func: Callable[..., Series] | tuple[Callable[..., Series], str],
        *args,
        **kwargs,
    ) -> Series: ...
    @overload
    def pipe(
        self: Resampler[Series],
        func: Callable[..., Scalar] | tuple[Callable[..., Scalar], str],
        *args,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def pipe(
        self: Resampler[Series],
        func: Callable[..., DataFrame] | tuple[Callable[..., DataFrame], str],
        *args,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self: Resampler[DataFrame],
        func: _FrameGroupByFuncArgs | None = ...,
        *args,
        **kwargs,
    ) -> Series | DataFrame: ...
    @overload
    def aggregate(
        self: Resampler[Series],
        func: _SeriesGroupByFuncArgs | None = ...,
        *args,
        **kwargs,
    ) -> Series | DataFrame: ...
    agg = aggregate
    apply = aggregate
    def transform(
        self, arg: Callable[[Series], Series], *args, **kwargs
    ) -> NDFrameT: ...
    def ffill(self, limit: int | None = ...) -> NDFrameT: ...
    def nearest(self, limit: int | None = ...) -> NDFrameT: ...
    def bfill(self, limit: int | None = ...) -> NDFrameT: ...
    def fillna(
        self,
        method: Literal["pad", "backfill", "ffill", "bfill", "nearest"],
        limit: int | None = ...,
    ) -> NDFrameT: ...
    @overload
    def interpolate(
        self,
        method: _Interpolation = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        *,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        downcast: Literal["infer"] | None = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: _Interpolation = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        inplace: Literal[False] = ...,
        limit_direction: Literal["forward", "backward", "both"] = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        downcast: Literal["infer"] | None = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def asfreq(self, fill_value: Scalar | None = ...) -> NDFrameT: ...
    def std(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def var(self, ddof: int = ..., *args, **kwargs) -> NDFrameT: ...
    def size(self) -> Series: ...
    def count(self) -> NDFrameT: ...
    def quantile(
        self,
        q: float | list[float] | npt.NDArray[np.float_] | Series[float] = ...,
        **kwargs,
    ) -> NDFrameT: ...
    def sum(
        self, _method: Literal["sum"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def prod(
        self, _method: Literal["prod"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def min(
        self, _method: Literal["min"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def max(
        self, _method: Literal["max"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def first(
        self, _method: Literal["first"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def last(
        self, _method: Literal["last"] = ..., min_count: int = ..., *args, **kwargs
    ) -> NDFrameT: ...
    def mean(self, _method: Literal["mean"] = ..., *args, **kwargs) -> NDFrameT: ...
    def sem(self, _method: Literal["sem"] = ..., *args, **kwargs) -> NDFrameT: ...
    def median(self, _method: Literal["median"] = ..., *args, **kwargs) -> NDFrameT: ...
    def ohlc(self, _method: Literal["ohlc"] = ..., *args, **kwargs) -> DataFrame: ...
    def nunique(self, _method: Literal["first"] = ..., *args, **kwargs) -> NDFrameT: ...
